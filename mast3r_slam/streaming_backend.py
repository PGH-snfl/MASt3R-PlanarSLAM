"""Recent-window optimization and bounded asynchronous ASMK retrieval."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import lietorch
import torch
from mast3r_slam.config import config
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.keyframe_archive import pack_frame, unpack_frame


class AsyncRetrieval:
    def __init__(self, database, pending_limit=2):
        self.database = database
        # Encoder/projection stays on GPU; codebook quantization and ASMK index
        # have a single CPU owner. Never run competing GPU inference in a worker.
        database.centroids = database.centroids.cpu()
        database.query_device = 'cpu'
        self.worker = ThreadPoolExecutor(1, thread_name_prefix='asmk-search')
        self.pending = deque()
        self.limit = pending_limit
        self.ids = []

    def ready(self):
        return len(self.pending) < self.limit

    def submit(self, frame, tag, add):
        if not self.ready():
            return False
        descriptors = self.database.prep_features(frame.feat)[0].detach().cpu().numpy().copy()
        cfg = config['retrieval']
        query = not add or int(tag) % max(1, config.get("streaming", {}).get("search_every", 2)) == 0
        def work():
            with torch.no_grad():
                result = self.database.update_descriptors(descriptors, add, cfg['k'], cfg['min_thresh'], query=query)
            mapped = [self.ids[int(i)] for i in result]
            if add:
                self.ids.append(int(tag))
            return tag, add, mapped
        self.pending.append(self.worker.submit(work))
        return True

    def poll(self):
        out = []
        while self.pending and self.pending[0].done():
            out.append(self.pending.popleft().result())
        return out

    def close(self):
        self.worker.shutdown(wait=True)
        for future in self.pending:
            future.result()


class TemporaryFrames:
    """Test a relocalization frame without evicting or indexing a real keyframe."""
    def __init__(self, base, frame):
        self.base, self.frame, self.idx = base, frame, len(base)

    def __getitem__(self, idx):
        return self.frame if int(idx) == self.idx else self.base[idx]

    def update_T_WCs(self, poses, indices):
        for idx, pose in zip(indices, poses):
            if int(idx) == self.idx:
                self.frame.T_WC = lietorch.Sim3(pose.data.clone())
            else:
                self.base.update_T_WCs(pose[None], [int(idx)])


class StreamingBackend:
    def __init__(self, model, store, K, retrieval, device):
        self.model, self.store, self.K, self.device = model, store, K, device
        self.offline = config.get('streaming', {}).get('offline', False)
        self.graph = FactorGraph(model, store, K, device)
        self.search = AsyncRetrieval(retrieval)
        self.candidates = deque()
        self.unindexed = deque()
        self.reloc_pending = False
        self.next_reloc = 0.
        self.last_verified = None
        from mast3r_slam.pose_graph import GlobalPoseGraph
        self.global_graph = GlobalPoseGraph(store, config.get('global_pose_graph', {}))
        self.search_interval = max(1, config.get('streaming', {}).get('search_every', 2))

    def _prune(self, anchor=None):
        # At most one historical anchor in the dense graph, even if the I/O
        # cache retains two. Prune immediately after a cache eviction.
        keep = set(self.store.recent)
        if anchor is not None:
            keep.add(anchor)
        self.graph.retain_frames(keep & self.store.resident_ids)

    def _solve(self, graph=None):
        graph = graph or self.graph
        if config['use_calib']:
            graph.solve_GN_calib()
        else:
            graph.solve_GN_rays()

    def new_keyframe(self, idx, link_previous=True):
        self._prune()
        if link_previous and idx > 0 and idx-1 in self.store.recent:
            if self.graph.add_factors([idx-1], [idx], config['local_opt']['min_match_frac']):
                self.global_graph.add(self.graph, idx-1, idx, 'temporal')
                self.store.record('edge', source=idx-1, target=idx, kind='temporal')
                self._solve()
        # Keep at most one unsubmitted frame ID; defer additional new keyframes
        # until the CPU index catches up (tracking continues against last KF).
        self.unindexed.append(idx)
        self._submit_index()

    def can_add(self):
        return not self.unindexed and self.search.ready()

    def _submit_index(self):
        if self.unindexed and self.search.ready():
            idx = self.unindexed[0]
            if self.search.submit(self.store[idx], idx, True):
                self.unindexed.popleft()

    def tick(self, frame, relocating=False):
        """Live: one ready candidate. Offline: wait and verify all retrieved candidates."""
        self.global_graph.poll()
        if self.offline:
            self._submit_index()
            if relocating and not self.reloc_pending and not self.candidates:
                self.reloc_pending = self.search.submit(frame, None, False)
            for future in self.search.pending:
                future.result()  # Backpressure: finish retrieval before advancing input.
        for tag, added, candidates in self.search.poll():
            if not added:
                self.reloc_pending = False
            relevant = (not added and relocating) or (added and not relocating and
                        tag in self.store.recent and tag % self.search_interval == 0)
            if relevant and not self.candidates:
                # Offline candidates are loaded sequentially within the same cache.
                chosen = [i for i in candidates if not added or i < tag-1]
                for idx in (chosen if self.offline else chosen[:self.store.cache_limit]):
                    self.candidates.append((None if not added else tag, idx))
                    self.store.prefetch(idx)
        self._submit_index()
        if not self.offline and relocating and not self.reloc_pending and not self.candidates and self.search.ready() and time.monotonic() >= self.next_reloc:
            self.reloc_pending = self.search.submit(frame, None, False)
            self.next_reloc = time.monotonic() + 1.0
        if self.offline:
            while self.candidates:
                if self._verify_next(frame, relocating):
                    return True
            return False
        return self._verify_next(frame, relocating)

    def _verify_next(self, frame, relocating):
        if not self.candidates:
            return False
        query, candidate = self.candidates[0]
        if (query is None) != relocating or (query is not None and query not in self.store.recent):
            self.candidates.popleft()
            if not self.candidates:
                self.store.discard_prefetches()
            return False
        if not self.store.prefetch(candidate):
            if self.offline:
                raise RuntimeError('Offline candidate cache cannot schedule a load')
            return False
        if self.offline and candidate in self.store.loads:
            self.store.loads[candidate].result()
        if not self.store.poll(candidate):
            return False
        self.candidates.popleft()
        self._prune(candidate)
        if relocating:
            temporary = TemporaryFrames(self.store, frame)
            graph = FactorGraph(self.model, temporary, self.K, self.device)
            # Candidate is the pinned (lowest global ID) reference. Geometric
            # verification uses today's frame, not the older retrieval query.
            ok = graph.add_factors([candidate], [len(self.store)], config['reloc']['min_match_frac'], is_reloc=True)
            if not ok:
                return False
            frame.T_WC = lietorch.Sim3(self.store[candidate].T_WC.data.clone())
            self._solve(graph)
            if not torch.isfinite(frame.T_WC.data).all():
                raise RuntimeError('Non-finite relocalization pose')
            # Registration needs the old candidate before append can evict it.
            self.global_graph.add(graph, candidate, len(self.store), 'relocalization')
            idx = self.store.append(frame)
            self.graph = FactorGraph(self.model, self.store, self.K, self.device)
            # Relocalization constraints have already solved the new pose. Start
            # a fresh temporal window rather than linking across tracking loss.
            self.store.record('edge', source=candidate, target=idx, kind='relocalization')
            self.new_keyframe(idx, link_previous=False)
            self.candidates.clear()
            self.store.discard_prefetches()
            print(f'Relocalized against archived/resident keyframe {candidate}', flush=True)
            return True
        if self.graph.add_factors([candidate], [query], config['local_opt']['min_match_frac'], is_reloc=True):
            self.global_graph.add(self.graph, candidate, query, 'loop')
            self.store.record('edge', source=candidate, target=query, kind='loop')
            self._solve()
            self.last_verified = (query, candidate)
            print(f'Loop verified: {query} -> {candidate}', flush=True)
        return False

    def publish(self, view, states):
        ids = list(self.store.recent)
        # Same lock order as the viewer. CPU viewer buffers do not duplicate
        # the full window on CUDA; only bounded GL textures are uploaded.
        with view.lock:
            for slot, idx in enumerate(ids):
                payload = self.store[idx]
                view[slot] = unpack_frame(pack_frame(payload), 'cpu')
            view.n_size.value = len(ids)
            if hasattr(view, 'history') and view.history_revision.value != self.store.revision:
                count = len(self.store.preview)
                view.history[:count] = torch.from_numpy(self.store.preview)
                view.history_count.value = count
                view.history_revision.value = self.store.revision
            remap = {idx: slot for slot, idx in enumerate(ids)}
            pairs = [(remap[i], remap[j]) for i, j in zip(self.graph.ii.cpu().tolist(), self.graph.jj.cpu().tolist()) if i in remap and j in remap]
            with states.lock:
                states.edges_ii[:] = [i for i, j in pairs]
                states.edges_jj[:] = [j for i, j in pairs]

    def close(self):
        try:
            try:
                self.search.close()
            finally:
                self.global_graph.close()
        finally:
            self.store.close()
