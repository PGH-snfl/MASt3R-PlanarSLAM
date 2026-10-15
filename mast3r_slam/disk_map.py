"""Persistent map chunks, independent of the bounded SLAM keyframe window.

Keyframe tensors are owned by KeyframeArchive. These smaller rendering chunks
contain all valid KF points, or voxel-selected non-KF points. Coordinates remain
local to their frame; updated anchor poses move both kinds without rewriting
geometry. Files are atomically replaced before publishing their index records.
"""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import time


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)


def write_chunk(path, arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def map_arrays(frame, keyframe, cfg):
    xyz = frame.X_canon.detach().cpu().numpy().reshape(-1, 3)
    rgb = frame.uimg.detach().cpu().numpy().reshape(-1, 3)
    confidence = frame.get_average_conf().detach().cpu().numpy().reshape(-1)
    source = (np.zeros(len(xyz), np.uint8) if frame.depth_source is None
              else frame.depth_source.detach().cpu().numpy().reshape(-1))
    return select_map_arrays(xyz, rgb, confidence, source, keyframe, cfg)


def select_map_arrays(xyz, rgb, confidence, source, keyframe, cfg):
    valid = np.isfinite(xyz).all(1) & np.isfinite(confidence) & (xyz[:, 2] > 0) & (confidence > 0)
    if not keyframe:
        valid &= confidence >= cfg.get('nonkeyframe_min_conf', 1.5)
    ids = np.flatnonzero(valid)
    if not keyframe and len(ids):
        voxel = float(cfg.get('nonkeyframe_voxel_m', .03))
        limit = int(cfg.get('nonkeyframe_max_points', 6000))
        if voxel <= 0 or limit < 1:
            raise ValueError('Map voxel size and nonkeyframe point limit must be positive')
        # Highest confidence in each camera-space voxel; evenly spaced image
        # samples if still over budget, so one highly textured patch cannot win.
        ids = ids[np.argsort(-confidence[ids], kind='stable')]
        cells = np.floor(xyz[ids]/voxel).astype(np.int64)
        cells -= cells.min(0)
        spans = cells.max(0)+1
        if int(spans[0])*int(spans[1])*int(spans[2]) < np.iinfo(np.int64).max:
            # Exact mixed-radix voxel IDs avoid slow structured-array sorting.
            keys = (cells[:, 0]*spans[1]+cells[:, 1])*spans[2]+cells[:, 2]
            _, first = np.unique(keys, return_index=True)
        else:
            _, first = np.unique(cells, axis=0, return_index=True)
        ids = np.sort(ids[first])
        if len(ids) > limit:
            ids = ids[np.linspace(0, len(ids)-1, limit, dtype=np.int64)]
    return dict(xyz=xyz[ids].astype(np.float32, copy=True),
                rgb=np.rint(np.clip(rgb[ids], 0, 1)*255).astype(np.uint8),
                confidence=confidence[ids].astype(np.float32, copy=True),
                source=source[ids].astype(np.uint8, copy=True))


class DiskMapWriter:
    def __init__(self, session, cfg):
        self.root = Path(session)/'map'
        (self.root/'frames').mkdir(parents=True, exist_ok=False)
        self.cfg = dict(cfg)
        self.worker = ThreadPoolExecutor(1, thread_name_prefix='map-save')
        self.pending = deque()
        self.index = (self.root/'index.jsonl').open('a', buffering=1)
        self.pose_token = None
        self.last_frame = -1
        self.counts = dict(keyframes=0, nonkeyframes=0, points=0)
        self.closed = False
        self.wait_ms = 0.
        self.prepare_ms = 0.
        atomic_json(self.root/'format.json', dict(version=1, config=self.cfg,
                    coordinates='camera local; T_world_anchor @ T_anchor_frame',
                    keyframes='full tensors in ../keyframes; all valid points in map chunks',
                    nonkeyframes='voxel-selected xyz, rgb, confidence, source; pose and timestamp in index'))

    def _drain(self):
        while self.pending and (self.pending[0].done() or len(self.pending) >= 2):
            started = time.perf_counter()
            self.pending.popleft().result()
            self.wait_ms += 1000*(time.perf_counter()-started)

    def sync_poses(self, store):
        token = (len(store), store.pose_revision)
        if token == self.pose_token:
            self._drain()
            return
        self._drain()
        import torch
        import lietorch
        matrices = lietorch.Sim3(torch.stack(store.poses)).matrix().reshape(-1, 4, 4).numpy().tolist()
        self.pending.append(self.worker.submit(atomic_json, self.root/'poses.json',
                            dict(revision=list(token), matrices=matrices,
                                 vectors=[p.tolist() for p in store.poses])))
        self.pose_token = token

    def add_frame(self, frame, anchor_id, anchor_pose, timestamp, keyframe_id=None):
        if frame.frame_id <= self.last_frame:
            raise ValueError('Map frames must be submitted once in increasing order')
        self._drain()  # Bound CPU payloads BEFORE copying GPU geometry.
        kind = 'keyframe' if keyframe_id is not None else 'nonkeyframe'
        # Own CPU snapshots: the caller may mutate/evict the frame as soon as
        # this method returns. No CUDA work or live Frame access in the worker.
        xyz = frame.X_canon.detach().cpu().numpy().reshape(-1, 3).copy()
        rgb = frame.uimg.detach().cpu().numpy().reshape(-1, 3).copy()
        confidence = frame.get_average_conf().detach().cpu().numpy().reshape(-1).copy()
        source = (np.zeros(len(xyz), np.uint8) if frame.depth_source is None
                  else frame.depth_source.detach().cpu().numpy().reshape(-1).copy())
        pose = frame.T_WC.matrix().detach().cpu().numpy().reshape(4, 4)
        relative = np.linalg.solve(np.asarray(anchor_pose).reshape(4, 4), pose)
        filename = f'{frame.frame_id:09d}.npz'
        record = dict(frame_id=int(frame.frame_id), kind=kind, keyframe_id=keyframe_id,
                      anchor_id=int(anchor_id), relative=relative.tolist(), timestamp=float(timestamp),
                      file=filename)
        def write():
            started = time.perf_counter()
            arrays = select_map_arrays(xyz, rgb, confidence, source, keyframe_id is not None, self.cfg)
            points = arrays['xyz']
            center = (points.min(0)+points.max(0))*.5 if len(points) else np.zeros(3)
            radius = float(np.linalg.norm(points-center, axis=1).max()) if len(points) else 0.
            preview_ids = np.linspace(0, len(points)-1, min(64, len(points)), dtype=np.int64)
            preview = np.column_stack((points[preview_ids], arrays['rgb'][preview_ids]/255.,
                                       arrays['confidence'][preview_ids])).astype(np.float32)
            record.update(points=len(points), center=center.tolist(), radius=radius, preview=preview.tolist())
            self.prepare_ms += 1000*(time.perf_counter()-started)
            write_chunk(self.root/'frames'/filename, arrays)
            self.index.write(json.dumps(record)+'\n')
            self.counts[kind+'s'] += 1
            self.counts['points'] += len(points)
        self.pending.append(self.worker.submit(write))
        self.last_frame = frame.frame_id

    def close(self):
        if self.closed:
            return
        try:
            for future in self.pending:
                future.result()
            atomic_json(self.root/'summary.json', self.counts)
        finally:
            self.worker.shutdown(wait=True)
            self.index.close()
            self.closed = True


class DiskMapReader:
    """Single reader thread owns index/poses and an LRU point-chunk cache."""
    def __init__(self, session, cfg):
        self.root = Path(session)/'map'
        self.cfg = dict(cfg)
        self.entries, self.cache = [], OrderedDict()
        self.offset = 0
        self.pose_stamp = None
        self.poses = []
        self.pose_vectors = []
        self.loads = 0
        self.point_budget = int(cfg.get('view_max_points', 1000000))
        self.chunk_limit = int(cfg.get('view_max_chunks', 24))
        if self.point_budget < 1 or self.chunk_limit < 1:
            raise ValueError('Map display budgets must be positive')
        self.worker = ThreadPoolExecutor(1, thread_name_prefix='map-read')
        self.future = None

    def refresh(self):
        path = self.root/'index.jsonl'
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(self.offset)
                while True:
                    line = stream.readline()
                    if not line.endswith(b'\n'):
                        break  # Writer has not committed this index record yet.
                    self.entries.append(json.loads(line))
                    self.offset = stream.tell()
        path = self.root/'poses.json'
        if path.exists():
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if stamp != self.pose_stamp:
                data = json.loads(path.read_text())
                self.poses = np.asarray(data['matrices'], dtype=np.float32)
                self.pose_vectors = data.get('vectors', [])
                self.pose_stamp = stamp

    def _matrix(self, entry):
        idx = entry['anchor_id']
        return self.poses[idx] @ np.asarray(entry['relative'], np.float32) if idx < len(self.poses) else None

    def _load(self, entry):
        fid = entry['frame_id']
        if fid not in self.cache:
            with np.load(self.root/'frames'/entry['file'], allow_pickle=False) as data:
                self.cache[fid] = {name: data[name] for name in ('xyz', 'rgb', 'confidence', 'source')}
            self.loads += 1
        self.cache.move_to_end(fid)
        while len(self.cache) > self.chunk_limit:
            self.cache.popitem(last=False)
        return self.cache[fid]

    def visible_candidates(self, eye, view_projection=None):
        """Rank visible chunks using the latest pose table, near ones first."""
        candidates = []
        matrices = {}
        planes = None
        if view_projection is not None:
            m = np.asarray(view_projection)
            planes = np.array([m[3]+m[i]*sign for i in range(3) for sign in (1, -1)])
            planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
        for entry in self.entries:
            matrix = self._matrix(entry)
            if matrix is None or not entry['points']:
                continue
            matrices[entry['frame_id']] = matrix
            center = matrix[:3, :3] @ entry['center']+matrix[:3, 3]
            radius = entry['radius']*np.linalg.norm(matrix[:3, 0])
            if planes is not None and np.any(planes[:, :3] @ center+planes[:, 3] < -radius):
                continue
            distance = max(0., np.linalg.norm(center-eye)-radius)
            candidates.append((distance, entry['kind'] != 'keyframe', -entry['frame_id'], entry))
        candidates.sort(key=lambda row: row[:3])
        return candidates, matrices

    def snapshot(self, eye, threshold=1.5, view_projection=None, *,
                 nonkeyframes_only=False, exclude_frame_ids=()):
        self.refresh()
        candidates, matrices = self.visible_candidates(eye, view_projection)
        excluded = set(exclude_frame_ids)
        selected, count = [], 0
        for _, _, _, entry in candidates:
            if entry['frame_id'] in excluded or (nonkeyframes_only and entry['kind'] == 'keyframe'):
                continue
            if len(selected) >= self.chunk_limit:
                break
            if count+entry['points'] <= self.point_budget:
                selected.append(entry)
                count += entry['points']
        ids = {e['frame_id'] for e in selected}
        for fid in list(self.cache):
            if fid not in ids:
                del self.cache[fid]
        output = []
        for entry in selected:
            data = self._load(entry)
            good = data['confidence'] >= threshold
            xyz = data['xyz'][good]
            matrix = matrices[entry['frame_id']]
            world = xyz @ matrix[:3, :3].T+matrix[:3, 3]
            output.append(np.column_stack((world, data['rgb'][good]/255.)).astype(np.float32))
        detailed = sum(len(x) for x in output)
        # Far/over-budget regions remain visible as a disk-backed overview.
        # It is rebuilt from local previews and current poses, never frozen in
        # stale world coordinates when a loop changes the anchors.
        overview_budget = int(self.cfg.get('view_overview_points', 100000))
        remaining = [e for e in self.entries if e['frame_id'] not in ids | excluded and e['frame_id'] in matrices]
        overview = []
        total = sum(len(e['preview']) for e in remaining)
        # Global deterministic subsampling, with an exact bound on output.
        pick = np.linspace(0, total-1, min(total, max(0, overview_budget)), dtype=np.int64)
        start = 0
        for entry in remaining:
            length = len(entry['preview'])
            lo, hi = np.searchsorted(pick, [start, start+length])
            indices = pick[lo:hi]-start
            start += length
            if not len(indices):
                continue
            preview = np.asarray(entry['preview'], np.float32)[indices]
            preview = preview[preview[:, 6] >= threshold]
            matrix = matrices[entry['frame_id']]
            preview[:, :3] = preview[:, :3] @ matrix[:3, :3].T+matrix[:3, 3]
            overview.append(preview[:, :6])
        output.extend(overview)
        points = np.concatenate(output) if output else np.empty((0, 6), np.float32)
        stats = dict(frames=len(self.entries), keyframes=sum(e['kind'] == 'keyframe' for e in self.entries),
                     nonkeyframes=sum(e['kind'] == 'nonkeyframe' for e in self.entries),
                     chunks=len(selected), detailed_points=detailed, overview_points=len(points)-detailed,
                     disk_loads=self.loads)
        return points, stats

    def request(self, eye, threshold, view_projection):
        if self.future is None:
            self.future = self.worker.submit(self.snapshot, np.array(eye, copy=True), threshold,
                                             np.array(view_projection, copy=True))

    def poll(self):
        if self.future is not None and self.future.done():
            future, self.future = self.future, None
            return future.result()
        return None

    def close(self):
        try:
            if self.future is not None:
                self.future.result()
        finally:
            self.worker.shutdown(wait=True)
