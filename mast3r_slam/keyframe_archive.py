"""Bounded live keyframes with stable IDs, durable eviction and async reload."""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
import json
from pathlib import Path
import threading

import lietorch
import numpy as np
import torch
from mast3r_slam.frame import Frame


CPU_FIELDS = {'uimg', 'depth', 'raw_depth', 'depth_source', 'feature_points', 'plane_depth_delta', 'surface_depth_delta'}


def pack_frame(frame):
    out = {}
    for field in fields(Frame):
        value = getattr(frame, field.name)
        if field.name == 'T_WC':
            value = value.data
        out[field.name] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    return out


def unpack_frame(payload, device):
    values = dict(payload)
    for name, value in values.items():
        if torch.is_tensor(value):
            values[name] = value.to('cpu' if name in CPU_FIELDS else device)
    values['T_WC'] = lietorch.Sim3(values['T_WC'])
    return Frame(**values)


def atomic_save(payload, path):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


class KeyframeArchive:
    """Only the main thread owns GPU frames. Workers own CPU data and files.

    len() is the number of permanent global IDs, NOT the resident window size.
    Archived frames must be prefetched before __getitem__; tracking never does
    synchronous disk reads. A full writer queue applies backpressure rather than
    silently dropping geometry or growing RAM indefinitely.
    """
    def __init__(self, directory, recent=8, cache=2, device='cuda:0', preview_limit=150000, save_on_append=False):
        if recent < 2 or cache < 1:
            raise ValueError('recent >= 2 and candidate_cache >= 1 required')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'keyframes').mkdir()
        self.recent_limit, self.cache_limit = recent, cache
        self.device, self.lock = device, threading.RLock()
        self.recent, self.cache = OrderedDict(), OrderedDict()
        self.poses = []
        self.K = None
        self.writer = ThreadPoolExecutor(1, thread_name_prefix='keyframe-save')
        self.reader = ThreadPoolExecutor(1, thread_name_prefix='keyframe-load')
        self.writes = deque()
        self.pending_writes, self.loads = {}, {}
        self.preview_limit = preview_limit
        self.preview = np.empty((0, 6), dtype=np.float32)
        self.preview_local = np.empty((0, 3), dtype=np.float32)
        self.preview_anchors = np.empty(0, dtype=np.int64)
        self.preview_seen = 0
        self.rng = np.random.default_rng(0)
        self.revision = 0
        self.pose_revision = 0
        self.closed = False
        self.save_on_append = save_on_append
        self.events = open(self.directory / 'events.jsonl', 'a', buffering=1)
        self.record('format', version=1, recent=recent, candidate_cache=cache)

    def record(self, event, **values):
        self.events.write(json.dumps(dict(event=event, **values)) + '\n')

    def __len__(self):
        return len(self.poses)

    @property
    def resident_ids(self):
        return set(self.recent) | set(self.cache)

    def __getitem__(self, idx):
        idx = int(idx)
        if idx in self.recent:
            return self.recent[idx]
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        raise KeyError(f'Keyframe {idx} is archived; prefetch and poll it first')

    def __setitem__(self, idx, frame):
        idx = int(idx)
        target = self.recent if idx in self.recent else self.cache
        if idx not in target:
            raise KeyError(idx)
        target[idx] = frame
        self.poses[idx] = frame.T_WC.data.detach().cpu().clone()

    def last_keyframe(self):
        return self[len(self)-1] if len(self) else None

    def set_intrinsics(self, K):
        self.K = K

    def get_intrinsics(self):
        return self.K

    def path(self, idx):
        return self.directory / 'keyframes' / f'{idx:08d}.pt'

    def _save(self, idx, frame):
        # Two CPU payloads at most; make room BEFORE creating the next copy.
        while self.writes and (self.writes[0][1].done() or len(self.writes) >= 2):
            old_idx, future = self.writes.popleft()
            future.result()  # Surface disk-full/permission errors immediately.
            if self.pending_writes.get(old_idx) is future:
                del self.pending_writes[old_idx]
        future = self.writer.submit(atomic_save, pack_frame(frame), self.path(idx))
        self.writes.append((idx, future))
        self.pending_writes[idx] = future

    def _preview_add(self, frame, anchor):
        if self.preview_limit <= 0:
            return
        # A bounded reservoir for display only. Full geometry remains on disk.
        xyz = frame.X_canon.detach()[::64]
        valid = (frame.C.detach().reshape(-1)[::64] > 0) & torch.isfinite(xyz).all(-1) & (xyz[:, 2] > 0)
        local = xyz[valid].cpu().numpy().reshape(-1, 3)
        xyz = (frame.T_WC * xyz)[valid].cpu().numpy().reshape(-1, 3)
        rgb = frame.uimg.reshape(-1, 3)[::64][valid.cpu()].cpu().numpy()
        points = np.concatenate((xyz, rgb), axis=1).astype(np.float32)
        if len(self.preview) < self.preview_limit:
            take = min(self.preview_limit-len(self.preview), len(points))
            self.preview = np.concatenate((self.preview, points[:take]))
            self.preview_local = np.concatenate((self.preview_local, local[:take]))
            self.preview_anchors = np.r_[self.preview_anchors, np.full(take, anchor)]
            self.preview_seen += take
            points = points[take:]
            local = local[take:]
        for point, camera_point in zip(points, local):
            self.preview_seen += 1
            slot = int(self.rng.integers(self.preview_seen))
            if slot < self.preview_limit:
                self.preview[slot] = point
                self.preview_local[slot] = camera_point
                self.preview_anchors[slot] = anchor
        self.revision += 1

    def append(self, frame):
        if len(self.recent) >= self.recent_limit:
            idx, old = next(iter(self.recent.items()))
            self._save(idx, old)
            self._preview_add(old, idx)
            del self.recent[idx]
        idx = len(self)
        self.recent[idx] = frame
        self.poses.append(frame.T_WC.data.detach().cpu().clone())
        self.record('keyframe', id=idx, frame_id=int(frame.frame_id), pose=self.poses[idx].tolist())
        if self.save_on_append:
            self._save(idx, frame)
        return idx

    def update_T_WCs(self, poses, indices):
        updates = {}
        for idx, pose in zip(indices, poses):
            idx = int(idx)
            self[idx].T_WC = lietorch.Sim3(pose.data.clone())
            self.poses[idx] = pose.data.detach().cpu().clone()
            updates[str(idx)] = self.poses[idx].tolist()
        if updates:
            # World-plane history must not survive a changed map coordinate fit.
            self.pose_revision += 1
            self.record('poses', poses=updates)
            self._refresh_preview([int(i) for i in updates])

    def _refresh_preview(self, indices):
        for idx in indices:
            mask = self.preview_anchors == idx
            if mask.any():
                T = lietorch.Sim3(self.poses[idx]).matrix().numpy().reshape(4, 4)
                self.preview[mask, :3] = self.preview_local[mask] @ T[:3, :3].T + T[:3, 3]
        self.revision += 1

    def apply_global_poses(self, pose_vectors):
        """Update even evicted poses without reading dense keyframes from disk."""
        data = np.asarray(pose_vectors, dtype=np.float32).reshape(-1, 8)
        if (len(data) != len(self) or not np.isfinite(data).all() or (data[:, 7] <= 0).any()
                or not np.allclose(np.linalg.norm(data[:, 3:7], axis=1), 1., atol=1e-4)):
            raise ValueError('Invalid global poses')
        for idx, value in enumerate(data):
            pose = torch.from_numpy(value.copy()).reshape(1, 8)
            self.poses[idx] = pose
            if idx in self.resident_ids:
                self[idx].T_WC = lietorch.Sim3(pose.to(self.device))
        self.pose_revision += 1
        self._refresh_preview(range(len(self)))
        self.record('poses', poses={str(i): p.tolist() for i, p in enumerate(self.poses)}, kind='global')

    def prefetch(self, idx):
        idx = int(idx)
        if idx in self.resident_ids or idx in self.loads:
            return True
        if not 0 <= idx < len(self) or len(self.loads) >= self.cache_limit:
            return False
        write = self.pending_writes.get(idx)
        path = self.path(idx)
        def read():
            if write is not None:
                write.result()
            return torch.load(path, map_location='cpu', weights_only=True)
        self.loads[idx] = self.reader.submit(read)
        return True

    def poll(self, idx):
        idx = int(idx)
        if idx in self.resident_ids:
            return True
        future = self.loads.get(idx)
        if future is None or not future.done():
            return False
        payload = future.result()
        del self.loads[idx]
        if len(self.cache) >= self.cache_limit:
            self.cache.popitem(last=False)
        payload['T_WC'] = self.poses[idx].clone()
        self.cache[idx] = unpack_frame(payload, self.device)
        return True

    def discard_prefetches(self):
        for future in self.loads.values():
            future.cancel()
        self.loads.clear()

    def close(self):
        if self.closed:
            return
        try:
            for idx, frame in self.recent.items():
                self._save(idx, frame)
            for _, future in self.writes:
                future.result()
            manifest = dict(version=1, keyframes=len(self), poses=[p.tolist() for p in self.poses],
                            intrinsics=None if self.K is None else self.K.cpu().tolist(),
                            note='Full keyframes; recent dense optimization plus optional CPU pose-only graph; not full-history bundle adjustment')
            temp = self.directory / 'manifest.tmp'
            temp.write_text(json.dumps(manifest))
            temp.replace(self.directory / 'manifest.json')
        finally:
            self.writer.shutdown(wait=True)
            self.reader.shutdown(wait=True, cancel_futures=True)
            self.events.close()
            self.closed = True
