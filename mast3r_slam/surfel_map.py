"""Read full organized keyframes for the original MASt3R surfel shader.

Only CPU image/geometry arrays are retained. Features are discarded immediately;
the GUI owns a separate bounded GL texture cache. All source files are read-only.
"""
from collections import OrderedDict
import numpy as np
import torch
from mast3r_slam.disk_map import DiskMapReader


class SurfelMapReader(DiskMapReader):
    def __init__(self, session, cfg):
        super().__init__(session, cfg)
        self.surfel_limit = int(cfg.get('view_surfel_keyframes', 12))
        if self.surfel_limit < 1:
            raise ValueError('Surfel keyframe limit must be positive')
        self.surfels = OrderedDict()
        self.surfel_loads = 0

    def _load_surfel(self, entry):
        fid = entry['frame_id']
        if fid not in self.surfels:
            while len(self.surfels) >= self.surfel_limit:
                self.surfels.popitem(last=False)
            path = self.root.parent/'keyframes'/f"{entry['keyframe_id']:08d}.pt"
            payload = torch.load(path, map_location='cpu', weights_only=True)
            h, w = map(int, payload['img_shape'].reshape(-1).tolist())
            xyz = payload['X_canon'].reshape(h, w, 3).numpy().astype('f4', copy=True)
            conf = (payload['C']/max(1, payload['N'])).reshape(h, w).numpy().astype('f4', copy=True)
            rgb = payload['uimg'].reshape(h, w, 3).numpy().astype('f4', copy=True)
            good = np.isfinite(xyz).all(2) & (xyz[..., 2] > 0) & np.isfinite(conf)
            conf[~good] = 0
            xyz[~good] = 0
            self.surfels[fid] = dict(frame_id=fid, shape=(h, w), xyz=xyz, confidence=conf, rgb=rgb)
            self.surfel_loads += 1
        self.surfels.move_to_end(fid)
        return self.surfels[fid]

    def snapshot(self, eye, threshold=1.5, view_projection=None):
        self.refresh()
        candidates, matrices = self.visible_candidates(eye, view_projection)
        # A wide depth bounding sphere often contains the eye even when most of
        # that frame is far away. Break this ambiguity by the capture viewpoint,
        # rather than letting the newest distant keyframes fill every slot.
        candidates.sort(key=lambda row: (
            np.linalg.norm(matrices[row[3]['frame_id']][:3, 3]-eye), -row[3]['frame_id']))
        selected = []
        for _, _, _, entry in candidates:
            if entry['kind'] != 'keyframe':
                continue
            path = self.root.parent/'keyframes'/f"{entry['keyframe_id']:08d}.pt"
            if path.is_file():
                selected.append(entry)
            if len(selected) == self.surfel_limit:
                break
        ids = {e['frame_id'] for e in selected}
        for fid in list(self.surfels):
            if fid not in ids:
                del self.surfels[fid]
        for entry in selected:
            self._load_surfel(entry)
        # Selected KFs are drawn only as surfels; other KFs remain in overview.
        points, stats = super().snapshot(eye, threshold, view_projection,
                                        nonkeyframes_only=True, exclude_frame_ids=ids)
        surfaces = [dict(self.surfels[e['frame_id']], matrix=self._matrix(e).copy()) for e in selected]
        stats.update(surfels=surfaces, surfel_keyframes=len(surfaces),
                     surfel_points=sum(int((s['confidence'] >= threshold).sum()) for s in surfaces),
                     surfel_loads=self.surfel_loads)
        return points, stats
