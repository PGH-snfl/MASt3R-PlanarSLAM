"""Short, raw-sensor-only surface denoising with pose/visibility checks.

History never contains corrected or model-filled depths. No holes are filled,
no confidence is increased, and discontinuities/photometric changes are skipped.
"""
from collections import deque
import cv2
import numpy as np
from mast3r_slam.temporal_plane_depth import TemporalPlaneDepth


class TemporalSurfaceDepth:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.window = int(cfg.get('window', 3))
        if not 2 <= self.window <= 5:
            raise ValueError('surface window must be between 2 and 5')
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.window)
        self.last = None
        self.rays = self.geometry = None

    def correct(self, raw, depth, source, confidence, rgb, K, pose, frame_id,
                epoch=0, timestamp=None, tracking_ok=True, total_cap=.01):
        raw = np.asarray(raw, np.float32)
        depth = np.asarray(depth, np.float32)
        K, pose = np.asarray(K, np.float32), np.asarray(pose, np.float32).reshape(4, 4)
        out = depth.copy()
        stats = dict(pixels=0, eligible=0, history=len(self.history), reset='', mean_change_mm=0., max_change_mm=0.)
        if not self.cfg.get('enabled', False):
            return out, np.zeros_like(raw), stats
        parsed = TemporalPlaneDepth._pose(pose)
        if not tracking_ok or parsed is None or timestamp is None or not np.isfinite(timestamp):
            self.reset()
            stats['reset'] = 'tracking_or_pose_invalid'
            return out, np.zeros_like(raw), stats
        geometry = (raw.shape, tuple(K.ravel()))
        reason = ''
        if self.last is not None:
            old_id, old_epoch, old_time, old_pose = self.last
            if frame_id == old_id and epoch == old_epoch:
                return out, np.zeros_like(raw), stats
            old = TemporalPlaneDepth._pose(old_pose)
            angle = np.arccos(np.clip((np.trace(old[1].T @ parsed[1])-1)/2, -1, 1))
            if epoch != old_epoch:
                reason = 'map_pose_updated'
            elif frame_id <= old_id or not 0 < timestamp-old_time <= self.cfg.get('max_gap_s', 1.):
                reason = 'time_or_frame_reset'
            elif (np.linalg.norm(parsed[2]-old[2]) > .25 or angle > np.deg2rad(15)
                  or abs(np.log(parsed[0]/old[0])) > .02):
                reason = 'pose_jump'
            elif geometry != self.geometry:
                reason = 'geometry_changed'
        if reason:
            self.reset()
            stats['reset'] = reason
        if self.rays is None:
            y, x = np.mgrid[:raw.shape[0], :raw.shape[1]].astype(np.float32)
            self.rays = np.stack(((x-K[0, 2])/K[0, 0], (y-K[1, 2])/K[1, 1], np.ones_like(x)), -1)
            self.geometry = geometry
        valid = np.isfinite(raw) & (raw > 0) & (source == 1) & (confidence >= 2.5)
        # All neighbours must be real, close-depth measurements. A 5x5 erosion
        # protects both sides of holes/silhouettes, including the image border.
        clean = np.where(valid, raw, 0)
        hi = cv2.dilate(clean, np.ones((3, 3), np.uint8))
        lo = cv2.erode(clean, np.ones((3, 3), np.uint8))
        stable = valid & ((hi-lo) < self.cfg.get('edge_threshold_m', .03))
        stable = cv2.erode(stable.astype('uint8'), np.ones((5, 5), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        rgb = np.asarray(rgb, np.float32)
        ids = np.flatnonzero(stable)
        stats['eligible'] = len(ids)
        current = self.rays.reshape(-1, 3)[ids]*raw.ravel()[ids, None]
        estimates = []
        for hist in self.history:
            transform = np.linalg.inv(hist['pose']) @ pose
            p = current @ transform[:3, :3].T + transform[:3, 3]
            z = p[:, 2]
            safe_z = np.maximum(z, 1e-6)
            u = (p[:, 0]/safe_z*K[0, 0]+K[0, 2]).astype('f4')[:, None]
            v = (p[:, 1]/safe_z*K[1, 1]+K[1, 2]).astype('f4')[:, None]
            # OpenCV remap has a SHRT_MAX dimension limit: use a 2D grid of
            # queries instead of an N x 1 image for full-resolution frames.
            count = len(ids)
            if not count:
                continue
            width = min(1024, count)
            padded = ((count+width-1)//width)*width
            uu = np.pad(u.ravel(), (0, padded-count), constant_values=-1).reshape(-1, width)
            vv = np.pad(v.ravel(), (0, padded-count), constant_values=-1).reshape(-1, width)
            def sample(a):
                result = cv2.remap(a, uu, vv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                return result.reshape((-1,)+a.shape[2:])[:count]
            observed = sample(hist['raw'])
            mask = sample(hist['stable'].astype('f4')) > .999
            color_error = np.max(np.abs(sample(hist['rgb'])-rgb.reshape(-1, 3)[ids]), axis=1)
            mask &= (z > 0) & (color_error < self.cfg.get('color_threshold', .08))
            gate = self.cfg.get('agreement_m', .02)
            mask &= np.abs(observed-z) <= gate
            previous = np.column_stack(((u.ravel()-K[0, 2])/K[0, 0]*observed,
                                        (v.ravel()-K[1, 2])/K[1, 1]*observed, observed))
            inverse = np.linalg.inv(transform)
            predicted = previous @ inverse[:3, :3].T + inverse[:3, 3]
            candidate = predicted[:, 2]
            mask &= np.isfinite(candidate) & (np.abs(candidate-raw.ravel()[ids]) <= gate)
            estimates.append(np.where(mask, candidate, np.nan))
        if len(estimates) >= 2 and len(ids):
            stack = np.array(estimates, dtype=np.float32)
            counts = np.isfinite(stack).sum(0)
            mean = np.nansum(stack, axis=0)/np.maximum(counts, 1)
            variance = np.nansum((stack-mean)**2, axis=0)/np.maximum(counts, 1)
            use = (counts >= 2) & (variance <= self.cfg.get('max_std_m', .008)**2)
            # Preserve areas already handled by the independent plane denoiser.
            use &= np.abs(depth.ravel()[ids]-raw.ravel()[ids]) < 1e-7
            delta = (mean-raw.ravel()[ids])*self.cfg.get('strength', .5)
            cap = min(total_cap, self.cfg.get('max_correction_m', .004))
            delta = np.clip(delta, -cap, cap)
            use &= np.abs(delta) > self.cfg.get('deadband_m', .0003)
            out.ravel()[ids[use]] += delta[use]
        self.history.append(dict(raw=clean.copy(), stable=stable, rgb=rgb.copy(), pose=pose.copy()))
        self.last = (frame_id, epoch, timestamp, pose.copy())
        change = out-depth
        affected = change != 0
        stats.update(history=len(self.history), pixels=int(affected.sum()))
        if affected.any():
            stats.update(mean_change_mm=float(np.abs(change[affected]).mean()*1000),
                         max_change_mm=float(np.abs(change[affected]).max()*1000))
        return out, change, stats

    def apply_frame(self, frame, epoch=0, timestamp=None, tracking_ok=True, total_cap=.01):
        if getattr(frame, '_surface_depth_done', False):
            return
        frame._surface_depth_done = True
        if frame.raw_depth is None or frame.depth_source is None:
            self.reset()
            return
        import torch
        corrected, delta, stats = self.correct(
            frame.raw_depth.numpy(), frame.depth.numpy(), frame.depth_source.numpy(),
            frame.get_average_conf().detach().cpu().numpy().reshape(frame.depth.shape),
            frame.uimg.numpy(), frame.K.detach().cpu().numpy(),
            frame.T_WC.matrix().detach().cpu().numpy(), frame.frame_id, epoch, timestamp, tracking_ok, total_cap)
        frame.surface_depth_stats = stats
        frame.surface_depth_delta = torch.from_numpy(delta)
        if stats['pixels']:
            from mast3r_slam.geometry import backproject, get_pixel_coords
            frame.depth = torch.from_numpy(corrected)
            xyz = frame.X_canon
            uv = get_pixel_coords(1, frame.depth.shape, device=xyz.device, dtype=xyz.dtype).reshape(-1, 2)
            frame.X_canon = backproject(uv, frame.depth.to(xyz).reshape(-1, 1), frame.K)
            if frame.feature_points is not None and len(frame.feature_points):
                points = frame.feature_points.clone()
                K = frame.K.detach().cpu()
                px = (points[:, 0]/points[:, 2]*K[0, 0]+K[0, 2]).round().long()
                py = (points[:, 1]/points[:, 2]*K[1, 1]+K[1, 2]).round().long()
                good = (points[:, 2] > 0) & (px >= 0) & (px < corrected.shape[1]) & (py >= 0) & (py < corrected.shape[0])
                points[good, :3] *= (frame.depth[py[good], px[good]]/points[good, 2])[:, None]
                frame.feature_points = points
