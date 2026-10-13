"""Bounded pose-compensated plane history, inspired by Dense Planar SLAM.

Only uncorrected measured plane fits enter history. This is a short-lived
denoiser, not sensor calibration or a globally persistent plane map. Tracking
loss, clock/pose jumps and backend pose revisions invalidate the history.
"""
from collections import deque
import numpy as np


class TemporalPlaneDepth:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.window = int(cfg.get('window', 8))
        self.minimum = int(cfg.get('min_observations', 3))
        self.limit = int(cfg.get('max_tracks', 32))
        if not 2 <= self.minimum <= self.window or self.limit < 1:
            raise ValueError('Temporal planes require 2 <= min_observations <= window and max_tracks >= 1')
        if not 0 <= cfg.get('strength', .6) <= 1 or cfg.get('max_correction_m', .003) <= 0:
            raise ValueError('Invalid temporal plane strength/correction cap')
        self.tracks = {}
        self.next_id = 0
        self.reset()

    def reset(self):
        self.tracks.clear()
        self.last_frame = self.last_pose = self.last_timestamp = self.epoch = None

    @staticmethod
    def _pose(matrix):
        matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        scale = np.linalg.norm(matrix[:3, 0])
        if not np.isfinite(matrix).all() or scale < 1e-6:
            return None
        rotation = matrix[:3, :3]/scale
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
                or np.linalg.det(rotation) < 0 or not np.allclose(matrix[3], [0, 0, 0, 1])):
            return None
        return scale, rotation, matrix[:3, 3]

    @staticmethod
    def _estimate(history):
        weights = np.array([x['weight'] for x in history])
        weights /= weights.sum()
        normals = np.array([x['normal'] for x in history])
        normal = weights @ normals
        length = np.linalg.norm(normal)
        offset = float(weights @ np.array([x['offset'] for x in history]))/length
        return normal/length, offset

    def correct(self, depth, spatial_delta, K, observations, pose, frame_id,
                epoch=0, timestamp=None, tracking_ok=True, total_cap=.01):
        z = np.asarray(depth, dtype=np.float32)
        out = z.copy()
        total_delta = np.asarray(spatial_delta, dtype=np.float32).copy()
        stats = dict(matched=0, used=0, pixels=0, tracks=0, mean_change_mm=0.,
                     max_change_mm=0., reset='', track_ids=[])
        if not self.cfg.get('enabled', False):
            return out, total_delta, stats
        parsed = self._pose(pose)
        if not tracking_ok or parsed is None:
            self.reset()
            stats['reset'] = 'tracking_lost' if not tracking_ok else 'invalid_pose'
            return out, total_delta, stats
        # Idempotence even if a caller accidentally submits the same frame twice.
        if self.last_frame == frame_id and self.epoch == epoch:
            stats['tracks'] = len(self.tracks)
            return out, total_delta, stats
        reason = ''
        if self.epoch is not None and self.epoch != epoch:
            reason = 'map_pose_updated'
        elif self.last_frame is not None and frame_id <= self.last_frame:
            reason = 'frame_reset'
        elif timestamp is not None and (not np.isfinite(timestamp) or
                (self.last_timestamp is not None and
                 not 0 < timestamp-self.last_timestamp <= self.cfg.get('max_gap_s', 2.))):
            reason = 'time_gap'
        if self.last_pose is not None:
            old = self._pose(self.last_pose)
            scale, rotation, translation = parsed
            angle = np.arccos(np.clip((np.trace(old[1].T @ rotation)-1)/2, -1, 1))
            if (np.linalg.norm(translation-old[2]) > self.cfg.get('max_pose_translation_m', .25)
                    or angle > np.deg2rad(self.cfg.get('max_pose_rotation_deg', 15.))
                    or abs(np.log(scale/old[0])) > .02):
                reason = 'pose_jump'
        if reason:
            self.reset()
            stats['reset'] = reason
        self.epoch, self.last_frame = epoch, frame_id
        self.last_pose = np.asarray(pose).copy()
        self.last_timestamp = timestamp if timestamp is not None and np.isfinite(timestamp) else None
        self.tracks = {i: t for i, t in self.tracks.items()
                       if frame_id-t['last'] <= self.cfg.get('max_age_frames', 15)}
        scale, rotation, translation = parsed
        h, w = z.shape
        K = np.asarray(K)
        used = set()
        changed = np.zeros(z.size, dtype=bool)
        for obs in observations:
            n = np.asarray(obs['normal'], dtype=np.float64)
            d = float(obs['offset'])
            center = np.asarray(obs['center'], dtype=np.float64)
            if not np.isfinite(np.r_[n, d, center, obs['rms'], obs['radius']]).all():
                continue
            world_n = rotation @ n
            world_d = scale*d-world_n @ translation
            world_c = scale*(rotation @ center)+translation
            # A configurable noise prior, NOT a factory-calibrated D457 model.
            sigma = max(float(obs['rms']), self.cfg.get('noise_floor_m', .001)
                        + self.cfg.get('noise_depth2', .001)*center[2]**2)
            observation = dict(normal=world_n, offset=world_d, center=world_c,
                               radius=scale*obs['radius'], weight=1/max((scale*sigma)**2, 1e-12))
            candidates = []
            for ident, track in self.tracks.items():
                if ident in used:
                    continue
                prior_n, prior_d = self._estimate(track['history'])
                sign = 1 if world_n @ prior_n >= 0 else -1
                angle = np.arccos(np.clip(abs(world_n @ prior_n), 0, 1))
                # Compare distance at the patch center, not at the world origin.
                distance = abs(prior_n @ world_c+prior_d)
                latest = track['history'][-1]
                separation = np.linalg.norm(world_c-latest['center'])
                overlap = self.cfg.get('overlap_ratio', .7)*(observation['radius']+latest['radius'])
                angle_gate = np.deg2rad(self.cfg.get('match_angle_deg', 2.))
                distance_gate = self.cfg.get('match_distance_m', .02)
                if angle < angle_gate and distance < distance_gate and separation < overlap:
                    candidates.append((angle/angle_gate+distance/distance_gate,
                                       ident, sign, prior_n, prior_d))
            candidates.sort(key=lambda x: x[0])
            # Similar candidates are ambiguous: don't merge adjacent surfaces.
            ambiguous = len(candidates) > 1 and candidates[1][0] < candidates[0][0]+.25
            if not candidates or ambiguous:
                if len(self.tracks) >= self.limit:
                    oldest = min(self.tracks, key=lambda i: self.tracks[i]['last'])
                    del self.tracks[oldest]
                ident = self.next_id
                self.next_id += 1
                self.tracks[ident] = dict(history=deque([observation], maxlen=self.window), last=frame_id)
            else:
                _, ident, sign, prior_n, prior_d = candidates[0]
                track = self.tracks[ident]
                observation['normal'] *= sign
                observation['offset'] *= sign
                stats['matched'] += 1
                history = track['history']
                distances = np.array([x['normal'] @ world_c+x['offset'] for x in history])
                stable = np.std(distances) <= self.cfg.get('max_history_std_m', .008)
                if len(history)+1 >= self.minimum and stable:
                    indices = obs['indices']
                    indices = indices[~changed[indices]]
                    yy, xx = np.divmod(indices, w)
                    rays = np.column_stack(((xx-K[0, 2])/K[0, 0],
                                            (yy-K[1, 2])/K[1, 1], np.ones(len(indices))))
                    # Reproject prior into TODAY's camera before denoising.
                    camera_n = rotation.T @ prior_n
                    camera_d = (prior_d+prior_n @ translation)/scale
                    dot = rays @ camera_n
                    current_dot = rays @ n
                    valid = ((np.abs(dot)/np.linalg.norm(rays, axis=1) >= .3)
                             & (np.abs(current_dot) > 1e-6))
                    old_z = np.divide(-camera_d, dot, out=np.zeros_like(dot), where=valid)
                    fit_z = np.divide(-d, current_dot, out=np.zeros_like(dot), where=valid)
                    shift = old_z-fit_z
                    valid &= ((old_z > 0) & (fit_z > 0) & np.isfinite(shift)
                              & (np.abs(shift) <= self.cfg.get('max_target_difference_m', .015)))
                    gain = self.cfg.get('strength', .6)
                    change = np.clip(gain*shift, -self.cfg.get('max_correction_m', .003),
                                     self.cfg.get('max_correction_m', .003))
                    valid &= np.abs(change) > self.cfg.get('deadband_m', .0002)
                    indices, change = indices[valid], change[valid]
                    previous = total_delta.ravel()[indices].copy()
                    updated = np.clip(previous+change, -total_cap, total_cap)
                    out.ravel()[indices] += (updated-previous).astype(np.float32)
                    total_delta.ravel()[indices] = updated
                    changed[indices] = True
                    stats['used'] += int(len(indices) > 0)
                # Raw fit only; never append a corrected plane or pixel depth.
                history.append(observation)
                track['last'] = frame_id
            used.add(ident)
            stats['track_ids'].append(ident)
        diff = out-z
        mask = diff != 0
        stats['tracks'] = len(self.tracks)
        stats['pixels'] = int(mask.sum())
        if mask.any():
            stats['mean_change_mm'] = float(np.abs(diff[mask]).mean()*1000)
            stats['max_change_mm'] = float(np.abs(diff[mask]).max()*1000)
        return out, total_delta, stats

    def apply_frame(self, frame, epoch=0, timestamp=None, tracking_ok=True, total_cap=.01):
        """Called after visual tracking, before keyframe insertion/graph building."""
        if getattr(frame, '_temporal_depth_done', False):
            return
        frame._temporal_depth_done = True
        observations = getattr(frame, '_plane_observations', [])
        # Masks have frame lifetime only, and must never live in the KF archive.
        frame._plane_observations = []
        if not tracking_ok:
            self.reset()
        if frame.depth is None or frame.plane_depth_delta is None:
            return
        import torch
        corrected, delta, stats = self.correct(
            frame.depth.cpu().numpy(), frame.plane_depth_delta.cpu().numpy(), frame.K.cpu().numpy(),
            observations, frame.T_WC.matrix().detach().cpu().numpy().reshape(4, 4),
            frame.frame_id, epoch, timestamp, tracking_ok, total_cap)
        frame.temporal_depth_stats = stats
        if stats['pixels']:
            from mast3r_slam.geometry import backproject, get_pixel_coords
            frame.depth = torch.from_numpy(corrected)
            frame.plane_depth_delta = torch.from_numpy(delta)
            xyz = frame.X_canon
            uv = get_pixel_coords(1, frame.depth.shape, device=xyz.device, dtype=xyz.dtype).reshape(-1, 2)
            z = frame.depth.to(device=xyz.device, dtype=xyz.dtype).reshape(-1, 1)
            frame.X_canon = backproject(uv, z, frame.K)
            if frame.feature_points is not None and len(frame.feature_points):
                # Keep the displayed accepted correspondences on the final map.
                points = frame.feature_points.clone()
                K = frame.K.detach().cpu()
                px = (points[:, 0]/points[:, 2]*K[0, 0]+K[0, 2]).round().long()
                py = (points[:, 1]/points[:, 2]*K[1, 1]+K[1, 2]).round().long()
                good = (points[:, 2] > 0) & (px >= 0) & (px < frame.depth.shape[1]) & (py >= 0) & (py < frame.depth.shape[0])
                points[good, :3] *= (frame.depth[py[good], px[good]]/points[good, 2])[:, None]
                frame.feature_points = points
            # Plane pose factors retain the independent current measured fit.
            # The temporally corrected map is not a new sensor measurement.
