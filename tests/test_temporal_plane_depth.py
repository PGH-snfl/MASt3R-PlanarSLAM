import copy
import unittest
import numpy as np
import torch
from mast3r_slam.config import config, load_config
from mast3r_slam.plane_depth import correct_large_planes
from mast3r_slam.temporal_plane_depth import TemporalPlaneDepth
from mast3r_slam.frame import Frame
from mast3r_slam.keyframe_archive import pack_frame, unpack_frame


class TemporalPlaneDepthTests(unittest.TestCase):
    def setUp(self):
        self.old = copy.deepcopy(config)
        load_config('config/d457_planar.yaml')
        self.cfg = config['temporal_plane_depth'].copy()
        self.filter = TemporalPlaneDepth(self.cfg)
        self.K = np.array([[120., 0, 80.], [0, 120., 64.], [0, 0, 1.]], np.float32)
        self.shape = (128, 160)
        yy, xx = np.mgrid[:128, :160]
        self.rays = np.stack(((xx-80)/120, (yy-64)/120, np.ones(self.shape)), -1)
        self.source = np.ones(self.shape, np.uint8)
        self.conf = np.full(self.shape, 3., np.float32)
        self.rng = np.random.default_rng(102)

    def tearDown(self):
        config.clear()
        config.update(self.old)

    def surface(self, pose=None, bias=0., noise=.003):
        pose = np.eye(4) if pose is None else pose
        s, R, t = self.filter._pose(pose)
        n = R.T @ np.array([0., 0., -1.])
        d = (2.-t[2])/s
        truth = -d/(self.rays @ n)
        noisy = (truth+bias+self.rng.normal(0, noise, self.shape)).astype(np.float32)
        center = (truth[..., None]*self.rays).reshape(-1, 3).mean(0)
        proposal = np.array([[*n, d, *center, 1]], np.float32)
        return truth, noisy, proposal

    def process(self, idx, pose=None, bias=0., epoch=0, noise=.003, **kwargs):
        pose = np.eye(4) if pose is None else pose
        truth, raw, p = self.surface(pose, bias, noise)
        obs = []
        spatial, delta, _, _ = correct_large_planes(raw, self.K, self.source, self.conf,
                                                   p, config['plane_depth'], observations=obs)
        result, total, stats = self.filter.correct(spatial, delta, self.K, obs, pose,
                                                  idx, epoch, idx*.1, **kwargs)
        return truth, raw, spatial, result, total, stats, obs

    def test_moving_camera_noise_reduction_and_caps(self):
        spatial_errors, temporal_errors = [], []
        used = 0
        for i in range(48):
            pose = np.eye(4)
            pose[2, 3] = .005*i  # Real forward movement must not be averaged away.
            bias = (.004 if i % 2 else -.004)
            truth, raw, spatial, out, total, stats, _ = self.process(i, pose, bias)
            self.assertLessEqual(float(np.abs(out-spatial).max()), .003001)
            self.assertLessEqual(float(np.abs(total).max()), .010001)
            np.testing.assert_allclose(out, raw+total, atol=5e-7)
            if i > 3:
                spatial_errors.append(np.mean((spatial[10:-10, 10:-10]-truth[10:-10, 10:-10])**2))
                temporal_errors.append(np.mean((out[10:-10, 10:-10]-truth[10:-10, 10:-10])**2))
            used += stats['used']
        self.assertGreater(used, 35)
        self.assertLess(np.sqrt(np.mean(temporal_errors)), .70*np.sqrt(np.mean(spatial_errors)))
        self.assertLess(abs(out[64, 80]-truth[64, 80]), .006)

    def test_rotation_translation_and_sim3_reprojection(self):
        for i in range(6):
            angle = i*.015
            R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                          [-np.sin(angle), 0, np.cos(angle)]])
            pose = np.eye(4)
            pose[:3, :3] = (1+i*.001)*R
            pose[:3, 3] = [.01*i, 0, .02*i]
            truth, _, spatial, out, _, stats, _ = self.process(i, pose, noise=0)
            self.assertEqual(stats['pixels'], 0)
            if i:
                self.assertEqual(stats['matched'], 1)
            np.testing.assert_allclose(out, truth, atol=2e-6)

    def test_model_holes_foreground_and_low_confidence_untouched(self):
        self.source[20:30, 20:40] = 3
        self.source[35:45, 20:40] = 2
        self.source[50:60, 20:40] = 0
        self.conf[70:80, 20:40] = 1.8
        for i in range(4):
            _, raw, spatial, out, total, stats, _ = self.process(i, bias=.004 if i % 2 else -.004)
            protected = (self.source != 1) | (self.conf < 2.5)
            np.testing.assert_array_equal(out[protected], raw[protected])
        self.assertGreater(stats['pixels'], 0)
        # Existing connected-component and edge tests cover physical foreground
        # geometry; temporal corrections must be a subset of their safe mask.
        obs = self.process(4)[-1]
        allowed = np.zeros(self.shape, bool)
        for item in obs:
            allowed.ravel()[item['indices']] = True
        self.assertFalse(allowed[protected].any())

    def test_history_invalidated_on_pose_revision_loss_and_time_gap(self):
        for i in range(3):
            self.process(i, bias=.004 if i % 2 else -.004)
        stats = self.process(3, epoch=1)[5]
        self.assertEqual(stats['reset'], 'map_pose_updated')
        self.assertEqual(stats['matched'], 0)
        stats = self.process(4, epoch=1, tracking_ok=False)[5]
        self.assertEqual(stats['reset'], 'tracking_lost')
        self.assertEqual(len(self.filter.tracks), 0)
        self.process(5, epoch=1)
        stats = self.process(40, epoch=1)[5]
        self.assertEqual(stats['reset'], 'time_gap')
        self.assertEqual(stats['pixels'], 0)

    def test_large_pose_jump_invalidates_history(self):
        self.process(0)
        pose = np.eye(4)
        pose[0, 3] = .5
        stats = self.process(1, pose)[5]
        self.assertEqual(stats['reset'], 'pose_jump')
        self.assertEqual(stats['pixels'], 0)

    def test_duplicate_id_disabled_and_invalid_pose_do_not_accumulate(self):
        truth, raw, spatial, out, delta, _, obs = self.process(0)
        for _ in range(5):
            _, _, stats = self.filter.correct(out, delta, self.K, obs, np.eye(4), 0)
            self.assertEqual(stats['pixels'], 0)
        self.assertEqual(len(next(iter(self.filter.tracks.values()))['history']), 1)
        disabled = TemporalPlaneDepth(dict(self.cfg, enabled=False))
        out2, _, _ = disabled.correct(out, delta, self.K, obs, np.eye(4), 1)
        np.testing.assert_array_equal(out2, out)
        bad = np.eye(4)
        bad[0, 0] = np.nan
        _, _, stats = self.filter.correct(out, delta, self.K, obs, bad, 1)
        self.assertEqual(stats['reset'], 'invalid_pose')
        self.assertFalse(self.filter.tracks)

    def test_nonoverlap_different_plane_and_bounded_history(self):
        self.filter = TemporalPlaneDepth(dict(self.cfg, max_tracks=2))
        _, _, spatial, _, delta, _, obs = self.process(0)
        for i in range(1, 30):
            shifted = copy.deepcopy(obs)
            shifted[0]['center'][0] += i*4
            _, _, stats = self.filter.correct(spatial, delta, self.K, shifted, np.eye(4), i)
            self.assertEqual(stats['matched'], 0)
            self.assertLessEqual(len(self.filter.tracks), 2)
        self.filter.reset()
        for i in range(20):
            self.filter.correct(spatial, delta, self.K, obs, np.eye(4), i)
        self.assertEqual(len(next(iter(self.filter.tracks.values()))['history']), self.cfg['window'])
        moved = copy.deepcopy(obs)
        moved[0]['offset'] += .1
        moved[0]['center'][2] += .1
        _, _, stats = self.filter.correct(spatial, delta, self.K, moved, np.eye(4), 21)
        self.assertEqual(stats['matched'], 0)

    def test_frame_archive_preserves_raw_depth_and_final_geometry(self):
        for i in range(5):
            _, raw, _ = self.surface(bias=.004 if i % 2 else -.004)
            frame = Frame(i, None, None, None, None, K=torch.from_numpy(self.K), depth=torch.from_numpy(raw.copy()))
            X = torch.zeros(raw.size, 3)
            X[:, 2] = 2
            frame.update_pointmap(X, torch.full((raw.size, 1), 4.))
            selected = [64*160+80, 64*160+90]
            frame.feature_points = torch.cat((frame.X_canon[selected], torch.ones(2, 3)), dim=1)
            self.filter.apply_frame(frame, timestamp=i*.1)
        self.assertGreater(frame.temporal_depth_stats['pixels'], 0)
        np.testing.assert_array_equal(frame.raw_depth.numpy(), raw)
        np.testing.assert_allclose(frame.depth.numpy(), raw+frame.plane_depth_delta.numpy(), atol=5e-7)
        torch.testing.assert_close(frame.X_canon[:, 2].reshape(self.shape), frame.depth)
        torch.testing.assert_close(frame.feature_points[:, :3], frame.X_canon[selected])
        before = frame.depth.clone()
        self.filter.apply_frame(frame, timestamp=1.)
        torch.testing.assert_close(before, frame.depth)
        self.assertFalse(frame._plane_observations)
        payload = pack_frame(frame)
        self.assertNotIn('_plane_observations', payload)
        restored = unpack_frame(payload, 'cpu')
        torch.testing.assert_close(restored.raw_depth, frame.raw_depth)
        self.assertEqual(restored.temporal_depth_stats, frame.temporal_depth_stats)


if __name__ == '__main__':
    unittest.main()
