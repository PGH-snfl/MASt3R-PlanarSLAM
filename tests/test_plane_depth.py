import copy
import unittest
import numpy as np
import torch
from mast3r_slam.config import config, load_config
from mast3r_slam.plane_depth import correct_large_planes
from mast3r_slam.frame import Frame
from mast3r_slam.keyframe_archive import pack_frame, unpack_frame


class PlaneDepthTests(unittest.TestCase):
    def setUp(self):
        self.old_config = copy.deepcopy(config)
        load_config('config/d457_planar.yaml')
        self.cfg = dict(config['plane_depth'])
        self.h, self.w = 128, 160
        self.K = np.array([[120., 0, 80.], [0, 120., 64.], [0, 0, 1.]], np.float32)
        self.rng = np.random.default_rng(2)
        self.depth = np.full((self.h, self.w), 2., np.float32)
        self.source = np.ones_like(self.depth, np.uint8)
        self.conf = np.full_like(self.depth, 3.)
        self.planes = np.array([[0, 0, -1, 2, 0, 0, 2, 1]], np.float32)

    def tearDown(self):
        config.clear()
        config.update(self.old_config)

    def correct(self, depth=None, **overrides):
        return correct_large_planes(self.depth if depth is None else depth, self.K,
                                    self.source, self.conf, self.planes, dict(self.cfg, **overrides))

    def test_large_plane_reduces_noise_and_caps_changes(self):
        noisy = self.depth+self.rng.normal(0, .005, self.depth.shape).astype(np.float32)
        out, delta, planes, stats = self.correct(noisy)
        mask = delta != 0
        self.assertGreater(stats['pixels'], self.depth.size*.4)
        self.assertGreaterEqual(stats['largest_area_m2'], .25)
        self.assertLess(np.sqrt(np.mean((out[mask]-2)**2)), .75*np.sqrt(np.mean((noisy[mask]-2)**2)))
        self.assertLessEqual(np.abs(delta).max(), .01)
        np.testing.assert_allclose(out, noisy+delta, atol=2e-7)
        # A stricter user cap must also be respected.
        _, delta, _, _ = self.correct(noisy, max_correction_m=.001)
        self.assertLessEqual(np.abs(delta).max(), .00100001)

    def test_area_screen_coverage_and_extent_are_separate_gates(self):
        noisy = self.depth+self.rng.normal(0, .005, self.depth.shape).astype(np.float32)
        for overrides in ({'min_area_m2': 100}, {'min_image_fraction': 1.1}, {'min_extent_m': 10}):
            out, delta, _, stats = self.correct(noisy, **overrides)
            np.testing.assert_array_equal(out, noisy)
            self.assertEqual(stats['pixels'], 0)
        self.source[:] = 0
        self.source[30:65, 30:65] = 1
        out, _, _, stats = self.correct(noisy)
        self.assertEqual(stats['accepted'], 0)
        np.testing.assert_array_equal(out, noisy)

    def test_disconnected_small_patches_are_not_combined(self):
        self.source[:] = 0
        for y in (5, 65):
            for x in (5, 55, 105):
                self.source[y:y+40, x:x+40] = 1
        noisy = self.depth+self.rng.normal(0, .005, self.depth.shape).astype(np.float32)
        out, _, _, stats = self.correct(noisy)
        self.assertEqual(stats['accepted'], 0)
        np.testing.assert_array_equal(out, noisy)

    def test_holes_model_depth_low_confidence_and_edges_stay_unchanged(self):
        noisy = self.depth+self.rng.normal(0, .004, self.depth.shape).astype(np.float32)
        noisy[35:70, 50:80] = 1.  # Foreground object must not be flattened onto wall.
        noisy[20:30, 20:30] = 0
        self.source[20:30, 20:30] = 0
        self.source[85:100, 20:50] = 3
        self.source[85:100, 60:80] = 2
        self.conf[85:100, 90:115] = 1.8
        source_before = self.source.copy()
        conf_before = self.conf.copy()
        out, delta, _, stats = self.correct(noisy)
        self.assertGreater(stats['pixels'], 0)
        np.testing.assert_array_equal(out[31:74, 46:84], noisy[31:74, 46:84])
        protected = (self.source != 1) | (self.conf < 2.5)
        np.testing.assert_array_equal(out[protected], noisy[protected])
        np.testing.assert_array_equal(self.source, source_before)
        np.testing.assert_array_equal(self.conf, conf_before)
        self.assertTrue(np.isfinite(out).all())

    def test_shallow_object_and_thin_rod_preserve_real_structure(self):
        noisy = self.depth+self.rng.normal(0, .004, self.depth.shape).astype(np.float32)
        noisy[35:60, 50:75] = 1.98
        noisy[90:110, 95] = 1.98
        out, _, _, stats = self.correct(noisy)
        self.assertGreater(stats['pixels'], 0)
        np.testing.assert_array_equal(out[35:60, 50:75], noisy[35:60, 50:75])
        np.testing.assert_array_equal(out[90:110, 95], noisy[90:110, 95])

    def test_disabled_zero_strength_bad_fit_and_grazing_planes(self):
        noisy = self.depth+self.rng.normal(0, .005, self.depth.shape).astype(np.float32)
        for overrides in ({'enabled': False}, {'strength': 0}, {'max_fit_rms_m': .0001}):
            out, _, _, stats = self.correct(noisy, **overrides)
            np.testing.assert_array_equal(out, noisy)
            self.assertEqual(stats['pixels'], 0)
        # A front-parallel observation cannot support a grazing side plane.
        self.planes[0, :4] = [1, 0, 0, -.01]
        out, _, _, stats = self.correct(noisy)
        np.testing.assert_array_equal(out, noisy)
        self.assertEqual(stats['pixels'], 0)

    def test_frame_uses_corrected_depth_once_and_archives_provenance(self):
        noisy = self.depth+self.rng.normal(0, .005, self.depth.shape).astype(np.float32)
        frame = Frame(0, None, None, None, None, K=torch.from_numpy(self.K), depth=torch.from_numpy(noisy.copy()))
        prediction = torch.zeros(self.depth.size, 3)
        prediction[:, 2] = 2
        frame.update_pointmap(prediction, torch.ones(self.depth.size, 1)*4)
        self.assertGreater(frame.plane_depth_stats['pixels'], 0)
        np.testing.assert_allclose(frame.depth.numpy(), noisy+frame.plane_depth_delta.numpy(), atol=2e-7)
        torch.testing.assert_close(frame.X_canon[:, 2].reshape(self.h, self.w), frame.depth)
        self.assertTrue(frame.depth_source.eq(1).all())
        self.assertTrue(frame.C.eq(3).all())
        before = frame.depth.clone()
        frame.update_pointmap(prediction*2, torch.ones(self.depth.size, 1)*4)
        torch.testing.assert_close(frame.depth, before)
        restored = unpack_frame(pack_frame(frame), 'cpu')
        torch.testing.assert_close(restored.plane_depth_delta, frame.plane_depth_delta)
        self.assertEqual(restored.plane_depth_stats, frame.plane_depth_stats)


if __name__ == '__main__':
    unittest.main()
