import unittest
from unittest.mock import patch
import numpy as np
import torch
from mast3r_slam.depth_fusion import fuse_depth, pair_depth_weight
from mast3r_slam.frame import Frame
from mast3r_slam.config import config
from mast3r_slam.planes import extract_planes


class FusionTests(unittest.TestCase):
    def setUp(self):
        self.sensor = np.full((32, 32), 2., np.float32)
        self.model = self.sensor.copy()
        self.conf = np.full_like(self.sensor, 8.)

    def test_good_sensor_is_preserved_exactly(self):
        self.model[:, :5] = 3.
        z, c, source, stats = fuse_depth(self.sensor, self.model, self.conf, {})
        np.testing.assert_array_equal(z, self.sensor)
        self.assertTrue((source == 1).all())
        self.assertGreater(stats['disagree'], 0)
        self.assertLess(c[0, 0], c[0, 20])

    def test_holes_filled_in_sensor_metric_scale(self):
        self.sensor[:4] = 0
        self.model[:] = 1.
        z, weights, source, stats = fuse_depth(self.sensor, self.model, self.conf, {})
        np.testing.assert_allclose(z, 2.)
        self.assertTrue((source[:4] == 3).all())
        self.assertLess(weights[0, 0], weights[10, 0])
        self.assertTrue(stats['aligned'])
        self.assertAlmostEqual(stats['scale'], 2.)

    def test_spike_replaced_only_with_neighborhood_and_model_agreement(self):
        self.sensor[10, 10] = 4.
        z, _, source, _ = fuse_depth(self.sensor, self.model, self.conf, {})
        self.assertEqual(z[10, 10], 2.)
        self.assertEqual(source[10, 10], 2)
        self.model[10, 10] = 3.
        z, _, source, _ = fuse_depth(self.sensor, self.model, self.conf, {})
        self.assertEqual(z[10, 10], 4.)
        self.assertEqual(source[10, 10], 1)

    def test_real_depth_step_is_not_smoothed(self):
        self.sensor[:, 16:] = 3.
        z, _, source, _ = fuse_depth(self.sensor, self.model, self.conf, {})
        np.testing.assert_array_equal(z, self.sensor)
        self.assertTrue((source == 1).all())

    def test_invalid_and_low_confidence_model_remain_invalid(self):
        self.sensor[:4] = 0
        self.model[0] = float('nan')
        self.model[1] = -1.
        self.model[2] = 100.
        self.conf[3] = 1.
        z, c, src, _ = fuse_depth(self.sensor, self.model, self.conf, {})
        self.assertTrue((z[:4] == 0).all())
        self.assertTrue((c[:4] == 0).all())
        self.assertTrue((src[:4] == 0).all())
        self.assertTrue(np.isfinite(z).all())

    def test_fully_missing_sensor_uses_conservative_metric_model(self):
        self.sensor[:] = 0
        z, c, src, stats = fuse_depth(self.sensor, self.model, self.conf, {})
        np.testing.assert_allclose(z, 2.)
        np.testing.assert_allclose(c, 1.6)
        self.assertTrue((src == 3).all())
        self.assertFalse(stats['aligned'])

    def test_frame_fused_geometry_reaches_planes_and_is_frozen(self):
        self.sensor[:8] = 0
        frame = Frame(0, None, None, None, None)
        frame.depth = torch.from_numpy(self.sensor)
        frame.K = torch.tensor([[25., 0, 16.], [0, 25., 16.], [0, 0, 1.]])
        X = torch.zeros(1024, 3)
        X[:, 2] = 2.
        with patch.dict(config, {'depth_fusion': {'enabled': True},
                                 'planes': {'enabled': True, 'sample_stride': 1, 'min_points': 30}}):
            frame.update_pointmap(X, torch.full((1024, 1), 8.))
            self.assertTrue((frame.X_canon[:, 2] == 2.).all())
            self.assertTrue((frame.depth_source[:8] == 3).all())
            self.assertGreater(frame.planes[:, 7].sum(), 0.)
            before = frame.X_canon.clone()
            frame.update_pointmap(X*100, torch.ones(1024, 1))
            torch.testing.assert_close(frame.X_canon, before)
            self.assertTrue((frame.depth_source[:8] == 3).all())

    def test_correspondence_weights_use_both_depths_and_index_mapping(self):
        a = torch.tensor([[3.], [1.8], [0.]])
        b = torch.tensor([[3.], [3.], [1.8]])
        w = pair_depth_weight(a, b, torch.tensor([1, 2, 0]))
        torch.testing.assert_close(w, torch.tensor([[.6], [0.], [.6]]))

    def test_model_only_planes_have_less_support_weight(self):
        K = np.array([[25., 0, 16.], [0, 25., 16.], [0, 0, 1.]])
        cfg = {'sample_stride': 1, 'min_points': 30}
        sensor = extract_planes(self.model, K, cfg, np.ones_like(self.model))
        model = extract_planes(self.model, K, cfg, np.full_like(self.model, .55))
        np.testing.assert_allclose(model[0, :4], sensor[0, :4], atol=1e-4)
        self.assertLess(model[0, 7], sensor[0, 7])


if __name__ == '__main__':
    unittest.main()
