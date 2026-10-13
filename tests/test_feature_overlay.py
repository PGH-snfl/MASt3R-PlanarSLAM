import unittest
from types import SimpleNamespace
import torch
from mast3r_slam.feature_overlay import tracking_points, FEATURE_LIMIT


class FeatureOverlayTests(unittest.TestCase):
    def test_mapping_cell_selection_and_depth_source_colors(self):
        frame = SimpleNamespace(img=torch.zeros(1, 3, 32, 32),
                                X_canon=torch.zeros(1024, 3), depth_source=torch.ones(32, 32, dtype=torch.uint8))
        frame.X_canon[:, 2] = 2.
        frame.X_canon[:, 0] = torch.arange(1024)
        frame.depth_source[0, 20] = 3
        # First two current-frame IDs are in the same display cell: retain
        # the second (stronger). Third maps to a model-supported point.
        points = tracking_points(frame, torch.tensor([1, 2, 20, 600]),
                                 torch.tensor([True, True, True, False]), torch.tensor([2., 5., 4., 100.]))
        self.assertEqual(points[:, 0].tolist(), [2., 20.])
        torch.testing.assert_close(points[:, 3:], torch.tensor([[0., 1., 1.], [1., .55, 0.]]))

    def test_empty_invalid_and_bounded_samples(self):
        frame = SimpleNamespace(img=torch.zeros(1, 3, 384, 512),
                                X_canon=torch.ones(384*512, 3), depth_source=None)
        ids = torch.arange(384*512)
        valid = torch.ones(len(ids), dtype=torch.bool)
        scores = torch.ones(len(ids))
        points = tracking_points(frame, ids, valid, scores)
        self.assertLessEqual(len(points), FEATURE_LIMIT)
        frame.X_canon[:] = float('nan')
        self.assertEqual(len(tracking_points(frame, ids, valid, scores)), 0)
        self.assertEqual(len(tracking_points(frame, ids, ~valid, scores)), 0)


if __name__ == '__main__':
    unittest.main()
