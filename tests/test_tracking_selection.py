import unittest
from unittest.mock import patch
import torch
import lietorch
from mast3r_slam.config import config, load_config
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.tracking_selection import select_pose_points, AdaptiveSelection
from mast3r_slam.runtime_metrics import DisplayGate, StageTimer


def observations(h=64, w=64, device='cpu'):
    v, u = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    z = 2.+.2*torch.sin(u/9.)+.1*torch.cos(v/7.)+.3*(u > w//2)
    xyz = torch.stack(((u-w/2)*z/100, (v-h/2)*z/100, z), -1).reshape(-1, 3)
    K = xyz.new_tensor([[100., 0, w/2], [0, 100., h/2], [0, 0, 1]])
    meas = torch.stack((u, v, z.log()), -1).reshape(-1, 3)
    valid = torch.ones(h*w, 1, device=device, dtype=torch.bool)
    confidence = torch.ones(h*w, 1, device=device)*3
    rgb = torch.zeros(h, w, 3, device=device)
    return xyz, K, meas, valid, confidence, rgb


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(enabled=True, cell_size=16, min_points=32, min_valid_fraction=.15)
        self.x, self.K, self.meas, self.valid, self.q, self.rgb = observations()

    def select(self):
        return select_pose_points(self.valid, self.q, self.x, self.rgb, (64, 64), self.cfg)

    def test_spatial_coverage_budget_and_no_mutation(self):
        original = self.x.clone()
        ids = self.select()
        self.assertEqual(len(ids), 16*8)
        self.assertEqual(len(ids.unique()), len(ids))
        cells = ((ids//64)//16)*4+(ids%64)//16
        self.assertTrue(torch.bincount(cells).eq(8).all())
        torch.testing.assert_close(self.x, original)

    def test_invalid_and_low_support_use_full(self):
        self.valid[:1000] = False
        self.x[1000:1100] = float('nan')
        self.assertTrue((self.select() >= 1100).all())
        self.valid[:] = False
        self.valid[-20:] = True
        self.assertIsNone(self.select())
        self.cfg['enabled'] = False
        self.assertIsNone(self.select())

    def test_near_far_and_boundary_survive_flat_high_confidence(self):
        self.x[:, 2] = 2
        self.x[5*64+5, 2] = 1
        self.x[10*64+10, 2] = 3
        self.rgb[:, 8:] = 1
        ids = self.select()
        self.assertIn(5*64+5, ids.tolist())
        self.assertIn(10*64+10, ids.tolist())
        first_cell = ids[(ids//64 < 16) & (ids%64 < 16)]
        self.assertTrue(((first_cell%64 == 7) | (first_cell%64 == 8)).any())

    def test_non_divisible_shape(self):
        x, _, _, valid, q, rgb = observations(67, 71)
        ids = select_pose_points(valid, q, x, rgb, (67, 71), self.cfg)
        self.assertTrue(((ids >= 0) & (ids < 67*71)).all())
        self.assertEqual(len(ids), len(ids.unique()))

    def tracker(self):
        load_config('config/d457_planar.yaml')
        tracker = FrameTracker(None, None, 'cpu')
        tracker.cfg = dict(config['tracking'], point_selection=self.cfg)
        tracker.metric_input = True
        return tracker

    def solve(self, tracker):
        identity = lietorch.Sim3.Identity(1)
        return tracker.solve_selected_pose(self.x, self.x, identity, identity, self.q, self.valid,
                    self.meas, self.valid, self.K, (64, 64), self.rgb, StageTimer())

    def test_rejected_sample_retries_full_from_original_pose(self):
        tracker = self.tracker()
        starts, sizes = [], []
        identity = lietorch.Sim3.Identity(1)
        wrong = lietorch.Sim3.Identity(1)
        wrong.data[0, 0] = 10.
        def opt(xf, xk, start, *args):
            sizes.append(len(xf))
            starts.append(start.data.clone())
            return (wrong, wrong) if len(sizes) == 1 else (identity, identity)
        with patch.object(tracker, 'opt_pose_calib_sim3', side_effect=opt):
            self.solve(tracker)
        self.assertEqual(sizes, [128, 4096])
        self.assertTrue(tracker.selection_stats['fallback'])
        torch.testing.assert_close(starts[0], starts[1])

    def test_sample_exception_retries_and_full_failure_propagates(self):
        tracker = self.tracker()
        with patch.object(tracker, 'opt_pose_calib_sim3', side_effect=RuntimeError('singular')) as solve:
            with self.assertRaisesRegex(RuntimeError, 'singular'):
                self.solve(tracker)
            self.assertEqual(solve.call_count, 2)

    def test_valid_sample_is_accepted_and_disabling_uses_full(self):
        tracker = self.tracker()
        identity = lietorch.Sim3.Identity(1)
        with patch.object(tracker, 'opt_pose_calib_sim3', return_value=(identity, identity)) as solve:
            self.solve(tracker)
            self.assertEqual(len(solve.call_args.args[0]), 128)
            self.assertEqual(solve.call_count, 1)
        tracker.cfg['point_selection'] = dict(enabled=False)
        with patch.object(tracker, 'opt_pose_calib_sim3', return_value=(identity, identity)) as solve:
            self.solve(tracker)
            self.assertEqual(len(solve.call_args.args[0]), 4096)

    def test_gate_keeps_pending_keyframe_change_and_drops_old_display_updates(self):
        gate = DisplayGate(5)
        self.assertEqual(gate.ready(0, True), (True, True))
        self.assertEqual(gate.ready(.05, True), (False, False))
        self.assertEqual(gate.ready(.1), (False, False))
        self.assertEqual(gate.ready(.21), (True, True))
        self.assertEqual(gate.ready(1.), (True, False))

    def test_adaptive_path_avoids_slower_selection_and_periodically_probes(self):
        policy = AdaptiveSelection()
        for expected, ms in ((False, 8), (False, 8), (True, 12), (True, 12)):
            self.assertEqual(policy.choose(), expected)
            policy.record(expected, ms)
        for _ in range(25):
            self.assertFalse(policy.choose())
            policy.record(False, 8)
        self.assertTrue(policy.choose())  # Call 30 probes the other path.
        for _ in range(5):
            policy.record(True, 5)
        self.assertTrue(policy.choose())

    def test_cached_reference_geometry_respects_new_match_mask(self):
        cache = {}
        select_pose_points(self.valid, self.q, self.x, self.rgb, (64, 64), self.cfg, cache)
        self.valid[::3] = False
        a = select_pose_points(self.valid, self.q, self.x, self.rgb, (64, 64), self.cfg, cache)
        b = self.select()
        torch.testing.assert_close(a, b)
        self.assertTrue(self.valid[a].all())

    def test_compact_full_valid_solve_matches_original_dense_solve(self):
        tracker = self.tracker()
        tracker.cfg['point_selection'] = dict(enabled=False)
        truth = lietorch.Sim3.exp(torch.tensor([[.03, -.01, .008, .01, 0., 0., 0.]]))
        identity = lietorch.Sim3.Identity(1)
        xf = truth.inv().act(self.x)
        self.valid[::2] = False
        def solve():
            return tracker.solve_selected_pose(xf, self.x, identity, identity, self.q, self.valid,
                        self.meas, self.valid, self.K, (64, 64), self.rgb, StageTimer())[0]
        dense = solve()
        # Force all-valid fallback, then physically remove only invalid rows.
        tracker.cfg['point_selection'] = dict(enabled=True, min_points=10000)
        compact = solve()
        self.assertEqual(tracker.selection_stats['used'], 2048)
        torch.testing.assert_close(compact.data, dense.data, atol=2e-6, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
