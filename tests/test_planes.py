import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
import lietorch
from mast3r_slam.planes import (extract_planes, transform_planes, plane_jacobian,
                                match_planes, plane_residuals, refine_graph)

torch.set_num_threads(2)


def plane_data():
    return torch.tensor([[0., 0., -1., 2., 0., 0., 2., .4],
                         [-1., 0., 0., 1., 1., 0., 2., .3],
                         [0., -1., 0., 1., 0., 1., 2., .3]], dtype=torch.float64)


class PlaneTests(unittest.TestCase):
    def test_ransac_with_holes_noise_and_outliers(self):
        rng = np.random.default_rng(5)
        depth = np.full((120, 160), 2., dtype=np.float32)+rng.normal(0, .003, (120, 160))
        depth[::7] = 0
        depth[10:15] = 4
        depth[30, 30] = np.nan
        K = np.array([[100., 0, 80], [0, 100., 60], [0, 0, 1]])
        p = extract_planes(depth, K, {"sample_stride": 2, "min_points": 80, "max_planes": 3})
        self.assertGreater(p[0, 7], .8)
        np.testing.assert_allclose(p[0, :4], [0, 0, -1, 2], atol=.003)
        self.assertEqual(extract_planes(np.zeros_like(depth), K, {})[:, 7].sum(), 0)

    def test_sim3_plane_transform_preserves_incidence(self):
        T = lietorch.Sim3.exp(torch.tensor([[.2, -.1, .3, .1, -.2, .03, .2]], dtype=torch.float64))
        planes = plane_data()
        transformed = transform_planes(planes, T)
        centers = T.act(planes[:, 4:7])
        torch.testing.assert_close((transformed[:, :3]*centers).sum(-1)+transformed[:, 3], torch.zeros(3, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_plane_jacobian_matches_finite_difference_including_scale(self):
        T = lietorch.Sim3.exp(torch.tensor([[.2, -.1, .3, .1, -.2, .03, .2]], dtype=torch.float64))
        p = plane_data()
        analytic = plane_jacobian(transform_planes(p, T))
        numerical = torch.zeros_like(analytic)
        for k in range(7):
            eps = torch.zeros(1, 7, dtype=torch.float64)
            eps[0, k] = 1e-6
            numerical[:, :, k] = (transform_planes(p, T.retr(eps))[:, :4]-transform_planes(p, T.retr(-eps))[:, :4])/(2e-6)
        torch.testing.assert_close(analytic, numerical, atol=1e-7, rtol=1e-6)

    def test_matching_sign_flip_and_distance_rejection(self):
        p = plane_data()
        q = p.clone()
        q[0, :4] *= -1
        identity = lietorch.Sim3.Identity(1, dtype=torch.float64)
        matches = match_planes(p, q, identity, {})
        self.assertEqual(len(matches), 3)
        residual, _ = plane_residuals(p, q, identity, matches, {})
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        q[:, 3] += 2
        self.assertEqual(match_planes(p, q, identity, {}), [])

    def test_graph_reduces_plane_error_and_preserves_fixed_pose(self):
        p = plane_data()
        identity = lietorch.Sim3.Identity(1, dtype=torch.float64)
        offset = lietorch.Sim3.exp(torch.tensor([[.06, -.03, .04, .02, -.02, .01, .01]], dtype=torch.float64))
        class Frames:
            def __init__(self):
                self.items = [SimpleNamespace(T_WC=identity, planes=p, depth=True),
                              SimpleNamespace(T_WC=offset, planes=p, depth=True)]
            def __getitem__(self, i):
                return self.items[i]
            def update_T_WCs(self, poses, ids):
                for k, i in enumerate(ids.tolist()):
                    self.items[i].T_WC = poses[k]
        frames = Frames()
        before = torch.linalg.norm(transform_planes(p, offset)[:, :4]-p[:, :4])
        n = refine_graph(frames, torch.tensor([0]), torch.tensor([1]), {"backend_iterations": 4, "weight": 10.})
        after = torch.linalg.norm(transform_planes(p, frames[1].T_WC)[:, :4]-p[:, :4])
        self.assertEqual(n, 3)
        self.assertLess(after, before*.6)
        torch.testing.assert_close(frames[0].T_WC.data, identity.data)

    def test_single_plane_does_not_invent_tangential_constraint(self):
        p = plane_data()[:1]
        T = lietorch.Sim3.exp(torch.tensor([[.05, 0, 0, 0, 0, .04, 0]], dtype=torch.float64))
        r, J = plane_residuals(p, p, T, [(0, 0, 1.)], {})
        torch.testing.assert_close(r, torch.zeros_like(r), atol=1e-10, rtol=0)
        self.assertEqual(J[:, :, 0].abs().sum().item(), 0.)

    def test_fused_attention_cpu_equivalence(self):
        from mast3r_slam import mast3r_utils  # sets up copied CroCo imports
        from models.blocks import Attention, CrossAttention
        from mast3r_slam.efficient_attention import enable_efficient_attention
        for cls in (Attention, CrossAttention):
            a = cls(32, num_heads=4).eval()
            b = copy.deepcopy(a)
            self.assertEqual(enable_efficient_attention(b), 1)
            x = torch.randn(1, 24, 32)
            args = (x, None) if cls is Attention else (x, x, x, None, None)
            torch.testing.assert_close(a(*args), b(*args), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
