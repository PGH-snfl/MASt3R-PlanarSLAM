import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import lietorch
import pyrealsense2 as rs
from mast3r_slam.config import load_config, config
from mast3r_slam.realsense_rgbd import resize_depth, RealSenseRGBDDataset, calibrated_intrinsics
from mast3r_slam.frame import Frame, create_frame


class RGBDTests(unittest.TestCase):
    def test_640_rgb_depth_intrinsics_and_metric_rays_agree(self):
        from mast3r_slam.geometry import project_calib
        intr = rs.intrinsics()
        intr.width, intr.height = 640, 480
        intr.fx, intr.fy, intr.ppx, intr.ppy = 400., 410., 319., 238.
        intr.model = rs.distortion.none
        intr.coeffs = [0.] * 5
        depth = np.full((480, 640), 2., np.float32)
        depth[:, :80] = 0
        # The processing raster, its valid-depth boundary and the pinhole
        # projection must agree at both sizes; a stale 512 K would fail this.
        for size, shape, boundary in [(512, (384, 512), 64), (640, (480, 640), 80)]:
            with self.subTest(size=size), patch.dict(config, {
                    'dataset': {'img_downsample': 1}, 'planes': {'enabled': False},
                    'depth_fusion': {'enabled': False}, 'plane_depth': {'enabled': False}}):
                calib = calibrated_intrinsics(intr, size)
                frame = create_frame(0, np.full((480, 640, 3), .5, np.float32),
                                     lietorch.Sim3.Identity(1), img_size=size, device='cpu')
                frame.K = torch.as_tensor(calib.K_frame, dtype=torch.float32)
                frame.depth = torch.from_numpy(resize_depth(depth, size))
                self.assertEqual(tuple(frame.img.shape[-2:]), shape)
                self.assertEqual(tuple(frame.depth.shape), shape)
                self.assertEqual(tuple(frame.uimg.shape[:2]), shape)
                self.assertTrue(frame.depth[:, :boundary].eq(0).all())
                self.assertTrue(frame.depth[:, boundary:].eq(2).all())
                np.testing.assert_allclose(calib.K_frame[:2], calib.K[:2] * (size/640))
                frame.update_pointmap(torch.zeros(shape[0]*shape[1], 3),
                                      torch.ones(shape[0]*shape[1], 1))
                pix, valid = project_calib(frame.X_canon, frame.K, shape)
                v, u = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
                keep = valid[:, 0] & (frame.C[:, 0] > 0)
                target = torch.stack([u, v], dim=-1).reshape(-1, 2).float()
                torch.testing.assert_close(pix[keep, :2], target[keep], atol=1e-4, rtol=1e-5)

    def test_invalid_rgbd_resolution_fails_before_opening_camera(self):
        with patch.dict(config, {'dataset': {'image_size': 600}, 'use_calib': True}), \
             patch.object(RealSenseRGBDDataset, 'devices') as devices:
            with self.assertRaisesRegex(ValueError, 'image_size'):
                RealSenseRGBDDataset()
            devices.assert_not_called()

    def test_depth_holes_do_not_poison_pose_jacobian(self):
        from mast3r_slam.geometry import project_calib
        points = torch.tensor([[0., 0., 0.], [0., 0., -1.], [float('nan'), 0., 1.], [1., 1., 2.]])
        p, J, valid = project_calib(points, torch.eye(3), (480, 640), jacobian=True)
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.isfinite(J).all())
        self.assertEqual(valid[:, 0].tolist(), [False, False, False, True])
        torch.testing.assert_close(J[:3], torch.zeros_like(J[:3]))

    def test_inverse_brown_rectification_matches_sdk(self):
        intr = rs.intrinsics()
        intr.width, intr.height = 640, 480
        intr.fx, intr.fy, intr.ppx, intr.ppy = 400., 410., 320., 240.
        intr.model = rs.distortion.inverse_brown_conrady
        intr.coeffs = [-.1, .02, .001, -.002, .001]
        calib = calibrated_intrinsics(intr)
        for u, v in [(0, 0), (639, 479), (320, 240), (99, 222)]:
            ray = [(u-intr.ppx)/intr.fx, (v-intr.ppy)/intr.fy, 1.]
            expected = rs.rs2_project_point_to_pixel(intr, ray)
            np.testing.assert_allclose([calib.mapx[v, u], calib.mapy[v, u]], expected, atol=1e-4)
        np.testing.assert_allclose(calib.K_frame[:2, :], calib.K[:2, :]/1.25)

    def test_depth_resize_and_invalid_pixels(self):
        depth = np.full((480, 640), 2., np.float32)
        depth[:, :80] = 0
        result = resize_depth(depth)
        self.assertEqual(result.shape, (384, 512))
        self.assertTrue(np.all(result[:, :64] == 0))
        self.assertTrue(np.all(result[:, 64:] == 2))

    def test_metric_pointmap_uses_depth_and_masks_holes(self):
        frame = Frame(0, None, None, None, None)
        frame.depth = torch.tensor([[1., 0.], [2., float('nan')]])
        frame.K = torch.eye(3)
        frame.update_pointmap(torch.randn(4, 3)*100, torch.ones(4, 1))
        torch.testing.assert_close(frame.X_canon, torch.tensor([[0., 0., 1.], [0., 0., 0.], [0., 2., 2.], [0., 0., 0.]]))
        self.assertEqual(frame.C[:, 0].tolist(), [3., 0., 3., 0.])
        frame.update_pointmap(torch.randn(4, 3)*100, torch.ones(4, 1))
        self.assertEqual(frame.X_canon[2, 2], 2.)

    def test_capture_converts_units_and_accepts_clock_reset(self):
        ds = RealSenseRGBDDataset.__new__(RealSenseRGBDDataset)
        ds.recorded_source = None
        ds.cfg = {"timeout_ms": 1000, "fps": 30}
        ds.depth_scale = .001
        ds.last_number = (1, 1)
        ds.last_timestamp = 10.
        ds.img_size = 512
        ds.pipeline = MagicMock()
        ds.align = MagicMock()
        sample = MagicMock()
        ds.align.process.return_value = sample
        color, depth = MagicMock(), MagicMock()
        color.profile.format.return_value = rs.format.rgb8
        sample.get_color_frame.return_value = color
        sample.get_depth_frame.return_value = depth
        color.get_frame_number.return_value = 2
        depth.get_frame_number.return_value = 2
        color.get_timestamp.return_value = 1000.
        color.get_data.return_value = np.full((480, 640, 3), 100, np.uint8)
        depth.get_data.return_value = np.full((480, 640), 1500, np.uint16)
        v, u = np.mgrid[:480, :640].astype(np.float32)
        ds.camera_intrinsics = MagicMock(mapx=u, mapy=v)
        ts, rgb, z = ds._capture()
        self.assertGreater(ts, 10.)
        np.testing.assert_allclose(z, 1.5)
        self.assertEqual(rgb.shape, (480, 640, 3))
        ds.recorded_source = MagicMock()
        ds.last_number = None
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            ds._capture()


if __name__ == '__main__':
    unittest.main()
