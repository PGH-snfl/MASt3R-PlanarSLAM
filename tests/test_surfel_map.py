import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from test_disk_map import make_frame
from mast3r_slam.disk_map import DiskMapWriter
from mast3r_slam.keyframe_archive import KeyframeArchive
from mast3r_slam.surfel_map import SurfelMapReader


class SurfelMapTests(unittest.TestCase):
    def test_full_pixel_grid_bounded_cache_pose_updates_and_no_duplicate_cloud(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)/'map'
            store = KeyframeArchive(session, recent=2, device='cpu', save_on_append=True)
            writer = DiskMapWriter(session, {})
            reader = SurfelMapReader(session, dict(view_surfel_keyframes=1, view_overview_points=0))
            try:
                for i, x in enumerate([0., 10., 20.]):
                    f = make_frame(i, x)
                    k = store.append(f)
                    writer.sync_poses(store)
                    writer.add_frame(f, k, f.T_WC.matrix().numpy(), f.timestamp, k)
                for _, future in store.writes:
                    future.result()
                for future in writer.pending:
                    future.result()
                for i, x in enumerate([0., 10., 20.]):
                    cloud, stats = reader.snapshot(np.array([x, 0., 1.]))
                    self.assertEqual(len(cloud), 0, 'Do not draw keyframes twice as points')
                    self.assertEqual(stats['surfel_keyframes'], 1)
                    surface, = stats['surfels']
                    self.assertEqual(surface['frame_id'], i)
                    self.assertEqual(surface['xyz'].shape, (64, 64, 3))
                    np.testing.assert_allclose(surface['rgb'], .5)
                    np.testing.assert_allclose(surface['confidence'], 3)
                    self.assertEqual(stats['surfel_points'], 4096)
                    self.assertEqual(len(reader.surfels), 1)
                loads = reader.surfel_loads
                poses = np.array([p.numpy() for p in store.poses]).reshape(-1, 8)
                poses[-1, 1] = .1
                store.apply_global_poses(poses)
                writer.sync_poses(store)
                for future in writer.pending:
                    future.result()
                _, stats = reader.snapshot(np.array([20., 0., 1.]))
                self.assertAlmostEqual(stats['surfels'][0]['matrix'][1, 3], .1)
                self.assertEqual(reader.surfel_loads, loads, 'Pose corrections must not reload geometry')
                n = make_frame(3, 20.)
                writer.add_frame(n, 2, store[2].T_WC.matrix().numpy(), .3)
                for future in writer.pending:
                    future.result()
                cloud, stats = reader.snapshot(np.array([20., 0., 1.]))
                self.assertGreater(len(cloud), 0)
                self.assertEqual(stats['nonkeyframes'], 1)
                self.assertEqual(stats['surfel_keyframes'], 1)
            finally:
                reader.close()
                writer.close()
                store.close()


if __name__ == '__main__':
    unittest.main()
