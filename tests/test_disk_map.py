import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import lietorch
import threading
from mast3r_slam.frame import Frame
from mast3r_slam.keyframe_archive import KeyframeArchive, pack_frame
from mast3r_slam.disk_map import DiskMapWriter, DiskMapReader, map_arrays


def make_frame(i, x=0.):
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    xyz = torch.stack(((xx-32)/50., (yy-32)/50., torch.ones(h, w)), -1).reshape(-1, 3)
    pose = lietorch.Sim3.Identity(1)
    pose.data[0, 0] = x
    return Frame(i, torch.zeros(3, h, w), torch.tensor([[h, w]]), torch.tensor([[h, w]]),
                 torch.full((h, w, 3), .5), pose, X_canon=xyz, C=torch.full((h*w, 1), 3.),
                 feat=torch.ones(1, 16, 1024), pos=torch.zeros(1, 16, 2, dtype=torch.long), N=1, N_updates=1,
                 K=torch.tensor([[50., 0, 32.], [0, 50., 32.], [0, 0, 1.]]),
                 depth=torch.ones(h, w), raw_depth=torch.ones(h, w),
                 depth_source=torch.ones(h, w, dtype=torch.uint8), timestamp=i*.1)


class DiskMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.session = Path(self.temp.name)/'session'
        self.store = KeyframeArchive(self.session, recent=2, device='cpu', preview_limit=0, save_on_append=True)
        self.cfg = dict(nonkeyframe_voxel_m=.03, nonkeyframe_max_points=64,
                        view_max_points=4300, view_max_chunks=2, view_overview_points=30)
        self.writer = DiskMapWriter(self.session, self.cfg)
        self.reader = DiskMapReader(self.session, self.cfg)

    def tearDown(self):
        self.reader.close()
        if not self.writer.closed:
            self.writer.close()
        self.store.close()
        self.temp.cleanup()

    def flush(self):
        for future in self.writer.pending:
            future.result()

    def add(self, frame, keyframe=False):
        idx = self.store.append(frame) if keyframe else len(self.store)-1
        self.writer.sync_poses(self.store)
        self.writer.add_frame(frame, idx, self.store[idx].T_WC.matrix().numpy(), frame.timestamp,
                              idx if keyframe else None)

    def test_all_keyframe_fields_saved_immediately_and_nonkeyframes_compact(self):
        frame = make_frame(0)
        frame.X_canon[0] = 0
        frame.X_canon[1] = float('nan')
        frame.C[2] = 0
        frame.C[3] = 1  # Retained in KF chunk, hidden by display threshold.
        self.add(frame, True)
        self.add(make_frame(1, .02))
        self.flush()
        for _, future in self.store.writes:
            future.result()
        self.assertTrue(self.store.path(0).exists())  # Before eviction/close.
        payload = torch.load(self.store.path(0), weights_only=True)
        self.assertEqual(set(payload), set(pack_frame(frame)))
        torch.testing.assert_close(payload['feat'], frame.feat)
        torch.testing.assert_close(payload['raw_depth'], frame.raw_depth)
        with np.load(self.session/'map/frames/000000000.npz') as data:
            self.assertEqual(len(data['xyz']), 4093)
            self.assertIn(1., data['confidence'])
        with np.load(self.session/'map/frames/000000001.npz') as data:
            self.assertLessEqual(len(data['xyz']), 64)
            self.assertEqual(set(data.files), {'xyz', 'rgb', 'confidence', 'source'})
            self.assertEqual(len(np.unique(np.floor(data['xyz']/.03).astype(int), axis=0)), len(data['xyz']))
        self.assertFalse(self.store.path(1).exists())

    def test_old_regions_reload_and_display_memory_is_bounded(self):
        for i in range(4):
            self.add(make_frame(2*i, x=i*10), True)
            self.add(make_frame(2*i+1, x=i*10+.1))
        self.flush()
        points, stats = self.reader.snapshot(np.array([0., 0., 0.]))
        self.assertEqual(stats['frames'], 8)
        self.assertEqual(stats['keyframes'], 4)
        self.assertEqual(stats['nonkeyframes'], 4)
        self.assertLessEqual(len(points), 4330)
        self.assertLessEqual(len(self.reader.cache), 2)
        self.assertIn(0, self.reader.cache)
        first = self.reader.loads
        self.reader.snapshot(np.array([30., 0., 0.]))
        self.assertNotIn(0, self.reader.cache)
        self.reader.snapshot(np.array([0., 0., 0.]))
        self.assertIn(0, self.reader.cache)
        self.assertGreater(self.reader.loads, first)
        self.assertLessEqual(len(self.store.recent), 2)

    def test_anchor_pose_updates_move_nonkeyframes_without_rewriting_chunks(self):
        self.add(make_frame(0), True)
        self.add(make_frame(1, .2))
        self.flush()
        before, _ = self.reader.snapshot(np.zeros(3))
        path = self.session/'map/frames/000000001.npz'
        stamp = path.stat().st_mtime_ns
        moved = lietorch.Sim3.Identity(1)
        moved.data[0, 0] = 5
        self.store.update_T_WCs(moved[None], [0])
        self.writer.sync_poses(self.store)
        self.flush()
        after, _ = self.reader.snapshot(np.array([5., 0, 0]))
        np.testing.assert_allclose(after[:, :3], before[:, :3]+[5, 0, 0], atol=1e-6)
        self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_reopen_map_and_partial_index_record(self):
        self.add(make_frame(0), True)
        self.flush()
        self.writer.close()
        a, _ = self.reader.snapshot(np.zeros(3))
        self.reader.close()
        self.reader = DiskMapReader(self.session, self.cfg)
        b, _ = self.reader.snapshot(np.zeros(3))
        np.testing.assert_array_equal(a, b)
        index = self.session/'map/index.jsonl'
        entry = dict(self.reader.entries[0], frame_id=99)
        text = json.dumps(entry)+'\n'
        with index.open('a') as f:
            f.write(text[:len(text)//2])
        self.reader.refresh()
        self.assertEqual(len(self.reader.entries), 1)
        with index.open('a') as f:
            f.write(text[len(text)//2:])
        self.reader.refresh()
        self.assertEqual(len(self.reader.entries), 2)

    def test_display_threshold_and_frustum_do_not_delete_saved_points(self):
        f = make_frame(0)
        f.C[:100] = 1
        self.add(f, True)
        self.flush()
        full, _ = self.reader.snapshot(np.zeros(3), threshold=0)
        filtered, _ = self.reader.snapshot(np.zeros(3), threshold=1.5)
        self.assertEqual(len(full)-len(filtered), 100)
        # Clip volume far away: only sparse overview, no dense chunk selection.
        projection = np.eye(4)
        projection[0, 3] = 100
        _, stats = self.reader.snapshot(np.zeros(3), view_projection=projection)
        self.assertEqual(stats['chunks'], 0)
        self.assertLessEqual(stats['overview_points'], 30)
        self.assertEqual(self.reader.entries[0]['points'], 4096)

    def test_write_failure_propagates_and_does_not_publish_a_chunk(self):
        f = make_frame(0)
        self.store.append(f)
        self.writer.sync_poses(self.store)
        self.flush()
        with patch('mast3r_slam.disk_map.write_chunk', side_effect=OSError('disk full')):
            self.writer.add_frame(f, 0, np.eye(4), 0, 0)
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.writer.close()
        self.reader.refresh()
        self.assertEqual(len(self.reader.entries), 0)

    def test_worker_prepares_owned_snapshot_without_losing_points(self):
        from mast3r_slam.disk_map import select_map_arrays
        entered, release = threading.Event(), threading.Event()
        worker_names = []
        def blocked(*args):
            worker_names.append(threading.current_thread().name)
            entered.set()
            if not release.wait(5):
                raise TimeoutError('test did not release writer')
            return select_map_arrays(*args)
        frame = make_frame(0)
        expected = map_arrays(frame, True, self.cfg)
        with patch('mast3r_slam.disk_map.select_map_arrays', side_effect=blocked):
            try:
                self.add(frame, True)
                self.assertTrue(entered.wait(5))
                self.assertLessEqual(len(self.writer.pending), 2)
                frame.X_canon.zero_()
                frame.uimg.zero_()
                frame.C.zero_()
                frame.depth_source.zero_()
            finally:
                release.set()
            self.flush()
        self.assertTrue(worker_names[0].startswith('map-save'))
        with np.load(self.session/'map/frames/000000000.npz') as data:
            for key in expected:
                np.testing.assert_array_equal(data[key], expected[key])


if __name__ == '__main__':
    unittest.main()
