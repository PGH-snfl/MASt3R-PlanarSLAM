import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
import lietorch
from mast3r_slam.config import load_config, config
from mast3r_slam.frame import Frame
from mast3r_slam.keyframe_archive import KeyframeArchive
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.streaming_backend import AsyncRetrieval, TemporaryFrames, StreamingBackend


def frame(i):
    pose = lietorch.Sim3.Identity(1)
    pose.data[0, 0] = i*.1
    return Frame(i, torch.full((3, 16, 16), i/100.), torch.tensor([[16, 16]]),
                 torch.tensor([[16, 16]]), torch.ones(16, 16, 3)*.4, pose,
                 X_canon=torch.ones(256, 3), C=torch.ones(256, 1)*3,
                 feat=torch.full((1, 1, 1024), float(i)), pos=torch.zeros(1, 1, 2, dtype=torch.long),
                 N=1, N_updates=1, K=torch.eye(3), depth=torch.ones(16, 16),
                 planes=torch.zeros(8, 8), depth_source=torch.ones(16, 16, dtype=torch.uint8))


def wait_for(test, predicate):
    end = time.monotonic()+10
    while not predicate() and time.monotonic() < end:
        time.sleep(.005)
    test.assertTrue(predicate())


class FakeDatabase:
    def __init__(self):
        self.centroids = torch.ones(4, 4)
        self.count = 0

    def prep_features(self, feat):
        return feat

    def update_descriptors(self, feat, add, k, threshold, query=True):
        time.sleep(.01)
        result = list(range(min(self.count, k))) if query else []
        self.count += int(add)
        return result


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.old_config = copy.deepcopy(config)
        load_config('config/d457_planar.yaml')
        self.temp = tempfile.TemporaryDirectory()
        self.store = KeyframeArchive(Path(self.temp.name)/'session', recent=3, cache=2, device='cpu', preview_limit=20)
        self.store.set_intrinsics(torch.eye(3))

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()
        config.clear()
        config.update(self.old_config)

    def test_120_keyframes_reload_and_bounded_memory(self):
        for idx in range(120):
            self.assertEqual(self.store.append(frame(idx)), idx)
            self.assertLessEqual(len(self.store.recent), 3)
            self.assertLessEqual(len(self.store.writes), 2)
            self.assertLessEqual(len(self.store.preview), 20)
        self.assertEqual(list(self.store.recent), [117, 118, 119])
        self.assertEqual(self.store.preview_seen, 117*4)
        with self.assertRaises(KeyError):
            self.store[0]
        for idx in (0, 50, 90):
            self.store.prefetch(idx)
            wait_for(self, lambda: self.store.poll(idx))
            torch.testing.assert_close(self.store[idx].feat, frame(idx).feat)
            torch.testing.assert_close(self.store[idx].depth_source, frame(idx).depth_source)
            self.assertLessEqual(len(self.store.cache), 2)
        self.assertNotIn(0, self.store.cache)
        self.store.close()
        manifest = json.loads((self.store.directory/'manifest.json').read_text())
        self.assertEqual(manifest['keyframes'], 120)
        self.assertEqual(len(list((self.store.directory/'keyframes').glob('*.pt'))), 120)

    def test_pose_update_survives_reload_and_archives_are_owned_copies(self):
        first = frame(0)
        self.store.append(first)
        changed = lietorch.Sim3.Identity(1)
        changed.data[0, 0] = 7.
        self.store.update_T_WCs(changed[None], [0])
        self.assertEqual(self.store.pose_revision, 1)
        for i in range(1, 4):
            self.store.append(frame(i))
        first.X_canon.fill_(999.)
        self.store.prefetch(0)
        wait_for(self, lambda: self.store.poll(0))
        self.assertEqual(float(self.store[0].T_WC.data[0, 0]), 7.)
        self.assertEqual(float(self.store[0].X_canon[0, 0]), 1.)

    def test_disk_failure_is_not_silently_dropped(self):
        for i in range(3):
            self.store.append(frame(i))
        with patch('mast3r_slam.keyframe_archive.atomic_save', side_effect=OSError('disk full')):
            self.store.append(frame(3))
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.store.close()
        self.assertTrue(self.store.closed)

    def test_factor_pruning_keeps_stable_global_ids_and_all_arrays(self):
        graph = FactorGraph(None, self.store, device='cpu')
        graph.ii = torch.tensor([0, 10, 11, 1])
        graph.jj = torch.tensor([1, 11, 12, 12])
        names = ('idx_ii2jj', 'idx_jj2ii', 'valid_match_j', 'valid_match_i', 'Q_ii2jj', 'Q_jj2ii')
        for name in names:
            setattr(graph, name, torch.arange(4*3).reshape(4, 3))
        graph.retain_frames({10, 11, 12})
        self.assertEqual(graph.ii.tolist(), [10, 11])
        self.assertEqual(graph.jj.tolist(), [11, 12])
        for name in names:
            self.assertEqual(getattr(graph, name).tolist(), [[3, 4, 5], [6, 7, 8]])
        graph.retain_frames(set())
        self.assertEqual(graph.idx_ii2jj.shape, (0, 3))

    def test_async_index_ids_and_queue_bound(self):
        search = AsyncRetrieval(FakeDatabase(), pending_limit=2)
        try:
            self.assertTrue(search.submit(frame(0), 10, True))
            self.assertTrue(search.submit(frame(1), 20, True))
            self.assertFalse(search.submit(frame(2), 30, True))
            wait_for(self, lambda: all(f.done() for f in search.pending))
            self.assertEqual(search.poll(), [(10, True, []), (20, True, [10])])
            self.assertTrue(search.submit(frame(2), None, False))
            wait_for(self, lambda: all(f.done() for f in search.pending))
            self.assertEqual(search.poll(), [(None, False, [10, 20])])
            self.assertEqual(search.ids, [10, 20])
        finally:
            search.close()

    def test_streaming_backend_crosses_capacity_and_relocalizes_to_disk_candidate(self):
        def match(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
            batch = len(feat_i)
            ids = torch.arange(256).repeat(batch, 1)
            valid = torch.ones(batch, 256, 1, dtype=torch.bool)
            scores = torch.ones(batch, 256, 1)*4
            return ids, ids.clone(), valid, valid.clone(), scores, scores.clone(), scores.clone(), scores.clone()
        backend = StreamingBackend(None, self.store, torch.eye(3), FakeDatabase(), 'cpu')
        try:
            with patch('mast3r_slam.global_opt.mast3r_match_symmetric', match), patch.object(backend, '_solve'):
                for idx in range(40):
                    current = frame(idx)
                    deadline = time.monotonic()+10
                    while not backend.can_add():
                        backend.tick(current)
                        self.assertLess(time.monotonic(), deadline)
                        time.sleep(.005)
                    backend.new_keyframe(self.store.append(current))
                    backend.tick(current)
                    self.assertLessEqual(len(self.store.resident_ids), 5)
                    self.assertLessEqual(backend.graph.ii.numel(), 6)
                deadline = time.monotonic()+10
                while backend.search.pending or backend.candidates:
                    backend.tick(current)
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.005)
                self.assertEqual(len(self.store), 40)
                self.assertIsNotNone(backend.last_verified)
                self.assertLess(backend.last_verified[1], 37)
                recovered = False
                while not recovered and time.monotonic() < deadline:
                    recovered = backend.tick(frame(901), relocating=True)
                    time.sleep(.005)
                self.assertTrue(recovered)
                self.assertEqual(len(self.store), 41)
                self.assertEqual(self.store.last_keyframe().frame_id, 901)
                self.assertLessEqual(len(self.store.recent), 3)
                self.assertEqual(backend.graph.ii.numel(), 0)
        finally:
            backend.close()

    def test_relocalization_temporary_frame_does_not_evict_window(self):
        for i in range(3):
            self.store.append(frame(i))
        pending = frame(100)
        temp = TemporaryFrames(self.store, pending)
        moved = lietorch.Sim3.Identity(1)
        moved.data[0, 0] = 9
        temp.update_T_WCs(moved[None], [3])
        self.assertEqual(len(self.store), 3)
        self.assertEqual(list(self.store.recent), [0, 1, 2])
        self.assertEqual(float(pending.T_WC.data[0, 0]), 9)


if __name__ == '__main__':
    unittest.main()
