import tempfile
import time
import unittest
from pathlib import Path
from concurrent.futures import Future
import numpy as np
import torch
import lietorch
from scipy.spatial.transform import Rotation
from mast3r_slam.temporal_surface import TemporalSurfaceDepth
from mast3r_slam.pose_graph import (robust_registration, optimize_pose_graph,
                                   matrices, vectors, GlobalPoseGraph, measurement_from_graph)
from mast3r_slam.keyframe_archive import KeyframeArchive, pack_frame, unpack_frame
from mast3r_slam.frame import Frame


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.filter = TemporalSurfaceDepth(dict(enabled=True))
        self.shape = (80, 96)
        self.K = np.array([[100, 0, 48], [0, 100, 40], [0, 0, 1]], np.float32)
        self.rgb = np.full((*self.shape, 3), .5, np.float32)
        self.source = np.ones(self.shape, np.uint8)
        self.conf = np.full(self.shape, 3, np.float32)
        self.rng = np.random.default_rng(8)

    def call(self, i, raw, pose=None, depth=None, **kw):
        return self.filter.correct(raw, raw if depth is None else depth, self.source, self.conf,
                                   self.rgb, self.K, np.eye(4) if pose is None else pose,
                                   i, timestamp=i*.1, **kw)

    def test_noise_reduction_with_motion_and_caps(self):
        before, after = [], []
        for i in range(12):
            pose = np.eye(4)
            pose[2, 3] = .005*i
            truth = 2-pose[2, 3]
            raw = (truth+self.rng.normal(0, .003, self.shape)).astype('f4')
            out, delta, stats = self.call(i, raw, pose)
            self.assertLessEqual(np.abs(delta).max(), .004001)
            if i > 3:
                before.append(np.mean((raw[8:-8, 8:-8]-truth)**2))
                after.append(np.mean((out[8:-8, 8:-8]-truth)**2))
                self.assertGreater(stats['pixels'], 100)
        self.assertLess(np.sqrt(np.mean(after)), .8*np.sqrt(np.mean(before)))
        self.assertEqual(len(self.filter.history), 3)
        np.testing.assert_array_equal(self.filter.history[-1]['raw'], raw)

    def test_holes_model_pixels_edges_motion_and_planes_protected(self):
        raw = np.full(self.shape, 2, np.float32)
        for i in range(3):
            self.call(i, raw)
        raw += .006
        raw[20:40, 20:40] = 1  # new foreground: incompatible depth must not average
        raw[50:60, 10:20] = 0
        self.source[5:15, 5:15] = 3
        self.source[5:15, 20:30] = 2
        self.rgb[50:65, 40:60] = 0  # changed appearance
        depth = raw.copy()
        depth[5:15, 40:60] -= .002  # pre-existing plane correction
        out, delta, stats = self.call(3, raw, depth=depth)
        for sl in [(slice(18,42),slice(18,42)), (slice(50,60),slice(10,20)),
                   (slice(5,15),slice(5,30)), (slice(50,65),slice(40,60)),
                   (slice(5,15),slice(40,60))]:
            np.testing.assert_array_equal(out[sl], depth[sl])
        self.assertGreater(stats['pixels'], 0)

    def test_revision_loss_jump_and_disabled(self):
        raw = np.full(self.shape, 2, np.float32)
        for i in range(3): self.call(i, raw)
        out, delta, stats = self.call(3, raw+.004, epoch=1)
        self.assertEqual(stats['reset'], 'map_pose_updated')
        self.assertFalse(delta.any())
        self.call(4, raw, tracking_ok=False)
        self.assertEqual(len(self.filter.history), 0)
        self.call(5, raw)
        p = np.eye(4); p[0, 3] = 1
        _, _, stats = self.call(6, raw, pose=p)
        self.assertEqual(stats['reset'], 'pose_jump')
        self.filter.cfg['enabled'] = False
        np.testing.assert_array_equal(self.call(7, raw+.003)[0], raw+.003)

    def test_frame_geometry_archive_and_original_depth_preserved(self):
        f = Frame(0, None, None, None, torch.from_numpy(self.rgb), lietorch.Sim3.Identity(1))
        f.K = torch.from_numpy(self.K)
        for i in range(4):
            f.frame_id = i
            f._surface_depth_done = False
            f.raw_depth = torch.full(self.shape, 2.+(.006 if i == 3 else 0))
            f.depth = f.raw_depth.clone()
            f.depth_source = torch.from_numpy(self.source)
            f.C = torch.ones(np.prod(self.shape), 1)*3; f.N = 1
            f.X_canon = torch.zeros(np.prod(self.shape), 3)
            self.filter.apply_frame(f, timestamp=i*.1)
        self.assertGreater(f.surface_depth_stats['pixels'], 0)
        torch.testing.assert_close(f.X_canon[:, 2].reshape(self.shape), f.depth)
        torch.testing.assert_close(f.raw_depth+f.surface_depth_delta, f.depth)
        recovered = unpack_frame(pack_frame(f), 'cpu')
        torch.testing.assert_close(recovered.surface_depth_delta, f.surface_depth_delta)


class GlobalGraphTests(unittest.TestCase):
    def test_dense_match_indices_register_target_camera_into_source(self):
        from types import SimpleNamespace
        from test_keyframe_archive import frame
        a,b=frame(0),frame(1)
        for f in (a,b):
            f.raw_depth=torch.full((16,16),2.)
            f.K=torch.tensor([[100.,0,8],[0,100.,8],[0,0,1]])
        ids=torch.arange(256).reshape(16,16)
        keep=(ids%16<15).reshape(1,-1,1)
        graph=SimpleNamespace(ii=torch.tensor([0]),jj=torch.tensor([1]),frames={0:a,1:b},
                              idx_ii2jj=(ids+1).reshape(1,-1),valid_match_j=keep,
                              Q_ii2jj=torch.ones(1,256,1))
        T,quality=measurement_from_graph(graph,0,1,{})
        np.testing.assert_allclose(T[:3,3],[.02,0,0],atol=1e-6)
        np.testing.assert_allclose(T[:3,:3],np.eye(3),atol=1e-6)
        self.assertEqual(quality['inliers'],240)

    def test_registration_outliers_and_transform_direction(self):
        rng = np.random.default_rng(5)
        source = rng.uniform([-1,-1,1],[1,1,3], (600,3))
        R = Rotation.from_rotvec([.02,-.03,.04]).as_matrix()
        target = source @ (1.01*R).T + [.1,-.03,.04]
        target += rng.normal(0,.002,target.shape)
        target[:100] = rng.uniform([-1,-1,1],[1,1,3],(100,3))
        T, quality = robust_registration(source,target,{})
        np.testing.assert_allclose(T[:3,:3],1.01*R,atol=.001)
        np.testing.assert_allclose(T[:3,3],[.1,-.03,.04],atol=.001)
        self.assertGreater(quality['inliers'], 490)
        self.assertIsNone(robust_registration(source, target[::-1], {}))
        self.assertIsNone(robust_registration(np.ones((600,3)), np.ones((600,3)), {}))

    def test_loop_distributes_error_and_preserves_anchor(self):
        poses=np.tile(np.eye(4),(12,1,1))
        poses[:,0,3]=np.arange(12)*.11
        edges=[]
        for i in range(11):
            z=np.eye(4);z[0,3]=.11
            edges.append(dict(source=i,target=i+1,kind='temporal',measurement=z))
        z=np.eye(4);z[0,3]=1.1
        edges.append(dict(source=0,target=11,kind='loop',measurement=z))
        out, stats=optimize_pose_graph(poses,edges,{})
        self.assertTrue(stats['accepted'],stats)
        self.assertLess(abs(out[-1,0,3]-1.1),abs(poses[-1,0,3]-1.1))
        np.testing.assert_array_equal(out[0],poses[0])
        self.assertLess(stats['cost_after'],stats['cost_before'])
        np.testing.assert_allclose(matrices(vectors(out)),out,atol=1e-10)
        rejected, stats=optimize_pose_graph(poses,edges,{'max_translation_m':1e-5,'max_log_scale':1e-5})
        self.assertFalse(stats['accepted'])
        np.testing.assert_array_equal(rejected,poses)

    def test_offline_settle_applies_pending_pose_before_archive_close(self):
        from test_keyframe_archive import frame
        with tempfile.TemporaryDirectory() as d:
            store = KeyframeArchive(Path(d)/'map', recent=2, device='cpu')
            graph = GlobalPoseGraph(store, dict(enabled=True))
            try:
                store.append(frame(0))
                store.append(frame(1))
                v = np.array([p.numpy() for p in store.poses]).reshape(-1, 8)
                v[1, 0] += .01
                graph.snapshot = (len(store), store.pose_revision)
                graph.pending = Future()
                graph.pending.set_result((matrices(v), dict(accepted=True)))
                graph.next_solve = time.monotonic() + 1000
                graph.settle()
                self.assertIsNone(graph.pending)
                self.assertEqual(graph.stats['status'], 'applied')
                self.assertAlmostEqual(float(store.poses[1][0, 0]), .11, places=5)
            finally:
                graph.close()
                store.close()

    def test_global_archived_pose_preview_and_stale_worker(self):
        from test_keyframe_archive import frame
        with tempfile.TemporaryDirectory() as d:
            store=KeyframeArchive(Path(d)/'map',recent=2,device='cpu',preview_limit=50)
            graph=GlobalPoseGraph(store,dict(enabled=True))
            try:
                for i in range(4): store.append(frame(i))
                original=store.preview.copy()
                v=np.array([p.numpy() for p in store.poses]).reshape(-1,8)
                v[:,0]+= .2
                store.apply_global_poses(v)
                np.testing.assert_allclose(store.preview[:,:3],original[:,:3]+[.2,0,0],atol=1e-6)
                np.testing.assert_allclose(store.preview[:,3:],original[:,3:])
                store.prefetch(0)
                for _ in range(100):
                    if store.poll(0):break
                    time.sleep(.01)
                self.assertAlmostEqual(float(store[0].T_WC.data[0,0]),.2,places=5)
                graph.snapshot=(len(store),store.pose_revision-1)
                graph.pending=Future()
                graph.pending.set_result((matrices(v),dict(accepted=True)))
                revision=store.pose_revision
                self.assertFalse(graph.poll())
                self.assertEqual(store.pose_revision,revision)
                self.assertEqual(graph.stats['status'],'stale')
                graph.snapshot=(len(store),store.pose_revision)
                graph.pending=Future()
                graph.pending.set_result((matrices(v),dict(accepted=True)))
                self.assertTrue(graph.poll())
                self.assertEqual(store.pose_revision,revision+1)
            finally:
                graph.close();store.close()


if __name__ == '__main__':
    unittest.main()
