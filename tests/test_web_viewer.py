"""CPU-only snapshot correctness, bounded publication and archive mount guards."""
from pathlib import Path
import sys
import importlib.util
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mast3r_slam.web_visualization import frame_cloud, map_cloud, sample_indices
from mast3r_slam.remote_archive import require_archive_mount, sshfs_mounts
from mount_laptop_maps import ssh_command, REMOTE_SETUP


class Pose:
    def __mul__(self, points):
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        return 2 * (points @ rotation.T) + torch.tensor([1., 2., 3.])


def frame():
    return SimpleNamespace(X_canon=torch.tensor([[1., 2., 3.], [0., 0., -1.], [float('nan'), 0., 1.], [1., 1., 2.]]),
                           C=torch.tensor([1., 1., 1., 0.]), T_WC=Pose(),
                           uimg=torch.tensor([[1., 0., 0.]] * 4))


class Tests(unittest.TestCase):
    def test_sim3_and_valid_points(self):
        cloud = frame_cloud(frame(), 10)
        np.testing.assert_allclose(cloud, [[-3., 4., 9., 1., 0., 0.]])

    def test_no_archive_loads_and_bounded_cloud(self):
        # Deliberately no __getitem__: the viewer must never page a disk frame.
        store = SimpleNamespace(recent={i: frame() for i in range(8)}, preview=np.ones((150000, 6), np.float32))
        result = map_cloud(store, 100)
        self.assertLessEqual(len(result), 100)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(map_cloud(SimpleNamespace(recent={}, preview=np.empty((0, 6))), 100).shape, (0, 6))
        self.assertEqual(len(sample_indices(0, 100)), 0)

    def test_missing_mount_never_falls_back(self):
        with patch.object(Path, 'read_text', return_value='1 0 8:1 / / rw - ext4 /dev/test rw\n'):
            with self.assertRaisesRegex(RuntimeError, 'Refusing to save maps on the server'):
                require_archive_mount('/tmp/not-mounted')

    def test_mountinfo_escaped_paths(self):
        info = '42 10 0:50 / /tmp/laptop\\040maps rw - fuse.sshfs :/tmp/maps rw\n'
        self.assertEqual(sshfs_mounts(info), {'/tmp/laptop maps'})

    def test_stale_mount_rejected_without_stat(self):
        info = '42 10 0:50 / /tmp/stale\\040maps rw - fuse.sshfs :/tmp/maps rw\n'
        with patch.object(sys, 'argv', ['remote', '/tmp/source', '/tmp/stale maps']), \
             patch.object(Path, 'read_text', return_value=info), \
             patch.object(Path, 'stat', side_effect=AssertionError('Must not stat stalled FUSE')):
            with self.assertRaisesRegex(SystemExit, 'Archive mount already exists'):
                exec(REMOTE_SETUP, {})

    def test_ssh_command_quotes_paths(self):
        import shlex
        args = SimpleNamespace(ssh_port=2200, server='cvr@example', server_mount="~/maps 'quoted' $(whoami)", identity=None)
        local = Path('/tmp/maps `literal`')
        cmd = ssh_command(args, local)
        parsed = shlex.split(cmd[-1])
        self.assertEqual(parsed[-2:], [str(local), args.server_mount])
        self.assertEqual(cmd[-3:-1], ['--', 'cvr@example'])

    @unittest.skipUnless(importlib.util.find_spec('viser'), 'optional viser dependency not installed')
    def test_headless_http_rate_gate_and_stop(self):
        import urllib.request
        from mast3r_slam.web_visualization import WebViewer
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        with patch.dict('os.environ', {'DISPLAY': ''}):
            viewer = WebViewer(port=port)
        try:
            self.assertEqual(urllib.request.urlopen(f'http://127.0.0.1:{port}', timeout=3).status, 200)
            self.assertFalse(viewer.ready(), 'No display copies without browser clients')
            with patch.object(viewer.server, 'get_clients', return_value={0: object()}):
                self.assertTrue(viewer.ready())
                self.assertFalse(viewer.ready(), 'Browser updates must be rate limited')
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                WebViewer(port=port)
        finally:
            viewer.close()
        with socket.socket() as probe:
            self.assertNotEqual(probe.connect_ex(('127.0.0.1', port)), 0)


if __name__ == '__main__':
    unittest.main()
