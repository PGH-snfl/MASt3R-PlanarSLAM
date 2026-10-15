"""Exercise the D457 launcher with synthetic RGB-D and substitute inference.

Requires a little CUDA memory, but never opens a camera or loads model weights.
All generated keyframe files live in an automatically removed temporary folder.
"""
from pathlib import Path
import sys
import tempfile
import json
import os
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import run_d457
from mast3r_slam.streaming_backend import StreamingBackend


class Camera:
    def __init__(self, serial=None, recording=None):
        self.recording = recording
        self._pending = object()
        self.first = self._pending
        self.camera_intrinsics = SimpleNamespace(K_frame=np.array([[400., 0, 256.], [0, 400., 192.], [0, 0, 1.]], dtype=np.float32))
        self.last_depth = (2.+np.random.default_rng(5).normal(0, .005, (384, 512))).astype(np.float32)
        self.base_depth = self.last_depth.copy()
        self.last_valid_fraction = 1.
    def get_img_shape(self):
        return (384, 512), (384, 512)
    def __getitem__(self, idx):
        if self.recording is not None:
            from mast3r_slam.recorded_rgbd import RecordingEnded
            if idx == 0:
                assert self._pending is self.first, 'First recording frame was dropped'
            if idx == 16:
                raise RecordingEnded('EOF')
        self.last_depth = self.base_depth+(0 if idx == 0 else .004*(-1)**idx)
        return idx*.1, np.full((480, 640, 3), .4, np.float32)
    def close(self):
        pass


def inference(model, frame):
    h, w = frame.img.shape[-2:]
    frame.feat = torch.ones(1, h*w//256, 1024, device=frame.img.device)
    frame.pos = torch.zeros(1, h*w//256, 2, device=frame.img.device, dtype=torch.long)
    X = torch.zeros(h*w, 3, device=frame.img.device)
    X[:, 2] = 2.
    return X, torch.ones(h*w, 1, device=frame.img.device)*3


class Tracker:
    def __init__(self, *args):
        self.plane_matches = []
    def track(self, frame):
        frame.update_pointmap(*inference(None, frame))
        return frame.frame_id % 3 == 0, None, False
    def reset_idx_f2k(self):
        pass


class Database:
    def __init__(self):
        self.centroids = torch.ones(1, 1)
    def prep_features(self, features):
        return features[:, :1]
    def update_descriptors(self, *args, **kwargs):
        return []


def main():
    web_mode = '--web' in sys.argv
    recording_mode = '--recording' in sys.argv
    with tempfile.TemporaryDirectory(prefix='mast3r-runner-test-') as temp:
        archive = Path(temp)/'session'
        sys.argv = ['run_d457.py', '--no-viz', '--max-frames', '16', '--fps', '30',
                    '--archive-dir', str(archive), '--recent-keyframes', '3', '--image-size', '512']
        if recording_mode:
            fixture = Path(temp)/'fixture.db3'
            fixture.touch()
            sys.argv += ['--recording', str(fixture)]
            sys.argv[sys.argv.index('--max-frames')+1] = '0'
        if web_mode:
            sys.argv.remove('--no-viz')
            sys.argv += ['--web-viz', '--web-port', '18081', '--web-points', '1000']
        mount = os.environ.get('MAST3R_TEST_MOUNT')
        if mount:
            arg_idx = sys.argv.index('--archive-dir')
            sys.argv[arg_idx:arg_idx+2] = ['--archive-mount', mount]
        overview_counts = []
        original_publish = StreamingBackend.publish
        def check_live_publish(backend, view, states):
            assert not hasattr(view, 'map_directory'), 'Live viewer must not page map files'
            original_publish(backend, view, states)
            assert len(view) <= 3
            count = view.history_count.value
            assert 0 <= count <= 150000
            overview_counts.append(count)
        with patch('mast3r_slam.realsense_rgbd.RealSenseRGBDDataset', Camera), \
             patch('mast3r_slam.mast3r_utils.load_mast3r', return_value=torch.nn.Identity()), \
             patch('mast3r_slam.mast3r_utils.load_retriever', side_effect=lambda _: Database()), \
             patch('mast3r_slam.mast3r_utils.mast3r_inference_mono', inference), \
             patch('mast3r_slam.tracker.FrameTracker', Tracker), \
             patch('mast3r_slam.efficient_attention.enable_efficient_attention', return_value=0), \
             patch.object(StreamingBackend, 'publish', check_live_publish), \
             patch('mast3r_slam.web_visualization.WebViewer.ready', return_value=True), \
             patch('mast3r_slam.global_opt.FactorGraph.add_factors', return_value=False):
            run_d457.main()
        if mount:
            archive, = [p for p in Path(mount).iterdir() if p.is_dir()]
        assert not overview_counts, 'Headless tracking must skip all viewer publication'
        metrics = [json.loads(line) for line in (archive/'performance.jsonl').read_text().splitlines()]
        assert len(metrics) == 16
        assert all('viewer_frame' not in m['stages_ms'] for m in metrics)
        assert all(('viewer_web' in m['stages_ms']) == web_mode for m in metrics)
        assert all(m['map_pending'] <= 2 for m in metrics)
        manifest = json.loads((archive/'manifest.json').read_text())
        assert manifest['keyframes'] > 3, manifest
        assert len(list((archive/'keyframes').glob('*.pt'))) == manifest['keyframes']
        first = torch.load(archive/'keyframes'/'00000000.pt', weights_only=True)
        assert first['depth_source'].eq(1).all()
        assert first['depth_stats']['sensor'] == 384*512
        assert first['plane_depth_stats']['pixels'] > 0
        assert first['plane_depth_delta'].abs().max() <= .01
        torch.testing.assert_close(first['depth'], torch.from_numpy(Camera().last_depth)+first['plane_depth_delta']+first['surface_depth_delta'])
        torch.testing.assert_close(first['X_canon'][:, 2].reshape(384, 512), first['depth'])
        corrected = 0
        for path in (archive/'keyframes').glob('*.pt'):
            payload = torch.load(path, weights_only=True)
            corrected += int(payload['temporal_depth_stats']['pixels'] > 0)
            torch.testing.assert_close(payload['depth'], payload['raw_depth']+payload['plane_depth_delta']+payload['surface_depth_delta'])
            torch.testing.assert_close(payload['X_canon'][:, 2].reshape(384, 512), payload['depth'])
            assert payload['raw_depth'].device.type == 'cpu'
            assert payload['plane_depth_delta'].abs().max() <= .010001
        assert corrected >= 5, corrected
        summary = json.loads((archive/'map/summary.json').read_text())
        assert summary['keyframes'] == manifest['keyframes'], summary
        assert summary['keyframes']+summary['nonkeyframes'] == 16, summary
        assert summary['nonkeyframes'] > 0, summary
        from mast3r_slam.disk_map import DiskMapReader
        reader = DiskMapReader(archive, {})
        try:
            points, stats = reader.snapshot(np.array([0, 0, 0]))
            assert stats['frames'] == 16 and len(points), stats
            assert len(points) <= 1100000
        finally:
            reader.close()
        print(f'PASS: headless skips viewer, bounded map queue, full disk map retained, temporal correction on {corrected} keyframes')


if __name__ == '__main__':
    main()
