"""Replay identical live RGB-D at 512/640 through the real SLAM pipeline.

Capture remains in RAM; temporary maps are removed. Only metrics are retained.
No ground-truth poses: this verifies operation/cost, not accuracy improvement.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_d457
from mast3r_slam.config import load_config, config
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.mast3r_utils import load_mast3r
from mast3r_slam.realsense_rgbd import RealSenseRGBDDataset, resize_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force-keyframes', action='store_true')
    parser.add_argument('--error-correction-ab', action='store_true',
                        help='Compare old/new error correction at 640 on identical input')
    args = parser.parse_args()
    if args.frames < 6:
        parser.error('at least 6 frames are needed')
    if args.output.exists():
        parser.error('output already exists')
    torch.set_num_threads(4)
    load_config('config/d457_planar.yaml')
    config['dataset']['image_size'] = 640
    camera = RealSenseRGBDDataset()
    samples = []
    try:
        intr = camera.camera_intrinsics
        for i in range(15):
            camera[i]
        for i in range(args.frames):
            timestamp, rgb = camera[i]
            samples.append((timestamp, rgb.copy(), camera.last_depth.copy()))
            time.sleep(.1)
    finally:
        camera.close()
    print(f'Captured {len(samples)} real RGB-D pairs in RAM; camera released', flush=True)
    model = load_mast3r(device='cuda:0').eval()
    results = []
    with tempfile.TemporaryDirectory(prefix='planar-resolution-') as temp:
        variants = [('baseline', 640), ('corrected', 640)] if args.error_correction_ab else [('512', 512), ('640', 640)]
        for variant, size in variants:
            calibrated = Intrinsics(size, intr.W, intr.H, intr.K_orig, intr.K,
                                   intr.distortion, intr.mapx, intr.mapy)
            resized = [(t, rgb, resize_depth(z, size)) for t, rgb, z in samples]
            shape = resized[0][2].shape

            class Replay:
                def __init__(self, *unused):
                    self.camera_intrinsics = calibrated
                    self.last_depth = resized[0][2].copy()
                    self.last_valid_fraction = float((self.last_depth > 0).mean())
                def get_img_shape(self):
                    return shape, samples[0][1].shape[:2]
                def __getitem__(self, idx):
                    t, rgb, z = resized[idx]
                    self.last_depth = z.copy()
                    self.last_valid_fraction = float((z > 0).mean())
                    return t, rgb.copy()
                def close(self):
                    pass

            session = Path(temp)/variant
            argv = ['run_d457.py', '--no-viz', '--profile', '--max-frames', str(args.frames),
                    '--fps', '1000', '--archive-dir', str(session), '--image-size', str(size)]
            if args.force_keyframes:
                argv += ['--config', 'config/d457_backend_check.yaml']
            if variant == 'baseline':
                argv += ['--disable-surface-depth', '--disable-global-poses']
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with patch.object(sys, 'argv', argv), \
                 patch('mast3r_slam.realsense_rgbd.RealSenseRGBDDataset', Replay), \
                 patch('mast3r_slam.mast3r_utils.load_mast3r', return_value=model):
                run_d457.main()
            rows = [json.loads(line) for line in (session/'performance.jsonl').read_text().splitlines()]
            assert len(rows) == args.frames
            saved = json.loads((session/'map/summary.json').read_text())
            assert saved['keyframes'] + saved['nonkeyframes'] == args.frames, saved
            for path in (session/'keyframes').glob('*.pt'):
                frame = torch.load(path, map_location='cpu', weights_only=True)
                assert tuple(frame['depth'].shape) == shape
                assert tuple(frame['img'].shape[-2:]) == shape
                assert frame['X_canon'].shape == (shape[0]*shape[1], 3)
                assert frame['feat'].shape[-2] == shape[0]*shape[1]//256
                assert torch.isfinite(frame['X_canon']).all()
                np.testing.assert_allclose(frame['K'], calibrated.K_frame, rtol=1e-6)
                surface_delta = frame.get('surface_depth_delta')
                if surface_delta is not None:
                    assert surface_delta.abs().max() <= .004001
            steady = rows[3:]
            result = dict(variant=variant, size=size, frames=len(rows), image_shape=shape,
                          tracking=sum(r['mode'] == 'TRACKING' for r in rows),
                          median_step_ms=float(np.median([r['step_ms'] for r in steady])),
                          p95_step_ms=float(np.percentile([r['step_ms'] for r in steady], 95)),
                          peak_allocated_mb=max(r['cuda_peak_allocated_mb'] for r in rows),
                          max_reserved_mb=max(r['cuda_reserved_mb'] for r in rows),
                          median_stages_ms={k: float(np.median([r['stages_ms'].get(k, 0) for r in steady]))
                                            for k in ('model', 'matching', 'depth_spatial', 'depth_surface', 'pose_total',
                                                      'backend_new_keyframe', 'backend_tick', 'map_submit')},
                          saved=saved, rows=rows)
            results.append(result)
            print('RESOLUTION_RESULT '+json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(results=results, forced_keyframes=args.force_keyframes,
                                         note='Same captured input; no ground truth. Viewer disabled. First 3 frames excluded from timing statistics.'), indent=2))
    print('PASS: 512/640 inference, tracking, saved RGB/depth/K/features/pointmap shapes and finite geometry')


if __name__ == '__main__':
    main()
