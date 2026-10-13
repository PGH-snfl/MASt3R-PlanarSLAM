"""Capture a short USB sequence in RAM, then replay identical input A/B/B/A.

Temporary map output is removed on exit; RGB-D camera frames are never saved.
This tests the real model/tracker/backend/writer with the viewer disabled.
No ground-truth trajectory is available, so it is not an accuracy benchmark.
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
from mast3r_slam.config import load_config
from mast3r_slam.realsense_rgbd import RealSenseRGBDDataset
from mast3r_slam.mast3r_utils import load_mast3r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=20)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    load_config('config/d457_planar.yaml')
    camera = RealSenseRGBDDataset()
    samples = []
    try:
        shape = camera.get_img_shape()
        for i in range(15):
            camera[i]
        intrinsics = camera.camera_intrinsics
        for i in range(args.frames):
            timestamp, rgb = camera[i]
            samples.append((timestamp, rgb.copy(), camera.last_depth.copy()))
            time.sleep(.1)
    finally:
        camera.close()
    print(f'Captured {len(samples)} RGB-D frames in RAM; camera released', flush=True)

    class Replay:
        def __init__(self, *unused):
            self.camera_intrinsics = intrinsics
            self.last_depth = samples[0][2].copy()
            self.last_valid_fraction = float((self.last_depth > 0).mean())
        def get_img_shape(self):
            return shape
        def __getitem__(self, idx):
            timestamp, rgb, depth = samples[idx]
            self.last_depth = depth.copy()
            self.last_valid_fraction = float((depth > 0).mean())
            return timestamp, rgb.copy()
        def close(self):
            pass

    model = load_mast3r(device='cuda:0').eval()
    summaries = []
    with tempfile.TemporaryDirectory(prefix='planar-paired-replay-') as temp:
        for trial, label in enumerate(('full', 'selected', 'selected', 'full')):
            session = Path(temp)/str(trial)
            argv = ['run_d457.py', '--no-viz', '--max-frames', str(args.frames), '--fps', '1000',
                    '--archive-dir', str(session)]
            if label == 'full':
                argv += ['--full-correspondences']
            if args.profile:
                argv += ['--profile']
            with patch.object(sys, 'argv', argv), \
                 patch('mast3r_slam.realsense_rgbd.RealSenseRGBDDataset', Replay), \
                 patch('mast3r_slam.mast3r_utils.load_mast3r', return_value=model):
                run_d457.main()
            rows = [json.loads(line) for line in (session/'performance.jsonl').read_text().splitlines()]
            steady = rows[3:]
            stages = {key: float(np.median([r['stages_ms'].get(key, 0) for r in steady]))
                      for key in ('model', 'matching', 'depth_spatial', 'pose_total', 'map_submit')}
            saved = json.loads((session/'map/summary.json').read_text())
            summary = dict(trial=trial, variant=label, frames=len(rows),
                           tracking=sum(r['mode'] == 'TRACKING' for r in rows),
                           fallback=sum(r['pose_points'].get('fallback', False) for r in rows),
                           median_step_ms=float(np.median([r['step_ms'] for r in steady])),
                           p95_step_ms=float(np.percentile([r['step_ms'] for r in steady], 95)),
                           median_stages_ms=stages, saved=saved)
            summaries.append(summary)
            print('REPLAY_RESULT '+json.dumps(summary), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print('REPLAY_SUMMARY '+json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
