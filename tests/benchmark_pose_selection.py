"""Same synthetic observations/initial pose through full and selected solves.

No camera or model. Measures selection + full-residual validation + solve, so
these timings must not be reported as end-to-end SLAM FPS.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import lietorch
from test_tracking_selection import observations
from mast3r_slam.config import load_config, config
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.runtime_metrics import StageTimer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--runs', type=int, default=12)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    load_config('config/d457_planar.yaml')
    tracker = FrameTracker(None, None, args.device)
    tracker.metric_input = True
    tracker.plane_matches = []
    base = copy.deepcopy(tracker.cfg)
    h, w = 384, 512
    xk, K, meas, valid, q, rgb = observations(h, w, args.device)
    truth = lietorch.Sim3.exp(xk.new_tensor([[.025, -.012, .008, .01, -.008, .004, 0.]]))
    identity = lietorch.Sim3.Identity(1, device=args.device)
    xf0 = truth.inv().act(xk)
    results = {k: [] for k in ('full', 'selected')}
    generator = torch.Generator(device=args.device).manual_seed(7)
    for run in range(args.runs+2):
        xf = xf0 + torch.randn(xf0.shape, device=args.device, generator=generator)*.001
        # 3% incorrect correspondences; identical data for both variants.
        bad = torch.rand(len(xf), device=args.device, generator=generator) < .03
        xf[bad] += .05
        for label in (('full', 'selected') if run%2 == 0 else ('selected', 'full')):
            tracker.cfg = copy.deepcopy(base)
            tracker.cfg['point_selection']['enabled'] = label == 'selected'
            tracker.cfg['point_selection']['adaptive'] = False
            timer = StageTimer(args.device.startswith('cuda'))
            timer.fence()
            started = time.perf_counter()
            pose, _ = tracker.solve_selected_pose(xf, xk, identity, identity, q, valid, meas,
                                                   valid, K, (h, w), rgb, timer)
            timer.fence()
            elapsed = (time.perf_counter()-started)*1000
            err = (truth.inv()*pose).log()[0]
            error_mm = float(err[:3].norm()*1000)
            error_deg = float(err[3:6].norm()*180/np.pi)
            assert error_mm < 3., (label, error_mm)
            assert error_deg < .1, (label, error_deg)
            if run >= 2:
                results[label].append(dict(ms=elapsed, error_mm=error_mm, error_deg=error_deg,
                                          **tracker.selection_stats))
    for label, rows in results.items():
        print(json.dumps(dict(variant=label, device=args.device, runs=len(rows),
                             median_ms=float(np.median([r['ms'] for r in rows])),
                             p95_ms=float(np.percentile([r['ms'] for r in rows], 95)),
                             max_translation_error_mm=max(r['error_mm'] for r in rows),
                             max_rotation_error_deg=max(r['error_deg'] for r in rows),
                             fallbacks=sum(r['fallback'] for r in rows),
                             selected=rows[0]['selected'], available=rows[0]['available'])))


if __name__ == '__main__':
    main()
