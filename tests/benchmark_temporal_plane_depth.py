"""Deterministic moving-camera plane benchmark; no camera/GPU/model required.

Reports synthetic errors, not D457 accuracy. Timings exclude plane detection,
model inference, pose tracking and CUDA transfers.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mast3r_slam.config import config, load_config
from mast3r_slam.plane_depth import correct_large_planes
from mast3r_slam.temporal_plane_depth import TemporalPlaneDepth


def main():
    load_config('config/d457_planar.yaml')
    temporal = TemporalPlaneDepth(config['temporal_plane_depth'])
    h, w = 384, 512
    K = np.array([[400., 0, 256.], [0, 400., 192.], [0, 0, 1.]], np.float32)
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack(((xx-256)/400, (yy-192)/400, np.ones((h, w))), -1)
    rng = np.random.default_rng(102)
    source = np.ones((h, w), np.uint8)
    confidence = np.full((h, w), 3., np.float32)
    raw_error, spatial_error, temporal_error, timings, counts = [], [], [], [], []
    for i in range(48):
        pose = np.eye(4)
        pose[2, 3] = .005*i
        truth = 2.-pose[2, 3]
        z = (truth+(.004 if i % 2 else -.004)+rng.normal(0, .003, (h, w))).astype(np.float32)
        planes = np.array([[0, 0, -1, truth, 0, 0, truth, 1]], np.float32)
        obs = []
        spatial, delta, _, _ = correct_large_planes(z, K, source, confidence, planes,
                                                    config['plane_depth'], observations=obs)
        start = time.perf_counter()
        out, _, stats = temporal.correct(spatial, delta, K, obs, pose, i, timestamp=.1*i)
        elapsed = time.perf_counter()-start
        if i > 3:
            crop = np.s_[10:-10, 10:-10]
            raw_error.append(np.mean((z[crop]-truth)**2))
            spatial_error.append(np.mean((spatial[crop]-truth)**2))
            temporal_error.append(np.mean((out[crop]-truth)**2))
            timings.append(elapsed*1000)
            counts.append(stats['pixels'])
    print(json.dumps(dict(input='synthetic world plane; known camera pose', shape=[h, w],
        frames=48, evaluation_frames=44, pixel_noise_std_mm=3, alternating_frame_bias_mm=4,
        camera_forward_step_mm=5, raw_rms_mm=float(np.sqrt(np.mean(raw_error))*1000),
        spatial_rms_mm=float(np.sqrt(np.mean(spatial_error))*1000),
        temporal_rms_mm=float(np.sqrt(np.mean(temporal_error))*1000),
        temporal_cpu_median_ms=float(np.median(timings)), temporal_cpu_p95_ms=float(np.percentile(timings, 95)),
        mean_corrected_pixels=float(np.mean(counts))), indent=2))


if __name__ == '__main__':
    main()
