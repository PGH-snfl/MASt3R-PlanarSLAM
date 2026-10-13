"""Explicit hardware test: drop sensor depth in RAM; temporary keyframe archive removed afterwards.

Run with the project Python, from the project root. Not a unittest discovery test.
"""
import sys
from pathlib import Path


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mast3r_slam.realsense_rgbd import RealSenseRGBDDataset
    from mast3r_slam import depth_fusion
    import run_d457
    read = RealSenseRGBDDataset.__getitem__
    fuse = depth_fusion.fuse_depth
    results = []

    def drop(self, idx):
        sample = read(self, idx)
        if idx in (2, 3):
            self.last_depth = self.last_depth.copy()
            self.last_depth[:] = 0
            self.last_valid_fraction = 0.
            print(f"TEST: all sensor depth dropped at frame {idx}", flush=True)
        return sample

    def observe(sensor, model, confidence, cfg):
        value = fuse(sensor, model, confidence, cfg)
        results.append(value[3])
        return value

    RealSenseRGBDDataset.__getitem__ = drop
    depth_fusion.fuse_depth = observe
    sys.argv = ["run_d457.py", "--no-viz", "--max-frames", "6", "--config", "config/d457_backend_check.yaml"]
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mast3r-depth-test-") as temp:
        sys.argv += ["--archive-dir", str(Path(temp)/"session")]
        run_d457.main()
    assert len(results) == 6, results
    for idx in (2, 3):
        assert results[idx]['sensor'] == 0 and results[idx]['filled'] > 0, results[idx]
    assert results[4]['sensor'] > 0, results[4]
    print("PASS: complete sensor-depth dropout used model geometry and sensor input recovered")


if __name__ == '__main__':
    main()
