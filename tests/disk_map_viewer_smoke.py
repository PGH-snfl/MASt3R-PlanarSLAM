"""Open/reload a saved map in the real OpenGL viewer, with no camera/model."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))


def main():
    os.chdir(ROOT)
    import torch
    import numpy as np
    from test_disk_map import make_frame
    from mast3r_slam.disk_map import DiskMapWriter
    from mast3r_slam.keyframe_archive import KeyframeArchive
    from mast3r_slam.frame import Mode
    from mast3r_slam.visualization import Window, run_visualization
    import view_saved_map
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='mast3r-map-view-') as temp:
        session = Path(temp)/'session'
        store = KeyframeArchive(session, recent=2, device='cpu', save_on_append=True)
        writer = DiskMapWriter(session, {})
        try:
            frame = make_frame(0)
            store.set_intrinsics(frame.K)
            store.append(frame)
            writer.sync_poses(store)
            writer.add_frame(frame, 0, np.eye(4), 0, 0)
            for _, future in store.writes:
                future.result()
            for future in writer.pending:
                future.result()
            loaded = threading.Event()
            seen, counts = set(), []
            original = Window.render_disk_map
            def checked(window):
                original(window)
                if window.disk_map_count:
                    seen.add(window.disk_map_stats['frames'])
                    counts.append(window.disk_map_count)
                    loaded.set()
            def run(cfg, states, keyframes, main2viz, viz2main):
                def update_and_stop():
                    if loaded.wait(15):
                        for i in (1, 2):
                            frame = make_frame(i, x=i*.02)
                            writer.add_frame(frame, 0, np.eye(4), i*.1)
                            for future in writer.pending:
                                future.result()
                            time.sleep(.8)
                    states.set_mode(Mode.TERMINATED)
                worker = threading.Thread(target=update_and_stop, daemon=True)
                worker.start()
                try:
                    return run_visualization(cfg, states, keyframes, main2viz, viz2main)
                finally:
                    worker.join(20)
            with patch.object(sys, 'argv', ['view_saved_map.py', str(session)]), \
                 patch.object(Window, 'render_disk_map', checked), \
                 patch('mast3r_slam.visualization.run_visualization', run):
                view_saved_map.main()
            assert loaded.is_set() and 3 in seen, (seen, counts)
            assert max(counts) <= 1100000, counts
            print(f'PASS: saved map reopened in actual GUI, streamed new nonkeyframes: {sorted(seen)}')
        finally:
            writer.close()
            store.close()


if __name__ == '__main__':
    main()
