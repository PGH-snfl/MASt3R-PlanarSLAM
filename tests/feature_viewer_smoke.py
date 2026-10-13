"""Render synthetic CPU geometry in the real viewer, without model/camera use."""
import os
from pathlib import Path
import sys
import threading
import time


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.chdir(root)
    import numpy as np
    import torch
    import torch.multiprocessing as mp
    import lietorch
    from mast3r_slam.config import load_config, config
    from mast3r_slam.frame import SharedStates, SharedKeyframes, Mode, create_frame
    from mast3r_slam.feature_overlay import tracking_points
    from mast3r_slam.visualization import run_visualization
    load_config('config/d457_planar.yaml')
    config['depth_fusion']['enabled'] = False
    config['planes']['enabled'] = False
    torch.set_num_threads(2)
    mp.set_start_method('spawn', force=True)
    with mp.Manager() as manager:
        h, w = 384, 512
        states = SharedStates(manager, h, w, device='cpu')
        keyframes = SharedKeyframes(manager, h, w, buffer=1, device='cpu')
        frame = create_frame(0, np.full((h, w, 3), .45, np.float32), lietorch.Sim3.Identity(1), device='cpu')
        frame.K = torch.tensor([[400., 0, 256.], [0, 400., 192.], [0, 0, 1.]])
        frame.depth = torch.full((h, w), 2.)
        frame.update_pointmap(torch.zeros(h*w, 3), torch.ones(h*w, 1))
        frame.depth_source[:, w//2:] = 3
        frame.feat = torch.zeros(1, h*w//256, 1024)
        frame.pos = torch.zeros(1, h*w//256, 2, dtype=torch.long)
        frame.feature_points = tracking_points(frame, torch.arange(h*w), torch.ones(h*w, dtype=torch.bool), torch.ones(h*w))
        keyframes.set_intrinsics(frame.K)
        keyframes.append(frame)
        keyframes.history = torch.zeros(100, 6).share_memory_()
        keyframes.history[:10, 2] = 2.
        keyframes.history[:10, 3:] = .5
        keyframes.history_count = manager.Value("i", 10)
        keyframes.history_revision = manager.Value("i", 0)
        # Live runner avoids copying encoder tensors that rendering never uses.
        frame.feat.fill_(7.)
        states.set_frame(frame, include_features=False)
        assert states.feat.eq(0).all()
        states.set_mode(Mode.TRACKING)
        ready = threading.Event()
        def stop():
            ready.wait(30)
            for idx in range(1, 5):
                time.sleep(.4)
                frame.frame_id = idx
                keyframes[0] = frame
            states.set_mode(Mode.TERMINATED)
        worker = threading.Thread(target=stop, daemon=True)
        worker.start()
        from unittest.mock import patch
        from mast3r_slam.visualization import Window
        render = Window._render_locked
        seen = set()
        def checked_render(window, t, dt):
            render(window, t, dt)
            assert window.disk_map is None, 'Live viewer opened a disk map reader'
            assert len(window.textures) <= 2, 'Evicted GL textures leaked'
            assert hasattr(window, 'history_buffer'), 'Archived map was not rendered'
            seen.update(k for k in window.textures if isinstance(k, int))
        with patch.object(Window, '_render_locked', checked_render), \
             patch('mast3r_slam.disk_map.DiskMapReader', side_effect=AssertionError('Live map disk read')):
            run_visualization(config, states, keyframes, manager.Queue(), manager.Queue(), ready)
        assert len(seen) >= 3, f'Rolling keyframes did not render: {seen}' 
        worker.join(1)
        assert ready.is_set(), 'Viewer never rendered'
        print('PASS: actual GUI rendered markers, archived map and rolling keyframes; textures bounded; clean shutdown')


if __name__ == '__main__':
    main()
