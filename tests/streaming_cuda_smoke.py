"""Small synthetic CUDA test; does not open the camera or load MASt3R weights."""
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import lietorch
from mast3r_slam.config import load_config, config
from mast3r_slam.frame import Frame
from mast3r_slam.keyframe_archive import KeyframeArchive
from mast3r_slam.global_opt import FactorGraph


def main():
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    load_config('config/d457_planar.yaml')
    config['planes']['enabled'] = False
    config['local_opt']['max_iters'] = 10
    if torch.cuda.mem_get_info()[0] < 512*1024**2:
        raise RuntimeError('Insufficient free GPU memory for isolated smoke test')
    K = torch.tensor([[20., 0, 8], [0, 20., 8], [0, 0, 1]], device='cuda')
    y, x = torch.meshgrid(torch.arange(16, device='cuda'), torch.arange(16, device='cuda'), indexing='ij')
    xyz = torch.stack(((x-8)/10., (y-8)/10., torch.full_like(x, 2.)), -1).reshape(-1, 3).float()
    with tempfile.TemporaryDirectory(prefix='mast3r-cuda-test-') as temp:
        store = KeyframeArchive(Path(temp)/'session', recent=3, device='cuda', preview_limit=20)
        store.set_intrinsics(K)
        try:
            for idx in range(40):
                pose = lietorch.Sim3.Identity(1, device='cuda')
                pose.data[0, 0] = idx*.002
                frame = Frame(idx, torch.zeros(3, 16, 16, device='cuda'), torch.tensor([[16, 16]], device='cuda'),
                              torch.tensor([[16, 16]], device='cuda'), torch.ones(16, 16, 3)*.4, pose,
                              X_canon=xyz.clone(), C=torch.ones(256, 1, device='cuda')*3,
                              feat=torch.zeros(1, 1, 1024, device='cuda'), pos=torch.zeros(1, 1, 2, device='cuda', dtype=torch.long),
                              N=1, N_updates=1, K=K, depth=torch.ones(16, 16)*2)
                store.append(frame)
            store.prefetch(0)
            deadline = time.monotonic()+10
            while not store.poll(0):
                if time.monotonic()>deadline: raise TimeoutError('Archive load')
                time.sleep(.01)
            graph = FactorGraph(None, store, K, 'cuda')
            graph.ii = torch.tensor([0, 37, 38], device='cuda')
            graph.jj = torch.tensor([37, 38, 39], device='cuda')
            graph.idx_ii2jj = torch.arange(256, device='cuda').repeat(3, 1)
            graph.idx_jj2ii = graph.idx_ii2jj.clone()
            graph.valid_match_i = torch.ones(3, 256, 1, device='cuda', dtype=torch.bool)
            graph.valid_match_j = graph.valid_match_i.clone()
            graph.Q_ii2jj = torch.ones(3, 256, 1, device='cuda')*4
            graph.Q_jj2ii = graph.Q_ii2jj.clone()
            graph.solve_GN_calib()
            for idx in (37, 38, 39):
                assert torch.isfinite(store[idx].T_WC.data).all()
                assert store[idx].T_WC.data[0, :3].abs().max() < .001, store[idx].T_WC.data
            graph.retain_frames(store.recent)
            assert graph.ii.tolist() == [37, 38]
            assert len(store.recent) == 3 and len(store.cache) == 1
            print('PASS: CUDA graph optimized sparse global IDs [0,37,38,39] against reloaded keyframe 0; bounded 3+1 frames')
        finally:
            store.close()


if __name__ == '__main__':
    main()
