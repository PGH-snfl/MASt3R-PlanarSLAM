"""Bounded display samples of accepted tracking correspondences, not landmarks."""
import numpy as np
import torch

FEATURE_LIMIT = 512


def tracking_points(frame, indices, valid, scores, cell_size=16):
    """One high-scoring current-frame correspondence per image cell."""
    h, w = frame.img.shape[-2:]
    mask = valid.detach().reshape(-1).cpu().numpy().astype(bool)
    ids = indices.detach().reshape(-1).cpu().numpy()[mask]
    score = scores.detach().reshape(-1).cpu().numpy()[mask]
    good = (ids >= 0) & (ids < h*w) & np.isfinite(score)
    ids, score = ids[good], score[good]
    if not len(ids):
        return torch.empty((0, 6), dtype=torch.float32)
    ids = ids[np.argsort(-score, kind="stable")]
    cells = (ids//w//cell_size)*((w+cell_size-1)//cell_size)+(ids % w)//cell_size
    _, first = np.unique(cells, return_index=True)
    ids = ids[np.sort(first)[:FEATURE_LIMIT]]
    point_ids = torch.as_tensor(ids, device=frame.X_canon.device, dtype=torch.long)
    xyz = frame.X_canon[point_ids].detach().cpu().numpy()
    colors = np.tile(np.array([0., 1., 1.], dtype=np.float32), (len(ids), 1))
    if frame.depth_source is not None:
        source = frame.depth_source.reshape(-1)[torch.as_tensor(ids)].cpu().numpy()
        colors[source >= 2] = [1., .55, 0.]
    good = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    return torch.from_numpy(np.concatenate([xyz[good], colors[good]], axis=1).astype(np.float32))
