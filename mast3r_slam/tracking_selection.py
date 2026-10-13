"""Spatially distributed pose observations. Never alters stored frame geometry."""
from collections import deque
from statistics import median
import torch
import torch.nn.functional as F


def select_pose_points(valid, confidence, xyz, rgb, shape, cfg, cache=None):
    """Up to eight points per cell: confidence, boundaries, near and far.

    Returns None when full observations should be used. Selection indexes the
    reference image, just like the tracker's correspondence arrays.
    """
    if not cfg.get('enabled', False):
        return None
    h, w = map(int, shape)
    mask = valid.reshape(h, w) & torch.isfinite(xyz).all(-1).reshape(h, w)
    mask &= (xyz[:, 2] > 0).reshape(h, w)
    count = int(mask.sum())
    cell = int(cfg.get('cell_size', 16))
    if cell < 4:
        raise ValueError('Pose selection cell_size must be >= 4')
    minimum = int(cfg.get('min_points', 512))
    if count < minimum or count/(h*w) < cfg.get('min_valid_fraction', .15):
        return None
    ph, pw = (-h) % cell, (-w) % cell

    def tiles(x, fill):
        x = F.pad(x, (0, pw, 0, ph), value=fill)
        return x.reshape((h+ph)//cell, cell, (w+pw)//cell, cell).permute(0, 2, 1, 3).reshape(-1, cell*cell)

    cache = {} if cache is None else cache
    if not cache:
        z = torch.nan_to_num(xyz[:, 2].reshape(h, w), nan=0., posinf=0., neginf=0.)
        reference_valid = torch.isfinite(xyz).all(-1).reshape(h, w) & (z > 0)
        color = rgb.to(device=xyz.device, dtype=xyz.dtype).reshape(h, w, 3).mean(-1)
        edge = torch.zeros_like(z)
        # Boundaries get their own quota rather than outvoting the whole image.
        for dim in (0, 1):
            dz = torch.diff(z, dim=dim).abs()
            dc = torch.diff(color, dim=dim).abs()
            if dim == 0:
                delta = (dz / z[:-1].clamp_min(.25)).clamp_max(1) + dc
                delta *= reference_valid[:-1] & reference_valid[1:]
                edge[:-1] = torch.maximum(edge[:-1], delta)
                edge[1:] = torch.maximum(edge[1:], delta)
            else:
                delta = (dz / z[:, :-1].clamp_min(.25)).clamp_max(1) + dc
                delta *= reference_valid[:, :-1] & reference_valid[:, 1:]
                edge[:, :-1] = torch.maximum(edge[:, :-1], delta)
                edge[:, 1:] = torch.maximum(edge[:, 1:], delta)
        cache.update(z=z, edge=edge, ids=tiles(torch.arange(h*w, device=xyz.device).reshape(h, w), -1))
    z, edge, ids = cache['z'], cache['edge'], cache['ids']
    available = tiles(mask, False)
    score = torch.nan_to_num(confidence.reshape(h, w), nan=0., posinf=0., neginf=0.).clamp_min(0)
    selected = []
    for priority, quota in ((score, 4), (edge*score.sqrt(), 2), (-z, 1), (z, 1)):
        values = tiles(priority, -float('inf')).masked_fill(~available, -float('inf'))
        v, ix = values.topk(quota, dim=1)
        selected.append(ids.gather(1, ix)[torch.isfinite(v)])
        available.scatter_(1, ix, False)
    result = torch.cat(selected).sort().values
    if len(result) < minimum or len(result) >= count:
        return None
    return result


class AdaptiveSelection:
    """Compare recent total pose costs, including selection, validation/retries.

    Start with two full and two selected solves; probe the other path every 30
    calls. This bounds experimentation overhead without assuming sparsity wins.
    """
    def __init__(self):
        self.costs = {'full': deque(maxlen=5), 'selected': deque(maxlen=5)}
        self.calls = 0

    def choose(self):
        self.calls += 1
        if len(self.costs['full']) < 2:
            return False
        if len(self.costs['selected']) < 2:
            return True
        faster = median(self.costs['selected']) < .95*median(self.costs['full'])
        return not faster if self.calls % 30 == 0 else faster

    def record(self, selected, ms):
        self.costs['selected' if selected else 'full'].append(ms)
