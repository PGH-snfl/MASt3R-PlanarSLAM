"""CPU sparse Sim(3) pose graph from bounded raw RGB-D registrations.

This is a pose-only graph, not dense bundle adjustment. Weights and acceptance
gates are conservative heuristics, not calibrated sensor covariances.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


def matrices(vectors):
    v = np.asarray(vectors, dtype=np.float64).reshape(-1, 8)
    result = np.tile(np.eye(4), (len(v), 1, 1))
    result[:, :3, :3] = Rotation.from_quat(v[:, 3:7]).as_matrix()*v[:, 7, None, None]
    result[:, :3, 3] = v[:, :3]
    return result


def vectors(mats):
    m = np.asarray(mats)
    scale = np.linalg.norm(m[:, :3, 0], axis=1)
    return np.column_stack((m[:, :3, 3], Rotation.from_matrix(m[:, :3, :3]/scale[:, None, None]).as_quat(), scale))


def fit_similarity(source, target):
    a, b = source-source.mean(0), target-target.mean(0)
    U, singular, Vt = np.linalg.svd(b.T @ a / len(a))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(U @ Vt)
    R = (U*sign) @ Vt
    if singular[1] < 1e-8 or np.mean(np.sum(a*a, axis=1)) < 1e-8:
        raise ValueError('degenerate correspondence geometry')
    scale = (singular*sign).sum()/np.mean(np.sum(a*a, axis=1))
    out = np.eye(4)
    out[:3, :3] = scale*R
    out[:3, 3] = target.mean(0)-scale*R @ source.mean(0)
    return out


def robust_registration(source, target, cfg):
    a, b = np.asarray(source, float), np.asarray(target, float)
    good = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (a[:, 2] > 0) & (b[:, 2] > 0)
    a, b = a[good], b[good]
    minimum = int(cfg.get('min_points', 100))
    if len(a) < minimum:
        return None
    rng = np.random.default_rng(19)
    threshold = cfg.get('inlier_distance_m', .04)
    best = np.zeros(len(a), bool)
    for _ in range(int(cfg.get('ransac_trials', 64))):
        ids = rng.choice(len(a), 3, replace=False)
        try:
            T = fit_similarity(a[ids], b[ids])
        except ValueError:
            continue
        scale = np.linalg.norm(T[:3, 0])
        if not .9 <= scale <= 1.1:
            continue
        mask = np.linalg.norm(a @ T[:3, :3].T+T[:3, 3]-b, axis=1) < threshold
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < max(minimum, cfg.get('min_inlier_fraction', .6)*len(a)):
        return None
    for _ in range(2):
        try:
            T = fit_similarity(a[best], b[best])
        except ValueError:
            return None
        residual = np.linalg.norm(a @ T[:3, :3].T+T[:3, 3]-b, axis=1)
        best = residual < threshold
        if best.sum() < max(minimum, cfg.get('min_inlier_fraction', .6)*len(a)):
            return None
    rms = float(np.sqrt(np.mean(residual[best]**2)))
    if rms > cfg.get('max_rmse_m', .02) or not .9 <= np.linalg.norm(T[:3, 0]) <= 1.1:
        return None
    return T, dict(inliers=int(best.sum()), samples=len(a), rmse_m=rms)


def measurement_from_graph(graph, source_id, target_id, cfg):
    """Return T_source^-1 T_target using matches and uncorrected real depths."""
    import torch
    rows = torch.where((graph.ii == source_id) & (graph.jj == target_id))[0]
    if not len(rows):
        return None
    a, b = graph.frames[source_id], graph.frames[target_id]
    if any(f.raw_depth is None or f.depth_source is None for f in (a, b)):
        return None
    row = int(rows[-1])
    ids_a = graph.idx_ii2jj[row].reshape(-1).detach().cpu().numpy()
    valid = graph.valid_match_j[row].reshape(-1).detach().cpu().numpy().copy()
    valid &= (ids_a >= 0) & (ids_a < a.raw_depth.numel())
    safe = np.clip(ids_a, 0, a.raw_depth.numel()-1)
    za, zb = a.raw_depth.numpy().ravel(), b.raw_depth.numpy().ravel()
    valid &= (a.depth_source.numpy().ravel()[safe] == 1) & (b.depth_source.numpy().ravel() == 1)
    valid &= np.isfinite(za[safe]) & np.isfinite(zb) & (za[safe] > 0) & (zb > 0)
    valid &= graph.Q_ii2jj[row].reshape(-1).detach().cpu().numpy() > 0
    ids_b = np.flatnonzero(valid)
    limit = int(cfg.get('max_points', 2048))
    if len(ids_b) > limit:
        ids_b = ids_b[np.linspace(0, len(ids_b)-1, limit, dtype=int)]
    def points(frame, ids):
        K = frame.K.detach().cpu().numpy()
        y, x = np.divmod(ids, frame.raw_depth.shape[1])
        z = frame.raw_depth.numpy().ravel()[ids]
        return np.column_stack(((x-K[0, 2])*z/K[0, 0], (y-K[1, 2])*z/K[1, 1], z))
    return robust_registration(points(b, ids_b), points(a, safe[ids_b]), cfg)


def optimize_pose_graph(initial, edges, cfg):
    """Optimize local increments; anchor every disconnected component."""
    started = time.monotonic()
    initial = np.asarray(initial, float)
    n = len(initial)
    ii = np.array([e['source'] for e in edges], int)
    jj = np.array([e['target'] for e in edges], int)
    Z = np.array([e['measurement'] for e in edges])
    info = dict(accepted=False, nodes=n, edges=len(edges), reason='no_loop')
    if not len(edges) or not any(e['kind'] != 'temporal' for e in edges):
        return initial, info
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(ii, jj):
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)
    free = [i for i in range(n) if root(i) != i]
    index = {node: k for k, node in enumerate(free)}
    if not free:
        return initial, info
    sigma = np.array([cfg.get('translation_sigma_m', .03)]*3 +
                     [np.deg2rad(cfg.get('rotation_sigma_deg', 2.))]*3 +
                     [cfg.get('scale_sigma', .02)])
    Zinv = np.linalg.inv(Z)
    def unpack(x):
        delta = np.tile(np.eye(4), (len(free), 1, 1))
        d = x.reshape(-1, 7)
        delta[:, :3, 3] = d[:, :3]
        delta[:, :3, :3] = Rotation.from_rotvec(d[:, 3:6]).as_matrix()*np.exp(d[:, 6, None, None])
        result = initial.copy()
        result[free] = delta @ initial[free]
        return result
    def residual(x):
        poses = unpack(x)
        error = Zinv @ np.linalg.inv(poses[ii]) @ poses[jj]
        s = np.linalg.norm(error[:, :3, 0], axis=1)
        r = np.column_stack((error[:, :3, 3], Rotation.from_matrix(error[:, :3, :3]/s[:, None, None]).as_rotvec(), np.log(s)))
        return (r/sigma).ravel()
    pattern = lil_matrix((7*len(edges), 7*len(free)), dtype=int)
    for k, (i, j) in enumerate(zip(ii, jj)):
        for node in (i, j):
            if node in index:
                pattern[k*7:(k+1)*7, index[node]*7:(index[node]+1)*7] = 1
    zero = np.zeros(7*len(free))
    before = residual(zero)
    result = least_squares(residual, zero, jac_sparsity=pattern.tocsr(), loss='huber', f_scale=1.,
                           max_nfev=int(cfg.get('max_iterations', 20)), ftol=1e-5, xtol=1e-5, gtol=1e-5)
    after = residual(result.x)
    def cost(r):
        a = np.abs(r)
        return float(np.where(a <= 1, .5*a*a, a-.5).sum())
    output = unpack(result.x)
    correction = output @ np.linalg.inv(initial)
    v = vectors(correction)
    moves = np.linalg.norm(output[:, :3, 3]-initial[:, :3, 3], axis=1)
    rotations = Rotation.from_quat(v[:, 3:7]).magnitude()
    # Reject large jumps and any solution that tears previously consistent edges.
    before_edges, after_edges = before.reshape(-1, 7), after.reshape(-1, 7)
    consistent = np.linalg.norm(before_edges, axis=1) < 2
    safe = (np.isfinite(output).all() and moves.max() <= cfg.get('max_translation_m', .5)
            and rotations.max() <= np.deg2rad(cfg.get('max_rotation_deg', 10))
            and np.abs(np.log(v[:, 7])).max() <= cfg.get('max_log_scale', .05)
            and np.all(np.linalg.norm(after_edges[consistent], axis=1) < 4))
    accepted = bool(safe and cost(after) < cost(before)-1e-7)
    info.update(accepted=accepted, reason='accepted' if accepted else 'cost_or_safety_gate',
                cost_before=cost(before), cost_after=cost(after), max_translation_m=float(moves.max()),
                elapsed_ms=1000*(time.monotonic()-started))
    return output if accepted else initial, info


class GlobalPoseGraph:
    def __init__(self, store, cfg):
        self.store, self.cfg = store, dict(cfg)
        self.enabled = cfg.get('enabled', False)
        self.edges = {}
        self.worker = ThreadPoolExecutor(1, thread_name_prefix='global-pose') if self.enabled else None
        self.pending = None
        self.dirty = False
        self.next_solve = 0.
        self.stats = dict(status='idle')

    def add(self, graph, i, j, kind):
        if not self.enabled or len(self.store) > self.cfg.get('max_nodes', 1500):
            return
        result = measurement_from_graph(graph, i, j, self.cfg)
        if result is None:
            self.store.record('global_constraint_rejected', source=i, target=j, kind=kind)
            return
        measurement, quality = result
        edge = dict(source=int(i), target=int(j), kind=kind, measurement=measurement.tolist(), **quality)
        key = (int(i), int(j), kind)
        if key in self.edges:
            return
        self.edges[key] = edge
        self.store.record('global_constraint', **edge)
        self.dirty |= kind != 'temporal'

    def poll(self):
        applied = False
        if self.pending is not None and self.pending.done():
            poses, info = self.pending.result()
            self.pending = None
            # Never overwrite a newer local optimization with stale work.
            if info['accepted'] and self.snapshot == (len(self.store), self.store.pose_revision):
                self.store.apply_global_poses(vectors(poses))
                applied = True
                info['status'] = 'applied'
            elif info['accepted']:
                info['status'] = 'stale'
                self.dirty = True
            else:
                info['status'] = 'rejected'
            self.stats = info
            self.store.record('global_optimization', **info)
        return applied

    def submit(self):
        if (not self.enabled or not self.dirty or self.pending is not None
                or time.monotonic() < self.next_solve):
            return
        if len(self.store) > self.cfg.get('max_nodes', 1500):
            self.stats = dict(status='node_limit', nodes=len(self.store))
            return
        self.snapshot = (len(self.store), self.store.pose_revision)
        poses = matrices(np.array([p.numpy() for p in self.store.poses]))
        self.pending = self.worker.submit(optimize_pose_graph, poses, list(self.edges.values()), self.cfg)
        self.next_solve = time.monotonic()+self.cfg.get('min_interval_s', 2.)
        self.dirty = False

    def close(self):
        if self.worker is not None:
            # Shutdown must not mutate poses after the disk map's final sync.
            self.worker.shutdown(wait=True)
            if self.pending is not None:
                self.pending.result()
