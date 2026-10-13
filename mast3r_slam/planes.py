"""Plane features and soft Sim(3) factors, inspired by PlanarSLAM (ICRA 2021).

Independent implementation: robust depth-plane fitting, normal/distance matching,
and point/plane pose refinement. This is not a reproduction of the complete
point-line-plane/Manhattan system. Plane rows are [nx,ny,nz,d,cx,cy,cz,support].
Camera coordinates use x right, y down, z forward; n.X + d = 0, metres.
"""
import numpy as np
import torch
from mast3r_slam.geometry import skew_sym


def extract_planes(depth, K, cfg, confidence=None):
    """Bounded, seeded CPU RANSAC; reject holes, lines and tiny patches."""
    depth = np.asarray(depth, dtype=np.float32)
    stride = int(cfg.get("sample_stride", 6))
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    good = np.isfinite(z) & (z > 0)
    quality = np.ones_like(z) if confidence is None else np.asarray(confidence)[::stride, ::stride]
    good &= np.isfinite(quality) & (quality > 0)
    weights = np.clip(quality[good], 0, 1).astype(np.float32)
    points = np.column_stack(((u[good]-K[0, 2])*z[good]/K[0, 0],
                              (v[good]-K[1, 2])*z[good]/K[1, 1], z[good])).astype(np.float32)
    total = len(points)
    minimum = max(int(cfg.get("min_points", 100)), int(total*cfg.get("min_fraction", .04)))
    result = np.zeros((int(cfg.get("max_planes", 8)), 8), dtype=np.float32)
    rng = np.random.default_rng(42)
    threshold = float(cfg.get("distance_threshold", .02))
    for slot in range(len(result)):
        if len(points) < max(minimum, 3):
            break
        triples = points[rng.integers(len(points), size=(int(cfg.get("ransac_trials", 96)), 3))]
        normals = np.cross(triples[:, 1]-triples[:, 0], triples[:, 2]-triples[:, 0])
        norm = np.linalg.norm(normals, axis=1)
        valid = norm > 1e-6
        if not valid.any():
            break
        normals = normals[valid]/norm[valid, None]
        offsets = -(normals*triples[valid, 0]).sum(1)
        distances = np.abs(points @ normals.T + offsets)
        best = np.argmax(((distances < threshold)*weights[:, None]).sum(0))
        mask = distances[:, best] < threshold
        if mask.sum() < minimum:
            break
        patch = points[mask]
        center = np.average(patch, axis=0, weights=weights[mask])
        _, singular, vh = np.linalg.svd((patch-center)*np.sqrt(weights[mask, None]), full_matrices=False)
        if singular[1]/np.sqrt(len(patch)) < .02:
            points = points[~mask]
            weights = weights[~mask]
            continue
        normal = vh[-1]
        if normal @ center > 0:
            normal = -normal
        offset = -normal @ center
        mask = np.abs(points @ normal + offset) < threshold
        if mask.sum() < minimum:
            break
        center = np.average(points[mask], axis=0, weights=weights[mask])
        result[slot] = np.r_[normal, offset, center, weights[mask].sum()/max(total, 1)]
        points = points[~mask]
        weights = weights[~mask]
    return result


def transform_planes(planes, pose):
    """n'=R n, d'=s d - n'.t for X'=s R X+t. Also transforms support centers."""
    matrix = pose.matrix().reshape(4, 4)
    s = torch.linalg.vector_norm(matrix[:3, 0])
    R = matrix[:3, :3]/s
    t = matrix[:3, 3]
    n = planes[:, :3] @ R.T
    d = s*planes[:, 3] - n @ t
    center = planes[:, 4:7] @ matrix[:3, :3].T + t
    return torch.cat((n, d[:, None], center, planes[:, 7:8]), dim=1)


def match_planes(source, target, pose, cfg):
    """One-to-one gated plane association at the current visual pose estimate."""
    if source is None or target is None:
        return []
    transformed = transform_planes(source, pose)
    a = transformed.detach().cpu().numpy()
    b = target.detach().cpu().numpy()
    cosine = np.cos(np.deg2rad(cfg.get("match_angle_deg", 15)))
    candidates = []
    for i in np.flatnonzero(a[:, 7] > 0):
        for j in np.flatnonzero(b[:, 7] > 0):
            dot = float(a[i, :3] @ b[j, :3])
            sign = 1.0 if dot >= 0 else -1.0
            distance = abs(a[i, 3]-sign*b[j, 3])
            center_distance = np.linalg.norm(a[i, 4:7]-b[j, 4:7])
            if abs(dot) >= cosine and distance < cfg.get("match_distance", .15) and center_distance < cfg.get("match_center_distance", 1.):
                candidates.append((1-abs(dot)+distance, i, j, sign))
    used_i, used_j, matches = set(), set(), []
    for _, i, j, sign in sorted(candidates):
        if i not in used_i and j not in used_j:
            matches.append((i, j, sign))
            used_i.add(i)
            used_j.add(j)
    return matches


def plane_jacobian(transformed):
    """Left Sim(3) perturbation: [translation, rotation, log scale]."""
    J = torch.zeros((len(transformed), 4, 7), device=transformed.device, dtype=transformed.dtype)
    J[:, :3, 3:6] = -skew_sym(transformed[:, :3])
    J[:, 3, :3] = -transformed[:, :3]
    J[:, 3, 6] = transformed[:, 3]
    return J


def plane_residuals(source, target, pose, matches, cfg):
    if not matches:
        return None
    ii, jj, signs = zip(*matches)
    transformed = transform_planes(source[list(ii)], pose)
    signs = source.new_tensor(signs)[:, None]
    residual = transformed[:, :4] - signs*target[list(jj), :4]
    J = plane_jacobian(transformed)
    sigma = source.new_tensor([cfg.get("sigma_normal", .05)]*3+[cfg.get("sigma_distance", .03)])
    support = torch.minimum(source[list(ii), 7], target[list(jj), 7])
    weight = (cfg.get("weight", 1.)*(4*support).clamp(.05, 1.)).sqrt()
    return residual/sigma*weight[:, None], J/sigma[None, :, None]*weight[:, None, None]


def robust_system(residual, J):
    r = residual.reshape(-1)
    J = J.reshape(-1, J.shape[-1])
    w = torch.clamp(1.345/r.abs().clamp_min(1e-12), max=1.)
    return J.T @ (w[:, None]*J), -(J.T @ (w*r)), .5*(w*r.square()).sum()


def metric_scale_factor(pose, sigma):
    scale = torch.linalg.vector_norm(pose.matrix().reshape(4, 4)[:3, 0])
    r = scale.log()/sigma
    J = torch.zeros((1, 7), device=r.device, dtype=r.dtype)
    J[0, 6] = 1./sigma
    return r.reshape(1), J


def refine_graph(frames, ii, jj, cfg):
    """Small plane graph with visual-pose priors after the original CUDA solve.

    The first keyframe is fixed. Priors preserve unobservable directions and
    keep this bounded correction close to the visual graph solution.
    """
    import lietorch
    edges = list(dict.fromkeys(tuple(sorted((int(a), int(b)))) for a, b in zip(ii.tolist(), jj.tolist()) if a != b))
    edges = edges[-int(cfg.get("backend_max_edges", 96)):]
    if not edges:
        return 0
    ids = sorted({x for edge in edges for x in edge})
    kfs = {i: frames[i] for i in ids}
    anchors = {i: lietorch.Sim3(kfs[i].T_WC.data.clone()) for i in ids}
    poses = {i: lietorch.Sim3(anchors[i].data.clone()) for i in ids}
    factors = []
    for i, j in edges:
        if kfs[i].planes is None or kfs[j].planes is None:
            continue
        matches = match_planes(kfs[i].planes, kfs[j].planes, poses[j].inv()*poses[i], cfg)
        if matches:
            factors.append((i, j, matches))
    if not factors:
        return 0
    free = [i for i in ids if i != ids[0]]
    indices = {i: slice(k*7, (k+1)*7) for k, i in enumerate(free)}
    proto = next(iter(poses.values())).data
    prior = proto.new_tensor([cfg.get("prior_translation", .03)]*3 + [cfg.get("prior_rotation", .04)]*3 + [cfg.get("prior_scale", .02)])
    sigma = proto.new_tensor([cfg.get("sigma_normal", .05)]*3+[cfg.get("sigma_distance", .03)])
    for _ in range(int(cfg.get("backend_iterations", 2))):
        H = torch.eye(len(free)*7, device=proto.device, dtype=proto.dtype)*1e-5
        g = proto.new_zeros(len(free)*7)
        for i in free:
            sl = indices[i]
            delta = (poses[i]*anchors[i].inv()).log().reshape(7)
            H[sl, sl] += torch.diag(1/prior.square())
            g[sl] -= delta/prior.square()
            if kfs[i].depth is not None:
                r, J = metric_scale_factor(poses[i], cfg.get("metric_scale_sigma", .02))
                h, b, _ = robust_system(r, J)
                H[sl, sl] += h
                g[sl] += b
        for i, j, matches in factors:
            a, b, signs = zip(*matches)
            pi = transform_planes(kfs[i].planes[list(a)], poses[i])
            pj = transform_planes(kfs[j].planes[list(b)], poses[j])
            sign = proto.new_tensor(signs)[:, None]
            r = ((pi[:, :4]-sign*pj[:, :4])/sigma).reshape(-1)
            Ji = (plane_jacobian(pi)/sigma[None, :, None]).reshape(-1, 7)
            Jj = (-sign[:, :, None]*plane_jacobian(pj)/sigma[None, :, None]).reshape(-1, 7)
            support = torch.minimum(pi[:, 7], pj[:, 7])
            confidence = (4*support).clamp(.05, 1.).sqrt().repeat_interleave(4)
            r, Ji, Jj = r*confidence, Ji*confidence[:, None], Jj*confidence[:, None]
            J = proto.new_zeros((len(r), len(free)*7))
            if i in indices:
                J[:, indices[i]] = Ji
            if j in indices:
                J[:, indices[j]] = Jj
            h, bvec, _ = robust_system(r, J)
            H += h*cfg.get("weight", 1.)
            g += bvec*cfg.get("weight", 1.)
        step = torch.linalg.solve(H, g).reshape(-1, 7)
        if not torch.isfinite(step).all():
            return 0
        for k, i in enumerate(free):
            delta = step[k]/torch.clamp(step[k].norm()/.05, min=1.)
            poses[i] = poses[i].retr(delta[None])
    updated = lietorch.Sim3(torch.stack([poses[i].data for i in free]))
    frames.update_T_WCs(updated, torch.tensor(free, device=proto.device))
    return sum(len(m) for _, _, m in factors)
