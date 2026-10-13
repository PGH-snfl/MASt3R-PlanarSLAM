"""Conservative depth denoising inside large, measured planar surfaces.

No temporal averaging or hole filling. All gates and fits use the unmodified
input. A pixel can be corrected once, along its camera ray, without increasing
its confidence or changing measured/model provenance.
"""
import cv2
import numpy as np


def correct_large_planes(depth, K, source, confidence, planes, cfg, observations=None):
    z = np.asarray(depth, dtype=np.float32)
    output = z.copy()
    delta = np.zeros_like(z)
    stats = dict(accepted=0, pixels=0, max_change_mm=0., mean_change_mm=0., largest_area_m2=0.)
    if not cfg.get('enabled', False):
        return output, delta, np.array(planes, copy=True), stats
    K = np.asarray(K, dtype=np.float32)
    source, confidence = np.asarray(source), np.asarray(confidence)
    if z.ndim != 2 or source.shape != z.shape or confidence.shape != z.shape:
        raise ValueError('Plane correction requires matching depth/source/confidence images')
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Plane correction requires finite calibrated intrinsics')
    alpha = float(cfg.get('strength', .60))
    cap = float(cfg.get('max_correction_m', .01))
    if not 0 <= alpha <= 1 or cap <= 0:
        raise ValueError('Plane correction requires strength in [0,1] and positive correction cap')
    p_out = np.array(planes, dtype=np.float32, copy=True)
    if alpha == 0 or not len(p_out):
        return output, delta, p_out, stats
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    rays = np.stack(((xx-K[0, 2])/K[0, 0], (yy-K[1, 2])/K[1, 1], np.ones_like(z)), -1)
    ray_length = np.linalg.norm(rays, axis=-1)
    valid = np.isfinite(z) & (z > 0)
    measured = valid & (source == 1) & np.isfinite(confidence) & (confidence >= cfg.get('min_sensor_weight', 2.5))
    safe_z = np.where(valid, z, 0)
    # Preserve foreground/background boundaries and a guard band around holes.
    local_range = cv2.dilate(safe_z, np.ones((3, 3), np.uint8))-cv2.erode(safe_z, np.ones((3, 3), np.uint8))
    boundary = (~valid) | (local_range > cfg.get('edge_threshold_m', .03))
    margin = max(1, int(cfg.get('boundary_margin_px', 3)))
    kernel = np.ones((2*margin+1, 2*margin+1), np.uint8)
    local_median = cv2.medianBlur(safe_z, 5)
    safe = (measured & ~cv2.dilate(boundary.astype(np.uint8), kernel).astype(bool)
            & (np.abs(safe_z-local_median) <= cfg.get("max_local_deviation_m", .015)))
    minimum = max(int(cfg.get('min_pixels', 2500)), int(np.ceil(z.size*cfg.get('min_image_fraction', .10))))
    min_area = float(cfg.get('min_area_m2', .25))
    tolerance = float(cfg.get('max_plane_residual_m', .03))
    max_rms = float(cfg.get('max_fit_rms_m', .01))
    claimed = np.zeros_like(valid)
    min_cos = float(cfg.get('min_view_cos', .3))
    for plane_index, plane in enumerate(p_out[:int(cfg.get('max_planes', 8))]):
        normal = plane[:3].copy()
        norm = np.linalg.norm(normal)
        if plane[7] <= 0 or not np.isfinite(plane).all() or norm < 1e-6:
            continue
        normal /= norm
        offset = float(plane[3]/norm)
        denom = rays @ normal
        facing = np.abs(denom)/ray_length >= min_cos
        target = np.divide(-offset, denom, out=np.zeros_like(z), where=facing)
        membership = measured & facing & (target > 0) & (np.abs(safe_z-target) <= tolerance) & ~claimed
        # A large bounding box is insufficient: require a connected measured patch.
        _, labels, sizes, _ = cv2.connectedComponentsWithStats(membership.astype(np.uint8), connectivity=8)
        components = np.flatnonzero(sizes[1:, cv2.CC_STAT_AREA] >= minimum)+1
        best_area = 0.
        original_center = plane[4:7].copy()
        for label in components:
            patch = (labels == label) & safe & ~claimed
            indices = np.flatnonzero(patch)
            if len(indices) < minimum:
                continue
            selection = indices[np.linspace(0, len(indices)-1, min(len(indices), 4096), dtype=np.int64)]
            points = rays.reshape(-1, 3)[selection]*safe_z.reshape(-1, 1)[selection]
            weights = confidence.reshape(-1)[selection].astype(np.float64)
            keep = np.ones(len(points), dtype=bool)
            # Refit on reliable measured depths, even when the initial proposal
            # was detected in sensor/model fused geometry.
            for _ in range(2):
                center = np.average(points[keep], axis=0, weights=weights[keep])
                _, _, basis = np.linalg.svd((points[keep]-center)*np.sqrt(weights[keep, None]), full_matrices=False)
                n = basis[-1]
                d = -float(n @ center)
                residual = np.abs(points @ n+d)
                keep = residual <= min(tolerance, max_rms*2.5)
                if keep.mean() < .9:
                    break
            if keep.mean() < .9:
                continue
            residual = points[keep] @ n+d
            rms = float(np.sqrt(np.average(residual**2, weights=weights[keep])))
            extent = np.percentile((points[keep]-center) @ basis[:2].T, 95, axis=0)-np.percentile((points[keep]-center) @ basis[:2].T, 5, axis=0)
            if rms > max_rms or extent.min() < cfg.get('min_extent_m', .35):
                continue
            dot = rays @ n
            facing = np.abs(dot)/ray_length >= min_cos
            plane_z = np.divide(-d, dot, out=np.zeros_like(z), where=facing)
            patch &= facing & (plane_z > 0) & (np.abs(safe_z-plane_z) <= tolerance)
            # A coherent offset may be a shallow object, not measurement noise.
            patch &= np.abs(local_median-plane_z) <= cfg.get("max_coherent_residual_m", .01)
            if patch.sum() < minimum:
                continue
            # Surface footprint per image pixel for a calibrated plane:
            # dA = z^2 / (fx*fy*|n dot ray|). Holes are not counted as area.
            area = float(np.sum(plane_z[patch]**2/np.abs(dot[patch]), dtype=np.float64)/(K[0, 0]*K[1, 1]))
            if area < min_area:
                continue
            interior = cv2.erode(patch.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            difference = plane_z-safe_z
            change = interior & (np.abs(difference) > cfg.get('deadband_m', .001))
            adjustment = np.clip(alpha*difference[change], -cap, cap)
            output[change] = z[change]+adjustment
            delta[change] = adjustment
            claimed |= patch
            stats['accepted'] += 1
            stats['largest_area_m2'] = max(stats['largest_area_m2'], area)
            # Keep plane-pose factors consistent with the measured surface fit.
            if n @ center > 0:
                n, d = -n, -d
            if observations is not None:
                # Transient support mask; never archive it or feed corrected
                # depths back into the temporal plane estimator.
                observations.append(dict(normal=n.copy(), offset=d, center=center.copy(),
                    rms=rms, radius=float(np.percentile(np.linalg.norm(points[keep]-center, axis=1), 90)),
                    indices=np.flatnonzero(interior), plane_index=plane_index))
            if area > best_area:
                best_area = area
                plane[:4] = np.r_[n, d]
                plane[4:7] = original_center-(original_center @ n+d)*n
    corrected = np.abs(delta) > 0
    stats['pixels'] = int(corrected.sum())
    if stats['pixels']:
        stats['max_change_mm'] = float(np.abs(delta[corrected]).max()*1000)
        stats['mean_change_mm'] = float(np.abs(delta[corrected]).mean()*1000)
    return output, delta, p_out, stats
