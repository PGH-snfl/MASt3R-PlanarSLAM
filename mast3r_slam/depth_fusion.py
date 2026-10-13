"""Sensor-first depth selection. Confidence scores are weights, not probabilities."""
import cv2
import numpy as np
import torch


def fuse_depth(sensor, predicted, confidence, cfg):
    """Return metric depth, optimization weights, provenance, diagnostics.

    Provenance: 0 invalid, 1 measured, 2 noisy measurement replaced, 3 hole filled.
    A disagreement alone cannot establish which source is wrong. Replace a
    measurement only when its neighborhood independently supports the model.
    """
    s = np.asarray(sensor, np.float32)
    p = np.asarray(predicted, np.float32)
    c = np.asarray(confidence, np.float32)
    if s.shape != p.shape or s.shape != c.shape:
        raise ValueError("Sensor/model depth and confidence must have the same shape")
    valid_s = np.isfinite(s) & (s > 0)
    s = np.where(valid_s, s, 0).astype(np.float32)
    valid_p = np.isfinite(p) & (p > 0) & np.isfinite(c)
    trusted_p = valid_p & (c >= cfg.get("min_model_conf", 2.))
    # Isolated spikes/holes are excluded from estimating the model's scale.
    median = cv2.medianBlur(s, 5)
    count = cv2.boxFilter(valid_s.astype(np.float32), -1, (5, 5), normalize=False)
    tolerance = cfg.get("sensor_noise_abs", .04) + cfg.get("sensor_noise_rel", .04)*median
    unstable = valid_s & (median > 0) & (count >= 12) & (np.abs(s-median) > tolerance)
    overlap = valid_s & trusted_p & ~unstable
    scale, aligned = 1., False
    if overlap.sum() >= cfg.get("min_scale_points", 200):
        ratios = np.log(s[overlap])-np.log(p[overlap])
        center = np.median(ratios)
        candidate = float(np.exp(center))
        inliers = np.abs(ratios-center) < cfg.get("scale_log_tolerance", .15)
        if (cfg.get("min_scale", .5) <= candidate <= cfg.get("max_scale", 2.)
                and inliers.mean() >= cfg.get("min_scale_inlier_fraction", .5)):
            scale, aligned = candidate, True
    p = np.where(valid_p, p, 0)*scale
    trusted_p &= (p >= cfg.get("min_depth", .25)) & (p <= cfg.get("max_depth", 8.))
    # Without sensor anchors use the checkpoint's metric output conservatively.
    if not aligned:
        trusted_p &= c >= cfg.get("unaligned_min_conf", 4.)
    threshold = cfg.get("agreement_abs", .06)+cfg.get("agreement_rel", .08)*s
    disagree = valid_s & trusted_p & (np.abs(s-p) > threshold)
    agrees_neighbors = np.abs(p-median) <= tolerance
    replace = (unstable & disagree & agrees_neighbors & trusted_p & aligned
               & (c >= cfg.get("replacement_min_conf", 4.)))
    fill = ~valid_s & trusted_p
    result = s.copy()
    result[replace | fill] = p[replace | fill]
    source = valid_s.astype(np.uint8)
    source[replace], source[fill] = 2, 3
    weights = np.where(valid_s, 3., 0.).astype(np.float32)
    weights[disagree | unstable] = 1.8
    model_weight = 1.6 + .4*np.clip((np.nan_to_num(c, nan=0.)-2.)/8., 0, 1)
    if not aligned:
        model_weight = np.full_like(c, 1.6)
    weights[replace | fill] = model_weight[replace | fill]
    stats = {"sensor": int((source == 1).sum()), "replaced": int(replace.sum()),
             "filled": int(fill.sum()), "invalid": int((source == 0).sum()),
             "disagree": int(disagree.sum()), "scale": scale, "aligned": aligned}
    return result, weights, source, stats


def pair_depth_weight(source_conf, target_conf, indices):
    """Reliability for correspondences indexed in the target pixel grid."""
    return (torch.minimum(source_conf[indices], target_conf)/3.).clamp(0, 1)
