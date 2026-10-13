import torch
import time
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.runtime_metrics import StageTimer
from mast3r_slam.tracking_selection import select_pose_points, AdaptiveSelection


class FrameTracker:
    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames
        self.device = device
        self.plane_cfg = config.get("planes", {})
        self.plane_matches = []
        self.metric_input = False
        self.selection_policy = AdaptiveSelection()
        self.selection_cache = {}
        self.selection_reference = None

        self.reset_idx_f2k()

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()
        reference = (keyframe.frame_id, keyframe.N_updates)
        if reference != self.selection_reference:
            self.selection_cache.clear()
            self.selection_reference = reference
        timer = StageTimer(config.get('runtime', {}).get('profile_cuda', False))
        self.timings = timer.times
        self.selection_stats = dict(available=0, selected=0, fallback=False, reason='not_solved')

        with timer.measure('inference_matching'):
            idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
                self.model, frame, keyframe, idx_i2j_init=self.idx_f2k, timer=timer
            )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        with timer.measure('depth_spatial'):
            frame.update_pointmap(Xff, Cff)
        self.metric_input = frame.depth is not None and keyframe.depth is not None
        self.plane_source, self.plane_target = frame.planes, keyframe.planes
        self.plane_matches = []
        if self.plane_cfg.get("enabled", False):
            from mast3r_slam.planes import match_planes
            self.plane_matches = match_planes(frame.planes, keyframe.planes,
                                              keyframe.T_WC.inv()*frame.T_WC, self.plane_cfg)

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K
        else:
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q
        if self.metric_input:
            # Keep the descriptor acceptance threshold separate from the
            # reliability used to weight accepted depth correspondences.
            Qk = Qk * (torch.minimum(Cf, Ck)/3.).clamp(0, 1)

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            with timer.measure('pose_total'):
                T_WCf, T_CkCf = self.solve_selected_pose(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt, meas_k,
                    valid_meas_k, K, img_size, keyframe.uimg, timer)
        except Exception as e:
            print(f"Pose solve failed at frame {frame.frame_id}: {type(e).__name__}: {e}", flush=True)
            return False, [], True

        frame.T_WC = T_WCf
        from mast3r_slam.feature_overlay import tracking_points
        display_valid = valid_opt
        if use_calib:
            _, visible = project_calib(T_CkCf.act(Xf), K, img_size,
                                       border=self.cfg["pixel_border"], z_eps=self.cfg["depth_eps"])
            display_valid = display_valid & visible & valid_meas_k
        frame.feature_points = tracking_points(frame, idx_f2k, display_valid, Qk)

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def solve_selected_pose(self, Xf, Xk, T_WCf, T_WCk, Qk, valid,
                            meas, valid_meas, K, shape, rgb, timer):
        cfg = self.cfg.get('point_selection', {})
        started = time.perf_counter()
        # Enable the new path only for calibrated sensor/model RGB-D tracking.
        eligible = self.metric_input and K is not None
        adaptive = eligible and cfg.get('enabled', False) and cfg.get('adaptive', False)
        attempt = not adaptive or self.selection_policy.choose()
        with timer.measure('pose_selection'):
            ids = select_pose_points(valid & valid_meas, Qk, Xk, rgb, shape, cfg,
                                     self.selection_cache) if eligible and attempt else None
        available = int(valid.sum())
        self.selection_stats = dict(available=available, selected=available if ids is None else len(ids),
                                    used=available if ids is None else len(ids),
                                    fallback=False, reason=('adaptive_full' if not attempt else 'full') if ids is None else 'sampled')

        def finish(result):
            if adaptive:
                self.selection_policy.record(ids is not None, 1000*(time.perf_counter()-started))
            return result

        def solve(indices=None):
            if indices is None and eligible and cfg.get('enabled', False):
                # Full fallback still uses EVERY valid observation, but avoids
                # building repeated Jacobians for masked-out zero-weight rows.
                indices = torch.where((valid & valid_meas).reshape(-1))[0]
                if not len(indices):
                    raise RuntimeError('No valid calibrated pose observations')
                self.selection_stats['used'] = len(indices)
            i = slice(None) if indices is None else indices
            if K is None:
                result = self.opt_pose_ray_dist_sim3(Xf[i], Xk[i], T_WCf, T_WCk, Qk[i], valid[i])
            else:
                result = self.opt_pose_calib_sim3(Xf[i], Xk[i], T_WCf, T_WCk, Qk[i], valid[i],
                                                meas[i], valid_meas[i], K, shape)
            if not all(torch.isfinite(p.data).all() for p in result):
                raise RuntimeError('Non-finite pose solution')
            return result

        if ids is not None:
            try:
                with timer.measure('pose_sampled_solve'):
                    candidate = solve(ids)
                with timer.measure('pose_validation'):
                    accepted = self.validate_sampled_pose(candidate[1], T_WCk.inv()*T_WCf,
                                                         Xf, Qk, valid & valid_meas, meas, K, shape, cfg)
                if accepted:
                    return finish(candidate)
                reason = 'full_residual_check'
            except (RuntimeError, ValueError) as e:
                reason = type(e).__name__
            self.selection_stats.update(fallback=True, reason=reason, used=available)
        # Always restart the fallback from the original incoming pose.
        with timer.measure('pose_full_solve'):
            result = solve()
        return finish(result)

    def validate_sampled_pose(self, pose, initial, Xf, Qk, valid, meas, K, shape, cfg):
        """Check ALL eligible matches once; the repeated Jacobian is sampled."""
        keep = valid.reshape(-1) & torch.isfinite(Xf).all(-1) & torch.isfinite(meas).all(-1)
        if not keep.any():
            return False
        x, target = Xf[keep], meas[keep]
        weights = Qk[keep].reshape(-1).clamp_min(0)
        sigma = target.new_tensor([self.cfg['sigma_pixel'], self.cfg['sigma_pixel'], self.cfg['sigma_depth']])

        def assess(p):
            pred, visible = project_calib(p.act(x), K, shape, border=self.cfg['pixel_border'],
                                         z_eps=self.cfg['depth_eps'])
            good = visible.reshape(-1) & torch.isfinite(pred).all(-1)
            r = torch.nan_to_num(((pred-target)/sigma).abs(), nan=50., posinf=50., neginf=50.).clamp_max(50.)
            # Missing projections are penalized, not silently excluded.
            r[~good] = 50.
            cost = ((r*r).sum(-1)*weights).sum()/weights.sum().clamp_min(1e-8)
            pixel = torch.linalg.vector_norm(pred[:, :2]-target[:, :2], dim=-1)
            depth = (pred[:, 2]-target[:, 2]).abs()
            inliers = good & (pixel <= cfg.get('max_pixel_error', 5.)) & (depth <= cfg.get('max_log_depth_error', .08))
            return cost, good.float().mean(), inliers.float().mean()

        cost, visible, inliers = assess(pose)
        before, _, _ = assess(initial)
        return bool(torch.isfinite(cost) & (cost <= before*1.02+1e-6)
                    & (visible >= cfg.get('min_visible_fraction', .8))
                    & (inliers >= cfg.get('min_inlier_fraction', .5)))

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J, pose=None):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        if self.metric_input and pose is not None:
            from mast3r_slam.planes import plane_residuals, robust_system, metric_scale_factor
            # Average visual energy makes the plane/visual balance independent
            # of input resolution and the number of valid depth pixels.
            n = sqrt_info.any(dim=-1).sum().clamp_min(1)
            H, g, cost = H/n, g/n, cost/n.item()
            if self.plane_matches:
                pr, pJ = plane_residuals(self.plane_source, self.plane_target, pose,
                                         self.plane_matches, self.plane_cfg)
                ph, pg, pcost = robust_system(pr, pJ)
                H, g, cost = H+ph, g+pg[:, None], cost+pcost.item()
            sr, sJ = metric_scale_factor(pose, self.plane_cfg.get("metric_scale_sigma", .02))
            sh, sg, scost = robust_system(sr, sJ)
            H, g, cost = H+sh, g+sg[:, None], cost+scost.item()
            H += torch.diag(H.diag().clamp_min(1.)*1e-6)

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J, T_CkCf)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J, T_CkCf)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
