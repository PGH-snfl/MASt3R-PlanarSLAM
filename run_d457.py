"""Live RGB-D + plane-constrained MASt3R with desktop or browser viewing."""
import argparse
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/d457_planar.yaml")
    parser.add_argument("--serial")
    parser.add_argument("--camera-host", help="Camera host or SSH tunnel address; use network RGB-D instead of local USB")
    parser.add_argument("--camera-port", type=int, default=9877)
    parser.add_argument("--camera-timeout", type=float, default=10., help="Network transfer deadline in seconds")
    parser.add_argument("--image-size", type=int, choices=(512, 640),
                        help="Inference long edge; overrides dataset.image_size (default 640)")
    parser.add_argument("--archive-dir", help="New session directory; default maps/<timestamp>")
    parser.add_argument("--recent-keyframes", type=int, help="GPU tracking window (default 8)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--check-camera", action="store_true", help="Check 30 RGB-D frames without loading the model")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--web-viz", action="store_true", help="Browser viewer over SSH; replaces the desktop/OpenGL window")
    parser.add_argument("--web-port", type=int, default=8080)
    parser.add_argument("--web-fps", type=float, default=2., help="Browser update limit; independent of SLAM FPS")
    parser.add_argument("--web-points", type=int, default=50000, help="Browser overview point budget, plus up to 4000 current points")
    parser.add_argument("--archive-mount", help="Require a mounted laptop archive folder; create a new session inside it")
    parser.add_argument("--max-frames", type=int, default=0, help="0: run until the window closes or Ctrl+C")
    parser.add_argument("--fps", type=float, default=5., help="Maximum processing rate; camera captures at its configured FPS")
    parser.add_argument("--disable-planes", action="store_true", help="Disable plane pose constraints and plane depth denoising")
    parser.add_argument("--no-sdpa", action="store_true")
    parser.add_argument("--disable-plane-depth", action="store_true", help="Disable large-plane depth denoising; keep plane pose constraints")
    parser.add_argument("--disable-temporal-depth", action="store_true", help="Disable temporal plane correction (A/B comparison)")
    parser.add_argument("--disable-surface-depth", action="store_true", help="Disable general surface temporal denoising for comparison")
    parser.add_argument("--disable-global-poses", action="store_true", help="Disable CPU global pose-only optimization for comparison")
    parser.add_argument("--sensor-only", action="store_true", help="Disable model depth fallback for comparison")
    parser.add_argument("--profile", action="store_true", help="Synchronize CUDA at stage boundaries for diagnostic timings")
    parser.add_argument("--full-correspondences", action="store_true", help="Disable pose point selection for A/B comparison")
    parser.add_argument("--viewer-fps", type=float, default=5., help="Maximum live data publication rate; latest frame only")
    args = parser.parse_args()
    import math
    native_viz = not args.no_viz and not args.web_viz
    if not 1 <= args.web_port <= 65535 or not math.isfinite(args.web_fps) or args.web_fps <= 0 or args.web_points < 100:
        parser.error('Invalid web port, FPS or point budget')
    if args.archive_mount and args.archive_dir:
        parser.error('Choose --archive-mount or --archive-dir')
    if args.archive_mount:
        from mast3r_slam.remote_archive import require_archive_mount
        args.archive_mount = str(require_archive_mount(args.archive_mount))
    if not 1 <= args.camera_port <= 65535 or not math.isfinite(args.camera_timeout) or args.camera_timeout <= 0:
        parser.error('Invalid camera port or timeout')
    if args.camera_host and (args.serial or args.list_devices):
        parser.error('--serial/--list-devices apply to local USB; select the camera on send_rgbd.py')
    if args.recent_keyframes is not None and args.recent_keyframes < 2:
        parser.error("--recent-keyframes must be at least 2")
    if args.fps <= 0 or args.viewer_fps <= 0 or args.max_frames < 0:
        parser.error("fps must be positive and max-frames must be nonnegative")
    # Existing resource/checkpoint/config paths are relative to the project.
    os.chdir(ROOT)
    import torch
    import torch.multiprocessing as mp
    import lietorch
    from mast3r_slam.config import load_config, config
    from mast3r_slam.realsense_rgbd import RealSenseRGBDDataset
    load_config(args.config)
    if args.image_size is not None:
        config['dataset']['image_size'] = args.image_size
    image_size = config['dataset'].get('image_size', 512)
    if image_size not in (512, 640):
        parser.error('dataset.image_size must be 512 or 640')
    config.setdefault('runtime', {})['profile_cuda'] = args.profile
    if args.full_correspondences:
        config['tracking'].setdefault('point_selection', {})['enabled'] = False
    if args.list_devices:
        devices = RealSenseRGBDDataset.devices()
        for name, serial in devices:
            print(f"{name}: {serial}")
        if not devices:
            print("No USB RealSense cameras detected")
        return
    if not config["use_calib"] or config["dataset"]["img_downsample"] != 1:
        parser.error("D457 requires use_calib: true and dataset.img_downsample: 1")
    config["planes"]["enabled"] &= not args.disable_planes
    if args.disable_plane_depth:
        config.setdefault("plane_depth", {})["enabled"] = False
    if args.disable_temporal_depth:
        config.setdefault("temporal_plane_depth", {})["enabled"] = False
    if args.disable_surface_depth:
        config.setdefault('temporal_surface', {})['enabled'] = False
    if args.disable_global_poses:
        config.setdefault('global_pose_graph', {})['enabled'] = False
    config.setdefault("depth_fusion", {}).setdefault("enabled", True)
    if args.sensor_only:
        config["depth_fusion"]["enabled"] = False
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    if args.camera_host:
        from mast3r_slam.network_rgbd import NetworkRGBDDataset
        dataset = NetworkRGBDDataset(args.camera_host, args.camera_port, args.camera_timeout)
    else:
        dataset = RealSenseRGBDDataset(args.serial)
    if args.check_camera:
        try:
            for i in range(30):
                timestamp, rgb = dataset[i]
                if i % 10 == 0:
                    print(f"RGB-D {i}: t={timestamp:.3f}, RGB={rgb.shape}, depth={dataset.last_depth.shape}, valid={dataset.last_valid_fraction:.1%}")
                    if args.camera_host:
                        print(f"Network: {dataset.last_network_stats}")
        finally:
            dataset.close()
        return
    if not torch.cuda.is_available():
        dataset.close()
        raise RuntimeError("The installed MASt3R CUDA backend requires an NVIDIA GPU")
    from mast3r_slam.frame import SharedKeyframes, SharedStates, Mode, create_frame
    from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, mast3r_inference_mono
    from mast3r_slam.efficient_attention import enable_efficient_attention
    from mast3r_slam.tracker import FrameTracker
    from mast3r_slam.keyframe_archive import KeyframeArchive
    from mast3r_slam.streaming_backend import StreamingBackend
    from mast3r_slam.temporal_plane_depth import TemporalPlaneDepth
    from mast3r_slam.temporal_surface import TemporalSurfaceDepth
    from mast3r_slam.disk_map import DiskMapWriter
    from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
    if native_viz:
        from mast3r_slam.visualization import run_visualization
    from mast3r_slam.runtime_metrics import StageTimer, DisplayGate
    display = DisplayGate(args.viewer_fps)
    metrics_file = None
    device = "cuda:0"
    mp.set_start_method("spawn", force=True)
    manager, states, viz, web = None, None, None, None
    backend, keyframes, disk_map = None, None, None
    processed, tracked, plane_count = 0, 0, 0
    try:
        if args.web_viz:
            from mast3r_slam.web_visualization import WebViewer
            web = WebViewer(args.web_port, args.web_fps, args.web_points)
        h, w = dataset.get_img_shape()[0]
        model = load_mast3r(device=device).eval()
        if not args.no_sdpa:
            print(f"Fused attention enabled in {enable_efficient_attention(model)} layers")
        retrieval = load_retriever(model)
        manager = mp.Manager()
        K = torch.as_tensor(dataset.camera_intrinsics.K_frame, device=device, dtype=torch.float32)
        recent = args.recent_keyframes or config['keyframes']['buffer']
        if recent < 2:
            parser.error('--recent-keyframes must be at least 2')
        from datetime import datetime
        archive_root = Path(args.archive_mount) if args.archive_mount else ROOT / 'maps'
        archive_dir = args.archive_dir or str(archive_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        keyframes = KeyframeArchive(archive_dir, recent=recent,
                                   cache=config.get('streaming', {}).get('candidate_cache', 2), device=device,
                                   preview_limit=150000, save_on_append=True)
        keyframes.set_intrinsics(K)
        disk_map = DiskMapWriter(archive_dir, config.get('map_storage', {}))
        metrics_file = (Path(archive_dir)/'performance.jsonl').open('w', buffering=1)
        import yaml
        (Path(archive_dir)/'config.yaml').write_text(yaml.safe_dump(config))
        view_keyframes = None
        if native_viz:
            view_keyframes = SharedKeyframes(manager, h, w, buffer=recent, device='cpu')
            view_keyframes.set_intrinsics(K.cpu())
            # Native display stays in memory; detailed disk paging is offline.
            view_keyframes.history = torch.zeros(keyframes.preview_limit, 6).share_memory_()
            view_keyframes.history_count = manager.Value('i', 0)
            view_keyframes.history_revision = manager.Value('i', -1)
        states = SharedStates(manager, h, w, device='cpu')
        main2viz = new_queue(manager, not native_viz)
        viz2main = new_queue(manager, not native_viz)
        tracker = FrameTracker(model, keyframes, device)
        backend = StreamingBackend(model, keyframes, K, retrieval, device)
        temporal_depth = TemporalPlaneDepth(config.get('temporal_plane_depth', {}))
        surface_depth = TemporalSurfaceDepth(config.get('temporal_surface', {}))
        depth_cap = config.get('plane_depth', {}).get('max_correction_m', .01)
        # Drop the startup frame: loading weights may have taken several seconds.
        dataset._pending = None
        last_pose = lietorch.Sim3.Identity(1, device=device)
        print(f"Ready: recent GPU keyframes={recent}, candidate cache={keyframes.cache_limit}. "
              f"Keyframes/map archive: {archive_dir} (continuous video is not recorded)", flush=True)
        if args.archive_mount:
            print('Archive destination: laptop via SSHFS; keep mount_laptop_maps.py running until Saved map.', flush=True)
        if native_viz:
            print("Live viewer: recent keyframes + 150,000-point overview; disk detail loading only in view_saved_map.sh", flush=True)
        while not args.max_frames or processed < args.max_frames:
            started = time.monotonic()
            if args.archive_mount:
                require_archive_mount(args.archive_mount, check_marker=False)
            if viz is not None and not viz.is_alive():
                if viz.exitcode != 0:
                    raise RuntimeError(f"Viewer failed with exit code {viz.exitcode}")
                break
            message = try_get_msg(viz2main)
            if message is not None:
                if message.is_terminated:
                    break
                states.paused.value = int(message.is_paused and not message.next)
            if states.is_paused():
                time.sleep(.02)
                continue
            timer = StageTimer(args.profile)
            wait_before = disk_map.wait_ms
            with timer.measure('camera'):
                timestamp, image = dataset[processed]
            if not config["depth_fusion"]["enabled"] and dataset.last_valid_fraction < .01:
                raise RuntimeError("Less than 1% valid depth; check D457 depth stream and viewing distance")
            frame = create_frame(processed, image, last_pose, img_size=image_size, device=device)
            if tuple(frame.img.shape[-2:]) != tuple(dataset.last_depth.shape):
                raise RuntimeError('RGB/depth processing shapes differ; check dataset.image_size')
            frame.K = K
            frame.timestamp = float(timestamp)
            frame.depth = torch.from_numpy(dataset.last_depth)
            add = False
            previous_count = len(keyframes)
            previous_loop = backend.last_verified
            previous_revision = keyframes.pose_revision
            mode = states.get_mode()
            if mode == Mode.INIT:
                with timer.measure('model'):
                    X, C = mast3r_inference_mono(model, frame)
                with timer.measure('depth_spatial'):
                    frame.update_pointmap(X, C)
                if (frame.C > 0).float().mean() < .01:
                    raise RuntimeError("Neither sensor nor model provides enough valid depth to initialize a map")
                with timer.measure('depth_temporal'):
                    temporal_depth.apply_frame(frame, keyframes.pose_revision, timestamp, total_cap=depth_cap)
                with timer.measure('depth_surface'):
                    surface_depth.apply_frame(frame, keyframes.pose_revision, timestamp, total_cap=depth_cap)
                with timer.measure('keyframe_archive'):
                    idx = keyframes.append(frame)
                with timer.measure('backend_new_keyframe'):
                    backend.new_keyframe(idx)
                states.set_mode(Mode.TRACKING)
            elif mode == Mode.TRACKING:
                with timer.measure('tracking_total'):
                    add, _, lost = tracker.track(frame)
                timer.times.update(getattr(tracker, 'timings', {}))
                plane_count += len(tracker.plane_matches)
                tracked += int(not lost)
                with timer.measure('depth_temporal'):
                    temporal_depth.apply_frame(frame, keyframes.pose_revision, timestamp,
                                               tracking_ok=not lost, total_cap=depth_cap)
                with timer.measure('depth_surface'):
                    surface_depth.apply_frame(frame, keyframes.pose_revision, timestamp,
                                              tracking_ok=not lost, total_cap=depth_cap)
                if lost:
                    states.set_mode(Mode.RELOC)
                if add and not lost and backend.can_add():
                    with timer.measure('keyframe_archive'):
                        idx = keyframes.append(frame)
                    with timer.measure('backend_new_keyframe'):
                        backend.new_keyframe(idx)
            else:
                with timer.measure('model'):
                    X, C = mast3r_inference_mono(model, frame)
                with timer.measure('depth_spatial'):
                    frame.update_pointmap(X, C)
                with timer.measure('depth_temporal'):
                    temporal_depth.apply_frame(frame, keyframes.pose_revision, timestamp,
                                               tracking_ok=False, total_cap=depth_cap)
                with timer.measure('depth_surface'):
                    surface_depth.apply_frame(frame, keyframes.pose_revision, timestamp,
                                              tracking_ok=False, total_cap=depth_cap)

            if not torch.isfinite(frame.T_WC.data).all():
                raise RuntimeError("Non-finite camera pose; stopping instead of adding invalid map geometry")
            reference = len(keyframes)-1
            relative = keyframes[reference].T_WC.inv()*frame.T_WC
            with timer.measure('backend_tick'):
                recovered = backend.tick(frame, relocating=(mode == Mode.RELOC))
                backend.global_graph.submit()
            if recovered:
                states.set_mode(Mode.TRACKING)
                tracker.reset_idx_f2k()
            elif states.get_mode() == Mode.TRACKING:
                frame.T_WC = keyframes[reference].T_WC*relative
            view_ready, view_changed = display.ready(time.monotonic(),
                len(keyframes) != previous_count or backend.last_verified != previous_loop
                or keyframes.pose_revision != previous_revision)
            if native_viz and view_ready and view_changed:
                with timer.measure('viewer_keyframes'):
                    backend.publish(view_keyframes, states)
            if not torch.isfinite(frame.T_WC.data).all():
                raise RuntimeError("Non-finite optimized camera pose")
            with timer.measure('map_submit'):
                disk_map.sync_poses(keyframes)
                if states.get_mode() == Mode.TRACKING:
                    anchor_id = len(keyframes)-1
                    anchor = keyframes[anchor_id]
                    keyframe_id = anchor_id if anchor.frame_id == frame.frame_id else None
                    disk_map.add_frame(frame, anchor_id, anchor.T_WC.matrix().detach().cpu().numpy(),
                                       timestamp, keyframe_id)
            if native_viz and view_ready:
                with timer.measure('viewer_frame'):
                    states.set_frame(frame, include_features=False)
            last_pose = lietorch.Sim3(frame.T_WC.data.clone())
            if web is not None and web.ready():
                with timer.measure('viewer_web'):
                    web.publish(frame, keyframes, states.get_mode().name)
            if viz is None and native_viz:
                ready = mp.Event()
                viz = mp.Process(target=run_visualization, args=(config, states, view_keyframes, main2viz, viz2main, ready))
                viz.start()
                deadline = time.monotonic()+45.
                while not ready.wait(timeout=.2):
                    if not viz.is_alive():
                        raise RuntimeError(f"Viewer failed to start (exit code {viz.exitcode})")
                    if time.monotonic() > deadline:
                        raise RuntimeError("Viewer did not render within 45 seconds; check the desktop OpenGL session or use --no-viz")
            processed += 1
            detected = int((frame.planes[:, 7] > 0).sum()) if frame.planes is not None else 0
            if frame.depth_stats is not None:
                s = frame.depth_stats
                total = frame.depth.numel()
                print(f"depth fusion: sensor={s['sensor']/total:.1%} replaced={s['replaced']/total:.1%}({s['replaced']}px) "
                      f"filled={s['filled']/total:.1%} invalid={s['invalid']/total:.1%} "
                      f"model_scale={s['scale']:.3f} aligned={s['aligned']}", flush=True)
            if frame.plane_depth_stats is not None:
                ps = frame.plane_depth_stats
                print(f"plane depth: accepted={ps['accepted']} corrected={ps['pixels']}px "
                      f"largest={ps['largest_area_m2']:.2f}m2 mean={ps['mean_change_mm']:.2f}mm "
                      f"max={ps['max_change_mm']:.2f}mm", flush=True)
            if frame.temporal_depth_stats is not None:
                ts = frame.temporal_depth_stats
                print(f"temporal depth: tracks={ts['tracks']} matched={ts['matched']} used={ts['used']} "
                      f"corrected={ts['pixels']}px mean={ts['mean_change_mm']:.2f}mm "
                      f"max={ts['max_change_mm']:.2f}mm reset={ts['reset'] or '-'}", flush=True)
            if frame.surface_depth_stats is not None:
                ss = frame.surface_depth_stats
                print(f"surface depth: corrected={ss['pixels']}px mean={ss['mean_change_mm']:.2f}mm "
                      f"max={ss['max_change_mm']:.2f}mm reset={ss['reset'] or '-'}", flush=True)
            print(f"frame={processed} mode={states.get_mode().name} keyframes={len(keyframes)} "
                  f"resident={len(keyframes.recent)}+{len(keyframes.cache)} edges={backend.graph.ii.numel()} "
                  f"planes={detected} matched={len(tracker.plane_matches)} graph_planes={getattr(backend.graph, 'last_plane_factors', 0)} "
                  f"depth={dataset.last_valid_fraction:.0%} step={time.monotonic()-started:.2f}s "
                  f"GPU={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
            timer.fence()
            selection = getattr(tracker, 'selection_stats', {}) if mode == Mode.TRACKING else {}
            metrics = dict(frame=frame.frame_id, mode=states.get_mode().name, cuda_synchronized=args.profile,
                           image_shape=[h, w], cuda_allocated_mb=torch.cuda.memory_allocated()/1024**2,
                           cuda_peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,
                           cuda_reserved_mb=torch.cuda.memory_reserved()/1024**2,
                           step_ms=1000*(time.monotonic()-started), stages_ms=timer.times,
                           pose_points=selection, map_wait_ms=disk_map.wait_ms-wait_before,
                           surface_depth=frame.surface_depth_stats, global_poses=backend.global_graph.stats,
                           map_pending=len(disk_map.pending))
            if args.camera_host:
                metrics['network'] = dict(dataset.last_network_stats)
            metrics_file.write(json.dumps(metrics)+'\n')
            if processed % 10 == 0 or args.profile:
                print('timing(ms): '+ ' '.join(f'{k}={v:.1f}' for k, v in timer.times.items())
                      + f' pose_points={selection} map_wait={metrics["map_wait_ms"]:.1f}', flush=True)
            time.sleep(max(0., 1./args.fps-(time.monotonic()-started)))
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        dataset.close()
        if web is not None:
            try:
                web.close()
            except Exception as exc:
                print(f'Browser shutdown error: {exc}; continuing map cleanup', flush=True)
        if states is not None:
            states.set_mode(Mode.TERMINATED)
        if viz is not None:
            viz.join(timeout=10)
            if viz.is_alive():
                print("Viewer did not shut down within 10 seconds; terminating it", flush=True)
                viz.terminate()
                viz.join()
            print(f"Viewer stopped (exit code {viz.exitcode})", flush=True)
        try:
            try:
                if disk_map is not None:
                    if keyframes is not None and len(keyframes):
                        disk_map.sync_poses(keyframes)
                    disk_map.close()
                    print(f"Saved map: {keyframes.directory} ({disk_map.counts})", flush=True)
            finally:
                if backend is not None:
                    backend.close()
                elif keyframes is not None:
                    keyframes.close()
        finally:
            if metrics_file is not None:
                metrics_file.close()
            if manager is not None:
                manager.shutdown()
        print(f"Summary: processed={processed}, tracked={tracked}, matched_plane_pairs={plane_count}", flush=True)


if __name__ == "__main__":
    main()
