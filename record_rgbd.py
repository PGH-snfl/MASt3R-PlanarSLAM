"""Record local RealSense RGB + metric depth with SDK calibration and timestamps.

The installed RealSense 2.58.4 SDK records ROS2 SQLite (.db3) files.
Process the file later with run_d457.py --recording PATH.
"""
import argparse
from datetime import datetime
import math
from pathlib import Path
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New .db3 file; existing files are never overwritten')
    parser.add_argument('--serial', help='Required when multiple RealSense cameras are connected')
    parser.add_argument('--seconds', type=float, default=0, help='0: record until Ctrl+C')
    parser.add_argument('--fps', type=int, choices=(15, 30), default=30)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds must be finite and nonnegative')
    output = (args.output or Path(__file__).resolve().parent / 'recordings' /
              datetime.now().strftime('rgbd_%Y%m%d_%H%M%S_%f.db3')).expanduser().absolute()
    if output.suffix.lower() != '.db3':
        parser.error('This recorder uses .db3 with the installed RealSense SDK')
    if output.exists():
        parser.error(f'Refusing to overwrite {output}')

    import pyrealsense2 as rs
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if args.serial:
        devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == args.serial]
    if len(devices) != 1:
        parser.error('Connect one RealSense camera, or select a connected camera with --serial')
    serial = devices[0].get_info(rs.camera_info.serial_number)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, args.fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    config.enable_record_to_file(str(output))
    pipeline = rs.pipeline(ctx)
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started = False
    try:
        pipeline.start(config)
        started = True
        begin = last_report = last_frame = time.monotonic()
        received = 0
        print(f'Recording: {output}', flush=True)
        print(f'Camera {serial}: RGB + depth 640x480 @ {args.fps} FPS. Ctrl+C to finish.', flush=True)
        while not stopping:
            now = time.monotonic()
            if args.seconds and now - begin >= args.seconds:
                break
            ready, frames = pipeline.try_wait_for_frames(1000)
            now = time.monotonic()
            if ready and frames.get_color_frame() and frames.get_depth_frame():
                received += 1
                last_frame = now
            if now - last_frame > 10:
                raise RuntimeError('No RGB-D frames for 10 seconds; check the USB camera')
            if now - last_report >= 1:
                print(f'{now-begin:.1f}s | received RGB-D pairs={received}', flush=True)
                last_report = now
    finally:
        try:
            if started:
                print('Finalizing recording; please wait...', flush=True)
                pipeline.stop()
        finally:
            for s, handler in previous.items():
                signal.signal(s, handler)
    print(f'Saved recording: {output}', flush=True)


if __name__ == '__main__':
    main()
