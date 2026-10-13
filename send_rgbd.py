"""Serve this computer's USB RealSense RGB-D to a SLAM server over Tailscale.

No model/GPU required; no image/video files are written. The SLAM server pulls
the latest pair on demand. Bind to this computer's Tailscale IP for remote use.
"""
import argparse
import math
import socket
import threading

from mast3r_slam.rgbd_transport import StreamError, serve_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind', default='127.0.0.1', help="This camera computer's Tailscale IPv4 address")
    parser.add_argument('--port', type=int, default=9877)
    parser.add_argument('--serial')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--camera-fps', type=int, choices=(6, 15, 30, 60), default=30)
    parser.add_argument('--max-fps', type=float, default=10., help='Maximum reply rate; capture remains at camera-fps')
    parser.add_argument('--rgb-codec', choices=('jpeg', 'png'), default='jpeg')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--timeout', type=float, default=10., help='Per-transfer timeout, not idle/model-load timeout')
    parser.add_argument('--list-devices', action='store_true')
    args = parser.parse_args()
    if (not 1 <= args.port <= 65535 or not 1 <= args.jpeg_quality <= 100 or
            not math.isfinite(args.max_fps) or args.max_fps <= 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0 or
            not 0 < args.width <= 1920 or not 0 < args.height <= 1080):
        parser.error('Invalid port, quality, rate, timeout or dimensions')
    from mast3r_slam.realsense_capture import RealSenseCapture
    if args.list_devices:
        devices = RealSenseCapture.devices()
        for name, serial in devices:
            print(f'{name}: {serial}')
        if not devices:
            print('No USB RealSense cameras detected')
        return
    stop = threading.Event()
    camera = None
    try:
        # Bind first so a wrong address/busy port does not open the camera.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind, args.port))
            listener.listen(1)
            listener.settimeout(1.)
            camera = RealSenseCapture(args.serial, args.width, args.height, args.camera_fps)
            print(f"Camera: {camera.metadata['camera']} serial={camera.metadata['serial']}; "
                  f"{args.width}x{args.height}@{args.camera_fps}; depth scale={camera.metadata['depth_scale']} m/unit", flush=True)
            print(f'RGB-D sender listening on {args.bind}:{args.port}; RGB={args.rgb_codec}, depth=lossless uint16. '
                  'No recording. One SLAM client at a time.', flush=True)
            while not stop.is_set():
                if camera.latest.error is not None:
                    raise StreamError(f'Camera capture failed: {camera.latest.error}')
                try:
                    conn, address = listener.accept()
                except socket.timeout:
                    continue
                with conn:
                    print(f'SLAM client connected: {address}', flush=True)
                    try:
                        serve_client(conn, camera.latest, stop, codec=args.rgb_codec,
                                     quality=args.jpeg_quality, max_fps=args.max_fps, timeout=args.timeout)
                    except (OSError, StreamError) as exc:
                        print(f'Client disconnected: {exc}', flush=True)
                    finally:
                        print('SLAM client ended; waiting for the next connection', flush=True)
    except KeyboardInterrupt:
        print('Sender stopped')
    finally:
        stop.set()
        if camera is not None:
            camera.close()


if __name__ == '__main__':
    main()
