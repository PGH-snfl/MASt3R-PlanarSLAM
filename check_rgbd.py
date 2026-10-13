"""Check a remote RGB-D sender without loading torch, a model or a viewer."""
import argparse
import json
import math
import time

import numpy as np
from mast3r_slam.rgbd_transport import RGBDClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True, help="Camera computer's Tailscale IP")
    parser.add_argument('--port', type=int, default=9877)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--fps', type=float, default=5.)
    parser.add_argument('--timeout', type=float, default=10.)
    args = parser.parse_args()
    if (args.frames <= 0 or not math.isfinite(args.fps) or args.fps <= 0 or
            not math.isfinite(args.timeout) or args.timeout <= 0 or not 1 <= args.port <= 65535):
        parser.error('frames, fps and timeout must be positive; port must be 1..65535')
    client = RGBDClient(args.host, args.port, args.timeout)
    requests, sizes = [], []
    started = time.monotonic()
    try:
        for i in range(args.frames):
            before = time.monotonic()
            meta, rgb, depth = client.next()
            stats = client.stats
            requests.append(stats['request_ms'])
            sizes.append(stats['bytes'])
            if i == 0:
                print(f"{meta.get('camera', 'RGB-D')} serial={meta.get('serial', '?')} "
                      f"{rgb.shape[1]}x{rgb.shape[0]}; RGB={meta['rgb_codec']}; depth={meta['depth_scale']} m/unit")
                print('K:', meta['K'])
            z = depth[depth > 0].astype(np.float32)*meta['depth_scale']
            print(f"frame={i+1} sequence={meta['sequence']} valid={(depth>0).mean():.1%} "
                  f"median_depth={float(np.median(z)) if z.size else 0.:.3f}m "
                  f"packet={stats['bytes']/1024:.1f}KiB request={stats['request_ms']:.1f}ms "
                  f"sender_age={stats['capture_age_ms']:.1f}ms skipped={stats['skipped']}", flush=True)
            if i+1 < args.frames:
                time.sleep(max(0., 1./args.fps-(time.monotonic()-before)))
    finally:
        client.close()
    elapsed = time.monotonic()-started
    print(json.dumps(dict(frames=len(sizes), elapsed_s=elapsed,
                          mean_packet_KiB=float(np.mean(sizes))/1024,
                          request_p50_ms=float(np.median(requests)),
                          request_p95_ms=float(np.percentile(requests, 95)),
                          payload_Mbps=sum(sizes)*8/elapsed/1e6)))
    print('request includes waiting/encoding/transfer/decoding; sender_age excludes network delay. '
          'These are not one-way camera-to-display latency measurements.')


if __name__ == '__main__':
    main()
