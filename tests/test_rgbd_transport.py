"""Protocol/geometry regression tests, including real TCP and slow consumers."""
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import time
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mast3r_slam.rgbd_transport import (HEADER, MAGIC, MAX_META, LatestFrame,
    RGBDClient, StreamError, encode_frame, receive_frame, serve_client)


def sample(sequence=0, w=64, h=48):
    meta = dict(version=1, geometry='rectified_color', session='test-session',
                width=w, height=h, K=[[w*.8, 0., w*.5], [0., w*.8, h*.5], [0., 0., 1.]],
                depth_scale=.001, sequence=sequence, timestamp=100.+sequence/30.,
                depth_timestamp=100.+sequence/30., capture_age_ms=0., pair_skew_ms=0.)
    rgb = np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)
    depth = np.random.default_rng(24).integers(0, 65536, (h, w), dtype=np.uint16)
    depth[0, :4] = [0, 1, 1500, 65535]
    return meta, rgb, depth


class Server:
    """Bounded test server; exceptions are collected, never swallowed."""
    def __init__(self, handler):
        self.socket = socket.socket()
        self.socket.bind(('127.0.0.1', 0))
        self.socket.listen(1)
        self.socket.settimeout(5)
        self.port = self.socket.getsockname()[1]
        self.error = None
        def run():
            try:
                with self.socket.accept()[0] as conn:
                    handler(conn)
            except BaseException as exc:
                self.error = exc
            finally:
                self.socket.close()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def join(self):
        self.thread.join(5)
        if self.thread.is_alive():
            raise AssertionError('Test server did not stop')
        if self.error:
            raise self.error


class TransportTests(unittest.TestCase):
    def test_fragmented_packet_preserves_depth_rgb_calibration_and_timestamps(self):
        meta, rgb, depth = sample()
        packet = encode_frame(meta, rgb, depth, codec='png')
        def handler(conn):
            self.assertEqual(conn.recv(1), b'F')
            for offset in range(0, len(packet), 97):
                conn.sendall(packet[offset:offset+97])
        server = Server(handler)
        client = RGBDClient('127.0.0.1', server.port)
        try:
            got, image, z = client.next()
            np.testing.assert_array_equal(rgb, image)
            np.testing.assert_array_equal(depth, z)
            self.assertTrue(z.flags.writeable)
            for key in ('K', 'timestamp', 'depth_scale', 'depth_timestamp'):
                self.assertEqual(got[key], meta[key])
        finally:
            client.close()
            server.join()

    def test_jpeg_color_order_and_lossless_depth(self):
        meta, rgb, depth = sample()
        rgb[:] = [240, 30, 5]
        packet = encode_frame(meta, rgb, depth, quality=95)
        server = Server(lambda conn: (conn.recv(1), conn.sendall(packet)))
        client = RGBDClient('127.0.0.1', server.port)
        try:
            _, image, z = client.next()
            self.assertLess(np.abs(image.astype(float)-rgb).max(), 5)
            np.testing.assert_array_equal(depth, z)
        finally:
            client.close()
            server.join()

    def test_latest_only_after_slow_inference_and_reconnect(self):
        latest, stop = LatestFrame(), threading.Event()
        latest.publish(*sample(0))
        server = Server(lambda conn: serve_client(conn, latest, stop, max_fps=1000))
        client = RGBDClient('127.0.0.1', server.port)
        try:
            self.assertEqual(client.next()[0]['sequence'], 0)
            # Simulate camera advancing while the consumer performs inference.
            for seq in range(1, 51):
                latest.publish(*sample(seq))
            self.assertEqual(client.next()[0]['sequence'], 50)
            self.assertEqual(client.stats['skipped'], 49)
        finally:
            client.close()
            server.join()
        # A new client starts a new sequence, without replaying old frames.
        latest.publish(*sample(51))
        server = Server(lambda conn: serve_client(conn, latest, stop, max_fps=1000))
        client = RGBDClient('127.0.0.1', server.port)
        try:
            self.assertEqual(client.next()[0]['sequence'], 51)
        finally:
            client.close()
            server.join()

    def test_stale_or_failed_camera_does_not_repeat_previous_frame(self):
        latest = LatestFrame()
        latest.publish(*sample())
        with self.assertRaisesRegex(StreamError, 'fresh'):
            latest.get(after=0, timeout=.01)
        with self.assertRaisesRegex(StreamError, 'fresh'):
            latest.get(timeout=.01, max_age=0)
        latest.fail(RuntimeError('unplugged'))
        with self.assertRaisesRegex(StreamError, 'unplugged'):
            latest.get()

    def test_timeout_discards_partial_packet_and_closes_connection(self):
        def handler(conn):
            conn.recv(1)
            conn.sendall(b'RGB')
            time.sleep(.12)
        server = Server(handler)
        client = RGBDClient('127.0.0.1', server.port, timeout=.04)
        try:
            with self.assertRaises(TimeoutError):
                client.next()
            self.assertEqual(client.sock.fileno(), -1)
        finally:
            client.close()
            server.join()

    def test_truncated_packet_raises(self):
        packet = encode_frame(*sample())
        server = Server(lambda conn: (conn.recv(1), conn.sendall(packet[:-10])))
        client = RGBDClient('127.0.0.1', server.port)
        try:
            with self.assertRaisesRegex(StreamError, 'closed'):
                client.next()
        finally:
            client.close()
            server.join()

    def test_excessive_header_rejected_without_reading_payload(self):
        server = Server(lambda conn: (conn.recv(1), conn.sendall(HEADER.pack(MAGIC, MAX_META+1, 1, 1))))
        client = RGBDClient('127.0.0.1', server.port)
        try:
            with self.assertRaisesRegex(StreamError, 'header'):
                client.next()
        finally:
            client.close()
            server.join()

    def test_expanded_depth_size_limit(self):
        packet = encode_frame(*sample())
        magic, nm, nr, nd = HEADER.unpack(packet[:HEADER.size])
        bad_depth = zlib.compress(bytes(64*48*20))
        packet = HEADER.pack(magic, nm, nr, len(bad_depth)) + packet[HEADER.size:HEADER.size+nm+nr] + bad_depth
        server = Server(lambda conn: (conn.recv(1), conn.sendall(packet)))
        client = RGBDClient('127.0.0.1', server.port)
        try:
            with self.assertRaisesRegex(StreamError, 'depth length'):
                client.next()
        finally:
            client.close()
            server.join()

    def test_camera_change_and_time_reset_rejected(self):
        for key, value in [('session', 'different-camera'), ('timestamp', 99.),
                           ('sequence', 0), ('depth_scale', .002)]:
            with self.subTest(key=key):
                first = encode_frame(*sample(0))
                meta, rgb, depth = sample(1)
                meta[key] = value
                second = encode_frame(meta, rgb, depth)
                def handler(conn):
                    for packet in (first, second):
                        conn.recv(1)
                        conn.sendall(packet)
                server = Server(handler)
                client = RGBDClient('127.0.0.1', server.port)
                try:
                    client.next()
                    with self.assertRaises(StreamError):
                        client.next()
                    self.assertEqual(client.sock.fileno(), -1)
                finally:
                    client.close()
                    server.join()


class DatasetTests(unittest.TestCase):
    def test_network_depth_and_k_match_usb_processing_at_both_sizes(self):
        from mast3r_slam.config import config
        from mast3r_slam.network_rgbd import NetworkRGBDDataset
        from mast3r_slam.realsense_rgbd import resize_depth
        for size in (512, 640):
            with self.subTest(size=size), patch.dict(config, {'use_calib': True,
                     'dataset': {'image_size': size, 'img_downsample': 1},
                     'realsense': {'min_depth': .25, 'max_depth': 5.}}):
                meta, rgb, depth = sample(w=640, h=480)
                depth[:] = 1500
                depth[:, :80] = 0
                depth[:, -10:] = 6000  # out-of-range measurements stay invalid
                packet = encode_frame(meta, rgb, depth, 'png')
                server = Server(lambda conn: (conn.recv(1), conn.sendall(packet)))
                ds = NetworkRGBDDataset('127.0.0.1', server.port)
                try:
                    timestamp, image = ds[0]
                    expected = depth.astype(np.float32)*.001
                    expected[expected > 5] = 0
                    np.testing.assert_array_equal(ds.last_depth, resize_depth(expected, size))
                    np.testing.assert_allclose(image, rgb.astype(np.float32)/255.)
                    np.testing.assert_allclose(ds.camera_intrinsics.K_frame[:2], np.array(meta['K'])[:2]*(size/640))
                    self.assertEqual(ds.get_img_shape()[0], (size*3//4, size))
                    self.assertEqual(timestamp, meta['timestamp'])
                    self.assertEqual(len(ds.timestamps), 1)
                finally:
                    ds.close()
                    server.join()


if __name__ == '__main__':
    unittest.main()
