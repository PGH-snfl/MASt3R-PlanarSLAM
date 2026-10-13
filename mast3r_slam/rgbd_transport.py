"""Bounded, request/response RGB-D transport; no torch, pickle or GPU required.

Each request is one byte ('F'). Each reply contains JSON calibration/timing,
RGB JPEG/PNG, and zlib-compressed little-endian uint16 depth. The camera side
keeps only the latest synchronized pair. One outstanding request per client
prevents an inference backlog from becoming a network frame backlog.
Use a trusted private connection (e.g. Tailscale); this protocol has no TLS/auth.
"""
import json
import math
import socket
import struct
import threading
import time
import zlib

import cv2
import numpy as np

HEADER = struct.Struct('!4sIII')
MAGIC = b'RGB1'
MAX_META = 16384
MAX_PIXELS = 1920 * 1080
MAX_RGB = 8 * 1024 * 1024
MAX_DEPTH = 5 * 1024 * 1024


class StreamError(RuntimeError):
    pass


def read_exact(sock, count, deadline):
    data = bytearray()
    while len(data) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('RGB-D packet deadline exceeded')
        sock.settimeout(remaining)
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise StreamError('Camera connection closed during a packet')
        data.extend(chunk)
    return bytes(data)


def validate_metadata(meta):
    try:
        if meta['version'] != 1 or meta['geometry'] != 'rectified_color':
            raise ValueError('unsupported protocol or image geometry')
        w, h = meta['width'], meta['height']
        if type(w) is not int or type(h) is not int or not (0 < w <= 1920 and 0 < h <= 1080):
            raise ValueError('invalid image dimensions')
        if w * h > MAX_PIXELS:
            raise ValueError('image too large')
        K = np.asarray(meta['K'], dtype=np.float64)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or
                K[0, 0] <= 0 or K[1, 1] <= 0 or
                not np.allclose(K[2], [0, 0, 1]) or K[0, 1] != 0 or K[1, 0] != 0):
            raise ValueError('invalid camera intrinsics')
        if not math.isfinite(meta['depth_scale']) or not 0 < meta['depth_scale'] <= .1:
            raise ValueError('invalid depth scale')
        if type(meta['sequence']) is not int or meta['sequence'] < 0:
            raise ValueError('invalid frame sequence')
        for key in ('timestamp', 'depth_timestamp', 'capture_age_ms', 'pair_skew_ms'):
            if not math.isfinite(meta[key]) or meta[key] < 0:
                raise ValueError('invalid timestamp or age')
        if not isinstance(meta['session'], str) or not 0 < len(meta['session']) <= 128:
            raise ValueError('invalid session')
        if meta['rgb_codec'] not in ('jpeg', 'png'):
            raise ValueError('unsupported RGB codec')
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise StreamError(f'Invalid RGB-D metadata: {exc}') from exc
    return h, w


def encode_frame(meta, rgb, depth, codec='jpeg', quality=95):
    if rgb.dtype != np.uint8 or depth.dtype != np.uint16 or rgb.shape != (*depth.shape, 3):
        raise StreamError('Expected matching uint8 RGB and uint16 depth')
    meta = dict(meta, rgb_codec=codec)
    if validate_metadata(meta) != depth.shape:
        raise StreamError('Calibration/image dimensions differ')
    options = [cv2.IMWRITE_JPEG_QUALITY, quality] if codec == 'jpeg' else [cv2.IMWRITE_PNG_COMPRESSION, 1]
    ok, encoded = cv2.imencode('.jpg' if codec == 'jpeg' else '.png',
                               cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), options)
    if not ok:
        raise StreamError('RGB encoding failed')
    rgb_bytes = encoded.tobytes()
    depth_bytes = zlib.compress(depth.astype('<u2', copy=False).tobytes(), level=1)
    meta_bytes = json.dumps(meta, allow_nan=False, separators=(',', ':')).encode()
    if len(meta_bytes) > MAX_META or len(rgb_bytes) > MAX_RGB or len(depth_bytes) > MAX_DEPTH:
        raise StreamError('RGB-D packet exceeds size limit')
    return HEADER.pack(MAGIC, len(meta_bytes), len(rgb_bytes), len(depth_bytes)) + meta_bytes + rgb_bytes + depth_bytes


def receive_frame(sock, timeout=10.):
    deadline = time.monotonic() + timeout
    magic, nm, nr, nd = HEADER.unpack(read_exact(sock, HEADER.size, deadline))
    if magic != MAGIC or not (0 < nm <= MAX_META and 0 < nr <= MAX_RGB and 0 < nd <= MAX_DEPTH):
        raise StreamError('Invalid RGB-D header or excessive packet length')
    try:
        meta = json.loads(read_exact(sock, nm, deadline))
    except (ValueError, UnicodeError) as exc:
        raise StreamError('Invalid RGB-D JSON') from exc
    h, w = validate_metadata(meta)
    rgb_bytes = read_exact(sock, nr, deadline)
    depth_bytes = read_exact(sock, nd, deadline)
    rgb_bgr = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
    if rgb_bgr is None or rgb_bgr.shape != (h, w, 3):
        raise StreamError('Corrupt RGB image or wrong dimensions')
    # Bound decompression, and reject trailing/concatenated/truncated streams.
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(depth_bytes, h * w * 2 + 1)
    except zlib.error as exc:
        raise StreamError('Corrupt depth payload') from exc
    if len(raw) != h * w * 2 or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise StreamError('Invalid decompressed depth length/stream')
    depth = np.frombuffer(raw, dtype='<u2').reshape(h, w).astype(np.uint16, copy=True)
    return meta, cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB), depth, HEADER.size + nm + nr + nd


class LatestFrame:
    """Single immutable frame slot, with errors propagated to waiting clients."""
    def __init__(self):
        self.condition = threading.Condition()
        self.value = None
        self.error = None
        self.closed = False

    def publish(self, meta, rgb, depth):
        with self.condition:
            self.value = (meta, rgb, depth, time.monotonic())
            self.condition.notify_all()

    def fail(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def get(self, after=-1, timeout=5., max_age=.5):
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise StreamError(f'Camera capture failed: {self.error}') from self.error
                if self.closed:
                    raise StreamError('Camera stopped')
                if self.value is not None:
                    meta, rgb, depth, captured = self.value
                    age = time.monotonic() - captured
                    if meta['sequence'] > after and age <= max_age:
                        return dict(meta, capture_age_ms=age * 1000), rgb, depth
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise StreamError('No fresh synchronized RGB-D pair within timeout')
                self.condition.wait(remaining)


def serve_client(sock, latest, stop, *, codec='jpeg', quality=95, max_fps=10., timeout=10.):
    """No unsolicited frames: idle during model loading, fresh pair on request."""
    sequence = -1
    next_send = 0.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    while not stop.is_set():
        sock.settimeout(1.)
        try:
            command = sock.recv(1)
        except socket.timeout:
            if latest.error is not None:
                raise StreamError(f'Camera capture failed: {latest.error}')
            continue
        if not command:
            return
        if command != b'F':
            raise StreamError('Unknown RGB-D request')
        if stop.wait(max(0., next_send - time.monotonic())):
            return
        meta, rgb, depth = latest.get(sequence, timeout=min(timeout, 5.))
        packet = encode_frame(meta, rgb, depth, codec, quality)
        sock.settimeout(timeout)
        sock.sendall(packet)
        sequence = meta['sequence']
        next_send = time.monotonic() + 1. / max_fps


class RGBDClient:
    """Pull fresh frames. Fail closed on timeout, clock reset or camera change."""
    def __init__(self, host, port=9877, timeout=10.):
        self.timeout = timeout
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.identity = None
        self.sequence = -1
        self.timestamp = None
        self.stats = {}

    def next(self):
        started = time.monotonic()
        try:
            self.sock.settimeout(self.timeout)
            self.sock.sendall(b'F')
            meta, rgb, depth, size = receive_frame(self.sock, self.timeout)
            identity = [meta[k] for k in ('session', 'width', 'height', 'K', 'depth_scale')]
            if self.identity is not None and self.identity != identity:
                raise StreamError('Camera/calibration changed; restart SLAM with a new map')
            if meta['sequence'] <= self.sequence or (self.timestamp is not None and meta['timestamp'] <= self.timestamp):
                raise StreamError('Camera sequence/timestamp stopped or moved backwards')
            self.identity = identity
            skipped = max(0, meta['sequence'] - self.sequence - 1) if self.sequence >= 0 else 0
            self.sequence, self.timestamp = meta['sequence'], meta['timestamp']
            self.stats = dict(sequence=self.sequence, skipped=skipped, bytes=size,
                              request_ms=(time.monotonic() - started) * 1000,
                              capture_age_ms=meta['capture_age_ms'], pair_skew_ms=meta['pair_skew_ms'])
            return meta, rgb, depth
        except Exception:
            # A partial packet cannot safely be retried on this byte stream.
            self.close()
            raise

    def close(self):
        self.sock.close()
