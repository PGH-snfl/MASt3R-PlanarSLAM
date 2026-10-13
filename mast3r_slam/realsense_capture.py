"""CPU-only RealSense aligned/rectified capture for the network sender."""
import math
import threading
import uuid

import cv2
import numpy as np
import pyrealsense2 as rs

from mast3r_slam.rgbd_transport import LatestFrame, StreamError


def rectification_maps(intr):
    """Use the SDK distortion model, including inverse/modified Brown."""
    K = np.array([[intr.fx, 0., intr.ppx], [0., intr.fy, intr.ppy], [0., 0., 1.]])
    v, u = np.mgrid[:intr.height, :intr.width]
    rays = np.column_stack(((u.ravel() - intr.ppx) / intr.fx,
                            (v.ravel() - intr.ppy) / intr.fy, np.ones(u.size)))
    pixels = np.asarray([rs.rs2_project_point_to_pixel(intr, ray.tolist())
                         for ray in rays], dtype=np.float32).reshape(intr.height, intr.width, 2)
    return K, pixels[..., 0].copy(), pixels[..., 1].copy()


class RealSenseCapture:
    @staticmethod
    def devices():
        return [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
                for d in rs.context().query_devices()]

    def __init__(self, serial=None, width=640, height=480, fps=30):
        devices = self.devices()
        if not devices:
            raise StreamError('No USB RealSense device found on this computer')
        if serial is None:
            if len(devices) != 1:
                raise StreamError('Multiple RealSense devices; select --serial')
            serial = devices[0][1]
        self.latest = LatestFrame()
        self.stop = threading.Event()
        self.thread = None
        self.running = False
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            profile = self.pipeline.start(cfg)
            self.running = True
            device = profile.get_device()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K, self.mapx, self.mapy = rectification_maps(intr)
            self.align = rs.align(rs.stream.color)
            self.metadata = dict(version=1, session=uuid.uuid4().hex,
                                 geometry='rectified_color', width=intr.width, height=intr.height,
                                 K=K.tolist(), depth_scale=device.first_depth_sensor().get_depth_scale(),
                                 camera=device.get_info(rs.camera_info.name), serial=serial, fps=fps)
            self.thread = threading.Thread(target=self._run, name='rgbd-capture', daemon=True)
            self.thread.start()
            self.latest.get(timeout=10.)
        except BaseException:
            self.close()
            raise

    def _run(self):
        sequence = 0
        last_pair, last_time = None, None
        try:
            while not self.stop.is_set():
                frames = self.align.process(self.pipeline.wait_for_frames(timeout_ms=5000))
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                pair = (color.get_frame_number(), depth.get_frame_number())
                if last_pair is not None and (pair[0] == last_pair[0] or pair[1] == last_pair[1]):
                    continue
                timestamp, depth_time = color.get_timestamp()/1000., depth.get_timestamp()/1000.
                if not math.isfinite(timestamp) or not math.isfinite(depth_time):
                    raise StreamError('Invalid camera timestamp')
                if last_time is not None and timestamp <= last_time:
                    raise StreamError('Camera clock reset; restart sender and SLAM')
                color_domain, depth_domain = str(color.get_frame_timestamp_domain()), str(depth.get_frame_timestamp_domain())
                if color_domain != depth_domain:
                    raise StreamError('RGB and depth timestamp domains differ')
                skew = abs(timestamp-depth_time) * 1000
                # The SDK frameset synchronizes streams; reject pairs more than
                # one capture interval apart rather than pairing unrelated frames.
                if skew > 1000. / self.metadata['fps'] + 2.:
                    continue
                rgb = cv2.remap(np.asanyarray(color.get_data()), self.mapx, self.mapy, cv2.INTER_LINEAR)
                depth_u16 = cv2.remap(np.asanyarray(depth.get_data()), self.mapx, self.mapy, cv2.INTER_NEAREST)
                meta = dict(self.metadata, sequence=sequence, timestamp=timestamp,
                            depth_timestamp=depth_time, color_frame_number=pair[0], depth_frame_number=pair[1],
                            timestamp_domain=color_domain, pair_skew_ms=skew)
                self.latest.publish(meta, rgb, depth_u16)
                last_pair, last_time = pair, timestamp
                sequence += 1
        except Exception as exc:
            if not self.stop.is_set():
                self.latest.fail(exc)

    def close(self):
        self.stop.set()
        self.latest.close()
        if self.thread is not None:
            self.thread.join(timeout=6.)
        if self.running:
            self.running = False
            self.pipeline.stop()
