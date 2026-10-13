"""USB RGB-D input, aligned and rectified into the MASt3R image geometry."""
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from mast3r_slam.config import config
from mast3r_slam.dataloader import Intrinsics, MonocularDataset
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.realsense_capture import rectification_maps


def resize_depth(depth, image_size=512):
    """Use exactly the RGB resize/crop, without interpolating invalid depths."""
    H, W = depth.shape
    rgb, (sw, sh, left, top) = resize_img(np.zeros((H, W, 3), np.float32), image_size, return_transformation=True)
    h, w = rgb["img"].shape[-2:]
    resized = cv2.resize(depth, (round(W/sw), round(H/sh)), interpolation=cv2.INTER_NEAREST)
    return resized[int(top):int(top)+h, int(left):int(left)+w].copy()


def calibrated_intrinsics(intr, image_size=512):
    """Rectified pinhole pixels -> raw color pixels using the device SDK model.

    RealSense inverse/modified Brown coefficients are not OpenCV coefficients.
    Build this LUT once; RGB and aligned depth then share the same remapping.
    Keeping the original K/FOV may leave invalid pixels near the border.
    """
    K, mapx, mapy = rectification_maps(intr)
    return Intrinsics(image_size, intr.width, intr.height, K, K.copy(),
                      np.asarray(intr.coeffs), mapx, mapy)


class RealSenseRGBDDataset(MonocularDataset):
    @staticmethod
    def devices():
        return [(d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
                for d in rs.context().query_devices()]

    def __init__(self, serial=None):
        super().__init__()
        self.img_size = config.get("dataset", {}).get("image_size", 512)
        if self.img_size not in (512, 640):
            raise ValueError("USB RGB-D dataset.image_size must be 512 or 640")
        self.dataset_path = None
        self.save_results = False
        self.running = False
        self.last_number = None
        self.last_timestamp = None
        self.last_depth = None
        self.last_valid_fraction = 0.
        self.cfg = config.get("realsense", {})
        devices = self.devices()
        if not devices:
            raise RuntimeError("No USB RealSense camera found. Connect D457 to this PC.")
        if serial is None:
            if len(devices) != 1:
                raise RuntimeError("Multiple RealSense cameras found; select --serial.")
            serial = devices[0][1]
        if serial not in [s for _, s in devices]:
            raise RuntimeError(f"RealSense serial {serial} is not connected.")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        w, h, fps = (int(self.cfg.get(k, v)) for k, v in [("width", 640), ("height", 480), ("fps", 30)])
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try:
            profile = self.pipeline.start(cfg)
            self.running = True
            self.align = rs.align(rs.stream.color)
            device = profile.get_device()
            self.depth_scale = device.first_depth_sensor().get_depth_scale()
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.camera_intrinsics = calibrated_intrinsics(intr, self.img_size)
            if self.camera_intrinsics is None:
                raise RuntimeError("USB RGB-D input requires use_calib: true")
            self._pending = self._capture()
            self.raw_shape = self._pending[1].shape[:2]
            print(f"USB RGB-D: {device.get_info(rs.camera_info.name)}; serial={serial}; "
                  f"{w}x{h}@{fps}; processing={self._pending[2].shape[1]}x{self._pending[2].shape[0]}; "
                  f"depth scale={self.depth_scale} m/unit", flush=True)
        except Exception:
            self.close()
            raise

    def _capture(self):
        deadline = time.monotonic()+self.cfg.get("timeout_ms", 5000)/1000.
        while time.monotonic() < deadline:
            timeout = max(1, int((deadline-time.monotonic())*1000))
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout)
            aligned = self.align.process(frames)
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color or not depth:
                continue
            number = (color.get_frame_number(), depth.get_frame_number())
            if self.last_number is not None and (number[0] == self.last_number[0] or number[1] == self.last_number[1]):
                continue
            timestamp = color.get_timestamp()/1000.
            if not math.isfinite(timestamp):
                raise RuntimeError("Non-finite camera timestamp")
            # Small resets of the device clock do not interrupt live tracking.
            if self.last_timestamp is not None and timestamp <= self.last_timestamp:
                timestamp = self.last_timestamp+1./self.cfg.get("fps", 30)
            self.last_number, self.last_timestamp = number, timestamp
            rgb = np.asanyarray(color.get_data()).copy()
            z = np.asanyarray(depth.get_data()).astype(np.float32)*self.depth_scale
            intr = self.camera_intrinsics
            rgb = cv2.remap(rgb, intr.mapx, intr.mapy, cv2.INTER_LINEAR)
            z = cv2.remap(z, intr.mapx, intr.mapy, cv2.INTER_NEAREST)
            good = np.isfinite(z) & (z >= self.cfg.get("min_depth", .25)) & (z <= self.cfg.get("max_depth", 5.))
            z[~good] = 0
            return timestamp, rgb.astype(np.float32)/255., resize_depth(z, self.img_size)
        raise RuntimeError("No new synchronized RGB-D pair received within timeout")

    def __getitem__(self, idx):
        sample = self._pending if self._pending is not None else self._capture()
        self._pending = None
        timestamp, rgb, self.last_depth = sample
        self.last_valid_fraction = float((self.last_depth > 0).mean())
        self.timestamps.append(timestamp)
        return timestamp, rgb

    def __len__(self):
        return 2**31-1

    def get_img_shape(self):
        return self._pending[2].shape, self.raw_shape

    def close(self):
        if self.running:
            self.running = False
            self.pipeline.stop()
