"""Read RGB frames forwarded by scripts/ros2_rgb_receiver.py."""

import socket
import struct

import numpy as np

from mast3r_slam.dataloader import Intrinsics, MonocularDataset
from mast3r_slam.mast3r_utils import resize_img


class ROS2Dataset(MonocularDataset):
    HEADER = struct.Struct("!dIII4d")

    def __init__(self, host="127.0.0.1", port=50051):
        self.connection = None
        super().__init__()
        self.dataset_path = None
        self.save_results = False
        try:
            self.connection = socket.create_connection((host, port), timeout=5)
        except OSError as exc:
            self.close()
            raise RuntimeError(
                f"Cannot connect to ROS 2 RGB receiver at {host}:{port}. "
                "Start or restart scripts/ros2_rgb_receiver.py with /usr/bin/python3 "
                "on the same PC as MASt3R-SLAM."
            ) from exc
        try:
            self.connection.settimeout(30)
            self.pending_frame = self._receive_frame()
        except OSError as exc:
            self.close()
            raise RuntimeError(
                "Connected to the ROS 2 RGB receiver, but could not receive a frame. "
                "Check /camera/color/image_raw and /camera/color/camera_info "
                "and the receiver logs."
            ) from exc

        _, self.initial_image, calibration = self.pending_frame
        height, width = self.initial_image.shape[:2]
        self.camera_intrinsics = Intrinsics.from_calib(
            self.img_size, width, height, calibration
        )

    def __len__(self):
        return 999999

    def _read_exact(self, length):
        data = bytearray()
        while len(data) < length:
            chunk = self.connection.recv(length - len(data))
            if not chunk:
                raise ConnectionError("ROS 2 RGB receiver disconnected")
            data.extend(chunk)
        return data

    def _receive_frame(self):
        self.connection.sendall(b"\x01")
        timestamp, height, width, channels, *calibration = self.HEADER.unpack(
            self._read_exact(self.HEADER.size)
        )
        if channels != 3 or not (0 < height <= 8192 and 0 < width <= 8192):
            raise ValueError("Invalid RGB image dimensions from ROS 2 receiver")
        image = np.frombuffer(
            self._read_exact(height * width * channels), dtype=np.uint8
        ).reshape(height, width, channels).copy()
        return timestamp, image, calibration

    def read_img(self, idx):
        try:
            frame = self.pending_frame
            if frame is None:
                frame = self._receive_frame()
            self.pending_frame = None
        except OSError as exc:
            self.close()
            raise RuntimeError(
                "ROS 2 RGB stream stopped. Check the receiver and camera topics."
            ) from exc
        timestamp, image, _ = frame
        self.timestamps.append(timestamp)
        return image

    def get_timestamp(self, idx):
        return self.timestamps[-1]

    def get_img_shape(self):
        image = resize_img(self.initial_image, self.img_size)
        return image["img"][0].shape[1:], self.initial_image.shape[:2]

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()
