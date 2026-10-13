#!/usr/bin/env python3
"""Forward Go2 RealSense RGB images from ROS 2 to MASt3R's Python 3.11 process."""
import select
import socket
import struct
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


class Receiver(Node):
    def __init__(self):
        super().__init__("mast3r_ros2_rgb_receiver")
        self.latest = None
        self.intrinsics = None
        self.sequence = 0
        self.condition = threading.Condition()
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            Image, "/camera/color/image_raw", self.callback, sensor_qos
        )
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.camera_info, sensor_qos)

    def camera_info(self, msg):
        with self.condition:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.condition.notify_all()

    def callback(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warning(f"Unsupported encoding: {msg.encoding}")
            return
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            image = image[:, :, ::-1]
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.condition:
            self.latest = (timestamp, image.copy())
            self.sequence += 1
            self.condition.notify_all()


def serve_client(node, conn):
    last_sequence = -1
    while rclpy.ok():
        request = conn.recv(1)
        if not request:
            return
        next_warning = time.monotonic() + 5.0
        with node.condition:
            while rclpy.ok() and (
                node.latest is None
                or node.intrinsics is None
                or node.sequence == last_sequence
            ):
                # A client can time out while ROS messages are unavailable.
                # Release that connection so subsequent SLAM runs can connect.
                readable, _, _ = select.select([conn], [], [], 0)
                if readable and not conn.recv(1, socket.MSG_PEEK):
                    return
                if time.monotonic() >= next_warning:
                    missing = []
                    if node.latest is None or node.sequence == last_sequence:
                        missing.append("new /camera/color/image_raw")
                    if node.intrinsics is None:
                        missing.append("/camera/color/camera_info")
                    node.get_logger().warning(
                        "Waiting for " + " and ".join(missing)
                        + "; check the Go2 camera and ROS 2 connection."
                    )
                    next_warning = time.monotonic() + 5.0
                node.condition.wait(timeout=1.0)
            if not rclpy.ok():
                return
            timestamp, image = node.latest
            sequence = node.sequence
            intrinsics = node.intrinsics
        conn.sendall(struct.pack("!dIII4d", timestamp, *image.shape, *intrinsics))
        conn.sendall(image.tobytes())
        last_sequence = sequence


def main():
    rclpy.init()
    node = Receiver()
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 50051))
    server.listen(1)
    node.get_logger().info("Waiting for MASt3R on 127.0.0.1:50051")
    while rclpy.ok():
        conn, _ = server.accept()
        try:
            serve_client(node, conn)
        except ConnectionError:
            pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
