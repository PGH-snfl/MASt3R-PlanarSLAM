#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Sender needs numpy, OpenCV and pyrealsense2; it does not import torch.
source scripts/python_env.sh
camera_python="${RGBD_PYTHON:-$project_python}"
exec "$camera_python" -u send_rgbd.py "$@"
