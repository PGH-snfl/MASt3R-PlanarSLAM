#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=4
exec .venv-map-appearance/bin/python -u offline_textured_mesh.py "$@"
