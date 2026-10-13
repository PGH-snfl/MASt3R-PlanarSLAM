#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/python_env.sh
export PATH="$(dirname -- "$project_python"):$PATH"
# Honor the host CUDA toolkit/GPU instead of forcing the original laptop's SM.
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"
export MAX_JOBS=2
exec .venv-map-appearance/bin/python -u offline_appearance.py "$@"
