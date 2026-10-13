#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source scripts/python_env.sh
exec "$project_python" -u view_saved_map.py "$@"
