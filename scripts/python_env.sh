#!/usr/bin/env bash
# Source after changing to the project directory. Explicit override wins.
project_python="${MAST3R_PYTHON:-}"
if [[ -z "$project_python" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    project_python="$CONDA_PREFIX/bin/python"
fi
if [[ -z "$project_python" ]]; then
    for candidate in "$HOME/miniconda3/envs/mast3r-slam/bin/python" "$HOME/anaconda3/envs/mast3r-slam/bin/python"; do
        if [[ -x "$candidate" ]]; then
            project_python="$candidate"
            break
        fi
    done
fi
if [[ -z "$project_python" ]]; then
    project_python="$(command -v python3)"
fi
