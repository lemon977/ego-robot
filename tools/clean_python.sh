#!/bin/sh
# Activation-free entry point for the project-local Clean Python environment.
set -eu

tool_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_root=$(CDPATH= cd -- "$tool_dir/.." && pwd -P)
environment_root="$project_root/assets/environments/clean-track4world-py311-v1"
python_executable="$environment_root/bin/python"
snapshot_manifest="$environment_root/.chaoyang-hardlink-snapshot/manifest.json.gz"

if [ ! -f "$snapshot_manifest" ]; then
    echo "Clean local environment is not published: $snapshot_manifest" >&2
    echo "Do not fall back to an external Conda prefix." >&2
    exit 78
fi
if [ ! -x "$python_executable" ]; then
    echo "Clean local Python is missing or not executable: $python_executable" >&2
    exit 78
fi

unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE
unset CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M LD_PRELOAD CUDA_HOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export CHA0YANG_PROJECT_ROOT="$project_root"
export CHA0YANG_CLEAN_ENV_ROOT="$environment_root"
export CONDA_PREFIX="$environment_root"
export CONDA_DEFAULT_ENV="chaoyang-clean-track4world-py311-v1"
export XDG_CACHE_HOME="$project_root/_run/caches/clean-track4world-py311-v1"
export MPLCONFIGDIR="$project_root/_run/caches/clean-track4world-py311-v1/matplotlib"
export LD_LIBRARY_PATH="$environment_root/lib"
export PATH="$environment_root/bin:/usr/bin:/bin"

exec "$python_executable" "$@"
