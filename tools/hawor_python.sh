#!/bin/sh
# Activation-free entry point for the project-local HaWoR Python environment.
set -eu

tool_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_root=$(CDPATH= cd -- "$tool_dir/.." && pwd -P)
environment_root="$project_root/assets/environments/hawor-py310-v1"
python_executable="$environment_root/bin/python"
snapshot_manifest="$environment_root/.chaoyang-hardlink-snapshot/manifest.json.gz"

if [ ! -f "$snapshot_manifest" ]; then
    echo "HaWoR local environment is not published: $snapshot_manifest" >&2
    echo "Authority remains ENV_PATH_MIGRATION_PENDING; do not fall back to another Conda prefix." >&2
    exit 78
fi
if [ ! -x "$python_executable" ]; then
    echo "HaWoR local Python is missing or not executable: $python_executable" >&2
    exit 78
fi

# Do not inherit an unrelated Python/Conda environment.  Base OS executables
# remain available for the upstream ffmpeg call and dynamic loader.
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE
unset CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M LD_PRELOAD CUDA_HOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CHA0YANG_PROJECT_ROOT="$project_root"
export CHA0YANG_HAWOR_ROOT="$project_root/third_party/HaWoR"
export CHA0YANG_HAWOR_ENV_ROOT="$environment_root"
export CONDA_PREFIX="$environment_root"
export CONDA_DEFAULT_ENV="chaoyang-hawor-py310-v1"
export XDG_CACHE_HOME="$project_root/_run/caches/hawor-py310-v1"
export HF_HOME="$project_root/_run/caches/hawor-py310-v1/huggingface"
export MPLCONFIGDIR="$project_root/_run/caches/hawor-py310-v1/matplotlib"
export NUMBA_CACHE_DIR="$project_root/_run/caches/hawor-py310-v1/numba"
export TORCH_EXTENSIONS_DIR="$project_root/_run/caches/hawor-py310-v1/torch-extensions"
export YOLO_CONFIG_DIR="$project_root/_run/caches/hawor-py310-v1/ultralytics"
export LD_LIBRARY_PATH="$environment_root/lib"
export PATH="$environment_root/bin:/usr/bin:/bin"

exec "$python_executable" "$@"
