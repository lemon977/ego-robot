#!/bin/sh
# Activation-free entry point for the project-local Clean + LaMa ONNX runtime.
set -eu

tool_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_root=$(CDPATH= cd -- "$tool_dir/.." && pwd -P)
base_environment="$project_root/assets/environments/clean-track4world-py311-v1"
overlay="$project_root/assets/environments/clean-lama-onnx-py311-v1/site-packages"
python_executable="$base_environment/bin/python"
model="$project_root/assets/models/lama-onnx/lama_fp32.onnx"

if [ ! -f "$base_environment/.chaoyang-hardlink-snapshot/manifest.json.gz" ]; then
    echo "Clean base environment is not published: $base_environment" >&2
    exit 78
fi
if [ ! -f "$overlay/onnxruntime/__init__.py" ]; then
    echo "Project-local ONNX Runtime overlay is missing: $overlay" >&2
    exit 78
fi
if [ ! -f "$model" ]; then
    echo "Project-local LaMa model is missing: $model" >&2
    exit 78
fi

unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE
unset CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M LD_PRELOAD CUDA_HOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export CHA0YANG_PROJECT_ROOT="$project_root"
export CHA0YANG_CLEAN_ENV_ROOT="$base_environment"
export CHA0YANG_CLEAN_LAMA_OVERLAY="$overlay"
export CHA0YANG_CLEAN_LAMA_MODEL="$model"
export CONDA_PREFIX="$base_environment"
export CONDA_DEFAULT_ENV="chaoyang-clean-lama-onnx-py311-v1"
export XDG_CACHE_HOME="$project_root/_run/caches/clean-lama-onnx-py311-v1"
export MPLCONFIGDIR="$project_root/_run/caches/clean-lama-onnx-py311-v1/matplotlib"
export PYTHONPATH="$overlay"
export LD_LIBRARY_PATH="$base_environment/lib"
export PATH="$base_environment/bin:/usr/bin:/bin"

exec "$python_executable" "$@"
