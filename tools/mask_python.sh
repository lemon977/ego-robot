#!/bin/sh
# Activation-free entry point for the project-local Mask/SAM3 CPU environment.
set -eu

tool_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_root=$(CDPATH= cd -- "$tool_dir/.." && pwd -P)
environment_root="$project_root/assets/environments/mask-sam3-py311-v1"
python_executable="$environment_root/bin/python"
snapshot_manifest="$environment_root/.chaoyang-hardlink-snapshot/manifest.json.gz"
overlay_manifest="$environment_root/.chaoyang-hardlink-snapshot/mask-sam3-import-closure-v1/MANIFEST.json"
path_hook="$environment_root/lib/python3.11/site-packages/chaoyang_mask_sam3_local_v1.pth"
cache_root="$project_root/_run/caches/mask-sam3-py311-v1"

if [ ! -f "$snapshot_manifest" ] || [ ! -f "$overlay_manifest" ] || [ ! -f "$path_hook" ]; then
    echo "Mask local environment is incomplete: $environment_root" >&2
    echo "External Python and PYTHONPATH fallback are forbidden." >&2
    exit 78
fi
if [ ! -x "$python_executable" ]; then
    echo "Mask local Python is missing or not executable: $python_executable" >&2
    exit 78
fi
mkdir -p "$cache_root/tmp" "$cache_root/matplotlib"

unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE
unset CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M LD_PRELOAD CUDA_HOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
export CUDA_VISIBLE_DEVICES=""
export CHA0YANG_PROJECT_ROOT="$project_root"
export CHA0YANG_MASK_ENV_ROOT="$environment_root"
export CONDA_PREFIX="$environment_root"
export CONDA_DEFAULT_ENV="chaoyang-mask-sam3-py311-v1"
export XDG_CACHE_HOME="$cache_root"
export MPLCONFIGDIR="$cache_root/matplotlib"
export TMPDIR="$cache_root/tmp"
export TMP="$cache_root/tmp"
export TEMP="$cache_root/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
export LD_LIBRARY_PATH="$environment_root/lib"
export PATH="$environment_root/bin:/usr/bin:/bin"

exec "$python_executable" "$@"
