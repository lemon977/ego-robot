#!/bin/sh
# Activation-free entry point for the project-local FoundationStereo CPU env.
set -eu

tool_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
project_root=$(CDPATH= cd -- "$tool_dir/.." && pwd -P)
environment_root="$project_root/assets/environments/foundationstereo-py311-v1"
python_executable="$environment_root/bin/python"
snapshot_manifest="$environment_root/.chaoyang-hardlink-snapshot/manifest.json.gz"
checkpoint_pin="$project_root/assets/models/checkpoints/foundationstereo/23-51-11/ASSET_PIN.json"
cache_root="$project_root/_run/caches/foundationstereo-py311-v1"

if [ ! -f "$snapshot_manifest" ] || [ ! -f "$checkpoint_pin" ]; then
    echo "FoundationStereo local runtime is incomplete: $environment_root" >&2
    echo "External Conda, checkpoint, cache, and PYTHONPATH fallback are forbidden." >&2
    exit 78
fi
if [ ! -x "$python_executable" ]; then
    echo "FoundationStereo local Python is missing: $python_executable" >&2
    exit 78
fi
mkdir -p "$cache_root/tmp" "$cache_root/torch" "$cache_root/huggingface"

unset PYTHONHOME PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE
unset CONDA_SHLVL CONDA_PROMPT_MODIFIER _CE_CONDA _CE_M LD_PRELOAD CUDA_HOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
# This authority is CPU-only until a separately leased checkpoint replay passes.
export CUDA_VISIBLE_DEVICES=""
export CHA0YANG_PROJECT_ROOT="$project_root"
export CHA0YANG_FOUNDATIONSTEREO_ROOT="$project_root/third_party/FoundationStereo"
export CHA0YANG_FOUNDATIONSTEREO_ENV_ROOT="$environment_root"
export CHA0YANG_FOUNDATIONSTEREO_CHECKPOINT_ROOT="$project_root/assets/models/checkpoints/foundationstereo/23-51-11"
export CONDA_PREFIX="$environment_root"
export CONDA_DEFAULT_ENV="chaoyang-foundationstereo-py311-v1"
export XDG_CACHE_HOME="$cache_root"
export TORCH_HOME="$cache_root/torch"
export HF_HOME="$cache_root/huggingface"
export FOUNDATION_STEREO_CACHE="$cache_root"
export TMPDIR="$cache_root/tmp"
export TMP="$cache_root/tmp"
export TEMP="$cache_root/tmp"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="$environment_root/lib"
export PATH="$environment_root/bin:/usr/bin:/bin"

exec "$python_executable" "$@"
