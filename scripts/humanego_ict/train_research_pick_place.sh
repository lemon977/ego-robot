#!/usr/bin/env bash
set -euo pipefail
ICT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ICT_PROJECT_ROOT}"
export PYTHONPATH="${ICT_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
if [[ $# -ne 5 ]]; then
  echo 'Usage: bash scripts/humanego_ict/train_research_pick_place.sh <raw|robotized> <1|8 GPUs> <new-training-task-packet.json> <research-manifest.json> <new-output-directory>' >&2
  exit 2
fi
ICT_BRANCH="$1"
ICT_GPUS="$2"
ICT_PACKET="$3"
ICT_MANIFEST="$4"
ICT_OUTPUT="$5"
case "${ICT_BRANCH}" in raw|robotized) ;; *) echo 'Invalid RGB branch' >&2; exit 2;; esac
case "${ICT_GPUS}" in 1|8) ;; *) echo 'Single-node 1 or 8 GPUs required' >&2; exit 2;; esac
ICT_CONFIG="${ICT_RESEARCH_CONFIG:-${ICT_PROJECT_ROOT}/docs/plans/PICK_PLACE_HUMANEGO_ICT_DERIVED_REVO2_V1/quick_completion_v1/train_research_20000.json}"
ICT_WEIGHTS="${ICT_RESEARCH_PRETRAINED:-${ICT_PROJECT_ROOT}/assets/models/humanego/pretrained_humanego/latest.pt}"
# Reject missing source/clock/causal/group/window evidence before any model or cache creation.
"${ICT_PYTHON:-python}" -B -m chaoyang.humanego_ict.research_training_cli preflight --manifest "${ICT_MANIFEST}" --branch "${ICT_BRANCH}"
ICT_ARGS=(train --task-packet "${ICT_PACKET}" --manifest "${ICT_MANIFEST}" --branch "${ICT_BRANCH}"
  --config "${ICT_CONFIG}" --output-dir "${ICT_OUTPUT}" --pretrained "${ICT_WEIGHTS}")
WORLD_SIZE="${ICT_GPUS}" ICT_RESEARCH_ADMISSION_ONLY=1 "${ICT_PYTHON:-python}" -B -m chaoyang.cli run \
  train_pick_place_humanego_ict_research_v1 "${ICT_ARGS[@]}"
export CUDA_VISIBLE_DEVICES="$("${ICT_PYTHON:-python}" -B -c 'import json,sys; from chaoyang.humanego_ict.runtime_gate import bind_cuda_visible_devices; print(bind_cuda_visible_devices(json.load(open(sys.argv[1]))))' "${ICT_PACKET}")"
ICT_CACHE_ROOT="${ICT_OUTPUT}/runtime_cache"
"${ICT_PYTHON:-python}" -B -c 'import sys; from pathlib import Path; from chaoyang.humanego_ict.runtime_gate import runtime_cache_paths; runtime_cache_paths(Path(sys.argv[1]), Path.cwd())' "${ICT_OUTPUT}"
mkdir -p -- "${ICT_CACHE_ROOT}/tmp" "${ICT_CACHE_ROOT}/torch" "${ICT_CACHE_ROOT}/inductor" "${ICT_CACHE_ROOT}/triton" "${ICT_CACHE_ROOT}/xdg"
export TMPDIR="${ICT_CACHE_ROOT}/tmp"
export TORCH_HOME="${ICT_CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${ICT_CACHE_ROOT}/inductor"
export TRITON_CACHE_DIR="${ICT_CACHE_ROOT}/triton"
export XDG_CACHE_HOME="${ICT_CACHE_ROOT}/xdg"
# These are direct paired candidates, not deployment authorization or a staged pretraining recipe.
exec "${ICT_PYTHON:-python}" -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${ICT_GPUS}" \
  -m chaoyang.cli run train_pick_place_humanego_ict_research_v1 "${ICT_ARGS[@]}"
