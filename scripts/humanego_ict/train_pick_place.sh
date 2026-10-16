#!/usr/bin/env bash
set -euo pipefail
ICT_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ICT_PROJECT_ROOT}"
export PYTHONPATH="${ICT_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
if [[ $# -lt 5 ]]; then
  echo 'Usage: bash scripts/humanego_ict/train_pick_place.sh <raw|robotized> <1|8 GPUs> <task-packet.json> <paired-manifest.json> <new-output-directory> [candidate-operation arguments]' >&2
  exit 2
fi
ICT_BRANCH="$1"
ICT_GPUS="$2"
ICT_PACKET="$3"
ICT_MANIFEST="$4"
ICT_OUTPUT="$5"
shift 5
case "${ICT_BRANCH}" in raw|robotized) ;; *) echo 'Invalid RGB branch' >&2; exit 2;; esac
case "${ICT_GPUS}" in 1|8) ;; *) echo 'This handoff validates single-node 1 or 8 GPUs only' >&2; exit 2;; esac
# A rejected preflight must stop before cache creation, torchrun or task writes.
"${ICT_PYTHON:-python}" -B -m chaoyang.humanego_ict.cli preflight --manifest "${ICT_MANIFEST}"
ICT_CONFIG="${ICT_PROJECT_ROOT}/docs/plans/PICK_PLACE_HUMANEGO_ICT_REVO2_V1/train_20000.json"
ICT_WEIGHTS="${ICT_PROJECT_ROOT}/assets/models/humanego/pretrained_humanego/latest.pt"
ICT_ARGS=(--task-packet "${ICT_PACKET}" --manifest "${ICT_MANIFEST}"
  --branch "${ICT_BRANCH}" --config "${ICT_CONFIG}" --output-dir "${ICT_OUTPUT}"
  --pretrained "${ICT_WEIGHTS}" "$@")
WORLD_SIZE="${ICT_GPUS}" ICT_ADMISSION_ONLY=1 "${ICT_PYTHON:-python}" -B -m chaoyang.cli run \
  train_pick_place_humanego_ict_revo2_candidate_v1 "${ICT_ARGS[@]}"
export CUDA_VISIBLE_DEVICES="$("${ICT_PYTHON:-python}" -B -c 'import json,sys; from chaoyang.humanego_ict.runtime_gate import bind_cuda_visible_devices; print(bind_cuda_visible_devices(json.load(open(sys.argv[1]))))' "${ICT_PACKET}")"
ICT_CACHE_ROOT="${ICT_OUTPUT}/runtime_cache"
"${ICT_PYTHON:-python}" -B -c 'import sys; from pathlib import Path; from chaoyang.humanego_ict.runtime_gate import runtime_cache_paths; runtime_cache_paths(Path(sys.argv[1]), Path.cwd())' "${ICT_OUTPUT}"
mkdir -p -- "${ICT_CACHE_ROOT}/tmp" "${ICT_CACHE_ROOT}/torch" "${ICT_CACHE_ROOT}/inductor" "${ICT_CACHE_ROOT}/triton" "${ICT_CACHE_ROOT}/xdg"
export TMPDIR="${ICT_CACHE_ROOT}/tmp"
export TORCH_HOME="${ICT_CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${ICT_CACHE_ROOT}/inductor"
export TRITON_CACHE_DIR="${ICT_CACHE_ROOT}/triton"
export XDG_CACHE_HOME="${ICT_CACHE_ROOT}/xdg"
# No registered operation / active packet -> existing Chaoyang CLI refuses
# before importing the candidate model. Registration is a separate finite task.
exec "${ICT_PYTHON:-python}" -B -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="${ICT_GPUS}" \
  -m chaoyang.cli run train_pick_place_humanego_ict_revo2_candidate_v1 \
  "${ICT_ARGS[@]}"
