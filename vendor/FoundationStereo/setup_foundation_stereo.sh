#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${REPO_DIR}/../.." && pwd)"
CONDA_BIN="/cpfs_infra/user/chenxianchi/miniconda3/bin/conda"
ENV_PREFIX="/cpfs_infra/user/chenxianchi/miniconda3/envs/foundation_stereo"
RUNTIME_CACHE="${PROJECT_ROOT}/_cache/foundationstereo"
CONDA_PACKAGE_CACHE="${RUNTIME_CACHE}/conda_pkgs"
PIP_PACKAGE_CACHE="${RUNTIME_CACHE}/pip"
UV_PACKAGE_CACHE="${RUNTIME_CACHE}/uv"
LOCAL_WHEELHOUSE="${RUNTIME_CACHE}/wheelhouse"
LOCAL_TORCH_WHEEL="${LOCAL_WHEELHOUSE}/torch-2.4.1-cp311-cp311-manylinux1_x86_64.whl"
MODEL_STORAGE="${PROJECT_ROOT}/assets/models/checkpoints/foundationstereo"
MODEL_DIR="${REPO_DIR}/pretrained_models"
CHECKPOINT="${MODEL_DIR}/23-51-11/model_best_bp2.pth"
MODEL_CFG="${MODEL_DIR}/23-51-11/cfg.yaml"
MODEL_URL="https://huggingface.co/Felix-Zhenghao/FoundationStereo/resolve/main/model_best_bp2.pth"
MODEL_SHA256="60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda not found: ${CONDA_BIN}" >&2
  exit 1
fi

mkdir -p \
  "${CONDA_PACKAGE_CACHE}" \
  "${PIP_PACKAGE_CACHE}" \
  "${UV_PACKAGE_CACHE}" \
  "${LOCAL_WHEELHOUSE}" \
  "${MODEL_STORAGE}"
if [[ ! -L "${MODEL_DIR}" && ! -e "${MODEL_DIR}" ]]; then
  ln -s "${MODEL_STORAGE}" "${MODEL_DIR}"
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  CONDA_PKGS_DIRS="${CONDA_PACKAGE_CACHE}" \
    "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.11 pip
fi

# uv resolves and downloads independent wheels concurrently. This is much
# faster and more reliable here than conda's serial pip phase. Re-running this
# command is safe and repairs an interrupted installation.
if [[ ! -x "${ENV_PREFIX}/bin/uv" ]]; then
  PIP_CACHE_DIR="${PIP_PACKAGE_CACHE}" \
    "${ENV_PREFIX}/bin/python" -m pip install uv
fi
UV_EXTRA_ARGS=()
if [[ -f "${LOCAL_TORCH_WHEEL}" ]]; then
  UV_EXTRA_ARGS+=("${LOCAL_TORCH_WHEEL}")
fi
UV_CACHE_DIR="${UV_PACKAGE_CACHE}" \
UV_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}" \
  "${ENV_PREFIX}/bin/uv" pip install \
  --python "${ENV_PREFIX}/bin/python" \
  --find-links "${LOCAL_WHEELHOUSE}" \
  "${UV_EXTRA_ARGS[@]}" \
  --requirements "${REPO_DIR}/requirements-pico.txt"

mkdir -p "${MODEL_DIR}/23-51-11"
if [[ ! -f "${MODEL_CFG}" ]]; then
  cp "${REPO_DIR}/configs/23-51-11-cfg.yaml" "${MODEL_CFG}"
fi

if ! echo "${MODEL_SHA256}  ${CHECKPOINT}" | sha256sum --check --status; then
  if command -v aria2c >/dev/null 2>&1; then
    ARIA_PROXY_ARGS=()
    if [[ -n "${HTTPS_PROXY:-}" ]]; then
      ARIA_PROXY_ARGS=(--all-proxy="${HTTPS_PROXY}")
    fi
    aria2c "${ARIA_PROXY_ARGS[@]}" --continue=true \
      --max-connection-per-server=8 --split=8 --min-split-size=8M \
      --file-allocation=none --dir="$(dirname "${CHECKPOINT}")" \
      --out="$(basename "${CHECKPOINT}")" "${MODEL_URL}"
  else
    curl --location --fail --retry 5 --retry-delay 5 --continue-at - \
      "${MODEL_URL}" --output "${CHECKPOINT}"
  fi
fi

if ! echo "${MODEL_SHA256}  ${CHECKPOINT}" | sha256sum --check --status; then
  echo "Checkpoint checksum failed: ${CHECKPOINT}" >&2
  exit 1
fi

"${ENV_PREFIX}/bin/python" - <<'PY'
import cv2
import torch
import xformers
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("OpenCV:", cv2.__version__)
print("xFormers:", xformers.__version__)
PY

echo "FoundationStereo is ready."
