#!/usr/bin/env bash
# Environment binding only. Registration, lease and torchrun stay in the runner.
set -euo pipefail
ICT_JOINT26_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${ICT_JOINT26_ROOT}"
# Parse an explicit interpreter here; only training arguments reach Python.
ICT_JOINT26_CLI_PYTHON=""
ICT_JOINT26_FORWARD_ARGS=()
while (($#)); do
  case "$1" in
    --python|--python=*)
      if [[ -n "$ICT_JOINT26_CLI_PYTHON" ]]; then
        echo '错误：--python 只能指定一次。' >&2; exit 2
      fi
      if [[ "$1" == --python ]]; then
        if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
          echo '错误：--python 需要激活后的 Conda Python 路径。' >&2; exit 2
        fi
        ICT_JOINT26_CLI_PYTHON="$2"; shift 2
      else
        ICT_JOINT26_CLI_PYTHON="${1#--python=}"; shift
        if [[ -z "$ICT_JOINT26_CLI_PYTHON" ]]; then
          echo '错误：--python 不能为空。' >&2; exit 2
        fi
      fi ;;
    *) ICT_JOINT26_FORWARD_ARGS+=("$1"); shift ;;
  esac
done
set -- "${ICT_JOINT26_FORWARD_ARGS[@]}"
if [[ -n "$ICT_JOINT26_CLI_PYTHON" ]]; then
  export ICT_PYTHON="$ICT_JOINT26_CLI_PYTHON"
fi
# A stale PYTHONHOME overrides even the explicitly selected interpreter.
unset PYTHONHOME
if [[ -n "${ICT_PYTHON:-}" && -z "$ICT_JOINT26_CLI_PYTHON" ]]; then
  export PYTHONPATH="${ICT_JOINT26_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
else
  # The dedicated environment must not inherit another environment's modules.
  export PYTHONPATH="${ICT_JOINT26_ROOT}/src"
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
usage() {
  echo '用法：bash scripts/humanego_ict/train_joint26_pick_place.sh --branch raw|robotized --manifest <新版26维缓存清单> --steps <正整数训练步数> --nproc 8 --output <项目内新目录> [--config <配置>] [--warmstart <MODEL张量>] [--preflight-only]' >&2
  echo '并行：--parallel --raw-gpu-ids 0,1,2,3 --robotized-gpu-ids 4,5,6,7 --manifest <清单> --steps <正整数> --output <新目录>' >&2
  echo '显式环境：--python "$CONDA_PREFIX/bin/python"；必须先激活 Conda。' >&2
  echo '批量：--global-batch 64 --micro-batch 8（--batch-size 是每卡 micro-batch 的别名）。' >&2
  echo '每分支全局 batch 保持64；梯度累积=64/(该分支GPU数×micro-batch)，必须整除。' >&2
  echo '训练默认按步数结束；可选 --wall-seconds <秒>；--json 保留机器接口。' >&2
  echo '不会自动准备数据，不接受旧50步×30维缓存；这是ICT模型，不是π0.5权重。' >&2
}
if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi
ICT_JOINT26_PREPARE=false
ICT_JOINT26_GPU_FLAG=''
for ICT_JOINT26_ARGUMENT in "$@"; do
  case "$ICT_JOINT26_ARGUMENT" in
    --help|-h) usage; exit 0 ;;
    --prepare) ICT_JOINT26_PREPARE=true ;;
  esac
  if [[ -n "$ICT_JOINT26_GPU_FLAG" ]]; then
    if [[ "$ICT_JOINT26_ARGUMENT" == *, ]]; then
      echo "GPU 参数 $ICT_JOINT26_GPU_FLAG 末尾多了逗号，请去掉：${ICT_JOINT26_ARGUMENT%,}" >&2
      exit 2
    fi
    ICT_JOINT26_GPU_FLAG=''
  fi
  case "$ICT_JOINT26_ARGUMENT" in
    --gpu-ids|--raw-gpu-ids|--robotized-gpu-ids) ICT_JOINT26_GPU_FLAG="$ICT_JOINT26_ARGUMENT" ;;
    --gpu-ids=*|--raw-gpu-ids=*|--robotized-gpu-ids=*)
      if [[ "$ICT_JOINT26_ARGUMENT" == *, ]]; then
        echo "GPU 参数末尾多了逗号，请去掉：${ICT_JOINT26_ARGUMENT%,}" >&2
        exit 2
      fi ;;
  esac
done
ICT_JOINT26_PYTHON="${ICT_PYTHON:-${ICT_JOINT26_ROOT}/.envs/humanego_ict/bin/python}"
if [[ "$ICT_JOINT26_PYTHON" != */* ]]; then
  ICT_JOINT26_PYTHON="$(command -v -- "$ICT_JOINT26_PYTHON" || true)"
fi
if [[ ! -x "$ICT_JOINT26_PYTHON" ]]; then
  echo '专用 ICT 环境尚未安装，或 ICT_PYTHON 不可执行。请先运行：' >&2
  echo "bash \"${ICT_JOINT26_ROOT}/scripts/humanego_ict/setup_joint26_env.sh\"" >&2
  exit 2
fi
if [[ -z "${ICT_PYTHON:-}" ]]; then
  ICT_JOINT26_REAL_PYTHON="$(realpath -- "$ICT_JOINT26_PYTHON")"
  if [[ "$ICT_JOINT26_REAL_PYTHON" != "$ICT_JOINT26_ROOT/.envs/humanego_ict/"* ]]; then
    echo '专用环境 Python 链接到了环境外，拒绝使用；请保留此环境并检查安装收据。' >&2
    exit 2
  fi
fi
export PYTHONNOUSERSITE=1
ICT_JOINT26_SCRATCH="$("$ICT_JOINT26_PYTHON" -B - "$ICT_JOINT26_ROOT" <<'PY'
import os, pathlib, sys, tempfile
root = pathlib.Path(sys.argv[1]).resolve()
base = pathlib.Path(os.environ.get('ICT_BOOTSTRAP_SCRATCH_ROOT', str(root/'_run/environment/humanego_ict/startup')))
if not base.is_absolute() or not base.resolve().is_relative_to(root) or base.resolve() == root:
    raise SystemExit('启动缓存必须是项目内的明确子目录。')
for ancestor in (base, *base.parents):
    if ancestor == root: break
    if ancestor.is_symlink(): raise SystemExit('启动缓存不能经过符号链接。')
base.mkdir(parents=True, exist_ok=True)
print(tempfile.mkdtemp(prefix='startup_', dir=base))
PY
)"
export ICT_ENV_CHECK_SCRATCH="$ICT_JOINT26_SCRATCH"
export TMPDIR="$ICT_JOINT26_SCRATCH" TMP="$ICT_JOINT26_SCRATCH" TEMP="$ICT_JOINT26_SCRATCH"
for ICT_JOINT26_CACHE in PIP_CACHE_DIR XDG_CACHE_HOME NUMBA_CACHE_DIR TORCH_HOME TORCH_EXTENSIONS_DIR TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR CUDA_CACHE_PATH HF_HOME HUGGINGFACE_HUB_CACHE MPLCONFIGDIR; do
  export "$ICT_JOINT26_CACHE=$ICT_JOINT26_SCRATCH/cache/${ICT_JOINT26_CACHE,,}"
done
ICT_JOINT26_CHECK_ARGS=(--startup-check --quiet-success)
if [[ "$ICT_JOINT26_PREPARE" == true ]]; then
  ICT_JOINT26_CHECK_ARGS+=(--prepare)
fi
"$ICT_JOINT26_PYTHON" -B "$ICT_JOINT26_ROOT/scripts/humanego_ict/check_joint26_env.py" "${ICT_JOINT26_CHECK_ARGS[@]}"
exec "$ICT_JOINT26_PYTHON" -B "${ICT_JOINT26_ROOT}/scripts/humanego_ict/joint26_runner_v1.py" "$@"
