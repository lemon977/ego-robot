#!/usr/bin/env bash
# Only this explicit setup command installs packages. Training never installs.
set -euo pipefail
ICT_SETUP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ICT_SETUP_PREFIX="$ICT_SETUP_ROOT/.envs/humanego_ict"
ICT_SETUP_CONDA="${CONDA_EXE:-}"
ICT_SETUP_OUTPUT=""
while (($#)); do
  case "$1" in
    --conda) ICT_SETUP_CONDA="${2:?--conda requires a path}"; shift 2 ;;
    --output) ICT_SETUP_OUTPUT="${2:?--output requires a project path}"; shift 2 ;;
    --help|-h)
      echo '用法：bash scripts/humanego_ict/setup_joint26_env.sh [--conda /path/to/conda] [--output <项目内新目录>]'
      echo '固定环境：.envs/humanego_ict；不修改现有 conda 环境；同一已完成环境可重复检查。'
      exit 0 ;;
    *) echo "错误：未知参数 $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$ICT_SETUP_CONDA" ]]; then
  ICT_SETUP_CONDA="$(command -v conda || true)"
fi
if [[ -z "$ICT_SETUP_CONDA" && -x /path/to/private_workspace/miniconda3/bin/conda ]]; then
  ICT_SETUP_CONDA=/path/to/private_workspace/miniconda3/bin/conda
fi
if [[ ! -x "$ICT_SETUP_CONDA" ]]; then
  echo '错误：找不到 Conda 可执行文件；请传 --conda /你的/miniconda3/bin/conda。' >&2
  exit 2
fi
[[ -n "$ICT_SETUP_OUTPUT" ]] || ICT_SETUP_OUTPUT="$ICT_SETUP_ROOT/_run/environment/humanego_ict/setup_$(date -u +%Y%m%dT%H%M%S%NZ)_$$"
# The base Python is used only for standard-library path/signature validation.
ICT_SETUP_BASE_PYTHON="$(dirname -- "$ICT_SETUP_CONDA")/python"
[[ -x "$ICT_SETUP_BASE_PYTHON" ]] || { echo '错误：Conda 同目录缺少基础 Python。' >&2; exit 2; }
"$ICT_SETUP_BASE_PYTHON" -B - "$ICT_SETUP_ROOT" "$ICT_SETUP_PREFIX" "$ICT_SETUP_OUTPUT" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
for raw in (*sys.argv[2:], str(pathlib.Path(sys.argv[2]) / '.wheels')):
    p = pathlib.Path(raw)
    if not p.is_absolute() or not p.resolve().is_relative_to(root) or p.resolve() == root:
        raise SystemExit('错误：环境及安装输出必须是项目内的明确子目录。')
    for ancestor in (p, *p.parents):
        if ancestor == root: break
        if ancestor.is_symlink(): raise SystemExit('错误：环境或输出目录不能经过符号链接。')
if pathlib.Path(sys.argv[3]).exists():
    raise SystemExit('错误：安装输出目录已存在；请使用新 --output。')
PY
mkdir -p -- "$ICT_SETUP_OUTPUT" "$ICT_SETUP_OUTPUT/tmp" "$ICT_SETUP_OUTPUT/cache/conda" "$ICT_SETUP_OUTPUT/cache/pip"
export CONDA_REGISTER_ENVS=false
export CONDA_PKGS_DIRS="$ICT_SETUP_OUTPUT/cache/conda"
export CONDA_ENVS_PATH="$ICT_SETUP_ROOT/.envs"
export CONDA_NO_PLUGINS=true
export PIP_CACHE_DIR="$ICT_SETUP_OUTPUT/cache/pip"
export TMPDIR="$ICT_SETUP_OUTPUT/tmp" TMP="$ICT_SETUP_OUTPUT/tmp" TEMP="$ICT_SETUP_OUTPUT/tmp"
export XDG_CACHE_HOME="$ICT_SETUP_OUTPUT/cache" NUMBA_CACHE_DIR="$ICT_SETUP_OUTPUT/cache/numba"
export TORCH_HOME="$ICT_SETUP_OUTPUT/cache/torch" TORCH_EXTENSIONS_DIR="$ICT_SETUP_OUTPUT/cache/torch_extensions"
export TRITON_CACHE_DIR="$ICT_SETUP_OUTPUT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$ICT_SETUP_OUTPUT/cache/inductor"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
ICT_SETUP_PROFILE="$ICT_SETUP_OUTPUT/REQUEST.json"
"$ICT_SETUP_BASE_PYTHON" -B - "$ICT_SETUP_ROOT" "$ICT_SETUP_PREFIX" "$ICT_SETUP_PROFILE" <<'PY'
import hashlib, json, pathlib, sys
root, prefix, output = map(pathlib.Path, sys.argv[1:])
files = [root/'scripts/humanego_ict/environment'/n for n in ('conda.yml','runtime-requirements.txt','torch-cu128-requirements.txt')]
locks = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
request = {'schema':'joint26-conda-environment-v1','prefix':str(prefix),'lock_sha256':locks}
marker = prefix/'JOINT26_ENVIRONMENT.json'
if prefix.exists():
    if not marker.is_file(): raise SystemExit('错误：固定环境路径已存在但无本项目安装收据；不覆盖它。')
    old=json.loads(marker.read_text())
    if old.get('lock_sha256') != locks: raise SystemExit('错误：现有环境锁不同；保留旧环境，请另开环境升级任务。')
output.write_text(json.dumps(request,ensure_ascii=False,indent=2)+'\n')
PY
if [[ ! -x "$ICT_SETUP_PREFIX/bin/python" ]]; then
  echo '【1/4】创建项目专用 Conda 环境：Python 3.11.11（不改 openwam_alpha）'
  timeout 1800 "$ICT_SETUP_CONDA" create --yes --prefix "$ICT_SETUP_PREFIX" \
    --override-channels --channel conda-forge --solver classic --no-default-packages \
    python=3.11.11 pip=23.3.2 2>&1 | tee "$ICT_SETUP_OUTPUT/conda.log"
  cp -- "$ICT_SETUP_PROFILE" "$ICT_SETUP_PREFIX/JOINT26_ENVIRONMENT.json"
fi
# Do not turn installed CUDA libraries into thousands of Git source files.
if [[ ! -e "$ICT_SETUP_PREFIX/.gitignore" ]]; then
  "$ICT_SETUP_BASE_PYTHON" -B - "$ICT_SETUP_PREFIX/.gitignore" <<'PY'
import pathlib, sys
with pathlib.Path(sys.argv[1]).open('x') as stream:
    stream.write('# Installed environment; rebuild using the project locks.\n*\n')
PY
fi
ICT_SETUP_PYTHON="$ICT_SETUP_PREFIX/bin/python"
"$ICT_SETUP_BASE_PYTHON" -B - "$ICT_SETUP_PREFIX" "$ICT_SETUP_PYTHON" <<'PY'
import pathlib, sys
prefix, interpreter = map(pathlib.Path, sys.argv[1:])
if not interpreter.resolve(strict=True).is_relative_to(prefix.resolve(strict=True)):
    raise SystemExit('错误：专用环境Python链接到环境外；拒绝安装，防止修改旧环境。')
PY
ICT_SETUP_READY="$("$ICT_SETUP_BASE_PYTHON" -B - "$ICT_SETUP_PREFIX/JOINT26_ENVIRONMENT.json" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text()).get('status', 'PARTIAL'))
PY
)"
if [[ "$ICT_SETUP_READY" == READY ]] && \
   "$ICT_SETUP_PYTHON" -B "$ICT_SETUP_ROOT/scripts/humanego_ict/check_joint26_env.py" --json \
     > "$ICT_SETUP_OUTPUT/ENVIRONMENT_CHECK.json"; then
  echo "同锁环境检查通过，直接复用，无需重新下载：$ICT_SETUP_PREFIX"
  exit 0
fi
echo '【2/4】安装锁定的 PyTorch / CUDA 12.8 依赖（首次下载较大）'
ICT_SETUP_WHEELS="$ICT_SETUP_PREFIX/.wheels"
mkdir -p -- "$ICT_SETUP_WHEELS"
# One pinned distribution per invocation: pip otherwise retains completed
# wheels in its temporary resolver directory until the WHOLE batch resolves.
download_locked() {
  local ICT_DL_FILE="$1" ICT_DL_LOG="$2" ICT_DL_BUDGET="$3"
  local ICT_DL_STARTED="$SECONDS" ICT_DL_PIN ICT_DL_REMAINING
  while IFS= read -r ICT_DL_PIN || [[ -n "$ICT_DL_PIN" ]]; do
    [[ "$ICT_DL_PIN" =~ ^[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+$ ]] || continue
    ICT_DL_REMAINING=$((ICT_DL_BUDGET - SECONDS + ICT_DL_STARTED))
    ((ICT_DL_REMAINING > 0)) || { echo '依赖下载达到有限预算，请保留日志后重试。' >&2; return 124; }
    echo "下载或复用：$ICT_DL_PIN" | tee -a "$ICT_DL_LOG"
    # Offline, verified local wheels take precedence. Do not compare a RECORD-
    # verified repack archive against an unrelated upstream archive hash.
    if timeout "$ICT_DL_REMAINING" "$ICT_SETUP_PYTHON" -B -m pip download --no-index --no-deps \
        --dest "$ICT_SETUP_WHEELS" --find-links "$ICT_SETUP_WHEELS" "$ICT_DL_PIN" \
        >> "$ICT_DL_LOG" 2>&1; then
      continue
    fi
    timeout "$ICT_DL_REMAINING" "$ICT_SETUP_PYTHON" -B -m pip download --no-deps \
      --retries 2 --timeout 60 --dest "$ICT_SETUP_WHEELS" --find-links "$ICT_SETUP_WHEELS" \
      --extra-index-url https://download.pytorch.org/whl/cu128 "$ICT_DL_PIN" \
      2>&1 | tee -a "$ICT_DL_LOG"
  done < "$ICT_DL_FILE"
}
download_locked "$ICT_SETUP_ROOT/scripts/humanego_ict/environment/torch-cu128-requirements.txt" \
  "$ICT_SETUP_OUTPUT/torch_download.log" 6000
timeout 1200 "$ICT_SETUP_PYTHON" -B -m pip install --no-index --find-links "$ICT_SETUP_WHEELS" \
  --report "$ICT_SETUP_OUTPUT/torch_install.json" \
  -r "$ICT_SETUP_ROOT/scripts/humanego_ict/environment/torch-cu128-requirements.txt" \
  2>&1 | tee "$ICT_SETUP_OUTPUT/torch.log"
echo '【3/4】安装数据读取与治理发布所需的锁定依赖'
download_locked "$ICT_SETUP_ROOT/scripts/humanego_ict/environment/runtime-requirements.txt" \
  "$ICT_SETUP_OUTPUT/runtime_download.log" 1200
timeout 1200 "$ICT_SETUP_PYTHON" -B -m pip install --no-index --find-links "$ICT_SETUP_WHEELS" \
  --report "$ICT_SETUP_OUTPUT/runtime_install.json" \
  -r "$ICT_SETUP_ROOT/scripts/humanego_ict/environment/runtime-requirements.txt" \
  2>&1 | tee "$ICT_SETUP_OUTPUT/runtime.log"
echo '【4/4】验证依赖一致性和实际导入'
"$ICT_SETUP_PYTHON" -B -m pip check | tee "$ICT_SETUP_OUTPUT/pip_check.log"
"$ICT_SETUP_PYTHON" -B -m pip freeze > "$ICT_SETUP_OUTPUT/pip_freeze.txt"
"$ICT_SETUP_CONDA" list --prefix "$ICT_SETUP_PREFIX" --explicit > "$ICT_SETUP_OUTPUT/conda_explicit.txt"
"$ICT_SETUP_PYTHON" -B "$ICT_SETUP_ROOT/scripts/humanego_ict/check_joint26_env.py" --json \
  > "$ICT_SETUP_OUTPUT/ENVIRONMENT_CHECK.json"
"$ICT_SETUP_BASE_PYTHON" -B - "$ICT_SETUP_PROFILE" "$ICT_SETUP_PREFIX/JOINT26_ENVIRONMENT.json" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
value['status'] = 'READY'
pathlib.Path(sys.argv[2]).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
PY
echo "环境安装及导入检查完成：$ICT_SETUP_PREFIX"
echo '训练 Bash 将自动使用此环境，无需 conda activate；目标服务器 CUDA / NCCL 仍需实际验证。'
echo "安装证据：$ICT_SETUP_OUTPUT"
