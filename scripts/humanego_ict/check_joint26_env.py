#!/usr/bin/env python3
"""Check the selected interpreter before importing project governance."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
from importlib import metadata, util
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODULES = {'torch': 'torch', 'numpy': 'numpy', 'PIL': 'Pillow',
                 'jsonschema': 'jsonschema', 'cv2': 'opencv-python-headless',
                 'scipy': 'scipy', 'trimesh': 'trimesh', 'numba': 'numba', 'yaml': 'PyYAML'}
PREPARE_MODULES = {key: value for key, value in TRAIN_MODULES.items() if key != 'torch'}
PREPARE_MODULES.update(mujoco='mujoco', mink='mink', qpsolvers='qpsolvers', daqp='daqp', yaml='PyYAML')
CACHE_KEYS = ('PIP_CACHE_DIR', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR', 'TORCH_HOME',
              'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
              'CUDA_CACHE_PATH', 'HF_HOME', 'HUGGINGFACE_HUB_CACHE', 'MPLCONFIGDIR')


def project_scratch(root=ROOT):
    """Every potentially importing/compiling package gets an owned cache."""
    root = Path(root).resolve()
    configured = os.environ.get('ICT_ENV_CHECK_SCRATCH')
    base = Path(configured or os.environ.get('TMPDIR', str(root / '_run/environment/humanego_ict/checks')))
    if not base.is_absolute() or not base.resolve().is_relative_to(root) or base.resolve() == root:
        if configured:
            raise ValueError('环境检查缓存必须是项目内的明确子目录。')
        base = root / '_run/environment/humanego_ict/checks'
    for ancestor in (base, *base.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError('环境检查缓存不能经过符号链接。')
    base.mkdir(parents=True, exist_ok=True)
    scratch = base if configured else Path(tempfile.mkdtemp(prefix='check_', dir=base))
    cache_paths = {key: scratch / 'cache' / key.lower() for key in CACHE_KEYS}
    for path in cache_paths.values():
        if not path.resolve().is_relative_to(root):
            raise ValueError('环境检查缓存不能解析到项目外。')
        for ancestor in (path, *path.parents):
            if ancestor == root:
                break
            if ancestor.is_symlink():
                raise ValueError('环境检查缓存不能经过符号链接。')
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                      PIP_DISABLE_PIP_VERSION_CHECK='1', TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    for key, path in cache_paths.items():
        os.environ[key] = str(path)
    return scratch


def check_environment(*, prepare=False, cuda=False, min_gpus=1, startup=False):
    required = PREPARE_MODULES if prepare else TRAIN_MODULES
    result = {'schema': 'humanego-joint26-environment-check-v1', 'status': 'FAILED_ENVIRONMENT',
              'python': sys.version.split()[0], 'executable': sys.executable, 'prefix': sys.prefix,
              'mode': 'prepare' if prepare else 'train', 'versions': {}, 'missing': [],
              'import_errors': {}, 'errors': [], 'cuda_checked': False, 'model_calls': 0}
    if sys.version_info < (3, 11):
        result['errors'].append('需要 Python 3.11 或更新版本；专用环境固定为 Python 3.11.11。')
    for module, distribution in required.items():
        try:
            available = util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            available = False
        if not available:
            result['missing'].append(distribution)
            continue
        try:
            result['versions'][module] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result['versions'][module] = '已找到，版本待导入确认'
    # Report all absent packages before attempting imports or project modules.
    if result['missing']:
        result['errors'].append('缺少依赖：' + '、'.join(result['missing']))
    if result['errors']:
        return result
    try:
        result['scratch'] = str(project_scratch())
    except (OSError, ValueError) as error:
        result['errors'].append(str(error))
        return result
    loaded = {}
    import_output = io.StringIO()
    for module in required:
        try:
            with redirect_stdout(import_output), redirect_stderr(import_output):
                loaded[module] = importlib.import_module(module)
            # jsonschema's version attribute emits a deprecation warning;
            # its distribution metadata already provides the exact version.
            actual = None if module == 'jsonschema' else getattr(loaded[module], '__version__', None)
            if isinstance(actual, str):
                result['versions'][module] = actual
        except Exception as error:
            result['import_errors'][module] = type(error).__name__ + ': ' + str(error)
    if import_output.getvalue():
        result['import_output'] = import_output.getvalue()[-4000:]
    if result['import_errors']:
        result['errors'].append('依赖导入失败：' + '、'.join(result['import_errors']))
        return result
    if not startup:
        try:
            checked = subprocess.run([sys.executable, '-B', '-m', 'pip', 'check'],
                                     capture_output=True, text=True, timeout=120)
            result['pip_check'] = {'exit_code': checked.returncode,
                                   'stdout': checked.stdout.strip(), 'stderr': checked.stderr.strip()}
            if checked.returncode:
                result['errors'].append('pip check 发现依赖冲突，请查看检查详情。')
        except (OSError, subprocess.TimeoutExpired) as error:
            result['errors'].append('pip check 未完成：' + str(error))
    if cuda:
        torch = loaded.get('torch')
        if torch is None:
            result['errors'].append('CUDA 检查仅适用于 GPU 训练环境，不适用于 --prepare。')
        else:
            try:
                result['cuda_checked'] = True
                result['cuda'] = {'available': bool(torch.cuda.is_available()),
                                  'gpu_count': int(torch.cuda.device_count()),
                                  'nccl_available': bool(torch.distributed.is_nccl_available()),
                                  'build': torch.version.cuda}
                if not result['cuda']['available'] or result['cuda']['gpu_count'] < min_gpus:
                    result['errors'].append('CUDA 不可用或可见 GPU 数少于 ' + str(min_gpus) + '。')
                if not result['cuda']['nccl_available']:
                    result['errors'].append('此 PyTorch 环境缺少 NCCL 支持。')
            except Exception as error:
                result['errors'].append('CUDA 检查失败：' + str(error))
    result['status'] = 'PASSED' if not result['errors'] else 'FAILED_ENVIRONMENT'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='输出一个完整检查 JSON')
    parser.add_argument('--prepare', action='store_true', help='检查 CPU 标签准备依赖，不要求 Torch/CUDA')
    parser.add_argument('--cuda', action='store_true', help='额外只读检查 CUDA/NCCL，不创建张量或模型')
    parser.add_argument('--min-gpus', type=int, default=1)
    parser.add_argument('--startup-check', action='store_true', help='启动前检查导入；独立检查还会执行 pip check')
    parser.add_argument('--quiet-success', action='store_true', help='成功不输出，用于保持训练 JSON 接口')
    args = parser.parse_args(argv)
    if args.min_gpus < 1:
        parser.error('--min-gpus 必须是正整数')
    result = check_environment(prepare=args.prepare, cuda=args.cuda,
                               min_gpus=args.min_gpus, startup=args.startup_check)
    if args.json:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    elif result['status'] != 'PASSED':
        print('环境检查未通过：' + '；'.join(result['errors']), file=sys.stderr)
        print('解释器：' + result['executable'] + '；Python ' + result['python'], file=sys.stderr)
        if result['versions']:
            print('已找到版本：' + '，'.join(key + '=' + value for key, value in result['versions'].items()), file=sys.stderr)
        for module, error in result['import_errors'].items():
            print(module + '：' + error, file=sys.stderr)
        if result.get('pip_check'):
            print(result['pip_check']['stdout'] or result['pip_check']['stderr'], file=sys.stderr)
        if args.prepare:
            print('标签准备请用 ICT_PYTHON 绑定 CPU 数学环境，并查看 ENV_SETUP_ZH.md；GPU 安装入口不安装数学／几何依赖。', file=sys.stderr)
        else:
            print('请执行：bash "' + str(ROOT / 'scripts/humanego_ict/setup_joint26_env.sh') + '"', file=sys.stderr)
    elif not args.quiet_success:
        print('环境检查通过：Python ' + result['python'] + '；解释器 ' + result['executable'])
        print('版本：' + '，'.join(key + '=' + value for key, value in result['versions'].items()))
    return 0 if result['status'] == 'PASSED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
