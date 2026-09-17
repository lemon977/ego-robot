from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_launcher_requires_lease_and_preserves_leased_device() -> None:
    source = (ROOT / "src/chaoyang/ops/foundationstereo_gpu_python.sh").read_text()
    assert 'lease="$project_root/_run/current/GPU_LEASE.json"' in source
    assert '${CUDA_VISIBLE_DEVICES:-}' in source
    assert 'export CUDA_VISIBLE_DEVICES=""' not in source
    assert 'PYTHONPATH="$project_root/src"' in source
