"""Small CPU-only admission tests, never loading a policy or checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from chaoyang.humanego_ict.runtime_gate import (
    authorize_training, bind_cuda_visible_devices, file_ref, project_output,
    runtime_cache_paths, storage_ready, verify_ref,
)


def test_gpu_mapping_matches_allocated_ids_before_import(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert bind_cuda_visible_devices({"allocated_gpu_ids": [2, 5]}) == "2,5"
    with pytest.raises(ValueError, match="DIFFERS_FROM_GPU_LEASE"):
        bind_cuda_visible_devices({"allocated_gpu_ids": [0, 1]})


def test_runtime_cache_cannot_escape_task_output(tmp_path):
    output = tmp_path / "attempt"
    output.mkdir()
    elsewhere = tmp_path / "other_task"
    elsewhere.mkdir()
    (output / "runtime_cache").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="CACHE_ESCAPES"):
        runtime_cache_paths(output, tmp_path)


def test_project_output_refuses_root_escape_and_symlink(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    for value in (root, outside, root / "alias" / "output"):
        with pytest.raises(ValueError, match="PROJECT_SUBDIRECTORY"):
            project_output(value, root)
    assert project_output(root / "_run" / "new", root) == root / "_run" / "new"


def test_exact_artifact_ref_rejects_drift(tmp_path):
    p = tmp_path / "input.json"
    p.write_text("{}")
    ref = file_ref(p)
    assert verify_ref(ref) == p
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="SHA_DRIFT"):
        verify_ref(ref)


def test_storage_refuses_no_space(tmp_path, monkeypatch):
    class Empty:
        f_bavail = 0
        f_frsize = 4096
    monkeypatch.setattr("chaoyang.humanego_ict.runtime_gate.os.statvfs", lambda p: Empty())
    with pytest.raises(ValueError, match="STORAGE_CAPACITY"):
        storage_ready(tmp_path / "output", 1)


def test_missing_operation_cannot_use_development_task(tmp_path):
    root = tmp_path / "repo"
    governance = root / "docs/governance"
    governance.mkdir(parents=True)
    packet = root / "tasks/current/dev/TASK_PACKET.json"
    packet.parent.mkdir(parents=True)
    packet.write_text('{"execution_mode":"CPU_DEVELOPMENT_ONLY"}')
    contract = governance / "ALGORITHM_CONTRACT.json"
    contract.write_text('{"special_status_contracts":{}}')
    (governance / "CURRENT_STATUS_RECEIPT.json").write_text(json.dumps({"files": {"algorithm_contract": file_ref(contract)}}))
    with pytest.raises(ValueError, match="OPERATION_NOT_REGISTERED"):
        authorize_training(root, packet, [], manifest=packet, config=packet,
                           pretrained=packet, output=root / "_run/out", updates=20000, world_size=8)


@pytest.mark.parametrize("arguments", [["--help"], ["preflight", "--manifest", "MISSING_TEST_MANIFEST.json"]])
def test_help_and_rejected_preflight_do_not_import_torch(arguments):
    script = """
import importlib.abc, sys
class DenyModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname.endswith('.FlowMatchingModel'):
            raise AssertionError('model import before admission: ' + fullname)
sys.meta_path.insert(0, DenyModelImports())
from chaoyang.humanego_ict.cli import main
code = main(sys.argv[1:])
assert 'torch' not in sys.modules
raise SystemExit(code)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, *arguments],
                            capture_output=True, text=True, check=False)
    assert result.returncode == (0 if arguments == ["--help"] else 2), result.stderr
    assert "model import before admission" not in result.stderr


def test_training_config_fixed_amount_for_one_and_eight_gpus():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "docs/plans/PICK_PLACE_HUMANEGO_ICT_REVO2_V1/train_20000.json").read_text())
    from chaoyang.humanego_ict.training import TrainingConfig
    config = TrainingConfig(**cfg["training"])
    for world, accumulation in ((1, 8), (8, 1)):
        config.validate(world)
        assert config.global_batch // (world * config.micro_batch) == accumulation
    assert config.global_batch * config.max_updates == 1_280_000
