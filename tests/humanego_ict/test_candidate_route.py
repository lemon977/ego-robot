"""CPU-only operation discovery and no-model-import admission checks."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from chaoyang.humanego_ict.contracts import candidate_contract
from chaoyang.humanego_ict.runtime_gate import OPERATION, ROUTE, verify_ref


def test_candidate_contract_and_cli_discovery(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    contract = candidate_contract(root)
    route = contract[ROUTE]
    assert route["physical_deployment_authorized"] is False
    assert route["current_runnable_operation"] is None
    assert route["training_steps_authorized_without_new_packet"] == 0
    for ref in route["code_closure"]:
        verify_ref(ref)
    (tmp_path / "docs/governance").mkdir(parents=True)
    (tmp_path / "docs/governance/ALGORITHM_CONTRACT.json").write_text(json.dumps({"special_status_contracts": contract}))
    import chaoyang.cli
    monkeypatch.setattr(chaoyang.cli, "get_paths", lambda: SimpleNamespace(repo_root=tmp_path))
    assert OPERATION in chaoyang.cli._maintained_operations()


def test_operation_help_and_rejected_input_do_not_import_torch():
    script = """
import importlib.abc, sys
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname.endswith('.FlowMatchingModel'):
            raise AssertionError('model imported before qualification')
sys.meta_path.insert(0, NoModels())
from chaoyang.ops.train_pick_place_humanego_ict_revo2_candidate_v1 import main
raise SystemExit(main(sys.argv[1:]))
"""
    help_result = subprocess.run([sys.executable, "-B", "-c", script, "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0, help_result.stderr
    args = ["--task-packet", "MISSING.json", "--manifest", "MISSING.json", "--branch", "raw",
            "--config", "MISSING.json", "--output-dir", "MISSING", "--pretrained", "MISSING.pt"]
    rejected = subprocess.run([sys.executable, "-B", "-c", script, *args], capture_output=True, text=True)
    assert rejected.returncode == 2, rejected.stderr
    assert "model imported" not in rejected.stderr
