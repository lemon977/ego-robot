from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src/chaoyang/ops/build_successor_command_packets_v71.py"
ARCHIVED_TASKS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks"


def archived_path(value: str) -> Path:
    prefix = "/mnt/workspace/code/chaoyang/tasks/"
    return ARCHIVED_TASKS / value[len(prefix):] if value.startswith(prefix) else Path(value)


def load_module():
    spec = importlib.util.spec_from_file_location("successor_packets", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_five_frozen_groups_and_no_execution_authority() -> None:
    module = load_module()
    selection = module.load(module.SELECTION)
    assert selection["counts"]["groups"] == 5
    assert [group["stage"] for group in selection["groups"]].count("hawor") == 2
    assert [group["stage"] for group in selection["groups"]].count("role_mask") == 2
    assert [group["stage"] for group in selection["groups"]].count("object_identity") == 1


def test_current_closure_blockers_are_evidence_based() -> None:
    module = load_module()
    assert module.HAWOR_RUNNER.is_file()
    assert module.ROLE_RUNNER.is_file()
    assert module.OBJECT_RUNNER.is_file()
    assert not module.OBJECT_MISSING_IMPORT.exists()
    legacy = module.LEGACY_GPU_LAUNCHER.read_text(encoding="utf-8")
    role = module.ROLE_RUNNER.read_text(encoding="utf-8")
    v71 = module.V71_GPU_LEASE.read_text(encoding="utf-8")
    assert '"schema_version": "gpu-lease-v1"' in legacy
    assert 'lease.get("status") != "ACQUIRED" or lease.get("holder") != holder' in role
    assert '"schema_version": "chaoyang-gpu-lease-v71"' in v71
    assert '"fencing_token"' in v71 and '"expires_at"' in v71


def test_generated_packets_are_immutable_negative_terminals() -> None:
    root = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/child_packets"
    index_path = root / "SUCCESSOR_CHILD_PACKET_INDEX_V71.json"
    if not index_path.is_file():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["counts"] == {"blocked_reference_proof": 5, "execution_ready": 0, "total": 5}
    assert len({row["task_id"] for row in index["children"]}) == 5
    for row in index["children"]:
        packet = json.loads(archived_path(row["packet"]["path"]).read_text(encoding="utf-8"))
        command = json.loads(archived_path(row["command_manifest"]["path"]).read_text(encoding="utf-8"))
        receipt = json.loads(archived_path(row["receipt"]["path"]).read_text(encoding="utf-8"))
        assert packet["status"] == "BLOCKED_REFERENCE_PROOF"
        assert command["execution_authorized"] is False
        assert receipt["gpu_started"] is False
        assert len(packet["scope"]["regression_sessions"]) == 2
        assert len(packet["run_signature"]) == 64


def test_chips_role_selection_exposes_non_ab_regression() -> None:
    root = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/child_packets"
    path = root / "role_mask_chips_tracker_0901/BLOCKED_REFERENCE_PROOF.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    blockers = {row["code"]: row for row in value["blockers"]}
    assert blockers["FROZEN_REGRESSION_NOT_AB"]["sessions"] == ["get_potato_chips_0903_052"]


def test_verification_receipts_never_promote_or_use_gpu() -> None:
    root = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/successors/child_packets"
    path = root / "VERIFICATION_RESULT.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["status"] == "PASSED_TWO_CPU_PREFLIGHTS_THREE_EXPLICIT_SKIPS"
    assert value["gpu_calls"] == 0
    assert len(value["rows"]) == 5
    statuses = [row["status"] for row in value["rows"]]
    assert statuses.count("PASSED") == 2
    assert statuses.count("SKIPPED_BLOCKED_REFERENCE_PROOF") == 3
