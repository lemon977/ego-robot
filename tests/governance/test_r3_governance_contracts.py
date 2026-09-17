from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from chaoyang.governance import common
from chaoyang.governance.bootstrap_current_governance import build_authority, build_task_state
from chaoyang.governance.current_baseline_v2 import build_registry
from chaoyang.governance.current_r3_contracts import (
    build_algorithm_contract,
    build_doc_authority_map,
    validate_doc_authority_map,
)


def _copy_required_static_governance_docs(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    source = ROOT / "docs/governance"
    for name in (
        "ROBOT_QUALITY_GATE_POLICY_V71_ZH.md",
        "GOVERNANCE_CONSISTENCY_REPAIR_20260917_ZH.md",
    ):
        (root / name).write_bytes((source / name).read_bytes())


def _configure_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "governance"
    _copy_required_static_governance_docs(root)
    names = {
        "GOVERNANCE_ROOT": root,
        "AUTHORITY_PATH": root / "CURRENT_AUTHORITY_INDEX.json",
        "TASK_STATE_PATH": root / "LONG_HORIZON_TASK_STATE.json",
        "STATUS_PATH": root / "CURRENT_PROJECT_STATUS_ZH.md",
        "RECEIPT_PATH": root / "CURRENT_STATUS_RECEIPT.json",
        "MIN_STATUS_PATH": root / "CURRENT_PROJECT_STATUS_MIN.json",
        "RC1_STATUS_MIN_PATH": root / "CURRENT_RC1_STATUS_MIN.json",
        "TASK_QUEUE_PATH": root / "TASK_QUEUE.json",
        "CHANGELOG_PATH": root / "STATE_CHANGELOG.jsonl",
        "LOCK_PATH": root / ".governance.lock",
        "BASELINE_REGISTRY_PATH": root / "CURRENT_BASELINE_REGISTRY_V2.json",
        "STAGE_BASELINES_PATH": root / "CURRENT_STAGE_BASELINES_ZH.md",
        "FILE_LAYOUT_PATH": root / "CURRENT_FILE_LAYOUT.json",
        "REGRESSION_MANIFEST_PATH": root / "CURRENT_REGRESSION_MANIFEST.json",
        "PLAN_REVISION_PATH": root / "PLAN_REVISION.json",
        "V71_TASK_PACKET_INDEX_PATH": root / "CURRENT_V71_TASK_PACKET_INDEX.json",
        "RC1_PLAN_PATH": root / "ABSENT_RC1_PLAN.md",
        "RC1_RELEASE_SPEC_PATH": root / "ABSENT_RC1_RELEASE_SPEC.json",
        "RC1_CONTRACT_PATH": root / "ABSENT_RC1_CONTRACT.json",
    }
    for name, value in names.items():
        monkeypatch.setattr(common, name, value)
    return root


def test_algorithm_contract_classifies_pose_similarity_as_soft() -> None:
    authority = build_authority()
    state = build_task_state()
    contract = build_algorithm_contract(build_registry(authority, state), authority)
    common.validate_schema("algorithm_contract.schema.json", contract)
    robot = next(item for item in contract["stages"] if item["stage"] == "Robot Visual")
    assert "strict_human_pose_similarity" in robot["quality_gates"]["SOFT_DIAGNOSTIC"]
    assert "strict_human_pose_similarity" not in robot["quality_gates"]["HARD_STRUCTURAL"]
    assert contract["special_status_contracts"]["mask_baseline"]["current"] == "SAM3.1"


def test_formal_h4_and_development_canary_are_separate() -> None:
    authority = build_authority()
    contract = build_algorithm_contract(build_registry(authority, build_task_state()), authority)
    formal = contract["special_status_contracts"]["sensor_h4_formal"]
    development = contract["special_status_contracts"]["sensor_h4_development_canary"]
    assert formal == {
        "execution_status": "BLOCKED_RESOURCE",
        "qa_status": "NOT_EVALUATED",
        "policy_status": "POLICY_DEFERRED",
        "pixel_mask_authority": False,
        "claim_limit": "The formal R7_0 preflight ran no SAM3.1 pixel inference.",
    }
    assert development["execution_status"] == "FAILED_QUALITY_C"
    assert development["may_override_formal_h4"] is False
    decision = contract["special_status_contracts"]["sensor_h4_bounded_policy_decision"]
    assert decision["terminal_status"] == "FAILED_QUALITY_C"
    assert decision["formal_h4_state_changed"] is False
    assert decision["pixel_mask_authority"] is False
    assert not common.validate_artifact_ref(decision["receipt"])


def test_real_depth_receipts_remain_local_development_blockers() -> None:
    authority = build_authority()
    contract = build_algorithm_contract(build_registry(authority, build_task_state()), authority)
    for key in ("depth10_real_development", "depth20_real_input_preflight"):
        value = contract["special_status_contracts"][key]
        assert value["terminal_status"] == "BLOCKED_PREREQ"
        assert value["authority_promoted"] is False
        assert not common.validate_artifact_ref(value["receipt"])


def test_development_watchers_cannot_register_authority() -> None:
    authority = build_authority()
    contract = build_algorithm_contract(build_registry(authority, build_task_state()), authority)
    watchers = contract["special_status_contracts"]["development_watchers"]
    assert watchers["robot_hard_soft_v75"]["may_register_authority"] is False
    assert watchers["occlusion_visible_surface_v75"]["may_register_authority"] is False


def test_visual_aux_and_robot_r3_contract_refs_are_closed() -> None:
    authority = build_authority()
    contract = build_algorithm_contract(build_registry(authority, build_task_state()), authority)
    special = contract["special_status_contracts"]
    for key in ("visual_aux_final_hardening", "robot_v75_output_contracts"):
        for field in ("code_closure", "test_closure", "schema_closure"):
            for reference in special[key].get(field, []):
                assert not common.validate_artifact_ref(reference)
    audit = special["r3_execution_status_gap_audit"]
    assert audit["counts"] == {"PASSED": 2, "BLOCKED": 7, "DEVELOPMENT": 6, "MISSING": 4}
    assert audit["authority_promoted"] is False
    for field in ("result", "execution_status_audit", "receipt"):
        assert not common.validate_artifact_ref(audit[field])


def test_doc_authority_map_rejects_duplicate_current_scope(tmp_path: Path) -> None:
    authority = build_authority()
    governance = tmp_path / "governance"
    _copy_required_static_governance_docs(governance)
    for name in ("CURRENT_PROJECT_STATUS_ZH.md", "CURRENT_STAGE_BASELINES_ZH.md"):
        (governance / name).write_text(name, encoding="utf-8")
    value = build_doc_authority_map(authority, governance, ROOT)
    common.validate_schema("doc_authority_map.schema.json", value)
    assert not validate_doc_authority_map(value)
    duplicate = copy.deepcopy(value["documents"][0])
    duplicate["document_id"] = "duplicate"
    value["documents"].append(duplicate)
    assert any("multiple CURRENT" in error for error in validate_doc_authority_map(value))


def test_doc_authority_map_contains_only_active_documents() -> None:
    authority = build_authority()
    value = build_doc_authority_map(authority, ROOT / "docs/governance", ROOT)
    documents = {item["document_id"]: item for item in value["documents"]}
    assert documents["repository_readme"]["status"] == "CURRENT"
    assert documents["agent_execution_protocol"]["status"] == "CURRENT"
    assert documents["rc1_final_delivery_plan"]["status"] == "CURRENT"
    assert all(item["status"] == "CURRENT" for item in value["documents"])
    assert all("/archive/" not in item["path"] for item in value["documents"])
    assert not validate_doc_authority_map(value)


def test_receipt_binds_r3_generated_contracts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _configure_tmp(monkeypatch, tmp_path)
    receipt = common.publish_bundle(
        build_authority(),
        build_task_state(),
        event_type="TEST_R3",
        expected_revision=0,
        generator_path=Path(__file__),
    )
    assert receipt["files"]["doc_authority_map"]["path"] == str(root / "DOC_AUTHORITY_MAP.json")
    assert receipt["files"]["algorithm_contract"]["path"] == str(root / "ALGORITHM_CONTRACT.json")
    for key in ("doc_authority_map", "algorithm_contract"):
        assert not common.validate_artifact_ref(receipt["files"][key])


def test_validator_absolute_script_has_no_cwd_import_dependency(tmp_path: Path) -> None:
    script = ROOT / "src/chaoyang/governance/validate_governance_state.py"
    process = subprocess.run(
        [sys.executable, str(script), "--allow-stale"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert "ModuleNotFoundError" not in process.stderr
    # Before or after a concurrent current publication, the validator may
    # return PASS or STATUS_CONFLICT, but it must emit structured JSON.
    payload = json.loads(process.stdout)
    assert payload["status"] in {"PASS", "STATUS_CONFLICT"}
