from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from chaoyang.governance import register_0915_campaign_task_v1 as subject


def _state(predecessor: str, status: str = "PASSED") -> dict:
    return {
        "next_task": None,
        "tasks": [{
            "task_id": predecessor,
            "status": status,
            "result": {"path": "/nonexistent", "bytes": 1, "sha256": "0" * 64},
        }],
    }


def test_later_stage_requires_immediate_predecessor_pass() -> None:
    state = _state("0915_foundationstereo_full_v1")
    predecessor = subject._validate_predecessor(state, "0915_post_geometry_robot_v1")
    assert predecessor["task_id"] == "0915_foundationstereo_full_v1"
    bad = copy.deepcopy(state)
    bad["tasks"][0]["status"] = "FAILED_RUNTIME_FINAL"
    with pytest.raises(RuntimeError, match="did not pass"):
        subject._validate_predecessor(bad, "0915_post_geometry_robot_v1")


def test_registration_rejects_any_live_task() -> None:
    state = _state("0915_hawor_full_v1")
    state["tasks"].append({"task_id": "other", "status": "RUNNING"})
    with pytest.raises(RuntimeError, match="no active"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")


def test_registration_rejects_duplicate_task() -> None:
    state = _state("0915_hawor_full_v1")
    state["tasks"].append({"task_id": "0915_sam31_mask_full_v1", "status": "PASSED"})
    with pytest.raises(RuntimeError, match="already registered"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")


def test_corrective_v2_accepts_only_exact_pre_execution_cli_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps({
        "returncodes": {"audit": 1},
        "self_containment": None,
        "prepared_manifest": None,
    }))
    (tmp_path / "AUDIT.log").write_text(
        "operation is not maintained by the current algorithm contract: "
        "audit_0915_processed_self_containment_v2\n"
    )
    state = _state("0915_input_prepare_cad_v1", "FAILED_RUNTIME_FINAL")
    state["tasks"][0]["result"]["path"] = str(result_path)
    monkeypatch.setattr(subject, "validate_artifact_ref", lambda _value: [])
    predecessor = subject._validate_predecessor(
        state, "0915_input_prepare_cad_v2",
    )
    assert predecessor["task_id"] == "0915_input_prepare_cad_v1"

    (tmp_path / "AUDIT.log").write_text("different failure\n")
    with pytest.raises(RuntimeError, match="exact pre-execution"):
        subject._validate_predecessor(state, "0915_input_prepare_cad_v2")


def test_vst_research_requires_exact_cancelled_image_domain_hold(
    tmp_path: Path, monkeypatch,
) -> None:
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps({
        "task_id": "0915_hawor_full_v1",
        "status": "CANCELLED",
        "first_blocker": "VST_IMAGE_DOMAIN_UNRESOLVED_POSSIBLE_REDUNDANT_UNDISTORTION",
    }))
    state = _state("0915_hawor_full_v1", "CANCELLED")
    state["tasks"][0]["result"]["path"] = str(result_path)
    monkeypatch.setattr(subject, "validate_artifact_ref", lambda _value: [])
    predecessor = subject._validate_predecessor(
        state, "0915_vst_image_domain_ab_v1",
    )
    assert predecessor["task_id"] == "0915_hawor_full_v1"

    bad = copy.deepcopy(state)
    bad["tasks"][0]["status"] = "PASSED"
    with pytest.raises(RuntimeError, match="requires the HaWoR task to be cancelled"):
        subject._validate_predecessor(bad, "0915_vst_image_domain_ab_v1")


def test_sam_registration_is_fail_closed_pending_canary_review() -> None:
    state = _state("0915_sam31_strict_role_canary_v5")
    with pytest.raises(RuntimeError, match="separate user review"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")
