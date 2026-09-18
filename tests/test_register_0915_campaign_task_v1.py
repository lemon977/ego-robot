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


def test_registration_requires_materialized_read_closure_and_weight(tmp_path: Path) -> None:
    (tmp_path / "runner.py").write_text("# ready\n", encoding="utf-8")
    packet = {
        "read_set": ["runner.py", "missing.schema.json"],
        "weights": ["model.pt"],
    }
    errors = subject._validate_materialized_read_closure(packet, repo_root=tmp_path)
    assert errors == [
        "read_set path is not materialized: missing.schema.json",
        "weight is not materialized: model.pt",
    ]
    (tmp_path / "missing.schema.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.pt").write_bytes(b"pinned")
    assert subject._validate_materialized_read_closure(
        packet, repo_root=tmp_path,
    ) == []


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


def test_sam_full_batch_registration_is_fail_closed_after_bounded_canaries() -> None:
    state = _state("0915_planar_object6d_single_session_canary_v1")
    with pytest.raises(RuntimeError, match="separate batch authorization"):
        subject._validate_predecessor(state, "0915_sam31_mask_full_v1")


def test_removal_envelope_requires_visual_rejection_and_exact_authorization(
    tmp_path: Path, monkeypatch,
) -> None:
    weak_path = tmp_path / "weak.json"
    weak_path.write_text(json.dumps({
        "session_id": "play_cards_0915_001",
        "session_admission": "AWAITING_USER_VISUAL_REVIEW",
    }))
    state = _state("0915_sam31_weak_role_canary_v1")
    state["tasks"][0]["result"]["path"] = str(weak_path)
    monkeypatch.setattr(subject, "validate_artifact_ref", lambda _value: [])
    real_load = subject.load_json

    def fake_load(path: Path) -> dict:
        if Path(path) == weak_path:
            return json.loads(weak_path.read_text())
        if str(path).endswith("0915_SAM31_WEAK_ROLE_CANARY_V1_USER_VISUAL_REVIEW.json"):
            return {"status": "REJECTED_QUALITY_AS_CLEAN_BASELINE"}
        if str(path).endswith("0915_REMOVAL_ENVELOPE_V1_USER_AUTHORIZATION.json"):
            return {
                "status": "CONFIRMED",
                "authorized_task": "0915_removal_envelope_single_session_canary_v1",
                "authorized_session": "play_cards_0915_001",
                "weights": "ABSENT",
            }
        return real_load(path)

    monkeypatch.setattr(subject, "load_json", fake_load)
    predecessor = subject._validate_predecessor(
        state, "0915_removal_envelope_single_session_canary_v1",
    )
    assert predecessor["task_id"] == "0915_sam31_weak_role_canary_v1"
