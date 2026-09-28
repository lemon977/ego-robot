from __future__ import annotations

import json
import sys
from datetime import datetime

import pytest

from chaoyang.governance import publish_human_to_robot_s2_checkpoint as module
from chaoyang.ops import validate_human_to_robot_s2_final as final_validation_module


class _Clock(datetime):
    current = datetime.fromisoformat("2026-09-23T17:45:07+08:00")

    @classmethod
    def now(cls, tz=None):
        value = cls.current
        return value if tz is None else value.astimezone(tz)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_final_validation_artifact_ref_accepts_non_authority_annotations(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"annotated-artifact")
    item = final_validation_module.artifact_ref(path)
    item["decoded_frames"] = 16
    assert final_validation_module._checked(item) == path.resolve()


def test_final_validation_artifact_ref_rejects_real_integrity_drift(tmp_path):
    path = tmp_path / "video.mp4"
    path.write_bytes(b"original")
    item = final_validation_module.artifact_ref(path)
    item["decoded_frames"] = 16
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="ARTIFACT_REF_DRIFT"):
        final_validation_module._checked(item)


@pytest.fixture
def checkpoint_repo(tmp_path, monkeypatch):
    _Clock.current = datetime.fromisoformat("2026-09-23T17:45:07+08:00")
    root = tmp_path / "attempt_0001"
    receipt = tmp_path / "CURRENT_STATUS_RECEIPT.json"
    state = tmp_path / "LONG_HORIZON_TASK_STATE.json"
    authority = tmp_path / "CURRENT_AUTHORITY_INDEX.json"
    plan = tmp_path / "docs/current/PLAN.md"
    work_entry = tmp_path / "docs/current/AI_WORK_ENTRY_ZH.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("before\n", encoding="utf-8")
    work_entry.write_text("before\n", encoding="utf-8")
    _write(receipt, {"governance_revision": 7})
    _write(authority, {})
    _write(
        state,
        {
            "next_task": {"task_id": module.TASK},
            "tasks": [
                {
                    "task_id": module.TASK,
                    "t0": "2026-09-23T10:45:07+08:00",
                    "updated_at": "2026-09-23T10:45:07+08:00",
                }
            ],
            "recent_events": [],
        },
    )
    _write(root / "checkpoints/H3_RESULT.json", {"checkpoint": "H3"})
    for lane in module.LANES:
        _write(
            root / f"lanes/{lane}/STATE.json",
            {
                "status": "RUNNING",
                "execution": "EXECUTED",
                "structure": "PASS",
                "quality": "REJECTED_QUALITY",
                "adoption": "NOT_ADOPTED",
                "checkpoint": "H3",
                "updated_at": "before",
            },
        )
    products = {}
    for index, session in enumerate(("a", "b", "c", "d")):
        path = root / f"product_{index}.json"
        _write(
            path,
            {
                "execution": "EXECUTED",
                "structure": "PASS",
                "decoded_frames": 1,
                "expected_frames": 1,
                "quality": "REJECTED_QUALITY",
                "adoption": "CANDIDATE_ONLY",
                "adapter_visible_frames": 0,
                "known_decision_coverage": 0.0,
                "unknown_decision_ratio": 1.0,
            },
        )
        products[session] = path
    _write(
        root / "formal_entry_resume/attempt_0002/RESULT.json",
        {"status": "PASS", "reused": 4, "mutated": 0},
    )
    _write(
        root / "h6_readiness/attempt_0001/RESULT.json",
        {"status": "READY_FOR_H6_FREEZE_NO_NEW_FULL_SESSION_ALGORITHM"},
    )
    _write(
        root / "evidence_ref_audit/attempt_0001/RESULT.json",
        {"status": "PASS", "error_count": 0},
    )
    for relative in (
        "src/chaoyang/pipeline/v5_product.py",
        "src/chaoyang/pipeline/occlusion_compositor_v1.py",
        "src/chaoyang/ops/run_human_to_robot_baseline_v1.py",
        "src/chaoyang/ops/run_human_to_robot_s2_local_huro_adapter_refresh.py",
        "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
        "shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "PRODUCTS", products)
    monkeypatch.setattr(module, "RECEIPT_PATH", receipt)
    monkeypatch.setattr(module, "TASK_STATE_PATH", state)
    monkeypatch.setattr(module, "AUTHORITY_PATH", authority)
    monkeypatch.setattr(module, "PLAN", plan)
    monkeypatch.setattr(module, "WORK_ENTRY", work_entry)
    monkeypatch.setattr(module, "datetime", _Clock)
    monkeypatch.setattr(
        module,
        "publish_bundle",
        lambda *args, **kwargs: {"governance_revision": kwargs["expected_revision"] + 1},
    )
    return root


def test_h6_freeze_uses_semantic_lane_snapshot_not_stale_artifact_ref(
    checkpoint_repo, monkeypatch
):
    monkeypatch.setattr(
        sys, "argv", ["checkpoint", "--checkpoint", "H6", "--expected-revision", "7"]
    )
    assert module.main() == 0
    result = json.loads(
        (checkpoint_repo / "checkpoints/H6_RESULT.json").read_text(encoding="utf-8")
    )
    assert result["decision"]["formal_entry_resume_status"] == "CLOSED_4_OF_4_NO_MUTATION"
    assert result["decision"]["evidence_reference_audit"]["bytes"] > 0
    assert any(
        "FAULT_PACKAGE_EXHAUSTED_2_OF_2" in reason
        for reason in result["decision"]["reasons"]
    )
    assert "CONTACT_R1_SCREENING_NOT_COMPLETED_NO_ADMISSIBLE_WINDOW_ESTABLISHED" in result["decision"]["reasons"]
    assert "007_clean_retry_without_frozen_top1_cable_instance" in result["decision"]["do_not_start"]
    assert "retrospective_occlusion_quality_threshold_from_H3_window" in result["decision"]["do_not_start"]
    successor = result["decision"]["single_successor_recommendation"]
    assert successor["status"] == "RECOMMENDED_AFTER_S2_NOT_EXECUTED_IN_S2"
    assert successor["scope"] == "play_cards_0915_031_frames_66_81_right_arm_only"
    assert "quality_contract" in successor["preconditions"][0]
    assert successor["fallback"].startswith("DO_NOT_AUTOMATICALLY_START")
    for snapshot in result["lane_states_before_checkpoint"].values():
        assert snapshot["checkpoint"] == "H3"
        assert not ({"path", "bytes", "sha256"} & snapshot.keys())
    for lane in module.LANES:
        state = json.loads(
            (checkpoint_repo / f"lanes/{lane}/STATE.json").read_text(encoding="utf-8")
        )
        assert state["checkpoint"] == "H6"
    assert "最新不可变检查点为 `H6`" in module.PLAN.read_text(encoding="utf-8")
    assert "S2 H6" in module.WORK_ENTRY.read_text(encoding="utf-8")


def test_h6_time_gate_fails_before_any_checkpoint_write(checkpoint_repo, monkeypatch):
    _Clock.current = datetime.fromisoformat("2026-09-23T11:45:07+08:00")
    monkeypatch.setattr(
        sys, "argv", ["checkpoint", "--checkpoint", "H6", "--expected-revision", "7"]
    )
    with pytest.raises(RuntimeError, match="H6_NOT_REACHED"):
        module.main()
    assert not (checkpoint_repo / "checkpoints/H6_RESULT.json").exists()


def test_h6_can_close_early_only_with_explicit_flag_and_full_readiness(
    checkpoint_repo, monkeypatch
):
    _Clock.current = datetime.fromisoformat("2026-09-23T11:45:07+08:00")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "checkpoint",
            "--checkpoint",
            "H6",
            "--expected-revision",
            "7",
            "--allow-early-close",
        ],
    )
    assert module.main() == 0
    result = json.loads(
        (checkpoint_repo / "checkpoints/H6_RESULT.json").read_text(encoding="utf-8")
    )
    assert result["early_close_authorized"] is True
    assert result["nominal_time_gate_seconds"] == 6 * 3600
    assert result["decision"]["formal_entry_resume_status"] == "CLOSED_4_OF_4_NO_MUTATION"


def test_h9_freezes_exact_component_bytes_and_updates_lane_checkpoint(
    checkpoint_repo, monkeypatch
):
    _Clock.current = datetime.fromisoformat("2026-09-23T20:45:07+08:00")
    _write(checkpoint_repo / "checkpoints/H6_RESULT.json", {"checkpoint": "H6"})
    monkeypatch.setattr(
        sys, "argv", ["checkpoint", "--checkpoint", "H9", "--expected-revision", "7"]
    )
    assert module.main() == 0
    result = json.loads(
        (checkpoint_repo / "checkpoints/H9_RESULT.json").read_text(encoding="utf-8")
    )
    assert result["freeze_decision"]["status"] == "FROZEN_FOR_S2_FINAL_VALIDATION"
    assert result["freeze_decision"]["formal_products"] == "4_STRUCTURAL_0_QUALITY_PASS"
    assert result["freeze_decision"]["partial_direction_ik"] == "NOT_RUN_IN_S2_DEFERRED_TO_VERSIONED_SUCCESSOR"
    assert result["freeze_decision"]["robot_r1"].startswith("0_SCREENED_0_EXECUTED_0_ADOPTED")
    assert result["freeze_decision"]["cable_top1_clean"] == "NOT_RUN_WITHOUT_FROZEN_INSTANCE_SUPPORT"
    assert result["freeze_decision"]["occlusion_temporal_quality"] == "INCONCLUSIVE_NO_RETROSPECTIVE_THRESHOLD"
    assert set(result["component_freeze"]) == {
        "renderer",
        "compositor",
        "formal_entry",
        "mount",
        "compare_refresh",
    }
    for item in result["component_freeze"].values():
        assert item["bytes"] > 0
        assert len(item["sha256"]) == 64
    for lane in module.LANES:
        state = json.loads(
            (checkpoint_repo / f"lanes/{lane}/STATE.json").read_text(encoding="utf-8")
        )
        assert state["checkpoint"] == "H9"
    assert "最新不可变检查点为 `H9`" in module.PLAN.read_text(encoding="utf-8")
    assert "S2 H9" in module.WORK_ENTRY.read_text(encoding="utf-8")
