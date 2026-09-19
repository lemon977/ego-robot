from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from chaoyang.ops.run_0915_robot15h_release_control_v1 import (
    _batch_reference,
    TASK_IDS,
    audit_campaign_references,
    audit_references,
    build_campaign_counts,
    build_capability_matrix,
    deadline_audit,
    ref,
    select_authoritative_evidence,
    validate_final_sidecars,
)


TZ = timezone(timedelta(hours=8))


def _artifact(path: Path, data: bytes = b"evidence") -> dict[str, object]:
    path.write_bytes(data)
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _evidence(
    task_id: str,
    terminal: str,
    sessions: list[dict[str, object]],
    tmp_path: Path,
) -> dict[str, object]:
    result_path = tmp_path / f"{task_id}.json"
    result_path.write_text(json.dumps({"task_id": task_id, "status": terminal}), encoding="utf-8")
    return {
        "task_id": task_id,
        "terminal_status": terminal,
        "result_ref": ref(result_path),
        "result": {"task_id": task_id, "status": terminal},
        "batch": {"sessions": sessions},
        "batch_ref": None,
    }


def _clock() -> dict[str, str]:
    return {
        "started_at": "2026-09-19T00:09:59+08:00",
        "freeze_algorithm_candidates_at": "2026-09-19T09:09:59+08:00",
        "start_new_sessions_deadline_at": "2026-09-19T13:39:59+08:00",
        "gpu_drain_deadline_at": "2026-09-19T14:09:59+08:00",
        "deadline_at": "2026-09-19T15:09:59+08:00",
    }


def test_matrix_separates_execution_export_and_quality(tmp_path: Path) -> None:
    evidence = [
        _evidence(
            "0915_robot15h_kai22_r0_wave0_v1",
            "PASSED",
            [{
                "session_id": "play_cards_0915_031",
                "task": "playing_cards",
                "status": "COMPLETED_DEVELOPMENT_BASELINE",
                "r0_exported": True,
                "r0_quality_admitted": False,
            }],
            tmp_path,
        ),
        _evidence(
            "0915_robot15h_foundationstereo_wave0_recovery_v1",
            "PASSED",
            [{
                "session_id": "get_potato_chips_0915_007",
                "task": "potato_chips",
                "status": "PASSED",
                "result": _artifact(tmp_path / "depth.json"),
            }],
            tmp_path,
        ),
    ]
    matrix = build_capability_matrix(evidence)
    rows = {(row["capability_id"], row["task"]): row for row in matrix["rows"]}
    r0 = rows[("kai22_r0", "playing_cards")]
    assert r0["executed"] is True
    assert r0["exported"] is True
    assert r0["quality_admitted"] is False
    assert r0["decision"] == "EXECUTED_NOT_QUALITY_ADMITTED"
    assert r0["w1_expansion_allowed"] is False
    depth = rows[("foundationstereo", "potato_chips")]
    assert depth["quality_admitted"] is True
    assert depth["w1_expansion_allowed"] is True
    assert depth["scope"] == "DEVELOPMENT_DEPTH_VISUAL_OBJECT6D_INPUT_ONLY"


def test_task_object_expansion_is_task_conditional(tmp_path: Path) -> None:
    evidence = [_evidence(
        "0915_robot15h_sam31_task_object_wave0_recovery_v1",
        "PASSED",
        [
            {
                "session_id": "play_cards_0915_031", "task": "playing_cards",
                "status": "PASSED_VISIBLE_OBJECT_MASK_PROXY", "execution_completed": True,
                "object_mask_consumer_allowed": True,
                "instance_counts": {"raw_selected": 3, "stable_consumer_allowed": 3},
            },
            {
                "session_id": "get_potato_chips_0915_007", "task": "potato_chips",
                "status": "REJECTED_QUALITY", "execution_completed": True,
                "object_mask_consumer_allowed": False,
                "instance_counts": {"raw_selected": 0, "stable_consumer_allowed": 0},
            },
        ],
        tmp_path,
    )]
    rows = {row["task"]: row for row in build_capability_matrix(evidence)["rows"]}
    assert rows["playing_cards"]["w1_expansion_allowed"] is True
    assert rows["potato_chips"]["executed"] is True
    assert rows["potato_chips"]["exported"] is False
    assert rows["potato_chips"]["w1_expansion_allowed"] is False


def test_structural_pass_does_not_promote_unimplemented_algorithm(tmp_path: Path) -> None:
    evidence = [_evidence(
        "0915_robot15h_geometry_object6d_wave0_v1",
        "PASSED",
        [{
            "session_id": "play_cards_0915_031", "task": "playing_cards",
            "status": "ELIGIBLE_FOR_SEPARATE_OBJECT6D_EXECUTION",
            "algorithm_attempted": False, "object6d_artifact_emitted": False,
        }],
        tmp_path,
    )]
    row = build_capability_matrix(evidence)["rows"][0]
    assert row["source_terminal_statuses"] == ["PASSED"]
    assert row["executed"] is False
    assert row["exported"] is False
    assert row["quality_admitted"] is False
    assert row["decision"] == "NOT_EXECUTED_OR_BLOCKED"
    assert row["w1_expansion_allowed"] is False


def test_explicit_successor_priority_ignores_failed_predecessor(tmp_path: Path) -> None:
    first = _evidence(
        "0915_robot15h_hawor_wave0_v1", "FAILED_RUNTIME_FINAL", [], tmp_path,
    )
    recovery = _evidence(
        "0915_robot15h_hawor_wave0_recovery_v1", "PASSED", [], tmp_path,
    )
    selected = select_authoritative_evidence([recovery, first])
    assert [item["task_id"] for item in selected] == ["0915_robot15h_hawor_wave0_recovery_v1"]


def test_successor_selection_preserves_distinct_waves(tmp_path: Path) -> None:
    first = _evidence(
        "0915_robot15h_foundationstereo_wave0_v1", "FAILED_RUNTIME_FINAL", [], tmp_path,
    )
    recovery = _evidence(
        "0915_robot15h_foundationstereo_wave0_recovery_v1", "PASSED", [], tmp_path,
    )
    w1 = _evidence(
        "0915_robot15h_foundationstereo_waves_v1", "PASSED", [], tmp_path,
    )
    selected = select_authoritative_evidence([w1, first, recovery])
    assert [item["task_id"] for item in selected] == [
        "0915_robot15h_foundationstereo_wave0_recovery_v1",
        "0915_robot15h_foundationstereo_waves_v1",
    ]


def test_h9_deadline_disables_late_expansion(tmp_path: Path) -> None:
    late = datetime(2026, 9, 19, 9, 10, tzinfo=TZ)
    audit = deadline_audit(TASK_IDS[0], _clock(), late)
    assert audit["freeze_deadline_met"] is False
    assert audit["phase_open"] is False
    evidence = [_evidence(
        "0915_robot15h_foundationstereo_wave0_recovery_v1", "PASSED",
        [{"session_id": "play_cards_0915_031", "status": "PASSED", "result": _artifact(tmp_path / "d.json")}],
        tmp_path,
    )]
    row = build_capability_matrix(evidence, freeze_deadline_met=False)["rows"][0]
    assert row["quality_admitted"] is True
    assert row["w1_expansion_allowed"] is False


def test_capability_specific_object_batch_is_collected(tmp_path: Path) -> None:
    batch = _artifact(tmp_path / "object-batch.json")
    assert _batch_reference({"object_identity_batch": batch}) == batch


def _valid_sidecars(tmp_path: Path, observed_at: str) -> dict[str, dict[str, object]]:
    video_ref = _artifact(tmp_path / "review.mp4", b"not-decoded-by-unit-test")
    return {
        "runtime": {
            "schema_version": "0915-robot15h-final-runtime-evidence-v1",
            "observed_at": observed_at,
            "all_owned_tasks_terminal": True,
            "owned_processes": [],
            "gpu_leases": [],
            "active_writers": [],
            "new_sessions_started_after_cutoff": False,
            "model_inference_active_after_drain": False,
            "selected_source_integrity": {
                "status": "PASSED",
                "sessions_checked": 12,
                "content_drift_count": 0,
                "0916_consumed": False,
            },
        },
        "video": {
            "schema_version": "0915-robot15h-final-video-evidence-v1",
            "status": "PASSED",
            "videos": [{**video_ref, "full_decode": True, "expected_frames": 12, "decoded_frames": 12}],
        },
        "tests": {
            "schema_version": "0915-robot15h-final-test-evidence-v1",
            "status": "PASSED",
            "checks": [
                {"kind": "governance", "status": "PASSED", "returncode": 0, "command": "validate-governance"},
                {"kind": "tests", "status": "PASSED", "returncode": 0, "command": "pytest"},
            ],
        },
        "commit": {
            "schema_version": "0915-robot15h-final-commit-evidence-v1",
            "status": "PASSED",
            "commit": "a" * 40,
            "git_status_porcelain": "",
            "remote_push_performed": False,
        },
    }


def test_final_sidecars_fail_closed_on_gpu_writer_or_dirty_tree(tmp_path: Path) -> None:
    now = datetime(2026, 9, 19, 14, 30, tzinfo=TZ)
    sidecars = _valid_sidecars(tmp_path, "2026-09-19T14:25:00+08:00")
    assert validate_final_sidecars(sidecars, clock=_clock(), now=now)["status"] == "PASSED"
    sidecars["runtime"]["gpu_leases"] = [{"gpu": 0}]
    sidecars["runtime"]["active_writers"] = [{"pid": 42}]
    sidecars["commit"]["git_status_porcelain"] = "?? untracked"
    audit = validate_final_sidecars(sidecars, clock=_clock(), now=now)
    assert audit["status"] == "REJECTED"
    assert any("gpu_leases" in error for error in audit["errors"])
    assert any("active_writers" in error for error in audit["errors"])
    assert any("worktree is dirty" in error for error in audit["errors"])


def test_final_sidecars_fail_closed_on_source_drift(tmp_path: Path) -> None:
    now = datetime(2026, 9, 19, 14, 30, tzinfo=TZ)
    sidecars = _valid_sidecars(tmp_path, "2026-09-19T14:25:00+08:00")
    integrity = sidecars["runtime"]["selected_source_integrity"]
    assert isinstance(integrity, dict)
    integrity["content_drift_count"] = 1
    integrity["status"] = "REJECTED"
    audit = validate_final_sidecars(sidecars, clock=_clock(), now=now)
    assert audit["status"] == "REJECTED"
    assert any("source integrity" in error for error in audit["errors"])


def test_reference_audit_detects_sha_drift(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    reference = _artifact(path, b"original")
    assert audit_references([{"artifact": reference}])["status"] == "PASSED"
    path.write_bytes(b"changed")
    audit = audit_references([{"artifact": reference}])
    assert audit["status"] == "REJECTED"
    assert audit["errors"]


def test_final_reference_audit_ignores_superseded_nested_staging_only(tmp_path: Path) -> None:
    predecessor_result = tmp_path / "predecessor-result.json"
    predecessor_batch = tmp_path / "predecessor-batch.json"
    missing_staging = tmp_path / ".session.staging-deleted" / "mask.json"
    predecessor_result.write_text("{}", encoding="utf-8")
    predecessor_batch.write_text("{}", encoding="utf-8")
    predecessor = {
        "task_id": "0915_robot15h_sam31_temporal_identity_v1",
        "result_ref": ref(predecessor_result),
        "batch_ref": ref(predecessor_batch),
        "result": {"staging_artifact": {
            "path": str(missing_staging), "bytes": 1, "sha256": "0" * 64,
        }},
        "batch": {},
    }
    recovery_result = tmp_path / "recovery-result.json"
    recovery_batch = tmp_path / "recovery-batch.json"
    current_artifact = tmp_path / "current-mask.json"
    recovery_result.write_text("{}", encoding="utf-8")
    recovery_batch.write_text("{}", encoding="utf-8")
    current_ref = _artifact(current_artifact)
    recovery = {
        "task_id": "0915_robot15h_sam31_temporal_identity_recovery_v1",
        "result_ref": ref(recovery_result),
        "batch_ref": ref(recovery_batch),
        "result": {"artifact": current_ref},
        "batch": {},
    }
    evidence = [predecessor, recovery]
    assert audit_campaign_references(evidence, {})["status"] == "PASSED"

    predecessor_result.write_text('{"drift": true}', encoding="utf-8")
    audit = audit_campaign_references(evidence, {})
    assert audit["status"] == "REJECTED"
    assert audit["errors"]


def test_campaign_counts_keep_export_quality_blocked_and_unselected_separate() -> None:
    def item(task_id: str, result: dict[str, object]) -> dict[str, object]:
        return {"task_id": task_id, "result": result}

    evidence = [
        item("0915_robot15h_window_start_inventory_v1", {
            "session_count": 220,
            "w0_session_ids": [f"w0-{index}" for index in range(4)],
            "w1_cumulative_session_ids": [f"selected-{index}" for index in range(12)],
        }),
        item("0915_robot15h_kai22_r0_wave0_v1", {"counts": {
            "attempted": 4, "r0_exported": 4, "r0_success": 0, "r0_rejected": 4,
        }}),
        item("0915_robot15h_robot_relative_refinement_v1", {"counts": {
            "sessions_total": 4, "r1_e_attempted": 0, "r1_e_exported": 0,
            "r1_e_adopted": 0, "r1_e_blocked_local_evidence": 4,
            "r1_h_attempted": 1, "r1_h_exported": 1, "r1_h_adopted": 0,
        }}),
        item("0915_robot15h_robot_virtual_arm_v1", {"counts": {
            "attempted": 4, "r2_exported": 4, "r2_quality_admitted": 0,
            "r2_quality_rejected": 4,
        }}),
        item("0915_robot15h_foundationstereo_wave0_recovery_v1", {"counts": {
            "total": 4, "passed": 4, "rejected_quality": 0, "failed_runtime": 0,
        }}),
        item("0915_robot15h_foundationstereo_waves_v1", {"counts": {
            "attempted": 8, "passed": 7, "rejected_quality": 1,
            "failed_runtime": 0, "unrun": 0,
        }}),
        item("0915_robot15h_robot_waves_v1", {"counts": {
            "total": 8, "attempted": 0, "blocked": 8, "unrun": 0,
        }}),
    ]
    summary = build_campaign_counts(evidence)
    assert summary["denominators"] == {
        "inventory_sessions": 220,
        "frozen_selected_sessions": 12,
        "w0_sessions": 4,
        "w1_additional_sessions": 8,
        "unselected_not_run": 208,
    }
    assert summary["robot_layers"]["R0"] == {
        "denominator": 12,
        "attempted": 4,
        "exported": 4,
        "quality_success": 0,
        "quality_rejected": 4,
        "blocked_upstream": 8,
        "unrun": 0,
    }
    assert summary["robot_layers"]["R1_H"]["quality_rejected"] == 1
    assert summary["robot_layers"]["R1_H"]["blocked_local_or_upstream"] == 11
