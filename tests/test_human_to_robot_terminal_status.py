from __future__ import annotations

import json

import pytest

from chaoyang.governance import build_human_to_robot_r2_terminal_status as module


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "STATUS.json")
    monkeypatch.setattr(module, "SYNC_VALIDATION", tmp_path / "REGRESSION.json")
    for task, count in ((module.TASK, 13), (module.S1_TASK, 15), (module.S2_TASK, 11)):
        base = tmp_path / f"_run/current/{task}/attempts/attempt_0001"
        base.mkdir(parents=True)
        (base / "RESULT.json").write_text(json.dumps({
            "task_id": task, "status": "TERMINAL_WITH_QUALITY_GAPS",
            "execution_summary": {"formal_clean_adopted": 0},
            "quality_axes": {"quality": "REJECTED_QUALITY"}, "claim_limit": "offline only",
        }))
        validation = {"video_count": count}
        if task == module.S2_TASK:
            validation["regression"] = {
                "project_local_main": {"tests": 1490, "failures": 0, "errors": 0, "skipped": 1},
                "s2_current_closure": {"tests": 41, "failures": 0, "errors": 0, "skipped": 0},
            }
        else:
            validation["pytest"] = {"passed": 1, "skipped": 0}
        (base / "FINAL_VALIDATION.json").write_text(json.dumps(validation))
        if task == module.S2_TASK:
            checkpoint = base / "checkpoints/H3_RESULT.json"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps({"checkpoint": "H3"}))
    return tmp_path


def state(*ids):
    return {"tasks": [{"task_id": task, "status": "REJECTED_QUALITY"} for task in ids],
            "next_task": None, "governance_revision": 99}


def test_successor_wins_even_when_r2_is_last_in_ledger(repo):
    result = module.build_status(state(module.S1_TASK, module.TASK), "now")
    assert result["latest_task"] == module.S1_TASK
    assert result["counts"]["validated_videos"] == 15
    assert result["preserved_r2"]["counts"]["formal_clean_adopted"] == 0
    assert result["training_eligible"] is False


def test_s2_terminal_supersedes_s1_and_uses_s2_regression_schema(repo):
    result = module.build_status(
        state(module.S1_TASK, module.TASK, module.S2_TASK), "now"
    )
    assert result["latest_task"] == module.S2_TASK
    assert result["counts"]["validated_videos"] == 11
    assert result["sealed_validation_test_summary"]["s2_current_closure"]["tests"] == 41


def test_active_s2_publishes_checkpoint_status_instead_of_leaving_old_s1(repo):
    ledger = state(module.S1_TASK, module.TASK, module.S2_TASK)
    ledger["tasks"][-1]["status"] = "PENDING"
    ledger["next_task"] = {"task_id": module.S2_TASK, "checkpoint": "H6"}
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    result = json.loads(module.OUTPUT.read_text())
    assert result["latest_task"] == module.S2_TASK
    assert result["status"] == "ACTIVE_H3_CHECKPOINT"
    assert result["counts"]["formal_product_quality_pass"] == 0
    assert result["counts"]["robot_r1_screened"] == 0
    assert "screening was not completed" in result["claim_limit"]


def test_active_product_first_projection_uses_actual_progress_not_old_s2(repo):
    ledger = state(module.S1_TASK, module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.PRODUCT_FIRST_TASK, "status": "PENDING"})
    ledger["next_task"] = {"task_id": module.PRODUCT_FIRST_TASK}
    path = repo / f"_run/current/{module.PRODUCT_FIRST_TASK}/attempts/attempt_0001/PROGRESS_H1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"task_id": module.PRODUCT_FIRST_TASK,
                                "counts": {"formal_quality_pass": 0, "cleanup_deleted_bytes": 762165394}}))
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    result = json.loads(module.OUTPUT.read_text())
    assert result["latest_task"] == module.PRODUCT_FIRST_TASK
    assert result["counts"]["cleanup_deleted_bytes"] == 762165394
    assert result["quality_axes"]["adoption"] == "NOT_ADOPTED"


def test_terminal_product_first_supersedes_old_s2_without_quality_promotion(repo):
    base = repo / f"_run/current/{module.PRODUCT_FIRST_TASK}/attempts/attempt_0001"
    base.mkdir(parents=True)
    validation = base / "validation_after_cleanup_batch2/RESULT.json"
    validation.parent.mkdir()
    validation.write_text(json.dumps({"status": "PASS_FOR_CHECKED_SCOPE", "slots": [1] * 15}))
    result = base / "RESULT.json"
    result.write_text(json.dumps({"task_id": module.PRODUCT_FIRST_TASK,
                                  "status": "TERMINAL_WITH_QUALITY_GAPS",
                                  "execution_summary": {"products_quality": "0/4", "cleanup_logical_deleted_bytes": 5219025177},
                                  "quality_axes": {"adoption": "NOT_ADOPTED"},
                                  "final_validation": module.artifact_ref(validation),
                                  "claim_limit": "no product quality promotion"}))
    ledger = state(module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.PRODUCT_FIRST_TASK,
                            "status": "REJECTED_QUALITY", "result": module.artifact_ref(result)})
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    status = json.loads(module.OUTPUT.read_text())
    assert status["latest_task"] == module.PRODUCT_FIRST_TASK
    assert status["counts"]["products_quality"] == "0/4"
    assert status["counts"]["cleanup_logical_deleted_bytes"] == 5219025177
    assert status["quality_axes"]["adoption"] == "NOT_ADOPTED"


def test_source_coverage_terminal_supersedes_product_first(repo):
    base = repo / f"_run/current/{module.SOURCE_COVERAGE_TASK}/attempts/attempt_0001"
    base.mkdir(parents=True)
    validation = base / "FINAL_VALIDATION.json"
    validation.write_text(json.dumps({"status": "PASS_FOR_CHECKED_STRUCTURE_NOT_PRODUCT_QUALITY"}))
    result = base / "RESULT.json"
    result.write_text(json.dumps({
        "task_id": module.SOURCE_COVERAGE_TASK,
        "status": "TERMINAL_WITH_QUALITY_GAPS",
        "execution_summary": {"031_projected_wrist_inside_image_frames": 0,
                              "products_quality": "0/4"},
        "quality_axes": {"quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED"},
        "final_validation": module.artifact_ref(validation),
        "claim_limit": "No product quality promotion",
    }))
    ledger = state(module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.PRODUCT_FIRST_TASK,
                            "status": "REJECTED_QUALITY"})
    ledger["tasks"].append({"task_id": module.SOURCE_COVERAGE_TASK,
                            "status": "REJECTED_QUALITY",
                            "result": module.artifact_ref(result)})
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    status = json.loads(module.OUTPUT.read_text())
    assert status["latest_task"] == module.SOURCE_COVERAGE_TASK
    assert status["counts"]["031_projected_wrist_inside_image_frames"] == 0
    assert status["quality_axes"]["quality"] == "REJECTED_QUALITY"


def test_active_source_coverage_does_not_fall_back_to_old_terminal(repo):
    ledger = state(module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.PRODUCT_FIRST_TASK,
                            "status": "REJECTED_QUALITY"})
    ledger["tasks"].append({"task_id": module.SOURCE_COVERAGE_TASK,
                            "status": "RUNNING"})
    assert not module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    assert not module.OUTPUT.exists()


def test_active_device_donor_probe_is_not_masked_by_prior_terminal(repo):
    ledger = state(module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.SOURCE_COVERAGE_TASK,
                            "status": "REJECTED_QUALITY"})
    ledger["tasks"].append({"task_id": module.DEVICE_DONOR_TASK,
                            "status": "PENDING", "task_packet": {"path": "packet"}})
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    status = json.loads(module.OUTPUT.read_text())
    assert status["latest_task"] == module.DEVICE_DONOR_TASK
    assert status["active_tasks"] == [module.DEVICE_DONOR_TASK]
    assert status["quality_axes"]["quality"] == "NOT_EVALUATED"


def test_terminal_device_donor_probe_publishes_only_probe_counts(repo):
    base = repo / f"_run/current/{module.DEVICE_DONOR_TASK}/attempts/attempt_0001"
    base.mkdir(parents=True)
    validation = base / "FINAL_VALIDATION.json"
    validation.write_text(json.dumps({"status": "PASS_FOR_PROBE_STRUCTURE_NOT_PRODUCT_QUALITY"}))
    result = base / "RESULT.json"
    result.write_text(json.dumps({
        "task_id": module.DEVICE_DONOR_TASK,
        "status": "TERMINAL_WITH_QUALITY_GAPS",
        "execution_summary": {"007_table_donor_geometry_pass_frames": [],
                              "products_quality": "0/4"},
        "quality_axes": {"quality": "FULL_CLEAN_NOT_PASSED", "adoption": "NOT_ADOPTED"},
        "final_validation": module.artifact_ref(validation),
        "claim_limit": "No accepted Clean",
    }))
    ledger = state(module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.DEVICE_DONOR_TASK,
                            "status": "REJECTED_QUALITY",
                            "result": module.artifact_ref(result)})
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    status = json.loads(module.OUTPUT.read_text())
    assert status["latest_task"] == module.DEVICE_DONOR_TASK
    assert status["counts"]["products_quality"] == "0/4"
    assert status["quality_axes"]["adoption"] == "NOT_ADOPTED"


def test_active_wearable_canary_supersedes_terminal_donor(repo):
    ledger = state(module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.DEVICE_DONOR_TASK,
                            "status": "REJECTED_QUALITY"})
    ledger["tasks"].append({"task_id": module.WEARABLE_TASK,
                            "status": "PENDING", "task_packet": {"path": "packet"}})
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    status = json.loads(module.OUTPUT.read_text())
    assert status["latest_task"] == module.WEARABLE_TASK
    assert status["counts"]["products_quality"] == "0/4"
    assert status["quality_axes"]["quality"] == "NOT_EVALUATED"


def test_passing_wearable_canary_cannot_promote_full_clean(repo):
    base = repo / f"_run/current/{module.WEARABLE_TASK}/attempts/attempt_0001"
    base.mkdir(parents=True)
    validation = base / "FINAL_VALIDATION.json"
    validation.write_text(json.dumps({"status": "PASS_FOR_SINGLE_FRAME_STRUCTURE_NOT_CLEAN_QUALITY"}))
    result = base / "RESULT.json"
    result.write_text(json.dumps({
        "task_id": module.WEARABLE_TASK, "status": "TERMINAL_SINGLE_FRAME",
        "execution_summary": {"007_raw_wearable_instances_supported": ["left_index_device"],
                              "products_quality": "0/4"},
        "quality_axes": {"quality": "PASS", "clean_quality": "NOT_EVALUATED",
                         "adoption": "NOT_ADOPTED"},
        "final_validation": module.artifact_ref(validation),
        "claim_limit": "Raw instance only",
    }))
    ledger = state(module.TASK, module.S2_TASK)
    ledger["tasks"].append({"task_id": module.WEARABLE_TASK,
                            "status": "PASSED", "result": module.artifact_ref(result)})
    assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    status = json.loads(module.OUTPUT.read_text())
    assert status["latest_task"] == module.WEARABLE_TASK
    assert status["counts"]["products_quality"] == "0/4"
    assert status["quality_axes"]["clean_quality"] == "NOT_EVALUATED"


def test_republication_cannot_overwrite_s1_with_r2(repo):
    ledger = state(module.TASK, module.S1_TASK)
    for at in ("first", "second"):
        assert module.publish_navigation_if_human_to_robot_r2(repo, ledger, at)
        assert json.loads(module.OUTPUT.read_text())["latest_task"] == module.S1_TASK


def test_old_installation_without_s1_still_publishes_r2(repo):
    assert module.build_status(state(module.TASK), "now")["counts"]["validated_videos"] == 13


def test_missing_successor_receipt_is_not_silent_r2_fallback(repo):
    (repo / f"_run/current/{module.S1_TASK}/attempts/attempt_0001/RESULT.json").unlink()
    with pytest.raises(FileNotFoundError):
        module.build_status(state(module.TASK, module.S1_TASK), "now")


def test_running_successor_does_not_publish_old_terminal_state(repo):
    ledger = state(module.TASK, module.S1_TASK)
    ledger["tasks"][-1]["status"] = "RUNNING"
    assert not module.publish_navigation_if_human_to_robot_r2(repo, ledger, "now")
    assert not module.OUTPUT.exists()


def test_other_active_task_not_hidden(repo):
    ledger = state(module.TASK, module.S1_TASK)
    ledger["tasks"].append({"task_id": "next_work", "status": "RUNNING"})
    result = module.build_status(ledger, "now")
    assert result["active_tasks"] == ["next_work"]
    assert result["current_index_status"] != "PASS_NO_ACTIVE_TASKS"


def test_mismatched_result_cannot_change_current_authority(repo):
    path = repo / f"_run/current/{module.S1_TASK}/attempts/attempt_0001/RESULT.json"
    value = json.loads(path.read_text()); value["task_id"] = module.TASK
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="TASK_MISMATCH"):
        module.build_status(state(module.TASK, module.S1_TASK), "now")


def test_stale_result_hash_rejected(repo):
    ledger = state(module.S1_TASK)
    path = repo / f"_run/current/{module.S1_TASK}/attempts/attempt_0001/RESULT.json"
    ledger["tasks"][0]["result"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="RESULT_BINDING_INVALID"):
        module.build_status(ledger, "now")
