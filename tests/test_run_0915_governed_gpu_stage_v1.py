from __future__ import annotations

import json

import pytest

from chaoyang.ops import run_0915_governed_gpu_stage_v1 as subject


def test_gpu_stages_are_finite_and_use_expected_pinned_launchers(tmp_path) -> None:
    assert set(subject.GPU_TASKS) == {
        "0915_hawor_full_v1",
        "0915_foundationstereo_full_v1",
        "0915_sam31_mask_full_v1",
    }
    hawor = subject.build_worker_command("0915_hawor_full_v1", tmp_path / "hawor")
    stereo = subject.build_worker_command("0915_foundationstereo_full_v1", tmp_path / "depth")
    sam = subject.build_worker_command("0915_sam31_mask_full_v1", tmp_path / "sam31")
    assert hawor[0].endswith("hawor_python.sh")
    assert stereo[0].endswith("foundationstereo_gpu_python.sh")
    assert sam[1].endswith("run_0915_sam31_persistent_masks_v2.py")


def test_sam_gpu_command_is_sam31_only(tmp_path) -> None:
    command = subject.build_worker_command("0915_sam31_mask_full_v1", tmp_path / "sam31")
    encoded = json.dumps(command).lower()
    assert "sam31" in encoded
    assert "sam2" not in encoded
    assert "cutie" not in encoded


def test_worker_output_placeholder_is_fully_resolved(tmp_path) -> None:
    for task_id in subject.GPU_TASKS:
        command = subject.build_worker_command(task_id, tmp_path / task_id)
        assert "{worker_output}" not in command


def test_hawor_resume_requires_interrupted_atomic_session_set(tmp_path) -> None:
    output = tmp_path / "hawor"
    committed = output / "sessions" / "playing_cards" / "play_cards_0915_001"
    committed.mkdir(parents=True)
    result = {
        "schema_version": "0915-hawor-persistent-session-v1",
        "session_id": "play_cards_0915_001",
        "status": "FAILED_QUALITY_C",
    }
    (committed / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    failed = {
        "schema_version": "0915-hawor-persistent-session-v1",
        "session_id": "play_cards_0915_002",
        "status": "FAILED_RUNTIME",
    }
    state = {
        "schema_version": "0915-hawor-persistent-state-v1",
        "state": "RUNNING",
        "session_count": 220,
        "completed": 2,
        "failed_runtime": 1,
        "results": [result, failed],
    }
    (output / "STATE.json").write_text(json.dumps(state), encoding="utf-8")
    assert subject.validate_resume_worker_output("0915_hawor_full_v1", output) == state


def test_hawor_resume_rejects_committed_state_drift(tmp_path) -> None:
    output = tmp_path / "hawor"
    output.mkdir()
    state = {
        "schema_version": "0915-hawor-persistent-state-v1",
        "state": "RUNNING",
        "session_count": 220,
        "completed": 1,
        "failed_runtime": 0,
        "results": [{"session_id": "missing", "status": "FAILED_QUALITY_C"}],
    }
    (output / "STATE.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="committed sessions differ"):
        subject.validate_resume_worker_output("0915_hawor_full_v1", output)


def test_resume_is_not_authorized_for_other_gpu_stages(tmp_path) -> None:
    output = tmp_path / "worker"
    output.mkdir()
    with pytest.raises(RuntimeError, match="only for HaWoR"):
        subject.validate_resume_worker_output("0915_sam31_mask_full_v1", output)
