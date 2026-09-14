from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/run_156_fail_forward_daemon.py"
SPEC = importlib.util.spec_from_file_location("ff156", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ff156 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ff156)


def test_extract_sessions_accepts_explicit_two_task_lists():
    rows = ff156.extract_sessions({
        "chips": ["get_potato_chips_0901_001"],
        "poker": ["play_cards_0903_001"],
    })
    assert rows == [
        {"task": "chips", "session": "get_potato_chips_0901_001"},
        {"task": "poker", "session": "play_cards_0903_001"},
    ]


def test_duplicate_cohort_is_rejected():
    with pytest.raises(ff156.ContractError, match="duplicate"):
        ff156.extract_sessions([
            {"task": "chips", "session": "get_potato_chips_0901_001"},
            {"task": "chips", "session": "get_potato_chips_0901_001"},
        ])


def test_grade_c_cannot_claim_training(tmp_path: Path):
    path = tmp_path / "RESULT.json"
    path.write_text(json.dumps({
        "schema_version": "fail-forward-stage-result-v1",
        "task": "chips", "session": "s", "stage": "MASK",
        "terminal": True, "grade": "C", "training_weight": None,
        "may_train": True, "may_be_delivery_candidate": False,
        "consumption_authorized": False, "defects": [], "metrics": {},
        "inputs": {}, "artifacts": {}, "hard_invalidators": [],
    }))
    with pytest.raises(ff156.ContractError, match="grade-C"):
        ff156.validate_receipt(
            path, {"task": "chips", "session": "s", "stage": "MASK"}
        )


def test_hard_invalidator_forces_c(tmp_path: Path):
    path = tmp_path / "RESULT.json"
    path.write_text(json.dumps({
        "schema_version": "fail-forward-stage-result-v1",
        "task": "poker", "session": "s", "stage": "CLEAN",
        "terminal": True, "grade": "B", "training_weight": 0.5,
        "may_train": True, "may_be_delivery_candidate": False,
        "consumption_authorized": True, "defects": [], "metrics": {},
        "inputs": {}, "artifacts": {},
        "hard_invalidators": ["wrong_session_or_wrong_scene_pixels"],
    }))
    with pytest.raises(ff156.ContractError, match="hard invalidator"):
        ff156.validate_receipt(
            path, {"task": "poker", "session": "s", "stage": "CLEAN"}
        )


def test_schedule_bootstraps_checkpoints_and_leaves_heldout_last():
    sessions = {}
    for task in ("chips", "poker"):
        for split, count in (("heldout", 1), ("test", 1), ("train", 31), ("validation", 1)):
            for index in range(count):
                name = f"{task}_{split}_{index:02d}"
                sessions[name] = {"task": task, "split": split}
    ordered = [name for name, _ in ff156.scheduled_sessions({"sessions": sessions})]
    assert ordered.index("chips_validation_00") < ordered.index("chips_train_00")
    assert ordered.index("chips_train_29") < ordered.index("poker_validation_00")
    assert ordered.index("poker_train_29") < ordered.index("chips_train_30")
    assert ordered[-2:] == ["chips_heldout_00", "poker_heldout_00"]
