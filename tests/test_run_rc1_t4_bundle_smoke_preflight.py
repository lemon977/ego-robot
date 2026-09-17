from __future__ import annotations

from chaoyang.ops.run_rc1_t4_bundle_smoke_preflight import evaluate


def _capacity(ready: bool) -> dict:
    return {
        "tasks": {
            "chips": {"capacity_ready_under_rc1": ready, "pair_terminal": "PASS" if ready else "BLOCKED_DATA_VOLUME"},
            "poker": {"capacity_ready_under_rc1": ready, "pair_terminal": "PASS" if ready else "BLOCKED_DATA_VOLUME"},
        }
    }


def test_current_negative_inputs_fail_closed_without_bundle() -> None:
    t2 = {"status": "BLOCKED_PREREQ", "fresh_clean_generated": False, "input_mode": "CAUSAL_TRAINING_INPUT"}
    t3 = {"status": "PASSED", "production_entry": {"full_session_training_eligibility": False}}
    status, blockers = evaluate(t2, t3, _capacity(False))
    assert status == "BLOCKED_PREREQ"
    codes = [row["code"] for row in blockers]
    assert "CAUSAL_CLEAN_TASK_NOT_PASSED" in codes
    assert "NO_FRESH_CAUSAL_CLEAN_RGB" in codes
    assert "CAUSAL_ROBOT_PRODUCTION_NOT_ELIGIBLE" in codes
    assert codes.count("PAIR_SOURCE_GROUP_CAPACITY_NOT_MET") == 2


def test_only_fully_causal_and_capacity_ready_inputs_pass_preflight() -> None:
    t2 = {"status": "PASSED", "fresh_clean_generated": True, "input_mode": "CAUSAL_TRAINING_INPUT"}
    t3 = {"status": "PASSED", "production_entry": {"full_session_training_eligibility": True}}
    status, blockers = evaluate(t2, t3, _capacity(True))
    assert status == "PASSED_PREFLIGHT"
    assert blockers == []


def test_offline_clean_mode_is_rejected_even_if_status_says_passed() -> None:
    t2 = {"status": "PASSED", "fresh_clean_generated": True, "input_mode": "OFFLINE_BIDIRECTIONAL_VISUALIZATION"}
    t3 = {"status": "PASSED", "production_entry": {"full_session_training_eligibility": True}}
    status, blockers = evaluate(t2, t3, _capacity(True))
    assert status == "BLOCKED_PREREQ"
    assert [row["code"] for row in blockers] == ["CAUSAL_CLEAN_INPUT_MODE_INVALID"]
