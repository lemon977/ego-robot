import json

from chaoyang.governance.common import artifact_ref
from chaoyang.governance.rc1_t5_consistency import validate_t5_current_flags


def _state(tmp_path):
    flags = {
        "CHECKPOINT_PAIR_CHIPS": "BLOCKED_DATA_VOLUME",
        "CHECKPOINT_PAIR_POKER": "BLOCKED_DATA_VOLUME",
        "RATE_FINALIZED": False,
        "RC1_RELEASE_STATUS": "INCOMPLETE",
    }
    summary_path = tmp_path / "QUALITY_SUMMARY.json"
    summary_path.write_text(json.dumps({"release_flags": flags}), encoding="utf-8")
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps({
        "task_id": "rc1_t5_batch_conversion",
        "status": "PASSED",
        "quality_summary": artifact_ref(summary_path),
        "release_status": "INCOMPLETE",
        "rate_finalized": False,
    }), encoding="utf-8")
    return {
        "tasks": [{"task_id": "rc1_t5_batch_conversion", "status": "PASSED", "result": artifact_ref(result_path)}],
        "rc1_pair_status": {"chips": "BLOCKED_DATA_VOLUME", "poker": "BLOCKED_DATA_VOLUME"},
        "rc1_release_flags": dict(flags),
    }


def test_t5_current_flags_match_immutable_receipt(tmp_path):
    assert validate_t5_current_flags(_state(tmp_path)) == []


def test_t5_current_flags_reject_stale_t0_projection(tmp_path):
    state = _state(tmp_path)
    state["rc1_release_flags"]["RC1_RELEASE_STATUS"] = "T0_COMPLETE_CAPACITY_LIMITED"
    assert any("RC1_RELEASE_STATUS" in error for error in validate_t5_current_flags(state))


def test_t5_current_flags_reject_unbound_pair_status(tmp_path):
    state = _state(tmp_path)
    state["rc1_pair_status"]["poker"] = "PASS"
    assert any("poker pair status" in error for error in validate_t5_current_flags(state))
