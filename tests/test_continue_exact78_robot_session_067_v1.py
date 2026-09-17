from __future__ import annotations

import json

from chaoyang.ops import continue_exact78_robot_session_067_v1 as continuation


def test_temporal_cli_receives_directory_not_result_file() -> None:
    assert continuation.TEMPORAL == continuation.TEMPORAL_ROOT / "RESULT.json"
    assert continuation.TEMPORAL_ROOT.name == "batch_067_v1"


def test_completed_arm_phase_can_be_adopted(tmp_path) -> None:
    root = tmp_path / "arm"
    root.mkdir()
    (root / "RESULT.json").write_text(
        json.dumps({"status": "PASS_BOUNDED_TWO_METHOD_ARM_RESULTS_PUBLISHED"}),
        encoding="utf-8",
    )
    adopted = continuation.load_completed_phase(
        root, "PASS_BOUNDED_TWO_METHOD_ARM_RESULTS_PUBLISHED"
    )
    assert adopted is not None
    assert adopted["status"] == "PASS_BOUNDED_TWO_METHOD_ARM_RESULTS_PUBLISHED"


def test_completed_phase_rejects_wrong_status(tmp_path) -> None:
    root = tmp_path / "arm"
    root.mkdir()
    (root / "RESULT.json").write_text(
        json.dumps({"status": "FAILED_RUNTIME_FINAL"}), encoding="utf-8"
    )
    try:
        continuation.load_completed_phase(
            root, "PASS_BOUNDED_TWO_METHOD_ARM_RESULTS_PUBLISHED"
        )
    except RuntimeError as error:
        assert "unexpected status" in str(error)
    else:
        raise AssertionError("wrong phase status must not be adopted")


def test_terminal_publisher_receives_unexpanded_base() -> None:
    assert continuation.TERMINAL_ROOT == (
        continuation.TERMINAL_BASE / continuation.TASK / continuation.SESSION
    )
