from __future__ import annotations

import pytest

from tools import run_exact78_robot_hand_bidirectional_round2_v52 as hand2


def test_classify_input_is_exhaustive_and_fail_closed() -> None:
    assert hand2.classify_input("PASS_HAND_ROUND1_FORWARD") == "CARRY_HAND_PASS"
    assert hand2.classify_input("HOLD_HAND_ROUND1_ELIGIBLE_ROUND2") == "RUN_HAND_ROUND2"
    with pytest.raises(ValueError, match="unexpected"):
        hand2.classify_input("FAILED_RUNTIME_RETRYABLE")


def test_unique_sessions_rejects_duplicate_or_missing_rows() -> None:
    assert hand2.unique_sessions([{"session": "a"}, {"session": "b"}], "rows") == {"a", "b"}
    with pytest.raises(ValueError, match="duplicate"):
        hand2.unique_sessions([{"session": "a"}, {"session": "a"}], "rows")
    with pytest.raises(ValueError, match="non-empty"):
        hand2.unique_sessions([{}], "rows")
