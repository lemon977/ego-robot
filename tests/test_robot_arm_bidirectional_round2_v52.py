from __future__ import annotations

import pytest

from tools import run_exact78_robot_arm_bidirectional_round2_v52 as arm2
from tools import run_robot_arm_segment_bidirectional_v3 as arm_impl


def test_only_method1_hold_consumes_method2() -> None:
    assert arm2.classify_placement_status("PASS_ARM_METHOD1_PLACEMENT_SELECTED") == "CARRY_METHOD1_PASS"
    assert arm2.classify_placement_status("HOLD_ARM_METHOD1_ALL_PLACEMENTS_ELIGIBLE_METHOD2") == "RUN_METHOD2"
    with pytest.raises(ValueError, match="unexpected"):
        arm2.classify_placement_status("FAILED_RUNTIME_RETRYABLE")


def test_unique_sessions_is_fail_closed() -> None:
    assert arm2.unique_sessions([{"session": "a"}, {"session": "b"}], "rows") == {"a", "b"}
    with pytest.raises(ValueError, match="duplicate"):
        arm2.unique_sessions([{"session": "a"}, {"session": "a"}], "rows")
    with pytest.raises(ValueError, match="non-empty"):
        arm2.unique_sessions([{}], "rows")


def test_successor_preflight_may_preserve_placement_lineage() -> None:
    old = {"path": "/tmp/old.json", "bytes": 1, "sha256": "a" * 64}
    new = {"path": "/tmp/new.json", "bytes": 2, "sha256": "b" * 64}
    assert arm2.preflight_lineage_allows({}, old, old)
    assert arm2.preflight_lineage_allows({"predecessor_preflight": old}, new, old)
    assert not arm2.preflight_lineage_allows({}, new, old)


def test_reverse_temporal_bound_infeasibility_is_quality_evidence() -> None:
    def fail():
        raise arm_impl.temporal.TemporalReviewError("empty previous-accepted bounds: max_excess=0.01")

    candidate, evidence = arm_impl.reverse_candidate_or_infeasible(fail)
    assert candidate is None
    assert evidence == {
        "status": "HOLD_NUMERIC_INFEASIBLE_TEMPORAL_BOUNDS",
        "all_gates": False,
        "candidate_available": False,
        "error_type": "TemporalReviewError",
        "error": "empty previous-accepted bounds: max_excess=0.01",
    }


def test_reverse_candidate_is_preserved_when_feasible() -> None:
    expected = ({1: "q"}, [{"frame": 1}], {"score": (0, 0, 0, 0), "all_gates": True})
    candidate, evidence = arm_impl.reverse_candidate_or_infeasible(lambda: expected)
    assert candidate == ("REVERSE_LOOKAHEAD", *expected)
    assert evidence is expected[2]
