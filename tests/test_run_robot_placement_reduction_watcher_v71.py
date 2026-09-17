from chaoyang.ops.run_robot_placement_reduction_watcher_v71 import decision, hand_round2_decision


def report(tasks, agreement=1.0):
    return {
        "sessions": [{"task": task} for task in tasks],
        "exact_winner_agreement": agreement,
        "counts": {"sessions": len(tasks)},
    }


def test_decision_requires_cross_task_coverage():
    result = decision(report(["chips"] * 12))
    assert result["status"] == "KEEP_FULL_CANDIDATE_SET"


def test_decision_requires_exact_replay_agreement():
    result = decision(report(["chips"] * 6 + ["poker"] * 6, 0.99))
    assert result["status"] == "KEEP_FULL_CANDIDATE_SET"


def test_decision_can_only_prepare_a_future_revision():
    result = decision(report(["chips"] * 6 + ["poker"] * 6))
    assert result["status"] == "READY_FOR_IMMUTABLE_SUCCESSOR_CANARY"
    assert "do not mutate R7.3" in result["next_action"]


def test_round2_skip_requires_cross_task_zero_improvement():
    audit = {
        "sessions": [{"task": "chips"}] * 6 + [{"task": "poker"}] * 6,
        "counts": {"sessions": 12, "summary_improved": 0},
    }
    assert hand_round2_decision(audit)["status"] == "READY_FOR_CONDITIONAL_SKIP_CANARY"
    audit["counts"]["summary_improved"] = 1
    assert hand_round2_decision(audit)["status"] == "KEEP_ROUND2"
