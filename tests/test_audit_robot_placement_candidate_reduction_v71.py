from chaoyang.ops.audit_robot_placement_candidate_reduction_v71 import audit_rows, select_reduced


def candidate(backoff, score):
    return {"backoff_m": backoff, "score": score, "status": "HOLD_NUMERIC_CANARY"}


def test_reduced_selection_uses_frozen_lexicographic_score():
    row = {
        "best_hold_backoff_m": 0.2,
        "evaluated_candidates": [
            candidate(0.258, [True, 22, 2.9, 0.002, 0.258]),
            candidate(0.2, [True, 5, 0.36, 0.06, 0.2]),
            candidate(0.3, [True, 62, 5.0, 0.04, 0.3]),
        ],
    }
    assert select_reduced(row, [0.2, 0.258])["backoff_m"] == 0.2


def test_audit_reports_exact_match_and_candidate_reduction():
    rows = [{
        "task": "chips",
        "session": "s1",
        "best_hold_backoff_m": 0.258,
        "evaluated_candidates": [
            candidate(0.258, [True, 2, 1.0, 0.002, 0.258]),
            candidate(0.2, [True, 2**31 - 1, 1e12, 0.06, 0.2]),
            candidate(0.3, [True, 9, 4.0, 0.04, 0.3]),
        ],
    }]
    report = audit_rows(rows, [0.2, 0.258])
    assert report["exact_winner_agreement"] == 1.0
    assert report["counts"]["full_candidate_evaluations"] == 3
    assert report["counts"]["reduced_candidate_evaluations"] == 2
    assert abs(report["candidate_evaluation_reduction_ratio"] - 1 / 3) < 1e-12
