from chaoyang.ops.run_robot_conversion_diagnosis_watcher_v71 import classify, summarize


def test_classification_separates_hard_geometry_from_soft_pose():
    assert classify({"strict_pose_match": False, "hard_geometry_pass": False}) == "HARD_GEOMETRY_FAIL"
    assert classify({
        "strict_pose_match": False,
        "hard_geometry_pass": True,
        "soft_gates": {"arm_pose_all_observed": True, "hand_anatomy_all_observed": False},
    }) == "HARD_PASS_SOFT_HAND_MISMATCH"
    assert classify({"strict_pose_match": True, "hard_geometry_pass": True}) == "STRICT_POSE_MATCH"


def test_summary_uses_frozen_matrix_split_and_reports_recovery_ratio():
    index = {"rows": [
        {"session": "a", "strict_pose_match": False, "hard_geometry_pass": True,
         "soft_gates": {"arm_pose_all_observed": False, "hand_anatomy_all_observed": False}},
        {"session": "b", "strict_pose_match": False, "hard_geometry_pass": False},
    ]}
    matrix = {"rows": [
        {"session_id": "a", "task": "chips", "split": "train"},
        {"session_id": "b", "task": "poker", "split": "validation"},
    ]}
    rows, summary = summarize(index, matrix)
    assert rows[0]["split"] == "train"
    assert summary["strict_failure_count"] == 2
    assert summary["strict_failure_hard_feasible"] == 1
    assert summary["strict_failure_hard_feasible_ratio"] == 0.5
