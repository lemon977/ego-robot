from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.audit_robot_hard_soft_gate_v71 import classify_gates


def _arm(strict: bool = False):
    return {
        "velocity": True,
        "acceleration": True,
        "missing_unknown_not_filled": True,
        "pose_branch_all_observed": strict,
    }


def _hand(strict: bool = False):
    return {
        "velocity": True,
        "acceleration": True,
        "missing_unknown_not_filled": True,
        "thumb_independent_q0_to_q5": True,
        "four_finger_chain_semantics": True,
        "anatomy_all_observed": strict,
    }


def _collision(passed: bool = True):
    return {
        "real_collision_geometry_loaded": True,
        "non_adjacent_self_intersection_absent": passed,
    }


def test_soft_pose_failure_does_not_become_hard_geometry_failure():
    result = classify_gates(_arm(), _hand(), _collision())
    assert result["status"] == "PASS_HARD_GEOMETRY_SOFT_POSE_REVIEW_REQUIRED"
    assert result["terminal_status"] == "PASSED"
    assert result["robot_tier"] == "NONE"
    assert result["candidate_robot_tier"] == "POSE_ONLY_VISUAL"
    assert result["hard_geometry_pass"] is True
    assert result["strict_pose_match"] is False
    assert result["soft_pose_similarity_pass"] is False


def test_collision_failure_remains_hard_failure():
    result = classify_gates(_arm(True), _hand(True), _collision(False))
    assert result["status"] == "FAILED_HARD_GEOMETRY"
    assert result["terminal_status"] == "FAILED_QUALITY_C"
    assert result["candidate_robot_tier"] == "NONE"
    assert result["hard_geometry_pass"] is False


def test_strict_pose_and_hard_geometry_are_reported_separately():
    result = classify_gates(_arm(True), _hand(True), _collision())
    assert result["status"] == "PASS_STRICT_POSE_AND_HARD_GEOMETRY_DEVELOPMENT"
    assert result["strict_pose_match"] is True
