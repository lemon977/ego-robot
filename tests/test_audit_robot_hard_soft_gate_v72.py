from chaoyang.ops.audit_robot_hard_soft_gate_v72 import classify_gates


def hand_gates() -> dict[str, bool]:
    return {
        "velocity": True,
        "acceleration": True,
        "missing_unknown_not_filled": True,
        "thumb_independent_q0_to_q5": True,
        "four_finger_chain_semantics": True,
        "anatomy_all_observed": False,
    }


def collision_gates() -> dict[str, bool]:
    return {
        "digital_collision_geometry_loaded": True,
        "non_adjacent_self_intersection_absent": True,
    }


def test_current_v77_arm_aliases_are_hard_pass() -> None:
    arm = {
        "arm_velocity": True,
        "arm_acceleration": True,
        "missing_frames_unknown_not_filled": True,
        "pose_branch_all_observed": False,
    }
    result = classify_gates(arm, hand_gates(), collision_gates())
    assert result["hard_geometry_pass"] is True
    assert result["terminal_status"] == "PASSED"
    assert result["strict_pose_match"] is False


def test_legacy_aliases_remain_supported() -> None:
    arm = {
        "velocity": True,
        "acceleration": True,
        "missing_unknown_not_filled": True,
        "pose_branch_all_observed": True,
    }
    result = classify_gates(arm, hand_gates() | {"anatomy_all_observed": True}, collision_gates())
    assert result["hard_geometry_pass"] is True
    assert result["strict_pose_match"] is True


def test_alias_disagreement_fails_closed() -> None:
    arm = {
        "velocity": True,
        "arm_velocity": False,
        "acceleration": True,
        "missing_unknown_not_filled": True,
    }
    result = classify_gates(arm, hand_gates(), collision_gates())
    assert result["hard_geometry_pass"] is False
    assert result["terminal_status"] == "FAILED_QUALITY_C"


def test_missing_arm_gate_fails_closed() -> None:
    arm = {"arm_velocity": True, "arm_acceleration": True}
    result = classify_gates(arm, hand_gates(), collision_gates())
    assert result["hard_geometry_pass"] is False
