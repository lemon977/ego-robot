from chaoyang.ops.audit_kaihand_tip_reachability_floor_v71 import conclusion


def test_reachability_conclusion_respects_frozen_threshold():
    assert conclusion(18.05, 15.0) == "TARGET_OUTSIDE_SAMPLED_ROBOT_REACHABLE_SET"
    assert conclusion(14.99, 15.0) == "TARGET_REACHABLE_UNDER_TIP_ONLY_OBJECTIVE"
