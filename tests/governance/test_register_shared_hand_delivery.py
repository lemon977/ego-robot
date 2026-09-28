from chaoyang.governance.register_human_to_robot_shared_hand_delivery import (
    LANES, TASK, REVISION,
)


def test_shared_hand_delivery_identity_and_lanes():
    assert TASK == "human_to_robot_shared_hand_delivery_20260924"
    assert REVISION == "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_20260924"
    assert LANES == ("hand_data", "robot", "clean", "delivery")
