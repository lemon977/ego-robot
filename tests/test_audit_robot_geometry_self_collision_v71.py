import numpy as np
import pytest

from chaoyang.ops.audit_robot_geometry_self_collision_v71 import (
    is_allowed_mount_contact,
    select_uniform_valid_frames,
)


def test_uniform_bilateral_valid_selection():
    valid = np.ones((2, 10), dtype=bool)
    valid[0, 4] = False
    selected = select_uniform_valid_frames(valid, 4)
    assert selected == [0, 3, 6, 9]
    assert all(valid[:, frame].all() for frame in selected)


def test_uniform_selection_rejects_insufficient_bilateral_frames():
    valid = np.zeros((2, 5), dtype=bool)
    valid[:, :2] = True
    with pytest.raises(ValueError, match="need 4"):
        select_uniform_valid_frames(valid, 4)


def test_only_same_side_tool_to_hand_base_is_mount_allowlisted():
    assert is_allowed_mount_contact(tianji_link="left_tool", hand_link_index=-1, side="left")
    assert not is_allowed_mount_contact(tianji_link="right_tool", hand_link_index=-1, side="left")
    assert not is_allowed_mount_contact(tianji_link="left_tool", hand_link_index=0, side="left")
