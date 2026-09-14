from __future__ import annotations

import numpy as np

from tools.render_chips_same_side_outward_frame0 import FRAME_ID, SESSION


def test_contract_is_one_chips_frame() -> None:
    assert FRAME_ID == 0
    assert SESSION == "get_potato_chips_0902_034"


def test_fixed_world_base_yields_per_frame_camera_base() -> None:
    c2w = np.eye(4)
    c2w[:3, 3] = [0.4, -0.2, 1.1]
    world_base = np.eye(4)
    world_base[:3, 3] = [-0.3, 0.7, 0.8]
    camera_base = np.linalg.inv(c2w) @ world_base
    assert np.allclose(c2w @ camera_base, world_base)


def test_world_hand_target_removes_camera_translation() -> None:
    hand_camera = np.asarray([0.2, 0.3, 0.5, 1.0])
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, -2.0, 0.4]
    hand_world = c2w @ hand_camera
    assert np.allclose(hand_world[:3], [1.2, -1.7, 0.9])
