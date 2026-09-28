from __future__ import annotations
import numpy as np
from chaoyang.pipeline.robot_rigid_correspondence_v1 import correspond_robot_pixel


def test_same_link_identity_and_wrong_link_rejected() -> None:
    z = np.ones((5, 5), float)
    ids = np.full((5, 5), 7, int)
    k = np.eye(3)
    frames = {7: np.eye(4)}
    good = correspond_robot_pixel((2, 2), z, ids, z, ids, k, frames, frames)
    assert good["valid"] and good["target_xy"] == [2, 2]
    other = ids.copy(); other[2, 2] = 8
    bad = correspond_robot_pixel((2, 2), z, ids, z, other, k, frames, frames)
    assert bad["reason"] == "OTHER_LINK_OR_DISOCCLUDED"


def test_reprojection_and_depth_must_close() -> None:
    z = np.ones((5, 5), float)
    ids = np.full((5, 5), 7, int)
    k = np.eye(3)
    frames = {7: np.eye(4)}
    moved = {7: np.eye(4)}
    moved[7][0, 3] = 1.0
    good = correspond_robot_pixel((2, 2), z, ids, z, ids, k, frames, moved)
    assert good["valid"] and good["target_xy"] == [3, 2]
    z_bad = z.copy(); z_bad[2, 3] = .5
    bad = correspond_robot_pixel((2, 2), z, ids, z_bad, ids, k, frames, moved)
    assert bad["reason"] == "DEPTH_INCONSISTENT_OR_OCCLUDED"
