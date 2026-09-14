from __future__ import annotations

import numpy as np

from tools.render_poker_same_side_outward_frame0 import (
    BLUE,
    FLANGE_IVORY,
    FRAME_ID,
    HAND_WHITE,
    RED,
    REFERENCE_004_MATERIAL_LINEAR,
    ROBOT_IVORY,
    SAME_SIDE,
    _angle_deg,
    outward_camera_rotation,
    same_direction_global_camera,
)


def test_contract_is_one_frame_and_same_side() -> None:
    assert FRAME_ID == 0
    assert SAME_SIDE == (0, 1)


def test_global_review_camera_preserves_exact_ego_rotation() -> None:
    ego = np.eye(4)
    ego[:3, :3] = np.asarray(((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)))
    ego[:3, 3] = [0.2, -0.4, 0.8]
    transform = same_direction_global_camera(ego)
    assert np.array_equal(transform[:3, :3], ego[:3, :3])
    assert np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])


def test_outward_camera_turns_front_view_around_robot_vertical_axis() -> None:
    front = np.asarray(
        ((0.0, 1.0, 0.0), (-0.8660254, 0.0, -0.5), (-0.5, 0.0, 0.8660254))
    )
    outward = outward_camera_rotation(front)
    assert np.allclose(outward[0], (0.0, -1.0, 0.0), atol=1e-7)
    assert np.allclose(outward[1], (0.8660254, 0.0, -0.5), atol=1e-7)
    assert np.allclose(outward[2], (0.5, 0.0, 0.8660254), atol=1e-7)
    assert np.isclose(np.linalg.det(outward), 1.0, atol=1e-7)


def test_review_palette_keeps_side_identity_and_white_hand_contrast() -> None:
    assert BLUE == (235, 130, 15)
    assert RED == (75, 86, 235)
    assert ROBOT_IVORY == (205, 225, 235)
    assert HAND_WHITE == (235, 242, 245)
    assert FLANGE_IVORY == (195, 218, 232)
    assert REFERENCE_004_MATERIAL_LINEAR["metallic"] == 0.72
    assert REFERENCE_004_MATERIAL_LINEAR["roughness"] == 0.27
    assert np.linalg.norm(np.asarray(BLUE, dtype=float) - RED) > 150.0
    assert np.mean(np.asarray(HAND_WHITE, dtype=float)) > 235.0
    assert np.linalg.norm(np.asarray(HAND_WHITE, dtype=float) - ROBOT_IVORY) > 20.0
    assert np.linalg.norm(np.asarray(FLANGE_IVORY, dtype=float) - ROBOT_IVORY) < 20.0


def test_thumb_web_angle_uses_true_vector_angle() -> None:
    assert np.isclose(_angle_deg(np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 1.0, 0.0))), 90.0)
