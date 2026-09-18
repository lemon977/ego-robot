from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from chaoyang.ops import run_hawor_short_gap_completion_v1 as module


def test_internal_false_runs_excludes_edges_and_preserves_two_frame_gap() -> None:
    mask = np.asarray([False, True, False, True, False, False, True, False])
    assert module.internal_false_runs(mask) == [(2, 3), (4, 6)]


def test_gap_gate_rejects_long_or_fast_motion() -> None:
    metrics = {
        "endpoint_confidence_min": 0.8,
        "wrist_world_step_mm": 2.0,
        "max_joint_world_step_mm": 4.0,
        "max_joint_2d_step_px": 8.0,
    }
    assert module.gap_is_eligible(2, 4, metrics) == (True, [])
    eligible, failures = module.gap_is_eligible(2, 5, metrics)
    assert not eligible and failures == ["GAP_TOO_LONG"]
    metrics["max_joint_2d_step_px"] = 80.0
    eligible, failures = module.gap_is_eligible(2, 4, metrics)
    assert not eligible and "JOINT_IMAGE_MOTION_HIGH" in failures


def test_pair_slerp_stays_on_so3_and_hits_midpoint() -> None:
    left = Rotation.from_euler("z", 0.0).as_matrix()
    right = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    middle = module.pair_slerp(left, right, 0.5)
    assert np.allclose(middle.T @ middle, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(middle), 1.0)
    assert np.isclose(Rotation.from_matrix(middle).as_euler("zyx", degrees=True)[0], 45.0)


def test_interpolate_gap_uses_world_translation_and_preserves_detector_boxes() -> None:
    frames = 3
    c2w = np.repeat(np.eye(4)[None], frames, axis=0)
    c2w[1, 0, 3] = 1.0
    source = {
        "c2w": c2w,
        "detector_boxes_xyxy": np.asarray([[[0, 0, 2, 2], [np.nan] * 4, [2, 2, 4, 4]]], dtype=float),
    }
    parameters = {
        "root_translation_camera": np.asarray([[[2, 0, 1], [np.nan] * 3, [2, 0, 1]]], dtype=float),
        "root_orient_camera": np.asarray([[np.eye(3), np.full((3, 3), np.nan), np.eye(3)]], dtype=float),
        "hand_pose_rotmat": np.asarray([[
            np.repeat(np.eye(3)[None], 15, axis=0),
            np.full((15, 3, 3), np.nan),
            np.repeat(np.eye(3)[None], 15, axis=0),
        ]], dtype=float),
        "betas": np.asarray([[[0] * 10, [np.nan] * 10, [2] * 10]], dtype=float),
    }
    continuity_boxes = source["detector_boxes_xyxy"].copy()
    module.interpolate_gap(source, parameters, continuity_boxes, 0, 1, 2)
    # Endpoint world translations are both x=2.  The moving camera at frame 1
    # therefore requires camera-space x=1, not naïve camera-space x=2.
    assert np.allclose(parameters["root_translation_camera"][0, 1], [1, 0, 1])
    assert np.allclose(parameters["betas"][0, 1], np.ones(10))
    assert np.allclose(continuity_boxes[0, 1], [1, 1, 3, 3])
