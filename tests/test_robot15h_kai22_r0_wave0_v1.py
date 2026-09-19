from __future__ import annotations

import numpy as np

from chaoyang.ops.run_hawor_bounded_parameter_successor import gate_failures
from chaoyang.ops.run_0915_robot15h_kai22_r0_wave0_v1 import (
    BOUNDED_THRESHOLDS,
    HANDS,
    MANO_JOINT_NAMES,
    ROOT,
    r0_from_hawor,
)
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


def _mano() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    chains = (
        (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16), (0, 17, 18, 19, 20),
    )
    for finger, chain in enumerate(chains):
        for step, joint in enumerate(chain[1:], 1):
            points[joint] = (0.012 * (finger - 2), 0.02 * step, 0.003 * step * step)
    return points


def test_bounded_gate_records_absent_side_without_exception() -> None:
    observed = {
        "observed_frames_raw": 0,
        "observed_frames_candidate": 0,
        "reprojection_p95_px": None,
        "raw_wrist_step_p95_mm": None,
        "candidate_wrist_step_p95_mm": None,
        "raw_all_joint_acceleration_p95_mm": None,
        "candidate_all_joint_acceleration_p95_mm": None,
        "raw_bone_length_cv_max": None,
        "candidate_bone_length_cv_max": None,
    }
    valid = {
        "observed_frames_raw": 10,
        "observed_frames_candidate": 10,
        "reprojection_p95_px": 2.0,
        "raw_wrist_step_p95_mm": 4.0,
        "candidate_wrist_step_p95_mm": 3.0,
        "raw_all_joint_acceleration_p95_mm": 4.0,
        "candidate_all_joint_acceleration_p95_mm": 3.0,
        "raw_bone_length_cv_max": 0.04,
        "candidate_bone_length_cv_max": 0.03,
    }
    value = {
        "sides": {"left": valid, "right": observed},
        "identity_switch_count": 0,
        "rotation_orthogonality_max": 0.0,
        "rotation_determinant_min": 1.0,
        "parameter_update_bounds": {
            "root_translation_update_max_mm": 0.0,
            "root_rotation_update_max_deg": 0.0,
            "pose_rotation_update_max_deg": 0.0,
            "beta_update_l2_max": 0.0,
        },
    }
    failures = gate_failures(value, BOUNDED_THRESHOLDS)
    assert failures == ["right:NO_OBSERVATION"]


def test_dynamic_r0_maps_anatomical_left_to_physical_right_and_keeps_missing_nan() -> None:
    frames = 5
    points = _mano()
    joints = np.full((2, frames, 21, 3), np.nan, np.float64)
    joints[0, 1] = points
    joints[0, 2] = points + np.asarray([0.001, 0.0, 0.0])
    observed = np.zeros((2, frames), bool)
    observed[0, 1:3] = True
    hawor = {
        "joints_3d_camera": joints,
        "observed": observed,
        "mano_joint_names": np.asarray(MANO_JOINT_NAMES),
        "anatomical_side_names": np.asarray(HANDS),
        "original_frame_indices": np.arange(frames),
    }
    result, arrays, _ = r0_from_hawor(
        hawor, np.arange(frames, dtype=np.float64) / 30.0,
        load_pinned_robot_assets(ROOT),
    )
    valid = arrays["valid_side_frame"]
    assert valid[:, 0].sum() == 0
    assert valid[:, 1].sum() == 2
    assert np.isnan(arrays["q22_init"][:, 0]).all()
    assert np.isfinite(arrays["q22_init"][valid]).all()
    assert result["side_contract"]["human_to_physical"] == {"left": "right", "right": "left"}
    assert result["coverage_preserved_exactly"] is True
