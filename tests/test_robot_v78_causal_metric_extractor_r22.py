from __future__ import annotations

import numpy as np

from chaoyang.ops.extract_robot_v78_causal_metrics_r22 import extract_metrics


def _transforms(translations: list[tuple[float, float, float]]) -> np.ndarray:
    result = np.repeat(np.eye(4)[None, :, :], len(translations), axis=0)
    result[:, :3, 3] = translations
    return result


def test_extract_metrics_does_not_bridge_invalid_or_nonconsecutive_steps() -> None:
    frames = np.array([10, 11, 13, 14])
    q = np.zeros((4, 2, 7))
    q[1, 0, 0] = 1.0
    q[2, 0, 0] = 10.0
    q[3, 0, 0] = 11.0
    wrists = np.repeat(np.eye(4)[None, None, :, :], 8, axis=0).reshape(4, 2, 4, 4)
    wrists[:, 0, 0, 3] = [0.0, 1.0, 10.0, 11.0]
    valid = np.ones((2, 4), dtype=bool)
    valid[0, 2] = False
    object_poses = _transforms([(0.0, 0.0, 0.0)] * 4)
    rows, summary = extract_metrics(
        {
            "q_arm": q,
            "T_actual_hand_root_world": wrists,
            "valid_side_frame": valid,
            "source_frames": frames,
        },
        {
            "frame_indices": frames,
            "valid": np.ones(4, dtype=bool),
            "observed": np.ones(4, dtype=bool),
            "T_object_to_world": object_poses,
        },
    )

    assert summary["sides"]["left"]["q_arm_consecutive_valid_total_variation_rad"] == 1.0
    assert summary["sides"]["left"]["actual_wrist_consecutive_valid_path_length_m"] == 1.0
    assert rows[2]["left_q_consecutive_total_variation_step_rad"] is None
    assert rows[3]["left_q_consecutive_total_variation_step_rad"] is None
    assert summary["sides"]["left"]["ik_valid_frames"] == 3


def test_extract_metrics_uses_direct_same_frame_object_and_reports_approach() -> None:
    frames = np.array([0, 1, 2])
    wrists = np.repeat(np.eye(4)[None, None, :, :], 6, axis=0).reshape(3, 2, 4, 4)
    wrists[:, 0, 0, 3] = [3.0, 2.0, 2.5]
    rows, summary = extract_metrics(
        {
            "q_arm": np.zeros((3, 2, 7)),
            "T_actual_hand_root_world": wrists,
            "valid_side_frame": np.ones((2, 3), dtype=bool),
            "source_frames": frames,
        },
        {
            "frame_indices": frames,
            "valid": np.ones(3, dtype=bool),
            "observed": np.array([True, True, False]),
            "T_object_to_world": _transforms([(0.0, 0.0, 0.0)] * 3),
        },
    )

    left = summary["sides"]["left"]
    assert left["wrist_object_same_frame_count"] == 2
    assert left["wrist_object_first_distance_m"] == 3.0
    assert left["wrist_object_min_distance_m"] == 2.0
    assert left["wrist_object_approach_delta_m"] == 1.0
    assert rows[2]["left_wrist_object_distance_m"] is None
