import numpy as np

from chaoyang.governance.build_robot100_visual_audit_v1 import selected_frames


def test_selected_frames_are_unique_bounded_and_include_anomalies() -> None:
    count = 20
    arrays = {
        "frame_id": np.arange(count),
        "posture_projected": np.zeros((count, 2), dtype=bool),
        "posture_projection_alpha": np.zeros((count, 2), dtype=float),
        "position_residual_mm": np.zeros((count, 2), dtype=float),
        "wrist_valid": np.ones((count, 2), dtype=bool),
        "time_edge_valid": np.ones(count, dtype=bool),
        "q_arm": np.zeros((count, 2, 7), dtype=float),
    }
    arrays["posture_projected"][4:9, 0] = True
    arrays["posture_projection_alpha"][6, 0] = 0.8
    arrays["position_residual_mm"][7, 1] = 200.0
    arrays["q_arm"][12:, 0, 0] = 0.5
    frames = selected_frames(arrays)
    assert len(frames) == len(set(frames)) <= 9
    assert frames == sorted(frames)
    assert all(0 <= frame < count for frame in frames)
    assert {0, count - 1, 6, 7, 12}.issubset(frames)


def test_selected_frames_handles_single_frame() -> None:
    arrays = {
        "frame_id": np.array([0]),
        "posture_projected": np.zeros((1, 2), dtype=bool),
        "posture_projection_alpha": np.zeros((1, 2), dtype=float),
        "position_residual_mm": np.zeros((1, 2), dtype=float),
        "wrist_valid": np.ones((1, 2), dtype=bool),
        "time_edge_valid": np.ones(1, dtype=bool),
        "q_arm": np.zeros((1, 2, 7), dtype=float),
    }
    assert selected_frames(arrays) == [0]
