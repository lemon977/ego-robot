import numpy as np

from chaoyang.ops.run_contact10_selected_camera_r23 import TemporalContact, contact_state


def test_invalid_observation_clears_velocity_history() -> None:
    tracker = TemporalContact()
    pose = np.eye(4)
    size = np.array([0.06, 0.09, 0.001])
    key = (0, 1)
    first = tracker.update(key, 10, 30.0, np.array([0.0, 0.0, 0.01]), pose, size)
    second = tracker.update(key, 11, 30.0, np.array([0.0, 0.0, 0.008]), pose, size)
    assert first["radial_velocity_mm_s"] is None
    assert second["radial_velocity_mm_s"] is not None
    tracker.update(key, 12, 30.0, None, None, None)
    reentry = tracker.update(key, 13, 30.0, np.array([0.0, 0.0, 0.004]), pose, size)
    assert reentry["radial_velocity_mm_s"] is None
    assert reentry["tangential_velocity_mm_s"] is None
    assert reentry["dt_s"] is None


def test_skipped_source_frame_does_not_fake_single_frame_velocity() -> None:
    tracker = TemporalContact()
    pose = np.eye(4)
    size = np.array([0.06, 0.09, 0.001])
    tracker.update((1, 4), 10, 30.0, np.array([0.0, 0.0, 0.01]), pose, size)
    skipped = tracker.update((1, 4), 15, 30.0, np.array([0.0, 0.0, 0.005]), pose, size)
    assert skipped["radial_velocity_mm_s"] is None
    assert skipped["dt_s"] is None
    adjacent = tracker.update((1, 4), 16, 30.0, np.array([0.0, 0.0, 0.004]), pose, size)
    assert adjacent["dt_s"] == 1 / 30.0


def test_coordinate_rotation_changes_proximity_and_contact_claim_stays_bounded() -> None:
    pose = np.eye(4)
    size = np.array([0.06, 0.09, 0.001])
    point = np.array([0.04, 0.0, 0.0])
    old = TemporalContact().update((0, 0), 0, 30.0, point, pose, size)
    selected_pose = pose.copy()
    selected_pose[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    new = TemporalContact().update((0, 0), 0, 30.0, point, selected_pose, size)
    assert old["distance_mm"] != new["distance_mm"]
    assert contact_state(25.0, None, None, False) == "UNKNOWN"
