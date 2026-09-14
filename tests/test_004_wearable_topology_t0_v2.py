import numpy as np

from tools.diagnose_004_wearable_topology_t0_v2 import (
    boundary_robust_band,
    finite_hand_geometry,
    narrowed_posthoc,
)


def projected_joints(wrist=(50.0, 50.0)):
    joints = np.full((21, 2), np.nan)
    joints[0] = wrist
    joints[5] = [45, 40]
    joints[9] = [50, 38]
    joints[13] = [55, 40]
    joints[17] = [58, 44]
    joints[1:5] = [[45, 35], [43, 30], [41, 25], [39, 20]]
    return joints


def test_band_measured_without_second_component():
    mask = np.ones((100, 100), bool)
    metric = boundary_robust_band(mask, finite_hand_geometry(projected_joints()))
    assert metric["boundary_robust_band_state"] == "MEASURED_ON_VISIBLE_IMAGE_INTERSECTION"
    assert metric["boundary_robust_wrist_band_coverage_proxy"] == 1.0


def test_outside_wrist_still_measured_on_visible_intersection():
    mask = np.ones((100, 100), bool)
    joints = projected_joints(wrist=(-2.0, 50.0))
    metric = boundary_robust_band(mask, finite_hand_geometry(joints))
    assert metric["projected_wrist_location"] == "OUTSIDE_IMAGE"
    assert metric["boundary_robust_band_state"] == "MEASURED_ON_VISIBLE_IMAGE_INTERSECTION"
    assert metric["boundary_robust_wrist_band_coverage_proxy"] == 1.0


def test_missing_geometry_remains_unmeasured():
    metric = boundary_robust_band(np.zeros((20, 20), bool), None)
    assert metric["boundary_robust_band_state"] == "GEOMETRY_UNAVAILABLE"
    assert metric["boundary_robust_wrist_band_coverage_proxy"] is None


def test_outside_alerts_are_pending_not_false_positive():
    records = [
        {"frame_index": i, "topology_state": "MULTI_SIGNIFICANT" if i in (64, 128, 230) else "ONE_SIGNIFICANT"}
        for i in range(460)
    ]
    out = narrowed_posthoc(records)
    assert out["outside_known_windows_disposition"]["VISUAL_REVIEW_SUPPORTS_ALERT"] == [230]
    assert out["outside_known_windows_disposition"]["PENDING_VISUAL_REVIEW"] == [[128, 128]]
