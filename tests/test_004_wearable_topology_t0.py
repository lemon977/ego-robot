import numpy as np

from tools.diagnose_004_wearable_topology_t0 import (
    adjacent_iou,
    component_and_wrist_metrics,
    compress_intervals,
    contact_phase,
    hand_geometry,
    posthoc_validation,
    quantile_summary,
)


def geometry():
    joints = np.full((21, 2), np.nan)
    joints[0] = [50, 60]
    joints[5] = [42, 45]
    joints[9] = [48, 42]
    joints[13] = [54, 44]
    joints[17] = [58, 48]
    joints[1:5] = [[40, 40], [38, 35], [36, 30], [34, 25]]
    return hand_geometry(joints, 100, 100)


def test_adjacent_iou_and_empty_rule():
    a = np.zeros((4, 4), bool)
    b = a.copy()
    assert adjacent_iou(a, b) == 1.0
    a[0, 0] = True
    b[0, :2] = True
    assert adjacent_iou(a, b) == 0.5


def test_components_and_read_only():
    mask = np.zeros((100, 100), bool)
    mask[20:50, 20:50] = True
    mask[60:90, 60:90] = True
    before = mask.copy()
    out = component_and_wrist_metrics(mask, geometry())
    assert out["topology_state"] == "MULTI_SIGNIFICANT"
    assert out["significant_component_count_1pct_min32"] == 2
    assert np.array_equal(mask, before)


def test_one_component_and_geometry_unavailable():
    mask = np.zeros((40, 40), bool)
    mask[3:30, 4:31] = True
    out = component_and_wrist_metrics(mask, None)
    assert out["topology_state"] == "ONE_SIGNIFICANT"
    assert out["wrist_band_coverage"] is None


def test_helpers():
    assert compress_intervals([5, 2, 3, 9, 9]) == [[2, 3], [5, 5], [9, 9]]
    assert quantile_summary([0.0, 1.0])["min"] == 0.0
    assert contact_phase(0.0, 0.1) == "NO_PROJECTED_OVERLAP"
    assert contact_phase(0.1, 0.0) == "PROJECTED_OVERLAP_ENTRY"
    assert contact_phase(0.2, 0.1) == "PROJECTED_OVERLAP_RISING"


def test_posthoc_windows_do_not_define_detection():
    records = [{"frame_index": i, "topology_state": "MULTI_SIGNIFICANT" if i in (63, 64, 96) else "ONE_SIGNIFICANT"} for i in range(460)]
    out = posthoc_validation(records)
    assert out["true_positive"] == 3
    assert out["false_positive"] == 0
    assert out["expected_frame_count"] == 39
