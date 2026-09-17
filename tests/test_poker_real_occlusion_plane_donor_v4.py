import numpy as np

from chaoyang.ops.run_poker_real_occlusion_plane_donor_v4 import _identity_checks


def _flow() -> dict:
    return {"flow_runtime_pass": True}


def test_identity_guard_accepts_same_instance_geometry() -> None:
    predicted = np.zeros((100, 120), bool)
    predicted[30:60, 40:70] = True
    candidate = predicted.copy()
    flow = _flow()
    checks = _identity_checks(candidate, predicted, flow)
    assert all(checks.values())
    assert flow["candidate_vs_flow_iou"] == 1.0


def test_identity_guard_rejects_visually_similar_sibling_location() -> None:
    predicted = np.zeros((100, 120), bool)
    predicted[30:60, 15:45] = True
    sibling = np.zeros_like(predicted)
    sibling[30:60, 75:105] = True
    checks = _identity_checks(sibling, predicted, _flow())
    assert checks["consecutive_flow_valid"]
    assert not checks["candidate_centroid_agrees_with_flow"]
    assert not checks["candidate_mask_iou_with_flow"]
