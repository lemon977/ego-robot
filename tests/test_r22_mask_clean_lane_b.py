import numpy as np

from chaoyang.ops.run_r22_mask_clean_lane_b import classify_temporal_donor, frozen_failure_canaries


def test_donor_classifier_rejects_future_invalid_and_object():
    shape = (2, 3)
    m_write = np.ones(shape, bool)
    kind = np.ones(shape, np.uint8)
    frame = np.array([[0, 2, -1], [0, 0, 0]], np.int32)
    x = np.array([[0, 0, 0], [1, 2, 8]], np.int32)
    y = np.array([[0, 0, 0], [1, 1, 1]], np.int32)
    objects = [np.array([[False, False, False], [False, True, False]]), np.zeros(shape, bool)]
    code, counts = classify_temporal_donor(1, m_write, kind, frame, x, y, objects)
    assert code.tolist() == [[1, 2, 3], [4, 1, 3]]
    assert counts == {
        "temporal": 6,
        "causal_support_surface_unknown": 2,
        "future_rejected": 1,
        "invalid_coordinate_rejected": 2,
        "task_object_rejected": 1,
    }


def test_failure_canaries_require_one_chips_role_and_one_poker_object(monkeypatch):
    monkeypatch.setattr("chaoyang.ops.run_r22_mask_clean_lane_b.ref", lambda path, verify=None: verify)
    selection = {"selections": [
        {"stage": "ROLE_MASK", "cluster": "REENTRY", "canary": {"task": "chips", "session_id": "c", "grade": "C", "result": {"path": "/c", "bytes": 1, "sha256": "x"}}},
        {"stage": "OBJECT_MASK", "cluster": "REENTRY", "canary": {"task": "poker", "session_id": "p", "grade": "C", "result": {"path": "/p", "bytes": 1, "sha256": "y"}}},
    ]}
    rows = frozen_failure_canaries(selection)
    assert [(x["stage"], x["task"]) for x in rows] == [("ROLE_MASK", "chips"), ("OBJECT_MASK", "poker")]
