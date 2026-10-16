import numpy as np
import pytest

from chaoyang.ops import run_robot100_anatomical_identity_v8 as subject
from chaoyang.pipeline import full_robot_review_inward_v3 as solver


def test_restore_anatomical_identity_swaps_only_side_inputs():
    source = {
        "human_to_physical": np.asarray([1, 0]),
        "q22": np.asarray([[[10], [20]], [[11], [21]]]),
        "finger_valid": np.asarray([[True, False], [False, True]]),
        "wrist_valid": np.asarray([[False, True], [True, False]]),
        "relative_wrist_T": np.arange(2 * 2 * 4 * 4).reshape(2, 2, 4, 4),
        "frame_id": np.asarray([0, 1]),
    }
    restored = subject.restore_anatomical_identity(source)
    np.testing.assert_array_equal(restored["human_to_physical"], [0, 1])
    for name in subject.SIDE_INPUT_FIELDS:
        np.testing.assert_array_equal(restored[name], source[name][:, [1, 0]])
    np.testing.assert_array_equal(restored["frame_id"], source["frame_id"])


def test_restore_refuses_unknown_predecessor_mapping():
    with pytest.raises(RuntimeError, match="PREDECESSOR_SIDE_MAPPING_DRIFT"):
        subject.restore_anatomical_identity({"human_to_physical": np.asarray([0, 1])})


def test_validate_motion_can_require_identity_without_weakening_default():
    eye = np.broadcast_to(np.eye(4), (2, 2, 4, 4)).copy()
    arrays = {
        "q22": np.zeros((2, 2, 22)),
        "finger_valid": np.ones((2, 2), dtype=bool),
        "wrist_valid": np.ones((2, 2), dtype=bool),
        "relative_wrist_T": eye,
        "timestamp_ns": np.asarray([0, 1_000_000], dtype=np.int64),
        "frame_id": np.asarray([0, 1], dtype=np.int64),
        "human_to_physical": np.asarray([0, 1], dtype=np.int64),
    }
    solver.validate_motion(arrays, expected_human_to_physical=[0, 1])
    with pytest.raises(ValueError, match="SIDE_MAPPING_MISMATCH"):
        solver.validate_motion(arrays)


def test_expected_mapping_must_be_bijection():
    arrays = {
        "q22": np.zeros((2, 2, 22)),
        "finger_valid": np.ones((2, 2), dtype=bool),
        "wrist_valid": np.ones((2, 2), dtype=bool),
        "relative_wrist_T": np.broadcast_to(np.eye(4), (2, 2, 4, 4)).copy(),
        "timestamp_ns": np.asarray([0, 1_000_000], dtype=np.int64),
        "frame_id": np.asarray([0, 1], dtype=np.int64),
        "human_to_physical": np.asarray([0, 1], dtype=np.int64),
    }
    with pytest.raises(ValueError, match="EXPECTED_SIDE_MAPPING_NOT_BIJECTION"):
        solver.validate_motion(arrays, expected_human_to_physical=[0, 0])
