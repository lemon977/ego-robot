from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline.human_contact_input_adapter_v1 import (  # noqa: E402
    ContactInputAdapterError,
    ContactState,
    build_direct_touch_evidence,
    freeze_four_touch_diagnostic_frames,
    signed_distance_points_to_oriented_box,
)


def test_oriented_box_signed_distance_has_metric_sign() -> None:
    transform = np.eye(4)
    size = np.asarray([0.088, 0.063, 0.001])
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0025], [0.05, 0.0, 0.0]])
    distance = signed_distance_points_to_oriented_box(points, transform, size)
    np.testing.assert_allclose(distance, [-0.0005, 0.002, 0.006], atol=1e-12)


def test_direct_touch_never_fills_invalid_object_pose() -> None:
    joints = np.zeros((2, 4, 21, 3), dtype=np.float64)
    observed = np.ones((2, 4), dtype=np.bool_)
    poses = np.repeat(np.eye(4)[None, None], 4, axis=0)
    valid = np.asarray([[True], [False], [True], [True]], dtype=np.bool_)
    sizes = np.asarray([[0.088, 0.063, 0.001]])
    output = build_direct_touch_evidence(
        joints, observed, poses, valid, sizes, tip_joint_indices=[4, 8, 12, 16, 20]
    )
    assert output.contact_state[0, 0, 0] == int(ContactState.TOUCH)
    assert output.contact_state[0, 1, 0] == int(ContactState.UNKNOWN)
    assert np.isnan(output.tip_signed_distance_m[:, 1]).all()
    assert not output.finger_contact[:, 1].any()


def test_far_direct_geometry_stays_unknown_not_no_contact_truth() -> None:
    joints = np.full((1, 1, 21, 3), 1.0, dtype=np.float64)
    output = build_direct_touch_evidence(
        joints,
        np.ones((1, 1), dtype=np.bool_),
        np.eye(4)[None, None],
        np.ones((1, 1), dtype=np.bool_),
        np.asarray([[0.1, 0.1, 0.1]]),
        tip_joint_indices=[4],
    )
    assert output.contact_state.item() == int(ContactState.UNKNOWN)
    assert output.contact_confidence.item() == 0.0


def test_four_frame_freeze_spans_separate_touch_segments() -> None:
    state = np.zeros((2, 20, 1), dtype=np.uint8)
    state[1, 3:9, 0] = int(ContactState.TOUCH)
    state[1, 15, 0] = int(ContactState.TOUCH)
    assert freeze_four_touch_diagnostic_frames(state) == (3, 6, 8, 15)


def test_four_frame_freeze_fails_closed_when_insufficient() -> None:
    state = np.zeros((1, 5, 1), dtype=np.uint8)
    state[0, 1:4, 0] = int(ContactState.TOUCH)
    with pytest.raises(ContactInputAdapterError, match="fewer than four"):
        freeze_four_touch_diagnostic_frames(state)
