from __future__ import annotations

import numpy as np
import pytest

from tools import run_robot_motion_transfer_arm_canary_v3 as arm3


def test_first_observed_is_independent_per_side() -> None:
    valid = np.asarray(
        [
            [False, False, True, True, False],
            [True, True, True, False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(arm3.first_observed_per_side(valid), np.asarray([2, 0]))


def test_missing_entire_side_and_wrong_shape_fail_closed() -> None:
    with pytest.raises(ValueError, match="no observed frame"):
        arm3.first_observed_per_side(np.asarray([[False, False], [True, False]], dtype=bool))
    with pytest.raises(ValueError, match="expected valid shape"):
        arm3.first_observed_per_side(np.ones((3, 2), dtype=bool))
