from __future__ import annotations

import inspect

import numpy as np

from chaoyang.ops.run_human_to_robot_007_multidonor_table_canary import (
    DONORS, color_offset, main, median_supported,
)


def test_frozen_donor_cohort_and_model_exclusion() -> None:
    assert len(DONORS) == 19
    source = inspect.getsource(main)
    assert "if donor_frame == frame" in source
    assert "table_mask(donor, human | role" in source
    assert "fit_heldout_homography" in source
    assert "& ~protect & ~object_mask" in source
    assert "candidate[~write] != raw[~write]" in source


def test_masked_median_ignores_invalid_donor() -> None:
    colors = [np.full((2, 2, 3), v, dtype=np.uint8) for v in (30, 40, 200, 250)]
    masks = [np.ones((2, 2), bool) for _ in colors]
    masks[-1][0, 0] = False
    median, count = median_supported(colors, masks)
    assert count[0, 0] == 3
    assert median[0, 0, 0] == 40
    assert count[1, 1] == 4
    assert median[1, 1, 0] == 120


def test_color_offset_uses_independent_holdout_and_is_bounded() -> None:
    target = np.full((64, 64, 3), 100, dtype=np.uint8)
    donor = np.full_like(target, 75)
    corrected, detail = color_offset(donor, target, np.ones((64, 64), bool),
                                     np.ones((64, 64), bool))
    assert corrected is not None
    assert detail["train_pixels"] >= 1000 and detail["holdout_pixels"] >= 250
    assert detail["offset_bgr"] == [25.0, 25.0, 25.0]
    assert detail["holdout_max_channel_p95"] == 0
