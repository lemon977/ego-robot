"""Boundary tests for the fixed cross-eye source-only diagnostic."""
from __future__ import annotations

import numpy as np
import pytest

from chaoyang.ops.run_human_to_robot_007_cross_eye_source_probe import (
    FRAMES, domain_residual, right_table_proxy,
)
from chaoyang.ops.finalize_human_to_robot_007_cross_eye_source_probe import verify_result


def test_frame_set_is_frozen() -> None:
    assert FRAMES == (181, 184, 192, 196)


def test_eye_binding_exposes_wrong_eye_or_frame() -> None:
    pinned = np.zeros((960, 1280, 3), np.uint8)
    assert domain_residual(pinned.copy(), pinned)["mean_abs_channel"] == 0
    wrong = np.full_like(pinned, 50)
    assert domain_residual(wrong, pinned)["mean_abs_channel"] > 2.5
    with pytest.raises(ValueError, match="PINNED_LEFT_DOMAIN_MISMATCH"):
        domain_residual(pinned[:500], pinned)


def test_proxy_is_not_all_frame_or_human_skin() -> None:
    image = np.full((960, 1280, 3), (75, 75, 75), np.uint8)
    image[400:500, 500:600] = (180, 190, 220)
    mask = right_table_proxy(image)
    assert mask[600, 600] > 0
    assert mask[450, 550] == 0


def test_finalizer_rejects_nonfrozen_result() -> None:
    with pytest.raises(RuntimeError, match="CROSS_EYE_FRAME_SET_DRIFT"):
        verify_result({"task_id": "wrong", "frames": []})
