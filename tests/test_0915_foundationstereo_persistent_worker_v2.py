from __future__ import annotations

import inspect

import numpy as np

from chaoyang.ops import run_0915_foundationstereo_persistent_worker_v2 as subject


def test_physical_left_depth_uses_unflipped_x_plus_disparity_domain() -> None:
    flipped = np.full((2, 8), 2.0, np.float32)
    disparity, depth, valid = subject.physical_left_depth(
        flipped, focal_px=10.0, baseline_m=0.2,
    )
    np.testing.assert_allclose(disparity, 2.0)
    assert valid[:, :6].all()
    assert not valid[:, 6:].any()
    np.testing.assert_allclose(depth[valid], 1.0)


def test_depth_quality_is_dynamic_and_fail_closed() -> None:
    rows = [
        {"valid_pixels": 6_000, "valid_fraction": 0.5, "depth_p50_m": 0.4}
        for _ in range(10)
    ]
    assert subject.quality_gate(
        rows, frame_count=10, rectification_passed=True,
    )["passed"] is True
    assert subject.quality_gate(
        rows, frame_count=11, rectification_passed=True,
    )["passed"] is False


def test_worker_has_no_fixed_001_or_379_identity() -> None:
    source = inspect.getsource(subject)
    assert "frame_count != 379" not in source
    assert 'session_id": "get_potato_chips_0915_001"' not in source
    assert "PRESENT_PRESERVED_NOT_CONSUMED" not in source
