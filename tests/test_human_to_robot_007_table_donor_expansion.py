from __future__ import annotations

import inspect

from chaoyang.ops.run_human_to_robot_007_device_donor_probe import fit_heldout_homography
from chaoyang.ops.run_human_to_robot_007_table_donor_expansion import (
    DONORS, TARGET, choose_best, main,
)


def test_fixed_grid_excludes_failed_four_and_keeps_denominator() -> None:
    assert TARGET == 184
    assert DONORS == tuple(range(8, 369, 8))
    assert len(DONORS) == 46
    assert not {0, 100, 250, 377}.intersection(DONORS)


def test_only_geometry_qualified_table_support_can_win() -> None:
    rows = [
        {"donor_frame": 8, "geometry_pass": False, "table_write_pixels": 999},
        {"donor_frame": 16, "geometry_pass": True, "table_write_pixels": 0},
        {"donor_frame": 32, "geometry_pass": True, "table_write_pixels": 15},
        {"donor_frame": 24, "geometry_pass": True, "table_write_pixels": 15},
    ]
    assert choose_best(rows) == 24
    assert choose_best(rows[:2]) is None


def test_existing_heldout_geometry_gate_is_not_lowered() -> None:
    source = inspect.getsource(fit_heldout_homography)
    assert 'summary["fit_inliers"] >= 25' in source
    assert 'summary["holdout_inliers"] >= 8' in source
    assert 'summary["holdout_median_px"] <= 2.0' in source
    assert 'summary["holdout_p95_px"] <= 4.0' in source
    runner = inspect.getsource(main)
    assert "fit_heldout_homography(donor, raw, donor_table, target_table)" in runner
    assert "support = warped_table & write" in runner
    assert "if DEST.exists()" in runner
