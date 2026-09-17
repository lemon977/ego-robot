from __future__ import annotations

import copy

import numpy as np
import pytest

from chaoyang.pipeline.contact_visible_tactile_hypothesis_v1 import (
    ContactHypothesisError,
    tactile_visible_surface_hypotheses,
)


def side() -> dict:
    grid = np.zeros((5, 4, 8), np.int16)
    grid[1, 0, 0] = 3
    return {
        "schema_version": "tactile-acquisition-aligned-frame-v2",
        "offline_source_valid": True,
        "offline_source_offset_ms": 2.0,
        "finger_grid_5x4x8": grid.tolist(),
        "valid_mask_5x4x8": np.ones_like(grid, bool).tolist(),
        "active_mask_5x4x8": np.ones_like(grid, bool).tolist(),
    }


def test_contact_requires_both_tactile_activity_and_visible_geometry() -> None:
    tactile = {"left": side(), "right": side()}
    tips = np.zeros((2, 5, 3), np.float64)
    tips[..., 2] = 0.505
    result = tactile_visible_surface_hypotheses(
        tactile=tactile, fingertip_xyz_m=tips,
        fingertip_observed=np.ones((2, 5), bool),
        visible_surface_centroid_m=np.asarray([0.0, 0.0, 0.5]),
        visible_surface_normal=np.asarray([0.0, 0.0, 1.0]),
    )
    supported = [row for row in result["hypotheses"]
                 if row["status"] == "TACTILE_SUPPORTED_HYPOTHESIS"]
    assert [(row["side"], row["finger"]) for row in supported] == [
        ("left", "index"), ("right", "index"),
    ]
    assert result["force_claim"] is False
    assert result["contact_ground_truth"] is False
    assert result["inputs"]["clean"] == "NOT_CONSUMED"


def test_contact_does_not_promote_tactile_without_visible_surface() -> None:
    tactile = {"left": side(), "right": side()}
    tips = np.zeros((2, 5, 3), np.float64)
    tips[..., 2] = 0.60
    result = tactile_visible_surface_hypotheses(
        tactile=tactile, fingertip_xyz_m=tips,
        fingertip_observed=np.ones((2, 5), bool),
        visible_surface_centroid_m=np.asarray([0.0, 0.0, 0.5]),
        visible_surface_normal=np.asarray([0.0, 0.0, 1.0]),
    )
    assert result["supported_count"] == 0


def test_contact_rejects_bad_tactile_timing() -> None:
    tactile = {"left": side(), "right": side()}
    broken = copy.deepcopy(tactile)
    broken["left"]["offline_source_offset_ms"] = 40.1
    with pytest.raises(ContactHypothesisError, match="40 ms"):
        tactile_visible_surface_hypotheses(
            tactile=broken,
            fingertip_xyz_m=np.zeros((2, 5, 3)),
            fingertip_observed=np.ones((2, 5), bool),
            visible_surface_centroid_m=np.zeros(3),
            visible_surface_normal=np.asarray([0.0, 0.0, 1.0]),
        )
