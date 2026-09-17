from __future__ import annotations

import numpy as np

from chaoyang.pipeline.object6d_visible_surface_v1 import (
    backproject_registered_visible_surface,
    estimate_visible_object,
)


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = 20, 30
    mask = np.ones((height, width), bool)
    depth = np.full((height, width), 0.5, np.float32)
    valid = np.ones_like(mask)
    intrinsics = np.asarray([[20.0, 0.0, 14.5], [0.0, 20.0, 9.5], [0.0, 0.0, 1.0]])
    homography = np.eye(3)
    return mask, depth, valid, intrinsics, homography


def test_registered_visible_surface_uses_only_masked_depth_pixels() -> None:
    mask, depth, valid, intrinsics, homography = fixture()
    mask[:, :15] = False
    points = backproject_registered_visible_surface(
        mask_primary=mask, depth_m=depth, depth_valid=valid,
        depth_intrinsics=intrinsics, h_depth_to_primary=homography,
    )
    assert len(points.points_depth_camera_m) == 20 * 15
    assert np.all(points.primary_pixels_xy[:, 0] >= 15)


def test_card_plane_is_observed_only_and_metric_consistent() -> None:
    mask, depth, valid, intrinsics, homography = fixture()
    result = estimate_visible_object(
        task="playing_cards", mask_primary=mask, depth_m=depth,
        depth_valid=valid, depth_intrinsics=intrinsics,
        h_depth_to_primary=homography,
    )
    assert result["status"] == "PASS_VISIBLE_SURFACE"
    assert result["geometry_kind"] == "VISIBLE_CARD_OR_STACK_PLANE"
    assert result["hidden_shape_inferred"] is False
    assert result["plane_residual_p90_m"] < 1e-9


def test_occluded_or_sparse_surface_remains_invalid() -> None:
    mask, depth, valid, intrinsics, homography = fixture()
    mask[:] = False
    mask[:3, :3] = True
    result = estimate_visible_object(
        task="potato_chips", mask_primary=mask, depth_m=depth,
        depth_valid=valid, depth_intrinsics=intrinsics,
        h_depth_to_primary=homography,
    )
    assert result["status"] == "INVALID_INSUFFICIENT_VISIBLE_SURFACE"
    assert result["hidden_shape_inferred"] is False
