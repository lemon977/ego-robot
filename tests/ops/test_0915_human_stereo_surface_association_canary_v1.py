from __future__ import annotations

import numpy as np

from chaoyang.ops.run_0915_human_stereo_surface_association_canary_v1 import (
    center_connected_surface_sample,
    fit_surface_alignment,
)


def _sample_inputs() -> dict[str, object]:
    depth = np.ones((480, 640), np.float32) * 0.50
    valid = np.zeros((480, 640), bool)
    lr = np.zeros((480, 640), bool)
    residual = np.ones((480, 640), np.float32) * np.nan
    hand = np.zeros((960, 1280), bool)
    hand[468:493, 628:653] = True
    objects = np.zeros_like(hand)
    return {
        "source_pixel_uv_1280": [640.5, 480.5],
        "depth_m": depth,
        "depth_valid": valid,
        "lr_consistent": lr,
        "lr_residual_px": residual,
        "intrinsics": np.asarray([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]]),
        "hand_mask_1280": hand,
        "sleeve_mask_1280": None,
        "object_union_1280": objects,
        "associated_hand": "right",
        "associated_finger": "index",
    }


def _admit_depth_pixel(inputs: dict[str, object], x: int, y: int, value: float) -> None:
    depth = inputs["depth_m"]
    valid = inputs["depth_valid"]
    lr = inputs["lr_consistent"]
    residual = inputs["lr_residual_px"]
    hand = inputs["hand_mask_1280"]
    assert isinstance(depth, np.ndarray) and isinstance(valid, np.ndarray)
    assert isinstance(lr, np.ndarray) and isinstance(residual, np.ndarray)
    assert isinstance(hand, np.ndarray)
    depth[y, x] = value
    valid[y, x] = True
    lr[y, x] = True
    residual[y, x] = 0.2
    sy, sx = int(round(2.0 * y + 0.5)), int(round(2.0 * x + 0.5))
    hand[max(0, sy - 1):sy + 2, max(0, sx - 1):sx + 2] = True


def test_center_connected_sampler_rejects_a_disconnected_depth_branch() -> None:
    inputs = _sample_inputs()
    # The projected anchor is at depth pixel (320, 240).  Its connected component
    # has eight pixels at 0.50 m; a larger but disconnected 0.62 m component must
    # not pull the sample to a different visible surface.
    for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1)]:
        _admit_depth_pixel(inputs, 320 + dx, 240 + dy, 0.50)
    for dx in range(2, 5):
        for dy in range(-2, 3):
            _admit_depth_pixel(inputs, 320 + dx, 240 + dy, 0.62)
    result = center_connected_surface_sample(**inputs)
    assert result["status"] == "OBSERVED_VISIBLE_SURFACE"
    assert result["fingertip_surface_observation"] is True
    assert abs(result["surface_point_xyz"][2] - 0.50) < 1e-6
    assert result["association_quality"]["center_connected_component_required"] is True


def test_center_connected_sampler_does_not_cross_object_pixels() -> None:
    inputs = _sample_inputs()
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            _admit_depth_pixel(inputs, 320 + dx, 240 + dy, 0.50)
    objects = inputs["object_union_1280"]
    assert isinstance(objects, np.ndarray)
    objects[468:493, 628:653] = True
    result = center_connected_surface_sample(**inputs)
    assert result["status"] == "UNKNOWN"
    assert result["fingertip_surface_observation"] is False


def test_surface_alignment_uses_frozen_frame_split_and_never_contact_rows() -> None:
    rows = []
    for frame in range(100):
        for hand_index, hand in enumerate(("left", "right")):
            source = 0.30 + 0.002 * frame + 0.001 * hand_index
            rows.append({
                "frame_id": frame,
                "hand_id": hand,
                "whole_frame_hand_contact_excluded": frame in {31, 32, 33},
                "support_count": 40,
                "mano_surface_depth_m": source,
                "stereo_surface_depth_m": 0.95 * source + 0.012,
            })
    result = fit_surface_alignment(rows)
    assert result["status"] == "PASS_DEVELOPMENT_ALIGNMENT"
    assert result["metric_translation_authorized"] is True
    assert result["contact_or_object_fit_used"] is False
    assert set(result["heldout"]["frame_ids"]) == set(range(0, 100, 5))
    assert result["train_row_count"] == 2 * (80 - 3)
    assert result["holdout_row_count"] == 40


def test_surface_alignment_rejects_holdout_tail_without_relaxing_gate() -> None:
    rows = []
    for frame in range(100):
        for hand in ("left", "right"):
            source = 0.30 + 0.002 * frame
            target = source + (0.05 if frame % 5 == 0 else 0.0)
            rows.append({
                "frame_id": frame,
                "hand_id": hand,
                "whole_frame_hand_contact_excluded": False,
                "support_count": 40,
                "mano_surface_depth_m": source,
                "stereo_surface_depth_m": target,
            })
    result = fit_surface_alignment(rows)
    assert result["status"] == "REJECTED_HELDOUT_ALIGNMENT"
    assert result["metric_translation_authorized"] is False
    assert result["heldout"]["p90_abs_residual_m"] > 0.030
