"""Original-V1 display candidate: clip the frozen feather to Robot support.

This module changes neither rasterization nor depth ordering.  Callers first
obtain ``draw`` from the frozen compositor, then use the original Clean image
as background.  Clean's protected pixels must already equal Raw pixels, just
as in the V1 render contract.  No model, renderer or file writer is created.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chaoyang.exact78_baseline.render import soft_alpha


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"BOUNDARY_ALPHA_MASK_DOMAIN_OR_TYPE:{name}")
    return array


def support_clipped_alpha(draw: np.ndarray) -> np.ndarray:
    """Keep V1's exact inner feather (3x3, sigma .7); set outside weight to 0."""
    value = np.asarray(draw)
    if value.ndim != 2 or value.dtype != np.bool_ or not all(value.shape):
        raise ValueError("BOUNDARY_ALPHA_DRAW_DOMAIN_OR_TYPE")
    # This is deliberately not hard alpha: all weights *inside* draw are the
    # frozen V1 soft_alpha weights, including its >= .5 inner boundary rule.
    return soft_alpha(value) * value


def compose_support_clipped(
    background: np.ndarray,
    layer: np.ndarray,
    draw: np.ndarray,
    protected_visible: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return support-clipped V1 composition plus numeric display evidence.

    RGB and BGR are both supported, but the two inputs must use the same order.
    Protected pixels are restored from background after composition, exactly
    as V1 restores its already-Raw-equal protected Clean pixels.  Inputs are
    never modified.  Returned invariants prove pixel behavior, not visual
    quality, contact or geometry.
    """
    background, layer = np.asarray(background), np.asarray(layer)
    if (background.ndim != 3 or background.shape[-1] != 3
            or background.dtype != np.uint8 or not all(background.shape[:2])):
        raise ValueError("BOUNDARY_ALPHA_BACKGROUND_DOMAIN_OR_TYPE")
    if layer.shape != background.shape or layer.dtype != np.uint8:
        raise ValueError("BOUNDARY_ALPHA_LAYER_DOMAIN_OR_TYPE")
    shape = background.shape[:2]
    draw = _mask(draw, shape, "draw")
    protected = _mask(protected_visible, shape, "protected_visible")
    weight = support_clipped_alpha(draw)
    # Preserve frozen over's float32 -> clip -> uint8 *truncation* semantics.
    # Introducing rounding here would cause a second, unrelated change.
    output = np.clip(
        background.astype(np.float32) * (1 - weight[..., None])
        + layer.astype(np.float32) * weight[..., None],
        0,
        255,
    ).astype(np.uint8)
    output[protected] = background[protected]
    changed = np.any(output != background, axis=-1)
    metadata = {
        "schema_version": "inpainting-v1-support-clipped-boundary-alpha-v1",
        "recipe": "FROZEN_SOFT_ALPHA_MULTIPLIED_BY_BINARY_DRAW",
        "frozen_gaussian_kernel": [3, 3],
        "frozen_gaussian_sigma": 0.7,
        "draw_pixels": int(draw.sum()),
        "protected_visible_pixels": int(protected.sum()),
        "weighted_draw_pixels": int(np.count_nonzero(weight)),
        "outside_draw_changed_pixels": int(np.count_nonzero(changed & ~draw)),
        "protected_changed_pixels": int(np.count_nonzero(changed & protected)),
        "changed_pixels": int(changed.sum()),
        "protected_restoration_source": "BACKGROUND_PIXELS_ALREADY_RAW_EQUAL",
        "geometry_decision": "UNKNOWN",
        "display_policy": "FROZEN_VISIBLE_OBJECT_FIRST_WITH_SUPPORT_CLIPPED_FEATHER",
        "physical_contact_ground_truth": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "visual_quality": "UNREVIEWED",
    }
    return output, metadata
