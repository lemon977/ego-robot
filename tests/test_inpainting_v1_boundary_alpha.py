"""Synthetic CPU tests; no actual robot render, model or production inputs."""

import numpy as np
import pytest

from chaoyang.exact78_baseline.render import over, soft_alpha
from chaoyang.inpainting_candidate.v1_boundary_alpha import (
    compose_support_clipped,
    support_clipped_alpha,
)


def _inputs(shape=(13, 17)):
    height, width = shape
    y, x = np.indices(shape)
    background = np.stack((x * 9 + 23, y * 7 + 25, (x + y) * 5 + 21), axis=-1).astype(np.uint8)
    draw = np.zeros(shape, bool)
    draw[3:height - 3, 4:width - 4] = True
    layer = np.zeros_like(background)
    layer[draw] = (39, 182, 215)
    protected = np.zeros(shape, bool)
    return background, layer, draw, protected


def test_outside_support_black_feather_pollution_is_removed():
    background, layer, draw, protected = _inputs()
    legacy = over(background, layer, draw)
    candidate, evidence = compose_support_clipped(background, layer, draw, protected)
    assert np.any(legacy[~draw] != background[~draw])
    np.testing.assert_array_equal(candidate[~draw], background[~draw])
    assert evidence["outside_draw_changed_pixels"] == 0
    assert evidence["control_ground_truth"] is False
    assert evidence["visual_quality"] == "UNREVIEWED"


def test_inside_support_is_bitwise_original_v1_not_hard_alpha():
    background, layer, draw, protected = _inputs()
    candidate, _ = compose_support_clipped(background, layer, draw, protected)
    legacy = over(background, layer, draw)
    np.testing.assert_array_equal(candidate[draw], legacy[draw])
    weight = support_clipped_alpha(draw)
    np.testing.assert_array_equal(weight[draw], soft_alpha(draw)[draw])
    assert np.any(weight[draw] < 1)
    assert not np.any(weight[~draw])


def test_protected_pixels_are_exactly_restored_even_inside_robot():
    background, layer, draw, protected = _inputs()
    protected[4:7, 5:12] = True
    candidate, evidence = compose_support_clipped(background, layer, draw, protected)
    np.testing.assert_array_equal(candidate[protected], background[protected])
    assert evidence["protected_changed_pixels"] == 0
    assert evidence["protected_visible_pixels"] == int(protected.sum())
    legacy = over(background, layer, draw)
    np.testing.assert_array_equal(candidate[draw & ~protected], legacy[draw & ~protected])


@pytest.mark.parametrize("kind", ["empty", "all", "top_left", "bottom_right", "one_pixel"])
@pytest.mark.parametrize("shape", [(9, 11), (1, 1), (1, 5), (5, 1)])
def test_empty_full_and_image_border_support(kind, shape):
    background, layer, draw, protected = _inputs(shape)
    draw[:] = False
    if kind == "all":
        draw[:] = True
    elif kind in {"top_left", "one_pixel"}:
        draw[0, 0] = True
    elif kind == "bottom_right":
        draw[-1, -1] = True
    layer[draw] = (13, 140, 249)
    candidate, metadata = compose_support_clipped(background, layer, draw, protected)
    np.testing.assert_array_equal(candidate[~draw], background[~draw])
    np.testing.assert_array_equal(candidate[draw], over(background, layer, draw)[draw])
    assert metadata["outside_draw_changed_pixels"] == 0
    assert candidate.shape == background.shape and candidate.dtype == np.uint8


def test_noncontiguous_inputs_and_all_protected_are_safe_and_unchanged():
    background, layer, draw, protected = _inputs()
    protected[:] = True
    inputs = [value[:, ::-1] for value in (background, layer, draw, protected)]
    snapshots = [value.copy() for value in inputs]
    candidate, evidence = compose_support_clipped(*inputs)
    np.testing.assert_array_equal(candidate, inputs[0])
    assert evidence["changed_pixels"] == 0
    for original, snapshot in zip(inputs, snapshots):
        np.testing.assert_array_equal(original, snapshot)


@pytest.mark.parametrize("argument,replacement", [
    (0, np.zeros((13, 17, 3), np.float32)),
    (0, np.zeros((13, 17), np.uint8)),
    (0, np.zeros((13, 17, 4), np.uint8)),
    (0, np.zeros((0, 17, 3), np.uint8)),
    (1, np.zeros((13, 17, 3), np.int16)),
    (1, np.zeros((12, 17, 3), np.uint8)),
    (2, np.zeros((13, 17), np.uint8)),
    (2, np.zeros((13, 16), bool)),
    (3, np.zeros((13, 17), np.float32)),
    (3, np.zeros((13, 17, 1), bool)),
])
def test_bad_input_domains_rejected(argument, replacement):
    inputs = list(_inputs())
    inputs[argument] = replacement
    with pytest.raises(ValueError, match="BOUNDARY_ALPHA_"):
        compose_support_clipped(*inputs)


@pytest.mark.parametrize("value", [
    np.zeros((5, 8), np.uint8),
    np.zeros((5, 8, 1), bool),
    np.zeros((0, 8), bool),
])
def test_bad_draw_rejected_directly(value):
    with pytest.raises(ValueError, match="BOUNDARY_ALPHA_DRAW_DOMAIN_OR_TYPE"):
        support_clipped_alpha(value)
