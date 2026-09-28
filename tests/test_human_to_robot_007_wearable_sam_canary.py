from __future__ import annotations

import numpy as np
import pytest

from chaoyang.ops.run_human_to_robot_007_wearable_sam_canary import (
    HEIGHT, WIDTH, PROTECT_POINTS, SPECS, evaluate_instance, normalize_outputs,
)


def test_frozen_prompts_inside_image_and_protect_points_not_positive():
    assert len(SPECS) == 6
    assert len({spec[0] for spec in SPECS}) == 6
    for _, _, (x, y, w, h), (px, py), negatives in SPECS:
        assert 0 <= x < x + w <= WIDTH and 0 <= y < y + h <= HEIGHT
        assert x <= px < x + w and y <= py < y + h
        assert (px, py) not in negatives
        assert all(0 <= nx < WIDTH and 0 <= ny < HEIGHT for nx, ny in negatives)
    assert all(point in SPECS[0][4] for point in PROTECT_POINTS)


def test_positive_device_with_clear_negatives_can_pass_one_frame_only():
    spec = SPECS[0]
    mask = np.zeros((HEIGHT, WIDTH), bool)
    x, y = spec[3]
    mask[y - 8:y + 8, x - 8:x + 8] = True
    row = evaluate_instance(mask, spec)
    assert row["status"] == "SUPPORTED_SINGLE_FRAME"
    assert row["positive_covered"]
    assert all(row["negative_excluded"])


def test_object_or_skin_leak_rejects_instance():
    spec = SPECS[0]
    mask = np.zeros((HEIGHT, WIDTH), bool)
    px, py = spec[3]
    mask[py - 8:py + 8, px - 8:px + 8] = True
    nx, ny = spec[4][0]
    mask[ny, nx] = True
    assert evaluate_instance(mask, spec)["status"] == "REJECTED_INSTANCE"
    mask[:] = True
    assert evaluate_instance(mask, spec)["status"] == "REJECTED_INSTANCE"


def test_missing_point_and_wrong_domain_reject():
    mask = np.zeros((HEIGHT, WIDTH), bool)
    assert evaluate_instance(mask, SPECS[0])["status"] == "REJECTED_INSTANCE"
    with pytest.raises(ValueError, match="INSTANCE_DOMAIN"):
        evaluate_instance(np.zeros((HEIGHT, WIDTH), np.uint8), SPECS[0])


def test_raw_instance_normalization_rejects_shape_drift():
    masks, ids = normalize_outputs({"out_binary_masks": np.zeros((1, HEIGHT, WIDTH), bool),
                                    "out_obj_ids": np.array([201])})
    assert masks.shape == (1, HEIGHT, WIDTH)
    assert ids.tolist() == [201]
    with pytest.raises(ValueError, match="SAM_MASK_DOMAIN"):
        normalize_outputs({"out_binary_masks": np.zeros((1, 10, 10), bool),
                           "out_obj_ids": np.array([201])})
