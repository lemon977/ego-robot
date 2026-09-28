"""Fixed failure cases for source-coverage recovery diagnostics."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from chaoyang.ops.run_human_to_robot_source_coverage_diagnostic import region_support, target_step_mm
from chaoyang.ops.run_human_to_robot_source_coverage_rebind_007 import merge_support
from chaoyang.ops.run_human_to_robot_source_projection_031 import inside_image


def test_nonempty_role_elsewhere_does_not_cover_frozen_left_region() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:7, 6:9] = True
    assert mask.any()
    assert region_support(mask, (0, 2, 4, 8)) == 0
    assert region_support(mask, (6, 2, 9, 8)) == 15


def test_region_domain_is_checked() -> None:
    with pytest.raises(ValueError, match="DOMAIN"):
        region_support(np.zeros((8, 10), bool), (0, 2, 11, 8))


def test_target_step_preserves_invalid_gap() -> None:
    xyz = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [2.1, 0., 0.]])
    result = target_step_mm(xyz, np.array([False, True, True, True]))
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert result[2] == 1000.0
    assert result[3] == pytest.approx(100.0)


def test_rebound_support_cannot_erase_protected_object() -> None:
    current = np.zeros((720, 960), bool)
    old = np.zeros((960, 1280), bool)
    write = np.zeros_like(old)
    protect = np.zeros_like(old)
    old[100:200, 100:200] = True
    protect[120:140, 120:140] = True
    with pytest.raises(ValueError, match="PROTECTION_CONFLICT"):
        merge_support(current, old, write, protect)


def test_rebound_context_covers_original_full_resolution_write_after_resize() -> None:
    current = np.zeros((720, 960), bool)
    old = np.zeros((960, 1280), bool)
    write = np.zeros_like(old)
    protect = np.zeros_like(old)
    write[133:138, 131:136] = True
    old[430:500, 70:220] = True
    model, new_write = merge_support(current, old, write, protect)
    roundtrip = cv2.resize(model.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST) > 0
    assert np.all(roundtrip[new_write])


def test_projected_wrist_outside_image_does_not_count_as_observed_wrist() -> None:
    uv = np.array([[[700., 995.], [750., 940.]]])
    result = inside_image(uv, 1280, 960)
    assert result.tolist() == [[False, True]]
