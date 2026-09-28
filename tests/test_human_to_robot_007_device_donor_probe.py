from __future__ import annotations

import cv2
import numpy as np
import pytest

from chaoyang.ops.run_human_to_robot_007_device_donor_probe import (
    COMPLAINT_POINTS, fit_heldout_homography, table_mask,
)


def test_table_mask_excludes_unsupported_pixels_and_room():
    image = np.full((960, 1280, 3), 80, np.uint8)
    exclude = np.zeros((960, 1280), bool)
    exclude[600:620, 600:620] = True
    mask = table_mask(image, exclude, donor_frame=0)
    assert not mask[100, 100]
    assert mask[700, 100]
    assert not mask[610, 610]
    assert not mask[580, 610]  # bounded exclusion margin around hand/device


def test_table_mask_rejects_wrong_pixel_domain():
    with pytest.raises(ValueError, match="DOMAIN"):
        table_mask(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool))


def test_blank_images_do_not_fabricate_donor_homography():
    image = np.full((960, 1280, 3), 80, np.uint8)
    mask = np.full((960, 1280), 255, np.uint8)
    h, report = fit_heldout_homography(image, image, mask, mask)
    assert h is None
    assert report["geometry_pass"] is False
    assert report["holdout_inliers"] == 0


def test_heldout_geometry_can_verify_real_planar_shift():
    image = np.full((960, 1280, 3), 80, np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(380):
        x, y = int(rng.integers(40, 1220)), int(rng.integers(40, 900))
        radius = int(rng.integers(4, 13))
        color = tuple(int(c) for c in rng.integers(25, 240, size=3))
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.line(image, (x - radius, y), (x + radius, y + radius), (20, 230, 40), 2)
    target = cv2.warpAffine(image, np.float32([[1, 0, 14], [0, 1, 9]]), (1280, 960))
    mask = np.full((960, 1280), 255, np.uint8)
    h, report = fit_heldout_homography(image, target, mask, mask)
    assert h is not None
    assert report["geometry_pass"] is True
    assert report["holdout_median_px"] <= 2.0


def test_complaint_points_are_frozen_inside_source_image():
    assert len(COMPLAINT_POINTS) == 7
    assert all(0 <= x < 1280 and 0 <= y < 960 for x, y in COMPLAINT_POINTS.values())
