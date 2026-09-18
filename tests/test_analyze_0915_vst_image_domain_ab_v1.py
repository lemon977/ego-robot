from __future__ import annotations

import cv2
import numpy as np

from chaoyang.ops import analyze_0915_vst_image_domain_ab_v1 as subject


def test_image_metrics_identical_and_changed() -> None:
    image = np.full((20, 30, 3), 100, np.uint8)
    identical = subject.image_metrics(image, image.copy())
    assert identical["mae"] == 0
    assert identical["psnr_db"] == 99.0
    changed = image.copy()
    changed[:, 15:] = 140
    metrics = subject.image_metrics(image, changed)
    assert metrics["mae"] > 0
    assert metrics["psnr_db"] < 99


def test_map_metrics_zero_for_resize_map_and_positive_for_warp() -> None:
    width, height = subject.TARGET_SIZE
    source_width, source_height = 2048, 1536
    scale_x, scale_y = source_width / width, source_height / height
    grid_x, grid_y = np.meshgrid(
        (np.arange(width, dtype=np.float32) + 0.5) * scale_x - 0.5,
        (np.arange(height, dtype=np.float32) + 0.5) * scale_y - 0.5,
    )
    stats, displacement = subject.map_metrics(
        grid_x, grid_y, source_width, source_height,
    )
    assert stats["p95"] == 0
    assert float(displacement.max()) == 0
    warped_x = grid_x + 16.0
    warped_y = grid_y + 8.0
    warped, _ = subject.map_metrics(
        warped_x, warped_y, source_width, source_height,
    )
    assert warped["p50"] > 10


def test_panel_adds_title_bar() -> None:
    image = np.zeros((960, 1280, 3), np.uint8)
    panel = subject.panel(image, "candidate", 7, (320, 240))
    assert panel.shape == (282, 320, 3)
    assert int(panel[:42].max()) > 0
