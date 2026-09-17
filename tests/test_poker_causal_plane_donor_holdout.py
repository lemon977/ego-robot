from __future__ import annotations

import cv2
import numpy as np

from chaoyang.ops.run_poker_causal_plane_donor_holdout import evaluate_pair


def test_known_planar_translation_can_be_recovered() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    rng = np.random.default_rng(7)
    image[60:180, 80:240] = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    mask = np.zeros((240, 320), dtype=np.uint8); mask[60:180, 80:240] = 255
    matrix = np.float32([[1, 0, 5], [0, 1, 3]])
    target = cv2.warpAffine(image, matrix, (320, 240))
    target_mask = cv2.warpAffine(mask, matrix, (320, 240), flags=cv2.INTER_NEAREST)
    metrics, _ = evaluate_pair(image, target, mask, target_mask)
    assert metrics["classification"] == "CAUSAL_WARP_ACCEPTED"
    assert metrics["coverage"] > 0.9
    assert metrics["mae_rgb"] < 15.0
