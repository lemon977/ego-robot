"""Conservative, seed-bracketed attachment-mask propagation.

This module is deliberately not a semantic detector.  It may only repair a
short gap between two already admitted masks.  Both temporal directions must
agree; frames outside a seed bracket remain UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AttachmentTrackGateV1:
    maximum_seed_gap: int = 6
    minimum_bidirectional_iou: float = 0.45
    minimum_area_ratio: float = 0.50
    maximum_area_ratio: float = 2.00


def read_binary_mask(path: str) -> np.ndarray:
    """Read masks losslessly, including uint16 files whose foreground is 1."""

    value = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    if value.ndim == 3:
        value = np.any(value != 0, axis=2)
    return np.asarray(value != 0, dtype=np.bool_)


def propagate_one_frame(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    """Warp a source mask into the adjacent target frame using backward flow."""

    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    if source.shape != target.shape or source.shape != source_mask.shape:
        raise ValueError("image/mask geometry mismatch")
    # Flow from target pixels to source pixels is directly usable by remap.
    backward = cv2.calcOpticalFlowFarneback(
        target, source, None, 0.5, 4, 31, 5, 7, 1.5, 0,
    )
    height, width = target.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    warped = cv2.remap(
        source_mask.astype(np.uint8),
        grid_x + backward[..., 0],
        grid_y + backward[..., 1],
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(bool)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    if not union:
        return 1.0 if not np.any(left) and not np.any(right) else 0.0
    return float(np.count_nonzero(left & right) / union)


def admit_bidirectional_candidate(
    forward: np.ndarray,
    reverse: np.ndarray,
    *,
    left_seed_area: int,
    right_seed_area: int,
    gate: AttachmentTrackGateV1 | None = None,
) -> tuple[bool, dict[str, float]]:
    limits = gate or AttachmentTrackGateV1()
    intersection = forward & reverse
    area = int(np.count_nonzero(intersection))
    reference = max(1.0, 0.5 * (left_seed_area + right_seed_area))
    ratio = area / reference
    iou = mask_iou(forward, reverse)
    passed = (
        area > 0
        and iou >= limits.minimum_bidirectional_iou
        and limits.minimum_area_ratio <= ratio <= limits.maximum_area_ratio
    )
    return passed, {
        "bidirectional_iou": iou,
        "intersection_area_px": float(area),
        "seed_mean_area_ratio": float(ratio),
    }


def seed_brackets(seed_frames: list[int], *, gate: AttachmentTrackGateV1 | None = None) -> list[tuple[int, int]]:
    limits = gate or AttachmentTrackGateV1()
    ordered = sorted(set(int(value) for value in seed_frames))
    return [
        (left, right)
        for left, right in zip(ordered[:-1], ordered[1:])
        if 1 < right - left <= limits.maximum_seed_gap
    ]
