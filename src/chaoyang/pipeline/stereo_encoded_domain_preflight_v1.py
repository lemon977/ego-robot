"""Encoded-pixel stereo diagnostics with no lens-model transformation.

The functions here operate only on already decoded physical-eye images.  They
measure whether matching correspondences are horizontally aligned enough for a
separate stereo model canary; they do not estimate metric depth or modify the
pixel domain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class EncodedStereoGateV1:
    minimum_total_robust_matches: int = 1_500
    minimum_matches_per_frame: int = 12
    minimum_qualifying_frame_fraction: float = 0.80
    minimum_mean_spatial_coverage: float = 0.08
    maximum_median_vertical_px: float = 2.5
    maximum_p90_vertical_px: float = 4.0
    maximum_p95_vertical_px: float = 5.0
    minimum_disparity_sign_consistency: float = 0.80


def split_source_index_eyes(
    frame_sbs: np.ndarray,
    *,
    eye_width: int,
    eye_height: int,
    physical_left_source_index: int,
    physical_right_source_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical left/right images from a two-eye horizontal SBS frame."""

    frame = np.asarray(frame_sbs)
    indices = (int(physical_left_source_index), int(physical_right_source_index))
    if sorted(indices) != [0, 1]:
        raise ValueError("physical eye source indices must be the permutation [0, 1]")
    if frame.shape[:2] != (eye_height, eye_width * 2):
        raise ValueError(
            f"SBS geometry is {frame.shape[1]}x{frame.shape[0]}, expected "
            f"{eye_width * 2}x{eye_height}"
        )
    halves = (frame[:, :eye_width], frame[:, eye_width:])
    return halves[indices[0]], halves[indices[1]]


def resize_only(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize decoded pixels without applying any camera/lens model."""

    interpolation = (
        cv2.INTER_AREA
        if image.shape[1] >= width and image.shape[0] >= height
        else cv2.INTER_LINEAR
    )
    return cv2.resize(image, (width, height), interpolation=interpolation)


def robust_correspondences(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    maximum_features: int = 3_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mutual ratio-tested, robust fundamental-matrix inliers."""

    detector = cv2.SIFT_create(int(maximum_features))
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    key_left, desc_left = detector.detectAndCompute(left_gray, None)
    key_right, desc_right = detector.detectAndCompute(right_gray, None)
    empty = np.empty((0, 2), np.float64)
    if desc_left is None or desc_right is None:
        return empty, empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward_pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    reverse_pairs = matcher.knnMatch(desc_right, desc_left, k=2)
    forward = [
        first for first, second in forward_pairs
        if first.distance < 0.70 * second.distance
    ]
    reverse = {
        first.queryIdx: first.trainIdx
        for first, second in reverse_pairs
        if first.distance < 0.70 * second.distance
    }
    mutual = [item for item in forward if reverse.get(item.trainIdx) == item.queryIdx]
    if len(mutual) < 8:
        return empty, empty
    left = np.asarray([key_left[item.queryIdx].pt for item in mutual], np.float64)
    right = np.asarray([key_right[item.trainIdx].pt for item in mutual], np.float64)
    matrix, inlier = cv2.findFundamentalMat(
        left, right, cv2.USAC_MAGSAC, 1.0, 0.999,
    )
    if matrix is None or inlier is None:
        return empty, empty
    keep = np.asarray(inlier).reshape(-1).astype(bool)
    return left[keep], right[keep]


def frame_metrics(
    left_points: np.ndarray,
    right_points: np.ndarray,
    *,
    width: int,
    height: int,
    grid_columns: int = 8,
    grid_rows: int = 6,
) -> dict[str, Any]:
    """Compute vertical error, signed disparity and spatial support for one frame."""

    left = np.asarray(left_points, np.float64).reshape(-1, 2)
    right = np.asarray(right_points, np.float64).reshape(-1, 2)
    if left.shape != right.shape:
        raise ValueError("left/right correspondence shapes differ")
    if not len(left):
        return {
            "robust_matches": 0,
            "median_abs_vertical_px": None,
            "p90_abs_vertical_px": None,
            "p95_abs_vertical_px": None,
            "median_signed_disparity_px": None,
            "positive_disparity_fraction": None,
            "spatial_grid_coverage": 0.0,
        }
    vertical = np.abs(left[:, 1] - right[:, 1])
    disparity = left[:, 0] - right[:, 0]
    cell_x = np.clip(
        (left[:, 0] * grid_columns / width).astype(int), 0, grid_columns - 1,
    )
    cell_y = np.clip(
        (left[:, 1] * grid_rows / height).astype(int), 0, grid_rows - 1,
    )
    occupied = len(set(zip(cell_x.tolist(), cell_y.tolist())))
    nonzero = disparity[np.abs(disparity) > 0.25]
    return {
        "robust_matches": int(len(left)),
        "median_abs_vertical_px": float(np.median(vertical)),
        "p90_abs_vertical_px": float(np.quantile(vertical, 0.90)),
        "p95_abs_vertical_px": float(np.quantile(vertical, 0.95)),
        "median_signed_disparity_px": float(np.median(disparity)),
        "positive_disparity_fraction": (
            float(np.mean(nonzero > 0.0)) if len(nonzero) else None
        ),
        "spatial_grid_coverage": float(
            occupied / float(grid_columns * grid_rows)
        ),
    }


def aggregate_metrics(
    frame_rows: list[dict[str, Any]],
    vertical_residuals: list[np.ndarray],
    signed_disparities: list[np.ndarray],
    *,
    gate: EncodedStereoGateV1 | None = None,
) -> dict[str, Any]:
    """Aggregate every decoded frame and return a finite routing decision."""

    if not frame_rows:
        raise ValueError("at least one frame row is required")
    limits = gate or EncodedStereoGateV1()
    vertical = np.concatenate([value for value in vertical_residuals if len(value)]) \
        if any(len(value) for value in vertical_residuals) else np.empty(0)
    disparity = np.concatenate([value for value in signed_disparities if len(value)]) \
        if any(len(value) for value in signed_disparities) else np.empty(0)
    nonzero = disparity[np.abs(disparity) > 0.25]
    total_matches = int(len(vertical))
    qualifying_fraction = float(np.mean([
        int(row["robust_matches"]) >= limits.minimum_matches_per_frame
        for row in frame_rows
    ]))
    mean_coverage = float(np.mean([
        float(row["spatial_grid_coverage"]) for row in frame_rows
    ]))
    median_vertical = float(np.median(vertical)) if len(vertical) else None
    p90_vertical = float(np.quantile(vertical, 0.90)) if len(vertical) else None
    p95_vertical = float(np.quantile(vertical, 0.95)) if len(vertical) else None
    positive_fraction = float(np.mean(nonzero > 0.0)) if len(nonzero) else None
    negative_fraction = float(np.mean(nonzero < 0.0)) if len(nonzero) else None
    sign_consistency = (
        max(positive_fraction, negative_fraction)
        if positive_fraction is not None and negative_fraction is not None else None
    )
    dominant_sign = (
        "POSITIVE_LEFT_MINUS_RIGHT"
        if positive_fraction is not None and positive_fraction >= negative_fraction
        else "NEGATIVE_LEFT_MINUS_RIGHT"
        if negative_fraction is not None else "UNKNOWN"
    )

    values = {
        "total_robust_matches": total_matches,
        "qualifying_frame_fraction": qualifying_fraction,
        "mean_spatial_grid_coverage": mean_coverage,
        "median_abs_vertical_px": median_vertical,
        "p90_abs_vertical_px": p90_vertical,
        "p95_abs_vertical_px": p95_vertical,
        "median_signed_disparity_px": (
            float(np.median(disparity)) if len(disparity) else None
        ),
        "positive_disparity_fraction": positive_fraction,
        "negative_disparity_fraction": negative_fraction,
        "dominant_disparity_sign": dominant_sign,
        "disparity_sign_consistency": sign_consistency,
    }
    checks = {
        "robust_matches": total_matches >= limits.minimum_total_robust_matches,
        "qualifying_frame_fraction": (
            qualifying_fraction >= limits.minimum_qualifying_frame_fraction
        ),
        "spatial_coverage": mean_coverage >= limits.minimum_mean_spatial_coverage,
        "median_vertical": (
            median_vertical is not None
            and median_vertical <= limits.maximum_median_vertical_px
        ),
        "p90_vertical": (
            p90_vertical is not None
            and p90_vertical <= limits.maximum_p90_vertical_px
        ),
        "p95_vertical": (
            p95_vertical is not None
            and p95_vertical <= limits.maximum_p95_vertical_px
        ),
        "disparity_sign": (
            sign_consistency is not None
            and sign_consistency >= limits.minimum_disparity_sign_consistency
        ),
    }
    evidence_sufficient = all(
        checks[name] for name in (
            "robust_matches", "qualifying_frame_fraction", "spatial_coverage",
        )
    )
    if not evidence_sufficient:
        decision = "REJECTED_QUALITY_INSUFFICIENT_CORRESPONDENCE_EVIDENCE"
    elif all(checks.values()):
        decision = "PASS_DIRECT_FOUNDATION_INPUT"
    else:
        decision = "NEEDS_ENCODED_EPIPOLAR_ALIGNMENT"
    return {
        "decision": decision,
        "gpu_successor_authorized": decision == "PASS_DIRECT_FOUNDATION_INPUT",
        "metrics": values,
        "gates": checks,
        "thresholds": asdict(limits),
    }

