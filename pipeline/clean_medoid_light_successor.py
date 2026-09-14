#!/usr/bin/env python3
"""Core operations for the auditable real-pixel Clean successor.

No function synthesizes texture or uses target gradients.  The temporal plate
is a per-pixel medoid selected from registered donor frames, photometric
normalization is fitted only on declared clean support, and composition changes
only the authorized human/shadow region plus its narrow seam.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class CleanSuccessorError(RuntimeError):
    pass


def donor_registration_support(
    semantic_excluded: np.ndarray,
    *,
    minimum_y: int,
    extra_dilation_px: int,
) -> np.ndarray:
    """Return table-only registration support with PICO contamination removed."""
    if semantic_excluded.ndim != 2 or semantic_excluded.dtype != bool:
        raise CleanSuccessorError("registration semantic exclusion must be a 2D bool mask")
    height, width = semantic_excluded.shape
    if not 0 <= minimum_y < height or extra_dilation_px < 0:
        raise CleanSuccessorError("invalid donor registration support parameters")
    excluded = semantic_excluded.astype(np.uint8)
    if extra_dilation_px:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * extra_dilation_px + 1, 2 * extra_dilation_px + 1),
        )
        excluded = cv2.dilate(excluded, kernel)
    support = np.zeros((height, width), dtype=bool)
    support[minimum_y:, :] = True
    support &= excluded == 0
    return support


def _require_stack(images: np.ndarray, valid: np.ndarray, excluded: np.ndarray) -> None:
    if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
        raise CleanSuccessorError("donor images must be uint8 NxHxWx3")
    if valid.shape != images.shape[:3] or excluded.shape != images.shape[:3]:
        raise CleanSuccessorError("donor validity/exclusion geometry mismatch")
    if len(images) < 3:
        raise CleanSuccessorError("at least three registered donor times are required")


def robust_real_pixel_temporal_medoid(
    images: np.ndarray,
    valid: np.ndarray,
    semantic_excluded: np.ndarray,
    *,
    foreground_bgr_l2_max: float,
    shadow_luma_delta_min: float,
    minimum_pure_observations: int,
) -> dict[str, Any]:
    """Choose one real donor pixel after semantic/foreground/shadow rejection.

    Global per-frame BGR/Luma offsets are removed for impurity classification
    only.  Published pixels remain byte values from one input donor frame.
    """
    _require_stack(images, valid, semantic_excluded)
    if minimum_pure_observations < 2 or minimum_pure_observations > len(images):
        raise CleanSuccessorError("invalid minimum pure-observation count")
    values = images.astype(np.float32)
    initial_legal = valid & ~semantic_excluded
    frame_bias = np.zeros((len(images), 3), np.float32)
    reference = np.nanmedian(np.where(valid[:, :, :, None], values, np.nan), axis=0)
    for index in range(len(images)):
        legal = initial_legal[index] & np.isfinite(reference).all(axis=2)
        if int(legal.sum()) < 256:
            raise CleanSuccessorError(f"donor frame {index} has insufficient clean support")
        frame_bias[index] = np.median(values[index][legal] - reference[legal], axis=0)
    normalized = values - frame_bias[:, None, None, :]
    center = np.nanmedian(np.where(valid[:, :, :, None], normalized, np.nan), axis=0)
    colour_distance = np.linalg.norm(normalized - center[None], axis=3)
    foreground_outlier = colour_distance > float(foreground_bgr_l2_max)

    luma = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0] for image in images]).astype(np.float32)
    luma_bias = np.zeros(len(images), np.float32)
    luma_reference = np.nanmedian(np.where(valid, luma, np.nan), axis=0)
    for index in range(len(images)):
        legal = initial_legal[index] & np.isfinite(luma_reference)
        luma_bias[index] = float(np.median(luma[index][legal] - luma_reference[legal]))
    normalized_luma = luma - luma_bias[:, None, None]
    luma_center = np.nanmedian(np.where(valid, normalized_luma, np.nan), axis=0)
    shadow_outlier = normalized_luma < luma_center[None] - float(shadow_luma_delta_min)

    pure = valid & ~semantic_excluded & ~foreground_outlier & ~shadow_outlier
    pure_count = pure.sum(axis=0)
    distance = colour_distance.copy()
    distance[~pure] = np.inf
    choice = np.argmin(distance, axis=0)
    row, column = np.indices(choice.shape)
    medoid = images[choice, row, column].copy()
    medoid_valid = pure_count >= int(minimum_pure_observations)
    medoid[~medoid_valid] = 0
    choice = choice.astype(np.uint16)
    choice[~medoid_valid] = 65535
    return {
        "plate": medoid,
        "choice_index": choice,
        "valid": medoid_valid,
        "pure_observation_count": pure_count.astype(np.uint16),
        "semantic_excluded": semantic_excluded,
        "foreground_outlier": foreground_outlier,
        "shadow_outlier": shadow_outlier,
        "diagnostics": {
            "donor_frame_count": len(images),
            "semantic_excluded_pixels_by_frame": semantic_excluded.reshape(len(images), -1).sum(axis=1).astype(int).tolist(),
            "foreground_outlier_pixels_by_frame": foreground_outlier.reshape(len(images), -1).sum(axis=1).astype(int).tolist(),
            "shadow_outlier_pixels_by_frame": shadow_outlier.reshape(len(images), -1).sum(axis=1).astype(int).tolist(),
            "frame_bgr_bias": frame_bias.tolist(),
            "frame_luma_bias": luma_bias.tolist(),
            "pure_observation_count_min": int(pure_count.min()),
            "pure_observation_count_p01": float(np.quantile(pure_count, 0.01)),
            "pure_observation_count_median": float(np.median(pure_count)),
            "invalid_output_pixels": int((~medoid_valid).sum()),
            "published_pixels_are_exact_registered_donor_samples": True,
        },
    }


def normalized_gaussian(image: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    weight = valid.astype(np.float32)
    denominator = cv2.GaussianBlur(weight, (0, 0), sigma)
    if image.ndim == 3:
        numerator = cv2.GaussianBlur(image * weight[:, :, None], (0, 0), sigma)
        return numerator / np.maximum(denominator[:, :, None], 1e-4)
    numerator = cv2.GaussianBlur(image * weight, (0, 0), sigma)
    return numerator / np.maximum(denominator, 1e-4)


def fit_legal_lab_illumination_field(
    plate: np.ndarray,
    target: np.ndarray,
    legal_support: np.ndarray,
    *,
    local_sigma_px: float = 13.0,
    field_sigma_px: float = 42.0,
    minimum_fit_pixels: int = 3500,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Fit low-frequency Lab gain/bias/field only from legal support."""
    if plate.shape != target.shape or plate.shape[:2] != legal_support.shape:
        raise CleanSuccessorError("photometric geometry mismatch")
    sample = legal_support.copy()
    sample[1::3, :] = False
    sample[:, 1::3] = False
    sample[:, 2::3] = False
    if int(sample.sum()) < minimum_fit_pixels:
        raise CleanSuccessorError(f"insufficient legal photometric pixels: {int(sample.sum())}")
    plate_lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    plate_low = normalized_gaussian(plate_lab, legal_support, local_sigma_px)
    target_low = normalized_gaussian(target_lab, legal_support, local_sigma_px)
    fitted = np.empty_like(plate_lab)
    details: dict[str, Any] = {
        "fit_pixels": int(sample.sum()),
        "fit_support_pixels": int(legal_support.sum()),
        "target_gradient_used": False,
        "poisson_used": False,
        "channels": [],
    }
    for channel in range(3):
        source_values = plate_low[:, :, channel][sample]
        target_values = target_low[:, :, channel][sample]
        if channel == 0:
            source_iqr = float(np.quantile(source_values, 0.75) - np.quantile(source_values, 0.25))
            target_iqr = float(np.quantile(target_values, 0.75) - np.quantile(target_values, 0.25))
            gain = float(np.clip(target_iqr / max(source_iqr, 3.0), 0.85, 1.15))
        else:
            gain = 1.0
        bias = float(np.median(target_values) - gain * np.median(source_values))
        low_prediction = gain * plate_low[:, :, channel] + bias
        residual = target_low[:, :, channel] - low_prediction
        field = normalized_gaussian(residual, legal_support, field_sigma_px)
        field = np.clip(field, -30.0, 30.0)
        fitted[:, :, channel] = np.clip(gain * plate_lab[:, :, channel] + bias + field, 0, 255)
        details["channels"].append({
            "channel": channel,
            "mode": "global_gain_bias_plus_legal_support_normalized_low_frequency_field",
            "gain": gain,
            "bias": bias,
            "field_min": float(field.min()),
            "field_max": float(field.max()),
        })
    error = np.linalg.norm(fitted - target_lab, axis=2)
    score = float(np.median(error[legal_support]))
    details["median_lab_error_on_legal_pixels"] = score
    return cv2.cvtColor(fitted.astype(np.uint8), cv2.COLOR_LAB2BGR), details, score


def localized_source_composite(
    raw: np.ndarray,
    corrected_donor: np.ndarray,
    donor_valid: np.ndarray,
    human: np.ndarray,
    protected_object: np.ndarray,
    seam_width_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose only human+seam and restore protected objects byte-exactly."""
    if seam_width_px < 0:
        raise CleanSuccessorError("negative seam width")
    if raw.shape != corrected_donor.shape or raw.shape[:2] != human.shape:
        raise CleanSuccessorError("composite geometry mismatch")
    distance = cv2.distanceTransform((~human).astype(np.uint8), cv2.DIST_L2, 5)
    if seam_width_px:
        alpha = np.where(human, 1.0, np.clip((seam_width_px - distance) / seam_width_px, 0.0, 1.0))
    else:
        alpha = human.astype(np.float32)
    alpha = alpha.astype(np.float32) * donor_valid.astype(np.float32)
    alpha[protected_object] = 0.0
    clean = np.clip(
        raw.astype(np.float32) * (1.0 - alpha[:, :, None])
        + corrected_donor.astype(np.float32) * alpha[:, :, None],
        0,
        255,
    ).astype(np.uint8)
    source_class = np.zeros(human.shape, np.uint8)
    source_class[(alpha > 0.0) & (alpha < 1.0)] = 2
    source_class[alpha >= 1.0] = 1
    clean[protected_object] = raw[protected_object]
    source_class[protected_object] = 4
    return clean, source_class, alpha


def audit_composite(
    raw: np.ndarray,
    clean: np.ndarray,
    source_class: np.ndarray,
    protected_object: np.ndarray,
) -> dict[str, Any]:
    if raw.shape != clean.shape or raw.shape[:2] != source_class.shape:
        raise CleanSuccessorError("audit geometry mismatch")
    changed = np.any(raw != clean, axis=2)
    allowed = np.isin(source_class, (1, 2, 3, 5))
    return {
        "changed_pixels_outside_authorized_domain": int((changed & ~allowed & ~protected_object).sum()),
        "changed_protected_object_pixels": int((changed & protected_object).sum()),
        "source_class_values": sorted(int(value) for value in np.unique(source_class)),
        "source_map_known": bool(np.isin(source_class, (0, 1, 2, 3, 4, 5)).all()),
    }
