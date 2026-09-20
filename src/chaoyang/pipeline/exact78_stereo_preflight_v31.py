"""Three-part Exact78 encoded-VST stereo preflight.

The checks deliberately separate pixel-domain identity, horizontal-epipolar
evidence, and metric conversion.  Passing one check cannot authorize the
others, and none of them claims external millimetre accuracy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import (
    EncodedStereoGateV1,
    aggregate_metrics,
)


@dataclass(frozen=True)
class ImageDomainEvidenceV31:
    sbs_width: int
    sbs_height: int
    eye_width: int
    eye_height: int
    model_width: int
    model_height: int
    physical_left_source_index: int
    physical_right_source_index: int
    decoded_vst_already_undistorted: bool
    lens_undistortion_applied: bool
    horizontal_reflection_for_disparity_sign: bool
    outputs_unflipped_to_physical_left: bool
    maximum_coordinate_roundtrip_error_px: float
    mismatched_rgb_roundtrip_pixels: int


def check_image_domain(evidence: ImageDomainEvidenceV31) -> dict[str, Any]:
    checks = {
        "physical_eye_indices": sorted(
            (evidence.physical_left_source_index, evidence.physical_right_source_index)
        ) == [0, 1],
        "sbs_geometry": (
            evidence.sbs_width == 2 * evidence.eye_width
            and evidence.sbs_height == evidence.eye_height
        ),
        "positive_geometry": min(
            evidence.eye_width,
            evidence.eye_height,
            evidence.model_width,
            evidence.model_height,
        ) > 0,
        "encoded_vst_declared_undistorted": evidence.decoded_vst_already_undistorted,
        "no_repeated_lens_undistortion": not evidence.lens_undistortion_applied,
        "physical_left_output_restored": (
            not evidence.horizontal_reflection_for_disparity_sign
            or evidence.outputs_unflipped_to_physical_left
        ),
        "coordinate_roundtrip_exact": (
            evidence.maximum_coordinate_roundtrip_error_px == 0.0
        ),
        "rgb_roundtrip_exact": evidence.mismatched_rgb_roundtrip_pixels == 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": "EXACT78_IMAGE_DOMAIN_CHECK_V31",
        "status": "PASS_IMAGE_DOMAIN" if passed else "REJECTED_IMAGE_DOMAIN",
        "passed": passed,
        "checks": checks,
        "evidence": asdict(evidence),
        "lens_transform_authorized": False,
        "output_pixel_domain": (
            f"PHYSICAL_LEFT_{evidence.model_width}x{evidence.model_height}"
            if passed
            else "UNAUTHORIZED"
        ),
    }


def check_stereo_geometry(
    frame_rows: list[dict[str, Any]],
    vertical_residuals: list[np.ndarray],
    signed_disparities: list[np.ndarray],
    *,
    reflected_for_model: bool,
    gate: EncodedStereoGateV1 | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_metrics(
        frame_rows, vertical_residuals, signed_disparities, gate=gate
    )
    metrics = aggregate["metrics"]
    checks = dict(aggregate["gates"])
    epipolar_checks = {
        key: checks[key]
        for key in (
            "robust_matches",
            "qualifying_frame_fraction",
            "spatial_coverage",
            "median_vertical",
            "p90_vertical",
            "p95_vertical",
            "disparity_sign",
        )
    }
    passed = all(epipolar_checks.values())
    physical_sign = metrics["dominant_disparity_sign"]
    expected_model_sign = (
        "POSITIVE_LEFT_MINUS_RIGHT"
        if reflected_for_model and physical_sign == "NEGATIVE_LEFT_MINUS_RIGHT"
        else physical_sign
    )
    model_sign_pass = expected_model_sign == "POSITIVE_LEFT_MINUS_RIGHT"
    return {
        "schema_version": "EXACT78_STEREO_GEOMETRY_CHECK_V31",
        "status": (
            "PASS_HORIZONTAL_EPIPOLAR"
            if passed and model_sign_pass
            else "REJECTED_MODEL_DISPARITY_SIGN"
            if passed
            else "REJECTED_STEREO_GEOMETRY"
        ),
        "passed": passed and model_sign_pass,
        "horizontal_epipolar_passed": passed,
        "model_disparity_sign_passed": model_sign_pass,
        "physical_domain_disparity_sign": physical_sign,
        "model_domain_disparity_sign": expected_model_sign,
        "horizontal_reflection_for_disparity_sign": reflected_for_model,
        "metrics": metrics,
        "checks": epipolar_checks,
        "thresholds": aggregate["thresholds"],
    }


def mirrored_intrinsics(intrinsics: np.ndarray, width: int) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or width < 1:
        raise ValueError("intrinsics must be 3x3 and width positive")
    output = matrix.copy()
    output[0, 2] = (width - 1.0) - output[0, 2]
    return output


def _valid_intrinsics(matrix: np.ndarray, width: int, height: int) -> bool:
    return bool(
        matrix.shape == (3, 3)
        and np.isfinite(matrix).all()
        and matrix[0, 0] > 0.0
        and matrix[1, 1] > 0.0
        and 0.0 <= matrix[0, 2] <= width - 1
        and 0.0 <= matrix[1, 2] <= height - 1
        and np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-9, rtol=0.0)
    )


def check_metric_conversion(
    *,
    physical_left_k: np.ndarray,
    physical_right_k: np.ndarray,
    model_left_k: np.ndarray,
    model_right_k: np.ndarray,
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    baseline_m: float,
    width: int,
    height: int,
    reflected_for_model: bool,
    outputs_unflipped_to_physical_left: bool,
    calibration_pixel_domain: str,
    expected_pixel_domain: str,
) -> dict[str, Any]:
    left = np.asarray(physical_left_k, np.float64)
    right = np.asarray(physical_right_k, np.float64)
    model_left = np.asarray(model_left_k, np.float64)
    model_right = np.asarray(model_right_k, np.float64)
    p_left = np.asarray(projection_left, np.float64)
    p_right = np.asarray(projection_right, np.float64)
    expected_model_left = mirrored_intrinsics(left, width) if reflected_for_model else left
    expected_model_right = mirrored_intrinsics(right, width) if reflected_for_model else right
    projection_shapes = p_left.shape == (3, 4) and p_right.shape == (3, 4)
    projection_finite = projection_shapes and np.isfinite(p_left).all() and np.isfinite(p_right).all()
    projection_intrinsics_match = bool(
        projection_finite
        and np.allclose(p_left[:, :3], left, atol=1e-6, rtol=1e-6)
        and np.allclose(p_right[:, :3], right, atol=1e-6, rtol=1e-6)
    )
    projection_baseline = None
    if projection_finite and abs(float(p_right[0, 0])) > 1e-12:
        projection_baseline = abs(float(p_right[0, 3] / p_right[0, 0]))
    checks = {
        "physical_left_k": _valid_intrinsics(left, width, height),
        "physical_right_k": _valid_intrinsics(right, width, height),
        "model_left_k": _valid_intrinsics(model_left, width, height),
        "model_right_k": _valid_intrinsics(model_right, width, height),
        "model_left_k_matches_adapter": np.allclose(
            model_left, expected_model_left, atol=1e-6, rtol=1e-6
        ),
        "model_right_k_matches_adapter": np.allclose(
            model_right, expected_model_right, atol=1e-6, rtol=1e-6
        ),
        "projection_intrinsics_match_physical_domain": projection_intrinsics_match,
        "baseline_positive_metre": np.isfinite(baseline_m) and baseline_m > 0.0,
        "projection_baseline_matches": bool(
            projection_baseline is not None
            and np.isclose(projection_baseline, baseline_m, atol=1e-6, rtol=1e-4)
        ),
        "calibration_domain_matches_output": calibration_pixel_domain == expected_pixel_domain,
        "unflip_restores_physical_left": (
            not reflected_for_model or outputs_unflipped_to_physical_left
        ),
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "schema_version": "EXACT78_METRIC_CONVERSION_CHECK_V31",
        "status": "PASS_LOCAL_STEREO_METRIC_DEV" if passed else "REJECTED_METRIC_CONVERSION",
        "passed": passed,
        "checks": checks,
        "baseline_m": float(baseline_m),
        "projection_baseline_m": projection_baseline,
        "calibration_pixel_domain": calibration_pixel_domain,
        "expected_pixel_domain": expected_pixel_domain,
        "depth_formula": "optical_Z_m = fx_px * baseline_m / positive_model_disparity_px",
        "output_intrinsics": left.tolist() if passed else None,
        "external_metric_authority": False,
    }


def combine_preflight(
    image_domain: Mapping[str, Any],
    stereo_geometry: Mapping[str, Any],
    metric_conversion: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "IMAGE_DOMAIN_CHECK": bool(image_domain.get("passed")),
        "STEREO_GEOMETRY_CHECK": bool(stereo_geometry.get("passed")),
        "METRIC_CONVERSION_CHECK": bool(metric_conversion.get("passed")),
    }
    all_passed = all(checks.values())
    return {
        "schema_version": "EXACT78_STEREO_PREFLIGHT_V31",
        "status": (
            "PASS_LOCAL_STEREO_METRIC_DEV"
            if all_passed
            else "DIAGNOSTIC_DISPARITY_ONLY"
            if checks["IMAGE_DOMAIN_CHECK"]
            else "REJECTED_IMAGE_DOMAIN"
        ),
        "checks": checks,
        "local_stereo_metric_dev": all_passed,
        "disparity_diagnostic_authorized": checks["IMAGE_DOMAIN_CHECK"],
        "external_metric_authority": False,
        "gpu_successor_authorized": all_passed,
        "image_domain": dict(image_domain),
        "stereo_geometry": dict(stereo_geometry),
        "metric_conversion": dict(metric_conversion),
    }
