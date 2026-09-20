from __future__ import annotations

import numpy as np

from chaoyang.pipeline.exact78_stereo_preflight_v31 import (
    ImageDomainEvidenceV31,
    check_image_domain,
    check_metric_conversion,
    check_stereo_geometry,
    combine_preflight,
    mirrored_intrinsics,
)
from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import frame_metrics


def image_evidence(**changes) -> ImageDomainEvidenceV31:
    values = {
        "sbs_width": 4096,
        "sbs_height": 1536,
        "eye_width": 2048,
        "eye_height": 1536,
        "model_width": 640,
        "model_height": 480,
        "physical_left_source_index": 1,
        "physical_right_source_index": 0,
        "decoded_vst_already_undistorted": True,
        "lens_undistortion_applied": False,
        "horizontal_reflection_for_disparity_sign": True,
        "outputs_unflipped_to_physical_left": True,
        "maximum_coordinate_roundtrip_error_px": 0.0,
        "mismatched_rgb_roundtrip_pixels": 0,
    }
    values.update(changes)
    return ImageDomainEvidenceV31(**values)


def stereo_rows(disparity_px: float):
    rows, vertical, disparity = [], [], []
    for frame in range(150):
        x = np.linspace(20.0, 620.0, 20)
        y = np.linspace(20.0, 460.0, 20)
        left = np.stack((x, y), axis=1)
        right = np.stack((x - disparity_px, y + 1.0), axis=1)
        rows.append(frame_metrics(left, right, width=640, height=480))
        vertical.append(np.abs(left[:, 1] - right[:, 1]))
        disparity.append(left[:, 0] - right[:, 0])
    return rows, vertical, disparity


def test_image_domain_rejects_repeated_lens_undistortion() -> None:
    result = check_image_domain(image_evidence(lens_undistortion_applied=True))
    assert result["status"] == "REJECTED_IMAGE_DOMAIN"
    assert result["checks"]["no_repeated_lens_undistortion"] is False


def test_negative_physical_disparity_passes_only_with_explicit_reflection() -> None:
    rows, vertical, disparity = stereo_rows(-18.0)
    passed = check_stereo_geometry(
        rows, vertical, disparity, reflected_for_model=True
    )
    rejected = check_stereo_geometry(
        rows, vertical, disparity, reflected_for_model=False
    )
    assert passed["status"] == "PASS_HORIZONTAL_EPIPOLAR"
    assert passed["physical_domain_disparity_sign"] == "NEGATIVE_LEFT_MINUS_RIGHT"
    assert passed["model_domain_disparity_sign"] == "POSITIVE_LEFT_MINUS_RIGHT"
    assert rejected["status"] == "REJECTED_MODEL_DISPARITY_SIGN"


def test_metric_conversion_closes_current_output_domain_and_mirror_intrinsics() -> None:
    left = np.asarray([[300.0, 0.0, 317.0], [0.0, 301.0, 241.0], [0.0, 0.0, 1.0]])
    right = left.copy()
    p_left = np.concatenate((left, np.zeros((3, 1))), axis=1)
    p_right = np.concatenate((right, np.zeros((3, 1))), axis=1)
    p_right[0, 3] = -left[0, 0] * 0.064
    result = check_metric_conversion(
        physical_left_k=left,
        physical_right_k=right,
        model_left_k=mirrored_intrinsics(left, 640),
        model_right_k=mirrored_intrinsics(right, 640),
        projection_left=p_left,
        projection_right=p_right,
        baseline_m=0.064,
        width=640,
        height=480,
        reflected_for_model=True,
        outputs_unflipped_to_physical_left=True,
        calibration_pixel_domain="PHYSICAL_LEFT_640x480",
        expected_pixel_domain="PHYSICAL_LEFT_640x480",
    )
    assert result["status"] == "PASS_LOCAL_STEREO_METRIC_DEV"
    assert result["external_metric_authority"] is False
    assert result["output_intrinsics"] == left.tolist()


def test_metric_failure_keeps_disparity_diagnostic_but_blocks_gpu_successor() -> None:
    rows, vertical, disparity = stereo_rows(-18.0)
    image = check_image_domain(image_evidence())
    geometry = check_stereo_geometry(rows, vertical, disparity, reflected_for_model=True)
    metric = {"passed": False, "status": "REJECTED_METRIC_CONVERSION"}
    result = combine_preflight(image, geometry, metric)
    assert result["status"] == "DIAGNOSTIC_DISPARITY_ONLY"
    assert result["disparity_diagnostic_authorized"] is True
    assert result["local_stereo_metric_dev"] is False
    assert result["gpu_successor_authorized"] is False
