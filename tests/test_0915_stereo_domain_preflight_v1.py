from __future__ import annotations

import inspect

import numpy as np
import pytest

from chaoyang.ops import analyze_0915_stereo_domain_preflight_v1 as subject


def test_source_eye_split_obeys_physical_source_indices() -> None:
    eye_width, eye_height = 4, 3
    source_zero = np.full((eye_height, eye_width, 3), 17, np.uint8)
    source_one = np.full((eye_height, eye_width, 3), 93, np.uint8)
    sbs = np.concatenate((source_zero, source_one), axis=1)

    physical_left, physical_right = subject.split_physical_eyes(
        sbs,
        eye_width=eye_width,
        eye_height=eye_height,
        source_indices=(1, 0),
    )

    assert np.all(physical_left == 93)
    assert np.all(physical_right == 17)


def test_vertical_residual_reports_median_p90_p95_and_coverage() -> None:
    left = np.asarray([
        [5.0, 5.0], [25.0, 25.0], [45.0, 45.0], [65.0, 65.0],
        [85.0, 85.0],
    ])
    residual = np.asarray([0.0, 1.0, 2.0, 3.0, 10.0])
    right = left.copy()
    right[:, 1] += residual

    result = subject.vertical_residual_summary(
        left, right, width=100, height=100, grid_shape=(2, 2),
    )

    assert result["robust_matches"] == 5
    assert result["median_vertical_error_px"] == 2.0
    assert result["p90_vertical_error_px"] == pytest.approx(7.2)
    assert result["p95_vertical_error_px"] == pytest.approx(8.6)
    assert result["spatial_grid_coverage"] == 0.5


def test_admission_is_fail_closed_and_raw_candidate_cannot_substitute() -> None:
    passing_quality = {"passed": True}
    assert subject.admission_decision(
        input_valid=True,
        source_indices=(1, 0),
        decode_valid=True,
        rectification_calibration_passed=True,
        calibrated_quality=passing_quality,
    )["gpu_depth_allowed"] is True

    missing_rectification = subject.admission_decision(
        input_valid=True,
        source_indices=(1, 0),
        decode_valid=True,
        rectification_calibration_passed=False,
        calibrated_quality=None,
    )
    assert missing_rectification["gpu_depth_allowed"] is False
    assert missing_rectification["selected_candidate"] is None
    assert missing_rectification["first_blocker"] == (
        "CALIBRATED_RECTIFICATION_HELDOUT_GATE_FAILED"
    )


def test_admission_rejects_wrong_eye_routing_even_with_good_quality() -> None:
    result = subject.admission_decision(
        input_valid=True,
        source_indices=(0, 1),
        decode_valid=True,
        rectification_calibration_passed=True,
        calibrated_quality={"passed": True},
    )
    assert result["gpu_depth_allowed"] is False
    assert result["first_blocker"] == "PHYSICAL_EYE_SOURCE_INDEX_MISMATCH"


def test_sampling_is_dynamic_and_source_has_no_rejected_geometry_literal() -> None:
    assert subject.sample_indices(7, 20) == list(range(7))
    assert subject.sample_indices(101, 3) == [0, 50, 100]
    source = inspect.getsource(subject)
    assert "2160" not in source
    assert "frame_count != 150" not in source
    assert "FoundationStereo" in source
    assert "model_inference_run\": False" in source
