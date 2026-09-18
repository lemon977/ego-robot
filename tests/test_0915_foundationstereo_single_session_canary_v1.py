from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

from chaoyang.ops import run_0915_foundationstereo_single_session_canary_v1 as subject


def test_physical_left_depth_keeps_unflipped_x_plus_disparity_domain() -> None:
    flipped = np.full((2, 8), 2.0, np.float32)
    disparity, depth, valid = subject.physical_left_depth(
        flipped, focal_px=10.0, baseline_m=0.2,
    )
    np.testing.assert_allclose(disparity, 2.0)
    assert valid[:, :6].all()
    assert not valid[:, 6:].any()
    np.testing.assert_allclose(depth[valid], 1.0)


def test_left_right_consistency_uses_x_right_equals_x_left_plus_d() -> None:
    left = np.full((2, 8), 2.0, np.float32)
    right = np.full((2, 8), 2.25, np.float32)
    geometric = np.ones((2, 8), bool)
    residual, testable, consistent = subject.left_right_consistency(
        left, right, geometric, max_residual_px=0.5,
    )
    assert testable[:, :6].all()
    assert not testable[:, 6:].any()
    np.testing.assert_allclose(residual[testable], 0.25)
    assert np.array_equal(consistent, testable)


def test_registration_map_is_nonlinear_map_not_identity_join() -> None:
    xx = np.broadcast_to(np.arange(1280, dtype=np.float32), (960, 1280)).copy()
    yy = np.broadcast_to(np.arange(960, dtype=np.float32)[:, None], (960, 1280)).copy()
    maps = subject.build_registration_maps(
        (xx, yy), source_width=2048, source_height=1536,
    )
    assert maps["depth_to_sam_resize_xy"].shape == (480, 640, 2)
    np.testing.assert_allclose(
        maps["depth_pixel_to_full_rectified_left_xy"][0, 0], [0.5, 0.5],
    )
    # OpenCV pixel-centre resize maps source 0.5 to target 0.125 here;
    # critically, this is not an identity registration.
    np.testing.assert_allclose(
        maps["depth_to_sam_resize_xy"][0, 0], [0.125, 0.125], atol=1e-6,
    )
    assert maps["sam_resize_in_bounds"].dtype == bool


def _rows(count: int = 10) -> list[dict]:
    return [{
        "geometric_valid_fraction": 0.80,
        "lr_testable_fraction": 0.75,
        "lr_consistent_fraction_of_testable": 0.90,
        "lr_residual_p90_px": 0.5,
        "final_valid_fraction": 0.65,
        "depth_p50_m": 0.55 + 0.001 * index,
        "formula_recompute_max_abs_error_m": 0.0,
        "edge": {"rgb_supported_disparity_edge_fraction": 0.50},
    } for index in range(count)]


def test_quality_gate_requires_decode_lr_temporal_and_edges() -> None:
    passed = subject.aggregate_quality(_rows(), 10)
    assert passed["passed"] is True
    assert set(passed["gates"]) == {
        "full_decode", "formula_recompute", "geometric_validity", "lr_testable_coverage",
        "lr_consistency", "final_validity", "temporal_distribution_stability",
        "rgb_edge_support",
    }
    failed = subject.aggregate_quality(_rows(9), 10)
    assert failed["gates"]["full_decode"] is False
    assert failed["passed"] is False


def test_runner_is_single_weight_t0_rectification_and_no_external_accuracy_claim() -> None:
    source = inspect.getsource(subject)
    assert subject.MODEL_WEIGHT.endswith("model_best_bp2.pth")
    assert "EXACT_T0_ACCEPTED_CALIBRATION_NO_REESTIMATION" in source
    assert "SAME_SESSION_INTRINSICS_HELDOUT_ESTIMATED_RECTIFIED" in source
    assert '"external_accuracy": "UNVERIFIED"' in source
    assert "ABSENT_NOT_FABRICATED" in source
    assert "PICO26" not in source
    assert "trackingData_hand" in source and "NOT_CONSUMED" in source


def test_runner_uses_pinned_environment_and_central_gpu_lease() -> None:
    assert subject.PINNED_PYTHON_LAUNCHER.name == "foundationstereo_gpu_python.sh"
    assert subject.CENTRAL_GPU_LEASE.name == "run_gpu_command_with_v71_lease.py"
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert '"--priority", "CANARY"' in source
    assert '"--", *worker_command' in source


def test_orchestrator_polls_worker_before_reading_released_lease() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    sleep_at = source.index("time.sleep(30)")
    exit_recheck_at = source.index("if process.poll() is not None:", sleep_at)
    lease_read_at = source.index(
        'lease_path = ROOT / "_run/current/GPU_LEASE.json"', sleep_at,
    )
    assert sleep_at < exit_recheck_at < lease_read_at


def test_namespaces_are_task_owned_and_visual_is_fixed() -> None:
    subject.validate_output_namespace(
        subject.OUTPUT_NAMESPACE / "attempts/attempt_0001",
        subject.VISUAL_NAMESPACE,
    )
    with pytest.raises(RuntimeError, match="output must stay"):
        subject.validate_output_namespace(
            subject.OUTPUT_NAMESPACE.parent / "another_task/attempt_0001",
            subject.VISUAL_NAMESPACE,
        )
    with pytest.raises(RuntimeError, match="visual root must equal"):
        subject.validate_output_namespace(
            subject.OUTPUT_NAMESPACE / "attempts/attempt_0001",
            subject.VISUAL_NAMESPACE.parent / "other",
        )


def test_edge_support_returns_bounded_fraction() -> None:
    image = np.zeros((960, 1280, 3), np.uint8)
    image[:, 640:] = 255
    disparity = np.ones((480, 640), np.float32)
    disparity[:, 320:] = 10.0
    metrics = subject.edge_support_metrics(image, disparity, np.ones_like(disparity, bool))
    assert metrics["strong_disparity_edge_pixels"] > 0
    assert 0.0 <= metrics["rgb_supported_disparity_edge_fraction"] <= 1.0
