from __future__ import annotations

import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import FOUNDATION_WEIGHT, build_packet
from chaoyang.ops.run_0915_robot15h_foundationstereo_wave0_v1 import aggregate


def test_foundation_wave0_binds_one_weight_and_forbids_lens_remap() -> None:
    packet = build_packet("0915_robot15h_foundationstereo_wave0_v1")
    assert packet["weights"] == [FOUNDATION_WEIGHT]
    assert "crop_resize_only_no_lens_remap" in packet["prerequisites"]
    assert "simultaneous_horizontal_reflection_no_camera_swap" in packet["prerequisites"]


def test_dynamic_full_decode_uses_session_denominator() -> None:
    rows = [{
        "geometric_valid_fraction": 0.8, "lr_testable_fraction": 0.8,
        "lr_consistent_fraction": 0.9, "final_valid_fraction": 0.7,
        "depth_p50_m": 0.5 + i * 0.001,
    } for i in range(7)]
    alignment = {"maximum_absolute_channel_error": 0, "mismatched_pixels": 0,
                 "coordinate_roundtrip_max_abs_error_px": 0.0}
    assert aggregate(rows, alignment, 7)["gates"]["full_decode"] is True
    assert aggregate(rows, alignment, 8)["gates"]["full_decode"] is False


def test_quality_rejection_does_not_gain_metric_contact_authority() -> None:
    rows = [{
        "geometric_valid_fraction": 0.01, "lr_testable_fraction": 0.01,
        "lr_consistent_fraction": 0.0, "final_valid_fraction": 0.0,
        "depth_p50_m": None,
    } for _ in range(3)]
    alignment = {"maximum_absolute_channel_error": 0, "mismatched_pixels": 0,
                 "coordinate_roundtrip_max_abs_error_px": 0.0}
    quality = aggregate(rows, alignment, 3)
    assert quality["passed"] is False
    assert np.isfinite(quality["metrics"]["median_geometric_valid_fraction"])


def test_foundation_recovery_is_one_session_and_gates_geometry() -> None:
    recovery = build_packet("0915_robot15h_foundationstereo_wave0_recovery_v1")
    assert recovery["weights"] == [FOUNDATION_WEIGHT]
    assert recovery["dag_dependencies"] == ["0915_robot15h_foundationstereo_wave0_v1"]
    assert "rerun_only_get_potato_chips_0915_042" in recovery["prerequisites"]
    geometry = build_packet("0915_robot15h_geometry_object6d_wave0_v1")
    assert "0915_robot15h_foundationstereo_wave0_recovery_v1" in geometry["dag_dependencies"]
    assert "0915_robot15h_foundationstereo_wave0_v1" not in geometry["dag_dependencies"]
