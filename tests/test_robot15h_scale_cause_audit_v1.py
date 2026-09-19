from __future__ import annotations

from chaoyang.governance.robot15h_task_specs_v1 import build_packet
from chaoyang.ops.run_0915_robot15h_scale_cause_audit_v1 import admitted_rows, affine_fit


def _row(frame: int, hand: str, x: float, y: float) -> dict[str, object]:
    return {
        "frame_id": frame, "hand_id": hand,
        "mano_surface_depth_m": x, "stereo_surface_depth_m": y,
        "status": "OBSERVED_SURFACE_PAIR", "whole_frame_hand_contact_excluded": False,
        "support_count": 30,
    }


def test_scale_audit_packet_forbids_contact_fit_and_bound_change() -> None:
    packet = build_packet("0915_robot15h_scale_cause_audit_v1")
    assert packet["weights"] == "ABSENT"
    assert "contact_and_object_fit_FORBIDDEN" in packet["prerequisites"]
    assert "scale_bound_change_FORBIDDEN" in packet["prerequisites"]
    assert "gpu_FORBIDDEN" in packet["prerequisites"]


def test_affine_slope_is_named_diagnostic_and_can_differ_by_stratum() -> None:
    left = [_row(i, "left", 0.2 + i * 0.01, 0.1 + 0.5 * (0.2 + i * 0.01)) for i in range(8)]
    right = [_row(i, "right", 0.3 + i * 0.01, 0.02 + 0.9 * (0.3 + i * 0.01)) for i in range(8)]
    rows = admitted_rows(left + right)
    assert abs(affine_fit(left)["scale"] - 0.5) < 1e-9
    assert abs(affine_fit(right)["scale"] - 0.9) < 1e-9
    assert affine_fit(rows)["status"] == "FIT_DIAGNOSTIC_ONLY"


def test_contact_excluded_or_low_support_rows_never_enter_scale_fit() -> None:
    rows = [_row(0, "left", 0.2, 0.2), _row(1, "left", 0.3, 0.3)]
    rows[0]["whole_frame_hand_contact_excluded"] = True
    rows[1]["support_count"] = 2
    assert admitted_rows(rows) == []
