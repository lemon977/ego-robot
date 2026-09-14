"""Fail-closed stage admission for new poker/chips sessions.

The evaluator consumes *summaries* produced by source/HaWoR/Mask/Clean/Robot
workers.  It never infers a PASS from a missing field.  Its purpose is to keep
failure ownership explicit: capture, HaWoR, Mask, Clean, or Robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


REQUIRED_SEGMENTS = (
    "EMPTY_TABLE_STATIC",
    "EMPTY_TABLE_VIEW_SWEEP",
    "OBJECT_CONTACT_FIT",
    "OBJECT_LIFT_NEGATIVE",
    "OBJECT_CONTACT_EVAL",
    "SLEEVE_FOREGROUND_CALIBRATION",
    "NORMAL_TASK_FIT",
    "NORMAL_TASK_EVAL",
    "END_EMPTY_TABLE",
)

MANO21_ORDER = "MANO21_WRIST0_TIPS_4_8_12_16_20_MCPS_2_5_9_13_17"


@dataclass(frozen=True)
class GateFailure:
    gate: str
    owner: str
    reason: str


def _require_bool(data: Mapping[str, Any], key: str, gate: str, owner: str) -> GateFailure | None:
    if data.get(key) is not True:
        return GateFailure(gate, owner, f"{key} must be true")
    return None


def _maximum(data: Mapping[str, Any], key: str, limit: float, gate: str, owner: str) -> GateFailure | None:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value > limit:
        return GateFailure(gate, owner, f"{key} must be <= {limit}; got {value!r}")
    return None


def _minimum(data: Mapping[str, Any], key: str, limit: float, gate: str, owner: str) -> GateFailure | None:
    value = data.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < limit:
        return GateFailure(gate, owner, f"{key} must be >= {limit}; got {value!r}")
    return None


def _source_failures(source: Mapping[str, Any]) -> list[GateFailure]:
    gate, owner = "S0_SOURCE", "CAPTURE"
    failures = [
        failure
        for failure in (
            _require_bool(source, "decode_ok", gate, owner),
            _maximum(source, "reset_count", 0, gate, owner),
        )
        if failure is not None
    ]
    frame_count = source.get("frame_count")
    timestamp_count = source.get("timestamp_count")
    intrinsics_count = source.get("intrinsics_count")
    c2w_count = source.get("c2w_count")
    count_types_ok = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (frame_count, timestamp_count, intrinsics_count, c2w_count)
    )
    counts_ok = (
        count_types_ok
        and frame_count > 0
        and timestamp_count == frame_count
        and c2w_count == frame_count
        and (
            intrinsics_count == frame_count
            or (source.get("intrinsics_session_constant") is True and intrinsics_count == 1)
        )
    )
    if not counts_ok:
        failures.append(
            GateFailure(
                gate,
                owner,
                "frame/timestamp/c2w counts must match; intrinsics must be per-frame "
                "or one explicitly session-constant calibration; "
                f"got frame={frame_count!r}, timestamp={timestamp_count!r}, "
                f"intrinsics={intrinsics_count!r}, c2w={c2w_count!r}",
            )
        )
    segments = source.get("segments")
    if (
        not isinstance(segments, list)
        or len(segments) != len(REQUIRED_SEGMENTS)
        or set(segments) != set(REQUIRED_SEGMENTS)
    ):
        missing = sorted(set(REQUIRED_SEGMENTS) - set(segments or []))
        extra = sorted(set(segments or []) - set(REQUIRED_SEGMENTS))
        failures.append(GateFailure(gate, owner, f"nine-segment contract mismatch; missing={missing}, extra={extra}"))
    return failures


def _capture_asset_failures(source: Mapping[str, Any]) -> list[GateFailure]:
    gate, owner = "S0_CAPTURE_ASSETS", "CAPTURE"
    checks = (
        _require_bool(source, "same_session_clean_plate", gate, owner),
        _require_bool(source, "metric_measurement_authority", gate, owner),
        _require_bool(source, "peripheral_fiducials_visible", gate, owner),
    )
    return [failure for failure in checks if failure is not None]


def _hawor_side_failures(side: str, data: Mapping[str, Any], *, robot: bool) -> list[GateFailure]:
    gate = "S1_HAWOR_ROBOT" if robot else "S1_HAWOR_MASK"
    prefix = f"{side}."
    limits = {
        "expected_active_frame_count": (1, "min"),
        "valid_fraction": (0.98 if robot else 0.95, "min"),
        "max_invalid_gap_frames": (3 if robot else 8, "max"),
        "confidence_median": (0.70 if robot else 0.65, "min"),
        "confidence_p05": (0.50 if robot else 0.45, "min"),
        "joint_overlay_p05_fraction": (0.90 if robot else 0.85, "min"),
        "reprojection_p95_px": (12.0 if robot else 20.0, "max"),
        "bone_length_cv_max": (0.08, "max"),
        "positive_depth_fraction": (0.995, "min"),
        "identity_switch_count": (0, "max"),
        "duplicate_track_frame_count": (0, "max"),
        "root_rotation_orthogonality_max": (1e-4, "max"),
    }
    if robot:
        limits["direct_observation_fraction"] = (0.90, "min")
        limits["contact_canary_direct_fraction"] = (1.0, "min")
        limits["wrist_step_p99_mm"] = (80.0, "max")
        limits["wrist_step_max_mm"] = (150.0, "max")
    failures: list[GateFailure] = []
    for key, (limit, direction) in limits.items():
        failure = (
            _minimum(data, key, limit, gate, "HAWOR")
            if direction == "min"
            else _maximum(data, key, limit, gate, "HAWOR")
        )
        if failure:
            failures.append(GateFailure(failure.gate, failure.owner, prefix + failure.reason))
    if data.get("named_joint_mapping_verified") is not True:
        failures.append(GateFailure(gate, "HAWOR", prefix + "named_joint_mapping_verified must be true"))
    if data.get("expected_active_definition_verified") is not True:
        failures.append(
            GateFailure(gate, "HAWOR", prefix + "expected_active_definition_verified must be true")
        )
    if data.get("visual_overlay_canary_verified") is not True:
        failures.append(GateFailure(gate, "HAWOR", prefix + "visual_overlay_canary_verified must be true"))
    if robot:
        if data.get("contact_canary_definition_verified") is not True:
            failures.append(
                GateFailure(gate, "HAWOR", prefix + "contact_canary_definition_verified must be true")
            )
        failure = _minimum(data, "contact_canary_frame_count", 1, gate, "HAWOR")
        if failure:
            failures.append(GateFailure(failure.gate, failure.owner, prefix + failure.reason))
    return failures


def _hawor_failures(hawor: Mapping[str, Any], *, robot: bool) -> list[GateFailure]:
    gate = "S1_HAWOR_ROBOT" if robot else "S1_HAWOR_MASK"
    failures: list[GateFailure] = []
    if hawor.get("joint_order") != MANO21_ORDER:
        failures.append(GateFailure(gate, "HAWOR", f"joint_order must be {MANO21_ORDER}"))
    if hawor.get("identity_assignment_verified") is not True:
        failures.append(GateFailure(gate, "HAWOR", "identity_assignment_verified must be true"))
    failure = _maximum(hawor, "dual_hand_collapse_frame_count", 0, gate, "HAWOR")
    if failure:
        failures.append(failure)
    sides = hawor.get("sides")
    if not isinstance(sides, Mapping):
        return failures + [GateFailure(gate, "HAWOR", "sides must be an object")]
    required_sides = hawor.get("required_sides")
    if not isinstance(required_sides, list) or not required_sides or any(side not in {"left", "right"} for side in required_sides):
        return failures + [GateFailure(gate, "HAWOR", "required_sides must contain left and/or right")]
    for side in required_sides:
        row = sides.get(side)
        if not isinstance(row, Mapping):
            failures.append(GateFailure(gate, "HAWOR", f"missing required side {side}"))
        else:
            failures.extend(_hawor_side_failures(side, row, robot=robot))
    return failures


def _mask_failures(mask: Mapping[str, Any]) -> list[GateFailure]:
    gate, owner = "S3_MASK", "MASK"
    checks = (
        _minimum(mask, "required_role_canary_recall", 1.0, gate, owner),
        _maximum(mask, "human_object_overlap_pixels", 0, gate, owner),
        _maximum(mask, "fixture_in_human_area_fraction", 0.002, gate, owner),
        _minimum(mask, "flow_aligned_iou_median", 0.85, gate, owner),
        _maximum(mask, "identity_switch_count", 0, gate, owner),
        _minimum(mask, "published_frame_fraction", 1.0, gate, owner),
        _require_bool(mask, "object_protection_verified", gate, owner),
    )
    return [failure for failure in checks if failure is not None]


def _clean_failures(clean: Mapping[str, Any]) -> list[GateFailure]:
    gate, owner = "S4_CLEAN", "CLEAN"
    checks = (
        _require_bool(clean, "same_session_same_camera_donor", gate, owner),
        _require_bool(clean, "donor_archive_crc_verified", gate, owner),
        _require_bool(clean, "donor_purity_verified", gate, owner),
        _maximum(clean, "changed_pixels_outside_authorized_domain", 0, gate, owner),
        _maximum(clean, "changed_protected_object_pixels", 0, gate, owner),
        _minimum(clean, "source_map_known_fraction", 0.999, gate, owner),
        _require_bool(clean, "per_pixel_source_map_verified", gate, owner),
        _maximum(clean, "residual_area_fraction", 0.001, gate, owner),
        _maximum(clean, "human_residual_ratio_max", 0.02, gate, owner),
        _maximum(clean, "tracker_residual_ratio_max", 0.02, gate, owner),
        _maximum(clean, "stable_background_changed_ratio_max", 0.05, gate, owner),
        _maximum(clean, "yellow_green_chroma_ghost_ratio_max", 0.0002, gate, owner),
        _maximum(clean, "boundary_to_context_gradient_ratio_max", 1.8, gate, owner),
        _maximum(clean, "boundary_luma_halo_delta_max", 12.0, gate, owner),
        _maximum(clean, "illumination_mismatch_lab_p95_max", 12.0, gate, owner),
        _maximum(clean, "shadow_residual_area_fraction", 0.001, gate, owner),
        _maximum(clean, "flow_stabilized_temporal_flicker_lab_p95", 6.0, gate, owner),
        _require_bool(clean, "consecutive_temporal_windows_verified", gate, owner),
        _maximum(clean, "object_codec_abs_error_p99", 16.0, gate, owner),
        _require_bool(clean, "codec_audit_verified", gate, owner),
        _require_bool(clean, "lossless_audit_master_available", gate, owner),
        _require_bool(clean, "visual_review_no_human_or_seam", gate, owner),
    )
    return [failure for failure in checks if failure is not None]


def _robot_failures(robot: Mapping[str, Any]) -> list[GateFailure]:
    gate, owner = "S5_ROBOT", "ROBOT"
    checks = (
        _require_bool(robot, "session_constant_base", gate, owner),
        _maximum(robot, "required_pose_position_max_mm", 10.0, gate, owner),
        _maximum(robot, "required_pose_rotation_max_deg", 5.0, gate, owner),
        _maximum(robot, "joint_velocity_max_rad_per_frame_30fps", 0.12, gate, owner),
        _maximum(robot, "joint_acceleration_max_rad_per_frame2_30fps", 0.06, gate, owner),
        _minimum(robot, "dense_signed_distance_min_mm", -1.0, gate, owner),
        _minimum(robot, "required_contact_signed_distance_min_mm", 0.0, gate, owner),
        _maximum(robot, "required_contact_signed_distance_max_mm", 3.0, gate, owner),
        _maximum(robot, "exact_self_collision_count", 0, gate, owner),
        _require_bool(robot, "joint_limits_pass", gate, owner),
        _require_bool(robot, "object_depth_occlusion_pass", gate, owner),
    )
    return [failure for failure in checks if failure is not None]


def evaluate_session(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return ordered stage decisions and the first accountable failure."""

    stage_builders = (
        ("source", "CAPTURE", _source_failures),
        ("capture_assets", "CAPTURE", _capture_asset_failures),
        ("hawor_mask", "HAWOR", lambda value: _hawor_failures(value, robot=False)),
        ("hawor_robot", "HAWOR", lambda value: _hawor_failures(value, robot=True)),
        ("mask", "MASK", _mask_failures),
        ("clean", "CLEAN", _clean_failures),
        ("robot", "ROBOT", _robot_failures),
    )
    stages: dict[str, Any] = {}
    first_failure: dict[str, str] | None = None
    for name, owner, builder in stage_builders:
        if name.startswith("hawor_"):
            value = metrics.get("hawor")
        elif name == "capture_assets":
            value = metrics.get("source")
        else:
            value = metrics.get(name)
        if value is None:
            stages[name] = {"status": "NOT_EVALUATED", "owner": owner, "failures": ["stage summary missing"]}
            if first_failure is None:
                first_failure = {"stage": name, "owner": owner, "reason": "stage summary missing"}
            continue
        if not isinstance(value, Mapping):
            failures = [GateFailure(name.upper(), owner, "stage summary must be an object")]
        else:
            failures = builder(value)
        stages[name] = {
            "status": "PASS" if not failures else "FAIL",
            "owner": owner,
            "failures": [failure.reason for failure in failures],
        }
        if failures and first_failure is None:
            first_failure = {"stage": name, "owner": failures[0].owner, "reason": failures[0].reason}

    source_pass = stages["source"]["status"] == "PASS"
    source_summary = metrics.get("source")
    if not isinstance(source_summary, Mapping):
        source_summary = {}
    clean_asset_ready = source_summary.get("same_session_clean_plate") is True
    robot_asset_ready = (
        source_summary.get("metric_measurement_authority") is True
        and source_summary.get("peripheral_fiducials_visible") is True
    )
    mask_seed_ready = source_pass and stages["hawor_mask"]["status"] == "PASS"
    robot_seed_ready = source_pass and robot_asset_ready and stages["hawor_robot"]["status"] == "PASS"
    return {
        "schema_version": "chaoyang-session-stage-admission-v1",
        "session_id": metrics.get("session_id"),
        "task_id": metrics.get("task_id"),
        "stages": stages,
        "first_failure": first_failure,
        "routing": {
            "mask_auto_seed_allowed": mask_seed_ready,
            "robot_auto_seed_allowed": robot_seed_ready,
            "manual_mask_route_allowed": source_pass,
            "clean_capture_ready": source_pass and clean_asset_ready,
            "robot_capture_ready": source_pass and robot_asset_ready,
            "downstream_must_not_blame_mask_when_source_or_hawor_failed": not (source_pass and robot_seed_ready),
        },
    }
