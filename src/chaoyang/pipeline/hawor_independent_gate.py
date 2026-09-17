"""Fail-closed HaWoR seed admission with independent identity/contour authority.

The HaWoR NPZ is allowed to prove numeric structure and provenance only.  It is
never allowed to prove that its own anatomical side labels or 2-D projection
match the image.  Those claims require a separately-authored review envelope.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import math
from pathlib import Path
from typing import Any

import numpy as np


INPUT_SCHEMA_VERSION = "hawor-independent-review-v1"
OUTPUT_SCHEMA_VERSION = "hawor-independent-gate-v1"
SIDES = ("left", "right")
MANO_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
MANO_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
STATUSES = frozenset({"PASS", "FAIL", "HOLD", "NOT_EVALUATED"})
INDEPENDENT_AUTHORITIES = frozenset({"INDEPENDENT_QA", "HUMAN_REVIEW"})
CONTOUR_METHODS = frozenset(
    {
        "FROZEN_HUMAN_SILHOUETTE_LABELS",
        "INDEPENDENT_SEGMENTATION_SILHOUETTE_LABELS",
    }
)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def longest_false_run(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def longest_missing_run_in_denominator(observed: np.ndarray, denominator: np.ndarray) -> int:
    """Count missing runs without joining separate visible/active intervals."""

    best = current = 0
    for is_observed, is_required in zip(observed, denominator, strict=True):
        if not is_required or is_observed:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def _percentile(values: np.ndarray, q: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, q)) if finite.size else None


def _gate(status: str, authority: str, *reason_codes: str, **metrics: Any) -> dict[str, Any]:
    return {
        "status": status,
        "authority": authority,
        "reason_codes": sorted(set(reason_codes)),
        "metrics": metrics,
    }


def _interval_mask(intervals: Sequence[Sequence[int]], frame_count: int) -> np.ndarray:
    result = np.zeros(frame_count, dtype=bool)
    for start, stop in intervals:
        result[int(start) : int(stop)] = True
    return result


def validate_independent_review(
    review: Mapping[str, Any], *, frame_count: int, task_id: str, session_id: str
) -> list[str]:
    """Validate review semantics beyond the JSON schema.

    A syntactically valid review still fails closed if it is produced by
    HaWoR, does not cover the full identity timeline, or lacks frozen contour
    labels.  This function deliberately does not infer independence from a
    filename.
    """

    errors: list[str] = []
    if review.get("schema_version") != INPUT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {INPUT_SCHEMA_VERSION}")
    if review.get("task_id") != task_id:
        errors.append("task_id differs from HaWoR result")
    if review.get("session_id") != session_id:
        errors.append("session_id differs from HaWoR result")
    if review.get("frame_count") != frame_count:
        errors.append("frame_count differs from HaWoR NPZ")

    provider = review.get("provider")
    if not isinstance(provider, Mapping):
        errors.append("provider must be an object")
    else:
        authority = provider.get("authority")
        family = provider.get("family")
        if authority not in INDEPENDENT_AUTHORITIES:
            errors.append("provider.authority must be INDEPENDENT_QA or HUMAN_REVIEW")
        if not isinstance(family, str) or not family.strip():
            errors.append("provider.family must be non-empty")
        elif "hawor" in family.casefold():
            errors.append("provider.family must be independent from HaWoR")
        if provider.get("independent_from_hawor") is not True:
            errors.append("provider.independent_from_hawor must be true")
        if not isinstance(provider.get("provider_id"), str) or not provider["provider_id"].strip():
            errors.append("provider.provider_id must be non-empty")

    evidence = review.get("evidence")
    roles: set[str] = set()
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must be a non-empty array")
    else:
        for index, row in enumerate(evidence):
            if not isinstance(row, Mapping):
                errors.append(f"evidence[{index}] must be an object")
                continue
            role = row.get("role")
            if isinstance(role, str):
                roles.add(role)
            if not isinstance(row.get("path"), str) or not row["path"]:
                errors.append(f"evidence[{index}].path must be non-empty")
            digest = row.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                errors.append(f"evidence[{index}].sha256 must be lowercase 64-hex")
    for required_role in (
        "FULL_TIMELINE_IDENTITY_REVIEW",
        "HAND_SILHOUETTE_LABELS",
        "CONTOUR_METRICS",
        "EXPECTED_ACTIVE_LABELS",
        "MISSING_VISIBILITY_LABELS",
    ):
        if required_role not in roles:
            errors.append(f"evidence lacks {required_role}")

    active = review.get("expected_active")
    if not isinstance(active, Mapping) or active.get("status") not in STATUSES:
        errors.append("expected_active.status is invalid")
    elif active.get("status") == "PASS":
        if active.get("authority") not in INDEPENDENT_AUTHORITIES | {"CAPTURE"}:
            errors.append("expected_active PASS lacks independent/capture authority")
        per_side = active.get("per_side_intervals")
        if not isinstance(per_side, Mapping):
            errors.append("expected_active.per_side_intervals must be an object")
        else:
            for side in SIDES:
                intervals = per_side.get(side)
                if not isinstance(intervals, list) or not intervals:
                    errors.append(f"expected_active.per_side_intervals.{side} must be non-empty")
                    continue
                previous_stop = -1
                for index, interval in enumerate(intervals):
                    if (
                        not isinstance(interval, list)
                        or len(interval) != 2
                        or not all(isinstance(value, int) and not isinstance(value, bool) for value in interval)
                    ):
                        errors.append(f"expected_active {side} interval {index} is not [int,int]")
                        continue
                    start, stop = interval
                    if not (0 <= start < stop <= frame_count) or start < previous_stop:
                        errors.append(f"expected_active {side} interval {index} is invalid/overlapping")
                    previous_stop = stop
    if isinstance(active, Mapping) and active.get("status") != "PASS" and not active.get("reason_codes"):
        errors.append("expected_active non-PASS requires reason_codes")

    identity = review.get("anatomical_identity")
    if not isinstance(identity, Mapping) or identity.get("status") not in STATUSES:
        errors.append("anatomical_identity.status is invalid")
    elif identity.get("status") == "PASS":
        if identity.get("full_timeline_reviewed") is not True:
            errors.append("identity PASS requires full_timeline_reviewed=true")
        if identity.get("reviewed_frame_count") != frame_count:
            errors.append("identity PASS requires reviewed_frame_count == frame_count")
        for key in ("identity_switch_frames", "duplicate_track_frames", "collapse_frames"):
            if identity.get(key) != []:
                errors.append(f"identity PASS requires {key}=[]")
    if isinstance(identity, Mapping) and identity.get("status") != "PASS" and not identity.get("reason_codes"):
        errors.append("anatomical_identity non-PASS requires reason_codes")

    contour = review.get("contour_projection")
    if not isinstance(contour, Mapping) or contour.get("status") not in STATUSES:
        errors.append("contour_projection.status is invalid")
    elif contour.get("status") == "PASS":
        if contour.get("method") not in CONTOUR_METHODS:
            errors.append("contour PASS requires a frozen independent silhouette method")
        if contour.get("mask_distance_threshold_px") != 20:
            errors.append("contour PASS requires mask_distance_threshold_px=20")
        if contour.get("robot_distance_threshold_px") != 12:
            errors.append("contour PASS requires robot_distance_threshold_px=12")
        frames = contour.get("sampled_frame_indices")
        if (
            not isinstance(frames, list)
            or len(frames) < 24
            or len(frames) != len(set(frames))
            or frames != sorted(frames)
            or any(not isinstance(value, int) or not 0 <= value < frame_count for value in frames)
        ):
            errors.append("contour PASS requires >=24 unique sorted in-range sampled frames")
        per_side = contour.get("per_side")
        if not isinstance(per_side, Mapping):
            errors.append("contour_projection.per_side must be an object")
        else:
            for side in SIDES:
                row = per_side.get(side)
                if not isinstance(row, Mapping):
                    errors.append(f"contour_projection.per_side.{side} must be an object")
                    continue
                for metric_name in (
                    "mask_coverage_fraction_p05_at_20px",
                    "robot_coverage_fraction_p05_at_12px",
                ):
                    coverage = row.get(metric_name)
                    if not isinstance(coverage, (int, float)) or isinstance(coverage, bool):
                        errors.append(f"contour {side} {metric_name} must be numeric")
                    elif not math.isfinite(float(coverage)) or not 0 <= float(coverage) <= 1:
                        errors.append(f"contour {side} {metric_name} must be finite in [0,1]")
    if isinstance(contour, Mapping) and contour.get("status") != "PASS" and not contour.get("reason_codes"):
        errors.append("contour_projection non-PASS requires reason_codes")

    visibility = review.get("missing_visibility")
    if not isinstance(visibility, Mapping) or visibility.get("status") not in STATUSES:
        errors.append("missing_visibility.status is invalid")
    elif visibility.get("status") == "PASS":
        if not isinstance(active, Mapping) or active.get("status") != "PASS":
            errors.append("missing_visibility PASS requires expected_active PASS")
        if visibility.get("all_missing_frames_classified") is not True:
            errors.append("missing_visibility PASS requires all_missing_frames_classified=true")
        if (
            visibility.get("boundary_support_rule")
            != "INDEPENDENT_SILHOUETTE_AT_LEAST_19_OF_21_JOINTS_IN_FRAME"
        ):
            errors.append("missing_visibility PASS requires the independent 19-of-21 boundary rule")
        classified = visibility.get("classified_missing_frames")
        if not isinstance(classified, Mapping):
            errors.append("missing_visibility.classified_missing_frames must be an object")
        else:
            for side in SIDES:
                frames = classified.get(side)
                if (
                    not isinstance(frames, list)
                    or len(frames) != len(set(frames))
                    or any(
                        not isinstance(value, int)
                        or isinstance(value, bool)
                        or not 0 <= value < frame_count
                        for value in frames
                    )
                ):
                    errors.append(
                        f"missing_visibility.classified_missing_frames.{side} must be unique in-range ints"
                    )
        partitions = visibility.get("per_side_visibility_intervals")
        active_intervals = active.get("per_side_intervals") if isinstance(active, Mapping) else None
        if not isinstance(partitions, Mapping):
            errors.append("missing_visibility.per_side_visibility_intervals must be an object")
        elif not isinstance(active_intervals, Mapping):
            errors.append("visibility partition requires expected-active intervals")
        else:
            for side in SIDES:
                row = partitions.get(side)
                if not isinstance(row, Mapping):
                    errors.append(f"visibility partition {side} must be an object")
                    continue
                class_masks: list[np.ndarray] = []
                for class_name in (
                    "visible_required",
                    "boundary_supported",
                    "occluded_or_out_of_frame",
                ):
                    intervals = row.get(class_name)
                    if not isinstance(intervals, list):
                        errors.append(f"visibility partition {side}.{class_name} must be intervals")
                        continue
                    previous_stop = -1
                    valid_intervals = True
                    for index, interval in enumerate(intervals):
                        if (
                            not isinstance(interval, list)
                            or len(interval) != 2
                            or not all(isinstance(value, int) and not isinstance(value, bool) for value in interval)
                        ):
                            errors.append(f"visibility {side}.{class_name}[{index}] is not [int,int]")
                            valid_intervals = False
                            continue
                        start, stop = interval
                        if not (0 <= start < stop <= frame_count) or start < previous_stop:
                            errors.append(f"visibility {side}.{class_name}[{index}] invalid/overlapping")
                            valid_intervals = False
                        previous_stop = stop
                    if valid_intervals:
                        class_masks.append(_interval_mask(intervals, frame_count))
                if len(class_masks) == 3:
                    stacked = np.stack(class_masks)
                    expected_mask = _interval_mask(active_intervals.get(side, []), frame_count)
                    if np.any(stacked.sum(axis=0) > 1):
                        errors.append(f"visibility classes overlap for {side}")
                    if not np.array_equal(stacked.any(axis=0), expected_mask):
                        errors.append(f"visibility classes do not exactly partition expected-active {side}")
    if isinstance(visibility, Mapping) and visibility.get("status") != "PASS" and not visibility.get("reason_codes"):
        errors.append("missing_visibility non-PASS requires reason_codes")
    return errors


def verify_review_evidence(
    review: Mapping[str, Any], *, evidence_root: Path
) -> list[str]:
    errors: list[str] = []
    evidence = review.get("evidence")
    if not isinstance(evidence, list):
        return ["review evidence is not an array"]
    for index, row in enumerate(evidence):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            continue
        path = Path(row["path"])
        if not path.is_absolute():
            path = evidence_root / path
        if not path.is_file():
            errors.append(f"evidence[{index}] missing: {path}")
            continue
        actual = sha256_file(path)
        if actual != row.get("sha256"):
            errors.append(f"evidence[{index}] sha256 mismatch")
    return errors


def load_hawor_npz(path: Path) -> dict[str, np.ndarray]:
    required = (
        "joints_3d_camera",
        "joints_3d_world",
        "joints_2d",
        "root_orient_camera",
        "observed",
        "provenance",
        "detector_confidence",
        "original_frame_indices",
        "fps",
        "mano_joint_names",
        "anatomical_side_names",
        "mano_wrist_index",
        "mano_tip_indices",
        "mano_mcp_indices",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"HaWoR NPZ missing arrays: {missing}")
        return {name: np.asarray(archive[name]) for name in required}


def _structure_errors(arrays: Mapping[str, np.ndarray], frame_count: int) -> list[str]:
    errors: list[str] = []
    shapes = {
        "joints_3d_camera": (2, frame_count, 21, 3),
        "joints_3d_world": (2, frame_count, 21, 3),
        "joints_2d": (2, frame_count, 21, 2),
        "root_orient_camera": (2, frame_count, 3, 3),
        "observed": (2, frame_count),
        "provenance": (2, frame_count),
        "detector_confidence": (2, frame_count),
        "original_frame_indices": (frame_count,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            errors.append(f"{name} shape {arrays[name].shape} != {shape}")
    if tuple(str(value) for value in arrays["mano_joint_names"].tolist()) != MANO_NAMES:
        errors.append("MANO21 joint names/order mismatch")
    if tuple(str(value) for value in arrays["anatomical_side_names"].tolist()) != SIDES:
        errors.append("anatomical side axis mismatch")
    if int(arrays["mano_wrist_index"].item()) != 0:
        errors.append("MANO wrist index is not 0")
    if tuple(int(value) for value in arrays["mano_tip_indices"].tolist()) != (4, 8, 12, 16, 20):
        errors.append("MANO tip indices mismatch")
    if tuple(int(value) for value in arrays["mano_mcp_indices"].tolist()) != (2, 5, 9, 13, 17):
        errors.append("MANO MCP indices mismatch")
    if not np.array_equal(arrays["original_frame_indices"], np.arange(frame_count)):
        errors.append("original frame mapping is not exact 0..N-1")
    return errors


def _side_metrics(
    arrays: Mapping[str, np.ndarray], side: int, active: np.ndarray
) -> dict[str, Any]:
    observed = arrays["observed"][side].astype(bool)
    eligible_observed = observed & active
    active_count = int(active.sum())
    fps = float(arrays["fps"].item())
    confidence = arrays["detector_confidence"][side].astype(np.float64)
    selected_confidence = confidence[eligible_observed]
    camera = arrays["joints_3d_camera"][side].astype(np.float64)
    world = arrays["joints_3d_world"][side].astype(np.float64)
    rotations = arrays["root_orient_camera"][side].astype(np.float64)

    active_indices = np.flatnonzero(active)
    active_observed = observed[active]
    missing_indices = active_indices[~active_observed]
    pair_active = active[:-1] & active[1:]
    pair_valid = pair_active & observed[:-1] & observed[1:]
    steps = np.linalg.norm(np.diff(world[:, 0], axis=0), axis=-1) * 1000.0
    selected_steps = steps[pair_valid]

    if eligible_observed.any():
        lengths = np.stack(
            [np.linalg.norm(world[:, child] - world[:, parent], axis=-1) for parent, child in MANO_EDGES],
            axis=1,
        )[eligible_observed]
        means = lengths.mean(axis=0)
        bone_cv = float(np.max(lengths.std(axis=0) / np.maximum(means, 1e-9)))
        selected_rotations = rotations[eligible_observed]
        orthogonality = float(
            np.max(
                np.abs(
                    np.transpose(selected_rotations, (0, 2, 1))
                    @ selected_rotations
                    - np.eye(3)
                )
            )
        )
        determinant_min = float(np.min(np.linalg.det(selected_rotations)))
        positive_depth = float((camera[eligible_observed, :, 2] > 0).mean())
    else:
        bone_cv = orthogonality = determinant_min = positive_depth = None

    finite_confidence = selected_confidence[np.isfinite(selected_confidence)]
    constant_one = bool(finite_confidence.size and np.allclose(finite_confidence, 1.0, atol=1e-7))
    p99_limit = 80.0 * 30.0 / fps
    max_limit = 150.0 * 30.0 / fps
    p99_step = _percentile(selected_steps, 99)
    max_step = float(selected_steps.max()) if selected_steps.size else None
    observed_fraction = float(eligible_observed.sum() / active_count) if active_count else 0.0
    provenance = arrays["provenance"][side].astype(str)
    direct = eligible_observed & (provenance == "OBSERVED")
    direct_fraction = float(direct.sum() / active_count) if active_count else 0.0
    metrics: dict[str, Any] = {
        "active_frames": active_count,
        "observed_frames": int(eligible_observed.sum()),
        "observed_fraction": observed_fraction,
        "direct_observation_fraction": direct_fraction,
        "longest_missing_gap_frames": longest_missing_run_in_denominator(observed, active),
        "missing_frame_indices": missing_indices.astype(int).tolist(),
        "confidence_median": _percentile(selected_confidence, 50),
        "confidence_p05": _percentile(selected_confidence, 5),
        "confidence_uninformative_constant_one": constant_one,
        "positive_depth_fraction": positive_depth,
        "bone_length_cv_max": bone_cv,
        "root_rotation_orthogonality_max": orthogonality,
        "root_rotation_determinant_min": determinant_min,
        "wrist_step_p99_mm": p99_step,
        "wrist_step_max_mm": max_step,
        "wrist_step_p99_limit_mm_at_fps": p99_limit,
        "wrist_step_max_limit_mm_at_fps": max_limit,
    }
    finite_common = all(
        isinstance(metrics[key], (int, float))
        and not isinstance(metrics[key], bool)
        and math.isfinite(float(metrics[key]))
        for key in (
            "confidence_median",
            "confidence_p05",
            "positive_depth_fraction",
            "bone_length_cv_max",
            "root_rotation_orthogonality_max",
            "root_rotation_determinant_min",
            "wrist_step_p99_mm",
            "wrist_step_max_mm",
        )
    )
    metrics["mask_numeric_pass"] = bool(
        active_count
        and finite_common
        and observed_fraction >= 0.95
        and metrics["longest_missing_gap_frames"] <= 8
        and float(metrics["confidence_median"]) >= 0.65
        and float(metrics["confidence_p05"]) >= 0.45
        and not constant_one
        and float(positive_depth) >= 0.995
        and float(bone_cv) <= 0.08
        and float(orthogonality) <= 1e-4
        and float(determinant_min) > 0
    )
    metrics["robot_numeric_pass"] = bool(
        active_count
        and finite_common
        and observed_fraction >= 0.98
        and metrics["longest_missing_gap_frames"] <= 3
        and float(metrics["confidence_median"]) >= 0.70
        and float(metrics["confidence_p05"]) >= 0.50
        and not constant_one
        and direct_fraction >= 0.90
        and float(positive_depth) >= 0.995
        and float(bone_cv) <= 0.08
        and float(p99_step) <= p99_limit
        and float(max_step) <= max_limit
        and float(orthogonality) <= 1e-4
        and float(determinant_min) > 0
    )
    return metrics


def _identity_diagnostic(arrays: Mapping[str, np.ndarray], active: np.ndarray) -> dict[str, Any]:
    observed = arrays["observed"].astype(bool)
    both = observed[0] & observed[1] & active
    joints = arrays["joints_2d"].astype(np.float64)
    if not both.any():
        return {
            "both_observed_frames": 0,
            "collapse_under_20px_frames": None,
            "wrist_separation_2d_px_min": None,
            "continuity_prefers_swap_transition_frames": [],
            "claim_limit": "HAWOR_NUMERIC_DIAGNOSTIC_NOT_ANATOMICAL_IDENTITY_AUTHORITY",
        }
    separation = np.linalg.norm(joints[0, :, 0] - joints[1, :, 0], axis=-1)
    transition_frames: list[int] = []
    for frame in range(1, joints.shape[1]):
        if not (both[frame - 1] and both[frame]):
            continue
        previous = joints[:, frame - 1, 0]
        current = joints[:, frame, 0]
        identity_cost = float(np.linalg.norm(current[0] - previous[0]) + np.linalg.norm(current[1] - previous[1]))
        swap_cost = float(np.linalg.norm(current[0] - previous[1]) + np.linalg.norm(current[1] - previous[0]))
        if swap_cost + 5.0 < identity_cost:
            transition_frames.append(frame)
    return {
        "both_observed_frames": int(both.sum()),
        "collapse_under_20px_frames": int(np.count_nonzero(separation[both] < 20.0)),
        "wrist_separation_2d_px_min": float(separation[both].min()),
        "continuity_prefers_swap_transition_frames": transition_frames,
        "claim_limit": "HAWOR_NUMERIC_DIAGNOSTIC_NOT_ANATOMICAL_IDENTITY_AUTHORITY",
    }


def _map_source_gate(value: Any) -> tuple[str, str]:
    text = str(value)
    if text.startswith("PASS"):
        return "PASS", text
    if text.startswith("FAIL"):
        return "FAIL", text
    return "HOLD", text


def evaluate_hawor_independent_gate(
    *,
    hawor_result: Mapping[str, Any],
    source_gate: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    artifact_errors: Sequence[str] = (),
    independent_review: Mapping[str, Any] | None = None,
    review_errors: Sequence[str] = (),
) -> dict[str, Any]:
    task_id = str(hawor_result.get("task_id", ""))
    session_id = str(hawor_result.get("session_id", ""))
    frame_count = int(hawor_result.get("frame_count", -1))
    structure_errors = _structure_errors(arrays, frame_count) if frame_count >= 0 else ["invalid frame_count"]
    if source_gate.get("task_id") != task_id:
        structure_errors.append("source task_id differs from HaWoR result")
    if source_gate.get("session_id") != session_id:
        structure_errors.append("source session_id differs from HaWoR result")
    provenance_values = sorted(set(arrays["provenance"].astype(str).reshape(-1).tolist()))
    provenance_errors: list[str] = []
    if not set(provenance_values) <= {"OBSERVED", "MISSING"}:
        provenance_errors.append("INTERPOLATED_OR_UNKNOWN_PROVENANCE_PRESENT")
    observed = arrays["observed"].astype(bool)
    provenance = arrays["provenance"].astype(str)
    if not np.array_equal(observed, provenance == "OBSERVED"):
        provenance_errors.append("OBSERVED_AND_PROVENANCE_DISAGREE")

    semantic_review_errors: list[str] = []
    if independent_review is not None:
        semantic_review_errors = validate_independent_review(
            independent_review,
            frame_count=frame_count,
            task_id=task_id,
            session_id=session_id,
        )
    all_review_errors = list(review_errors) + semantic_review_errors

    expected_masks = {side: np.ones(frame_count, dtype=bool) for side in SIDES}
    active_gate = _gate(
        "HOLD",
        "NONE",
        "EXPECTED_ACTIVE_AUTHORITY_MISSING",
        denominator="FULL_SESSION_DIAGNOSTIC_ONLY",
    )
    if independent_review is not None and not all_review_errors:
        expected = independent_review["expected_active"]
        if expected["status"] == "PASS":
            expected_masks = {
                side: _interval_mask(expected["per_side_intervals"][side], frame_count)
                for side in SIDES
            }
            active_gate = _gate(
                "PASS",
                str(expected["authority"]),
                per_side_intervals=expected["per_side_intervals"],
            )
        else:
            active_gate = _gate(
                str(expected["status"]),
                str(expected.get("authority", "NONE")),
                *[str(code) for code in expected.get("reason_codes", ["EXPECTED_ACTIVE_NOT_PASS"])],
            )

    # The task-level EXPECTED_ACTIVE interval and image-visibility denominator
    # are intentionally distinct.  PICO/OpenXR may remain valid while a hand is
    # outside the RGB image or fully occluded.  Only a separately authored,
    # exact visibility partition can exclude such frames from HaWoR recall.
    denominator_masks = {side: expected_masks[side].copy() for side in SIDES}
    visibility_class_masks = {
        side: {
            class_name: np.zeros(frame_count, dtype=bool)
            for class_name in (
                "visible_required",
                "boundary_supported",
                "occluded_or_out_of_frame",
            )
        }
        for side in SIDES
    }
    expected_missing_frames = {
        side: np.flatnonzero(expected_masks[side] & ~observed[index]).astype(int).tolist()
        for index, side in enumerate(SIDES)
    }
    visibility_exact = False
    visibility_denominator_authorized = False
    if independent_review is not None and not all_review_errors:
        visibility_review = independent_review["missing_visibility"]
        classified = visibility_review.get("classified_missing_frames", {})
        visibility_exact = all(
            sorted(classified.get(side, [])) == expected_missing_frames[side]
            for side in SIDES
        )
        if visibility_review["status"] == "PASS" and active_gate["status"] == "PASS":
            partitions = visibility_review["per_side_visibility_intervals"]
            visibility_class_masks = {
                side: {
                    class_name: _interval_mask(partitions[side][class_name], frame_count)
                    for class_name in (
                        "visible_required",
                        "boundary_supported",
                        "occluded_or_out_of_frame",
                    )
                }
                for side in SIDES
            }
            if visibility_exact:
                denominator_masks = {
                    side: (
                        visibility_class_masks[side]["visible_required"]
                        | visibility_class_masks[side]["boundary_supported"]
                    )
                    for side in SIDES
                }
                visibility_denominator_authorized = True

    sides = {
        side: _side_metrics(arrays, index, denominator_masks[side])
        for index, side in enumerate(SIDES)
    }
    mask_numeric = all(sides[side]["mask_numeric_pass"] for side in SIDES)
    robot_numeric = all(sides[side]["robot_numeric_pass"] for side in SIDES)

    reported_sides = hawor_result.get("sides")
    report_disagreements: list[str] = []
    if isinstance(reported_sides, Mapping) and all(denominator_masks[side].all() for side in SIDES):
        for side in SIDES:
            reported = reported_sides.get(side)
            if not isinstance(reported, Mapping):
                report_disagreements.append(f"reported side missing: {side}")
                continue
            for key in ("mask_numeric_gate", "robot_numeric_gate"):
                recomputed_key = key.replace("_gate", "_pass")
                if reported.get(key) is not sides[side][recomputed_key]:
                    report_disagreements.append(f"{side}.{key} differs from NPZ replay")

    identity_gate = _gate(
        "HOLD",
        "NONE",
        "HUMAN_REVIEW_REQUIRED",
        "INDEPENDENT_ANATOMICAL_IDENTITY_AUTHORITY_MISSING",
    )
    contour_mask_gate = _gate(
        "HOLD",
        "NONE",
        "HUMAN_REVIEW_REQUIRED",
        "INDEPENDENT_HAND_SILHOUETTE_LABELS_MISSING",
        coverage_fraction_p05_min=0.85,
        distance_threshold_px=20,
    )
    contour_robot_gate = _gate(
        "HOLD",
        "NONE",
        "HUMAN_REVIEW_REQUIRED",
        "INDEPENDENT_HAND_SILHOUETTE_LABELS_MISSING",
        coverage_fraction_p05_min=0.90,
        distance_threshold_px=12,
    )
    visibility_gate = _gate(
        "PASS" if not any(expected_missing_frames[side] for side in SIDES) else "HOLD",
        "PRODUCER" if not any(expected_missing_frames[side] for side in SIDES) else "NONE",
        *(
            []
            if not any(expected_missing_frames[side] for side in SIDES)
            else ["HUMAN_REVIEW_REQUIRED", "MISSING_FRAMES_NOT_CLASSIFIED_AS_OCCLUSION_OR_OUT_OF_FRAME"]
        ),
        expected_active_missing_frames=expected_missing_frames,
        denominator="FULL_SESSION_DIAGNOSTIC_ONLY",
        visibility_aware_denominator_authorized=False,
    )
    if independent_review is not None and not all_review_errors:
        provider_authority = str(independent_review["provider"]["authority"])
        identity = independent_review["anatomical_identity"]
        identity_gate = _gate(
            str(identity["status"]),
            provider_authority,
            *[str(code) for code in identity.get("reason_codes", [])],
            full_timeline_reviewed=identity.get("full_timeline_reviewed"),
            identity_switch_frames=identity.get("identity_switch_frames", []),
            duplicate_track_frames=identity.get("duplicate_track_frames", []),
            collapse_frames=identity.get("collapse_frames", []),
        )
        contour = independent_review["contour_projection"]
        contour_status = str(contour["status"])
        contour_authority = provider_authority
        contour_reasons = [str(code) for code in contour.get("reason_codes", [])]
        per_side = contour.get("per_side", {})
        mask_coverage = all(
            isinstance(per_side.get(side), Mapping)
            and float(per_side[side].get("mask_coverage_fraction_p05_at_20px", -1)) >= 0.85
            for side in SIDES
        )
        robot_coverage = all(
            isinstance(per_side.get(side), Mapping)
            and float(per_side[side].get("robot_coverage_fraction_p05_at_12px", -1)) >= 0.90
            for side in SIDES
        )
        contour_mask_gate = _gate(
            "PASS" if contour_status == "PASS" and mask_coverage else ("FAIL" if contour_status == "FAIL" else "HOLD"),
            contour_authority,
            *(contour_reasons + ([] if mask_coverage else ["MASK_CONTOUR_COVERAGE_P05_LT_0_85"])),
            per_side=per_side,
            sampled_frame_indices=contour.get("sampled_frame_indices", []),
            distance_threshold_px=contour.get("mask_distance_threshold_px"),
        )
        contour_robot_gate = _gate(
            "PASS" if contour_status == "PASS" and robot_coverage else ("FAIL" if contour_status == "FAIL" else "HOLD"),
            contour_authority,
            *(contour_reasons + ([] if robot_coverage else ["ROBOT_CONTOUR_COVERAGE_P05_LT_0_90"])),
            per_side=per_side,
            sampled_frame_indices=contour.get("sampled_frame_indices", []),
            distance_threshold_px=contour.get("robot_distance_threshold_px"),
        )
        visibility = independent_review["missing_visibility"]
        classified = visibility.get("classified_missing_frames", {})
        visibility_reasons = [str(code) for code in visibility.get("reason_codes", [])]
        if visibility["status"] == "PASS" and not visibility_exact:
            visibility_reasons.append("MISSING_FRAME_CLASSIFICATION_NOT_EXACT")
        if visibility["status"] == "PASS" and active_gate["status"] != "PASS":
            visibility_reasons.append("EXPECTED_ACTIVE_SCOPE_NOT_AUTHORIZED")
        visibility_status = (
            "PASS"
            if visibility_denominator_authorized
            else ("FAIL" if visibility["status"] == "FAIL" else "HOLD")
        )
        visibility_gate = _gate(
            visibility_status,
            provider_authority,
            *visibility_reasons,
            expected_active_missing_frames=expected_missing_frames,
            classified_missing_frames=classified,
            per_side={
                side: {
                    "expected_active_frames": int(expected_masks[side].sum()),
                    "visible_required_frames": int(
                        visibility_class_masks[side]["visible_required"].sum()
                    ),
                    "boundary_supported_frames": int(
                        visibility_class_masks[side]["boundary_supported"].sum()
                    ),
                    "occluded_or_out_of_frame_frames": int(
                        visibility_class_masks[side]["occluded_or_out_of_frame"].sum()
                    ),
                    "missing_in_visible_denominator_frames": sides[side][
                        "missing_frame_indices"
                    ],
                    "excluded_missing_frames": sorted(
                        set(expected_missing_frames[side])
                        & set(
                            np.flatnonzero(
                                visibility_class_masks[side]["occluded_or_out_of_frame"]
                            )
                            .astype(int)
                            .tolist()
                        )
                    ),
                }
                for side in SIDES
            },
            denominator=(
                "EXPECTED_VISIBLE_ACTIVE"
                if visibility_denominator_authorized
                else "EXPECTED_ACTIVE_DIAGNOSTIC_ONLY"
            ),
            denominator_formula=(
                "EXPECTED_ACTIVE intersect (VISIBLE_REQUIRED union BOUNDARY_SUPPORTED)"
            ),
            boundary_support_rule=visibility.get("boundary_support_rule"),
            visibility_aware_denominator_authorized=visibility_denominator_authorized,
        )
    elif independent_review is not None and all_review_errors:
        identity_gate = _gate("HOLD", "NONE", "INDEPENDENT_REVIEW_INVALID")
        contour_mask_gate = _gate("HOLD", "NONE", "INDEPENDENT_REVIEW_INVALID")
        contour_robot_gate = _gate("HOLD", "NONE", "INDEPENDENT_REVIEW_INVALID")
        visibility_gate = _gate("HOLD", "NONE", "INDEPENDENT_REVIEW_INVALID")

    source_gates = source_gate.get("gates", {})
    g0_status, g0_raw = _map_source_gate(source_gates.get("G0_SOURCE_MEDIA", "NOT_EVALUATED"))
    g1_status, g1_raw = _map_source_gate(source_gates.get("G1_CAMERA_WORLD", "NOT_EVALUATED"))
    g2_status, g2_raw = _map_source_gate(source_gates.get("G2_PICO21_PRELIMINARY", "NOT_EVALUATED"))
    integrity_errors = list(artifact_errors) + structure_errors + provenance_errors + report_disagreements
    integrity_gate = _gate(
        "PASS" if not integrity_errors else "FAIL",
        "INDEPENDENT_QA",
        *integrity_errors,
    )
    provenance_gate = _gate(
        "PASS" if not provenance_errors else "FAIL",
        "INDEPENDENT_QA",
        *provenance_errors,
        values=provenance_values,
        interpolation_frames=0 if not provenance_errors else None,
        per_side_direct_frames={
            side: int(np.count_nonzero(observed[index] & (provenance[index] == "OBSERVED")))
            for index, side in enumerate(SIDES)
        },
    )
    numeric_mask_gate = _gate(
        "PASS" if mask_numeric else "FAIL",
        "INDEPENDENT_QA",
        *([] if mask_numeric else ["HAWOR_MASK_NUMERIC_THRESHOLD_FAILED"]),
        per_side=sides,
        denominator=(
            "EXPECTED_VISIBLE_ACTIVE"
            if visibility_denominator_authorized
            else (
                "EXPECTED_ACTIVE_DIAGNOSTIC_ONLY"
                if active_gate["status"] == "PASS"
                else "FULL_SESSION_DIAGNOSTIC_ONLY"
            )
        ),
        formal_denominator_authorized=visibility_denominator_authorized,
    )
    numeric_robot_gate = _gate(
        "PASS" if robot_numeric else "FAIL",
        "INDEPENDENT_QA",
        *([] if robot_numeric else ["HAWOR_ROBOT_NUMERIC_THRESHOLD_FAILED"]),
        per_side=sides,
        denominator=(
            "EXPECTED_VISIBLE_ACTIVE"
            if visibility_denominator_authorized
            else (
                "EXPECTED_ACTIVE_DIAGNOSTIC_ONLY"
                if active_gate["status"] == "PASS"
                else "FULL_SESSION_DIAGNOSTIC_ONLY"
            )
        ),
        formal_denominator_authorized=visibility_denominator_authorized,
    )

    mask_ready = all(
        gate["status"] == "PASS"
        for gate in (
            integrity_gate,
            active_gate,
            numeric_mask_gate,
            provenance_gate,
            identity_gate,
            contour_mask_gate,
            visibility_gate,
        )
    ) and g0_status == "PASS"
    robot_ready = all(
        gate["status"] == "PASS"
        for gate in (
            integrity_gate,
            active_gate,
            numeric_robot_gate,
            provenance_gate,
            identity_gate,
            contour_robot_gate,
            visibility_gate,
        )
    ) and g0_status == "PASS" and g1_status == "PASS"

    primary_blame: list[dict[str, str]] = []
    if g0_status == "FAIL":
        primary_blame.append({"owner": "CAPTURE", "reason_code": g0_raw})
    if integrity_gate["status"] == "FAIL":
        primary_blame.append({"owner": "HAWOR_ARTIFACT", "reason_code": "HAWOR_ARTIFACT_INTEGRITY_FAILED"})
    if numeric_mask_gate["status"] == "FAIL" or numeric_robot_gate["status"] == "FAIL":
        # PICO can keep a valid 3-D hand pose while that hand is occluded or
        # outside the RGB frame.  PICO validity therefore cannot, by itself,
        # turn a HaWoR visual miss into an algorithm failure.  A formal blame
        # decision needs an independently declared visible/active denominator
        # and exact missing-frame visibility classification.
        if not visibility_denominator_authorized:
            primary_blame.append(
                {
                    "owner": "AUTHORITY",
                    "reason_code": "NUMERIC_FAILURE_DENOMINATOR_OR_VISIBILITY_UNRESOLVED",
                }
            )
        else:
            primary_blame.append(
                {
                    "owner": "HAWOR_ALGORITHM",
                    "reason_code": "HAWOR_NUMERIC_THRESHOLD_FAILED_ON_AUTHORIZED_VISIBLE_ACTIVE_SCOPE",
                }
            )
    if not primary_blame and any(
        gate["status"] != "PASS" for gate in (active_gate, identity_gate, contour_mask_gate, visibility_gate)
    ):
        primary_blame.append({"owner": "AUTHORITY", "reason_code": "HUMAN_OR_INDEPENDENT_LABEL_REVIEW_REQUIRED"})
    world_blame = None
    if g1_status != "PASS":
        world_blame = {"owner": "CAPTURE_WORLD", "reason_code": g1_raw}

    hard_fail = bool(
        g0_status == "FAIL"
        or integrity_gate["status"] == "FAIL"
        or (
            numeric_mask_gate["status"] == "FAIL"
            and visibility_denominator_authorized
        )
    )
    status = "FAIL_HAWOR_OR_CAPTURE" if hard_fail else (
        "PASS_FORMAL_HAWOR_SEED" if mask_ready else "HOLD_HUMAN_REVIEW_REQUIRED"
    )
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": status,
        "task_id": task_id,
        "session_id": session_id,
        "frame_count": frame_count,
        "claim_limit": (
            "HaWoR numeric/provenance and separately supplied independent identity/contour authority only; "
            "no Object6D, contact, Mask pixel, Clean, Robot retarget, or training admission claim."
        ),
        "input_validation": {
            "valid": not integrity_errors and not all_review_errors,
            "artifact_errors": list(artifact_errors),
            "structure_errors": structure_errors,
            "provenance_errors": provenance_errors,
            "reported_metric_disagreements": report_disagreements,
            "independent_review_supplied": independent_review is not None,
            "independent_review_errors": all_review_errors,
        },
        "gates": {
            "source_media_g0": _gate(g0_status, "CAPTURE", *([] if g0_status == "PASS" else [g0_raw]), raw=g0_raw),
            "world_camera_g1": _gate(g1_status, "CAPTURE", *([] if g1_status == "PASS" else [g1_raw]), raw=g1_raw),
            "pico21_g2_advisory": _gate(g2_status, "CAPTURE", *([] if g2_status == "PASS" else [g2_raw]), raw=g2_raw),
            "artifact_and_mano21_integrity": integrity_gate,
            "expected_active_scope": active_gate,
            "numeric_mask_seed": numeric_mask_gate,
            "numeric_robot_seed": numeric_robot_gate,
            "direct_vs_interpolated_provenance": provenance_gate,
            "anatomical_identity_continuity": identity_gate,
            "independent_contour_mask": contour_mask_gate,
            "independent_contour_robot": contour_robot_gate,
            "occlusion_out_of_frame": visibility_gate,
        },
        "diagnostics": {
            "identity_from_hawor_only": _identity_diagnostic(
                arrays, expected_masks["left"] & expected_masks["right"]
            ),
            "self_detector_or_self_projection_is_independent_evidence": False,
        },
        "routes": {
            "mask_numeric_candidate": bool(
                g0_status == "PASS" and integrity_gate["status"] == "PASS" and mask_numeric
            ),
            "mask_formal_seed_ready": mask_ready,
            "robot_numeric_candidate": bool(
                g0_status == "PASS" and integrity_gate["status"] == "PASS" and robot_numeric
            ),
            "robot_world_route_ready": bool(g1_status == "PASS"),
            "robot_formal_motion_seed_ready": robot_ready,
            "human_review_required": not mask_ready and not hard_fail,
        },
        "attribution": {
            "primary": primary_blame,
            "world_route_only": world_blame,
            "policy": (
                "G1 blocks only world/Robot routes. PICO G2 is advisory unless explicitly consumed. "
                "Missing independent labels are AUTHORITY HOLD, not HaWoR failure. PICO 3-D validity does "
                "not prove RGB visibility. Numeric failure becomes HaWoR blame only on an authorized visible/active "
                "denominator with exact visibility classification; thresholds are never relaxed to preserve yield."
            ),
        },
    }
