"""Tiered, fail-closed admission for Kai22 R0 artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chaoyang.pipeline.temporal_authority_v1 import validate_online_current_inputs


SCHEMA_VERSION = "KAI22_R0_TIERED_ADMISSION_V1"
LEVELS = ("KINEMATIC_ONLY", "DEVELOPMENT_R0", "H50_READY")


def _longest_contiguous(mask: np.ndarray, frame_ids: np.ndarray) -> int:
    best = current = 0
    previous: int | None = None
    for selected, frame_id in zip(mask.tolist(), frame_ids.tolist(), strict=True):
        if selected and (previous is None or frame_id == previous + 1):
            current += 1
        elif selected:
            current = 1
        else:
            current = 0
        best = max(best, current)
        previous = frame_id
    return best


def evaluate_kai22_r0_tiers(
    *,
    frame_ids: np.ndarray,
    q22_valid: np.ndarray,
    fk_finite: np.ndarray,
    joint_semantics_pass: bool,
    consumer_local_window_mask: np.ndarray,
    consumer_local_window_quality_pass: bool,
    temporal_authority_audit: Mapping[str, Any],
    h50_current_input_fields: Sequence[str],
    timestamps_valid: bool,
    h50_minimum_contiguous_frames: int = 51,
) -> dict[str, Any]:
    """Return the highest honest R0 layer without using Contact as a gate."""

    ids = np.asarray(frame_ids, np.int64)
    q_valid = np.asarray(q22_valid, bool)
    fk_valid = np.asarray(fk_finite, bool)
    local = np.asarray(consumer_local_window_mask, bool)
    if ids.ndim != 1 or q_valid.shape != ids.shape or fk_valid.shape != ids.shape or local.shape != ids.shape:
        raise ValueError("all Kai22 tier axes must have shape [T]")
    if ids.size == 0 or np.any(np.diff(ids) <= 0):
        raise ValueError("frame_ids must be non-empty and strictly increasing")
    if not isinstance(joint_semantics_pass, bool) or not isinstance(consumer_local_window_quality_pass, bool):
        raise ValueError("quality flags must be booleans")
    if not isinstance(timestamps_valid, bool):
        raise ValueError("timestamps_valid must be boolean")
    if h50_minimum_contiguous_frames <= 0:
        raise ValueError("h50_minimum_contiguous_frames must be positive")

    kinematic_reasons: list[str] = []
    if not joint_semantics_pass:
        kinematic_reasons.append("JOINT_SEMANTICS_NOT_VERIFIED")
    if not q_valid.any():
        kinematic_reasons.append("NO_VALID_Q22_FRAME")
    if np.any(q_valid & ~fk_valid):
        kinematic_reasons.append("FK_NONFINITE_FOR_VALID_Q22")
    kinematic_pass = not kinematic_reasons

    development_reasons: list[str] = []
    if not kinematic_pass:
        development_reasons.append("KINEMATIC_ONLY_PREREQUISITE_FAILED")
    if not consumer_local_window_quality_pass:
        development_reasons.append("CONSUMER_LOCAL_WINDOW_QUALITY_FAILED")
    admitted_local = local & q_valid & fk_valid
    if not admitted_local.any():
        development_reasons.append("NO_ADMITTED_LOCAL_WINDOW_FRAME")
    if not timestamps_valid:
        development_reasons.append("TIMESTAMP_AXIS_INVALID")
    development_pass = not development_reasons

    temporal_failures = validate_online_current_inputs(
        temporal_authority_audit, h50_current_input_fields
    )
    longest = _longest_contiguous(admitted_local, ids)
    h50_reasons: list[str] = []
    if not development_pass:
        h50_reasons.append("DEVELOPMENT_R0_PREREQUISITE_FAILED")
    if longest < h50_minimum_contiguous_frames:
        h50_reasons.append("H50_CONTIGUOUS_WINDOW_TOO_SHORT")
    h50_reasons.extend(f"TEMPORAL_INPUT:{reason}" for reason in temporal_failures)
    h50_pass = not h50_reasons

    if h50_pass:
        highest = "H50_READY"
    elif development_pass:
        highest = "DEVELOPMENT_R0"
    elif kinematic_pass:
        highest = "KINEMATIC_ONLY"
    else:
        highest = "NONE"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASSED_TIERED_ACCOUNTING",
        "highest_admitted_level": highest,
        "levels": {
            "KINEMATIC_ONLY": {
                "status": "PASS" if kinematic_pass else "BLOCKED",
                "reason_codes": kinematic_reasons,
                "valid_q22_frames": int(q_valid.sum()),
                "finite_fk_frames": int((q_valid & fk_valid).sum()),
            },
            "DEVELOPMENT_R0": {
                "status": "PASS" if development_pass else "BLOCKED",
                "reason_codes": development_reasons,
                "admitted_local_window_frames": int(admitted_local.sum()),
            },
            "H50_READY": {
                "status": "PASS" if h50_pass else "BLOCKED",
                "reason_codes": h50_reasons,
                "minimum_contiguous_frames": h50_minimum_contiguous_frames,
                "longest_contiguous_frames": longest,
                "required_current_input_fields": list(h50_current_input_fields),
            },
        },
        "contact_required": False,
        "object6d_required": False,
        "control_ground_truth": False,
        "training_eligible": h50_pass,
        "physical_deployment_authorized": False,
        "calibration_authority": "DEVELOPMENT_ONLY",
    }
