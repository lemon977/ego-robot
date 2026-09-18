"""Fail-closed, image-plane-only interaction evidence.

``Interaction Evidence v0a`` deliberately stops before depth, occlusion,
Object6D, contact, or Robot authority.  It combines directly observed HaWoR
fingertips with role-aware task-object masks and, optionally, aligned
processed tactile activity.  Every output is either an explicitly named weak
2D observation/hypothesis or ``UNKNOWN``.

In particular, an inferred HaWoR point is never upgraded to strict evidence,
and an ``unknown`` object-mask frame invalidates every fingertip/object pair
for that frame.  Tactile activity can only support a hypothesis when its
source is valid, its timestamp is aligned, and direct 2D adjacency is present.
It is not force or contact truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "INTERACTION_EVIDENCE_V0A"
COORDINATE_DOMAIN = "IMAGE_2D_ONLY"
AUTHORITY = "DEVELOPMENT_WEAK_EVIDENCE_ONLY"
MASK_TRACK_STATES = frozenset({"seeded", "tracked", "reseeded", "unknown"})
HAWOR_OBSERVATION_STATES = frozenset({"direct_observed", "inferred", "unknown"})


class InteractionEvidenceError(ValueError):
    """Raised when an input cannot safely support v0a evidence."""


class AdjacencyStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    ADJACENT_2D = "ADJACENT_2D"
    NOT_ADJACENT_2D = "NOT_ADJACENT_2D"


class ApproachStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    APPROACHING_2D = "APPROACHING_2D"
    NOT_APPROACHING_2D = "NOT_APPROACHING_2D"


class CoMotionStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    CO_MOVING_2D = "CO_MOVING_2D"
    NOT_CO_MOVING_2D = "NOT_CO_MOVING_2D"


class TactileStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    TACTILE_SUPPORTED_HYPOTHESIS = "TACTILE_SUPPORTED_HYPOTHESIS"
    NOT_TACTILE_SUPPORTED = "NOT_TACTILE_SUPPORTED"


@dataclass(frozen=True)
class InteractionEvidenceV0AResult:
    """Per-frame arrays for ``(frame, object, fingertip)`` pairs."""

    frame_timestamps_s: np.ndarray
    object_mask_state: np.ndarray
    fingertip_observation_state: np.ndarray
    boundary_distance_px: np.ndarray
    adjacency_status: np.ndarray
    approach_status: np.ndarray
    co_motion_status: np.ndarray
    tactile_status: np.ndarray
    tactile_time_offset_ms: np.ndarray


def _require_bool_array(value: Any, *, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.bool_:
        raise InteractionEvidenceError(f"{label} must be bool with shape {shape}")
    return result


def _require_state_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    allowed: frozenset[str],
    label: str,
) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape:
        raise InteractionEvidenceError(f"{label} must have shape {shape}")
    parsed = result.astype(str)
    unexpected = sorted(set(parsed.ravel().tolist()) - allowed)
    if unexpected:
        raise InteractionEvidenceError(f"{label} has unsupported states: {unexpected}")
    return parsed


def _positive_finite(value: float, *, label: str, allow_zero: bool = False) -> float:
    parsed = float(value)
    lower_ok = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not np.isfinite(parsed) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise InteractionEvidenceError(f"{label} must be finite and {qualifier}")
    return parsed


def _mask_centroid_xy(mask: np.ndarray) -> np.ndarray:
    rows_yx = np.argwhere(mask)
    return np.asarray([rows_yx[:, 1].mean(), rows_yx[:, 0].mean()], dtype=np.float64)


def _point_mask_distance_px(point_xy: np.ndarray, mask: np.ndarray) -> float:
    rows_yx = np.argwhere(mask)
    pixels_xy = rows_yx[:, [1, 0]].astype(np.float64)
    return float(np.min(np.linalg.norm(pixels_xy - point_xy[None, :], axis=1)))


def estimate_interaction_evidence_v0a(
    *,
    fingertip_xy_px: np.ndarray,
    fingertip_observation_state: np.ndarray,
    object_masks: np.ndarray,
    object_mask_state: np.ndarray,
    frame_timestamps_s: np.ndarray,
    adjacency_threshold_px: float = 4.0,
    approach_min_delta_px: float = 0.5,
    co_motion_min_displacement_px: float = 0.5,
    co_motion_min_cosine: float = 0.8,
    co_motion_max_residual_px: float = 3.0,
    tactile_active: np.ndarray | None = None,
    tactile_source_valid: np.ndarray | None = None,
    tactile_timestamps_s: np.ndarray | None = None,
    tactile_max_offset_s: float = 0.040,
) -> InteractionEvidenceV0AResult:
    """Estimate bounded 2D evidence without making a contact claim.

    Shapes are ``fingertip_xy_px=(T,K,2)``, ``object_masks=(T,O,H,W)``,
    fingertip state ``(T,K)``, and object-mask state ``(T,O)``.  Optional
    tactile arrays have shape ``(T,K)`` and must be supplied as a complete
    triple.  ``tactile_active`` is an upstream activity flag, not a force.

    ``direct_observed`` HaWoR coordinates must be finite and inside the image.
    ``inferred`` coordinates may be finite but are intentionally ignored by
    all evidence outputs.  A non-empty mask may never be labelled ``unknown``
    and an admitted mask state may never carry an empty mask.
    """

    tips = np.asarray(fingertip_xy_px, dtype=np.float64)
    masks = np.asarray(object_masks)
    times = np.asarray(frame_timestamps_s, dtype=np.float64)
    if tips.ndim != 3 or tips.shape[-1] != 2:
        raise InteractionEvidenceError("fingertip_xy_px must have shape (T,K,2)")
    if masks.ndim != 4 or masks.dtype != np.bool_:
        raise InteractionEvidenceError("object_masks must be bool with shape (T,O,H,W)")
    frame_count, fingertip_count, _ = tips.shape
    if masks.shape[0] != frame_count or masks.shape[2] <= 0 or masks.shape[3] <= 0:
        raise InteractionEvidenceError("object_masks frame/image axes are invalid")
    object_count = masks.shape[1]
    if object_count <= 0 or fingertip_count <= 0 or frame_count <= 0:
        raise InteractionEvidenceError("frame, object, and fingertip axes must be non-empty")
    if times.shape != (frame_count,) or not np.isfinite(times).all():
        raise InteractionEvidenceError("frame_timestamps_s must be finite with shape (T,)")
    if frame_count > 1 and np.any(np.diff(times) <= 0.0):
        raise InteractionEvidenceError("frame_timestamps_s must be strictly increasing")

    tip_states = _require_state_array(
        fingertip_observation_state,
        shape=(frame_count, fingertip_count),
        allowed=HAWOR_OBSERVATION_STATES,
        label="fingertip_observation_state",
    )
    mask_states = _require_state_array(
        object_mask_state,
        shape=(frame_count, object_count),
        allowed=MASK_TRACK_STATES,
        label="object_mask_state",
    )
    direct = tip_states == "direct_observed"
    if not np.isfinite(tips[direct]).all():
        raise InteractionEvidenceError("direct_observed fingertips must be finite")
    image_height, image_width = masks.shape[2:]
    if np.any(
        direct
        & (
            (tips[:, :, 0] < 0.0)
            | (tips[:, :, 0] >= image_width)
            | (tips[:, :, 1] < 0.0)
            | (tips[:, :, 1] >= image_height)
        )
    ):
        raise InteractionEvidenceError("direct_observed fingertips must be inside the image")

    mask_nonempty = np.any(masks, axis=(2, 3))
    if np.any((mask_states == "unknown") & mask_nonempty):
        raise InteractionEvidenceError("unknown mask state cannot carry mask pixels")
    if np.any((mask_states != "unknown") & ~mask_nonempty):
        raise InteractionEvidenceError("admitted mask state cannot carry an empty mask")

    adjacency_threshold = _positive_finite(
        adjacency_threshold_px, label="adjacency_threshold_px", allow_zero=True
    )
    approach_delta = _positive_finite(
        approach_min_delta_px, label="approach_min_delta_px", allow_zero=True
    )
    motion_min = _positive_finite(
        co_motion_min_displacement_px,
        label="co_motion_min_displacement_px",
        allow_zero=False,
    )
    motion_residual = _positive_finite(
        co_motion_max_residual_px,
        label="co_motion_max_residual_px",
        allow_zero=True,
    )
    cosine_gate = float(co_motion_min_cosine)
    if not np.isfinite(cosine_gate) or not -1.0 <= cosine_gate <= 1.0:
        raise InteractionEvidenceError("co_motion_min_cosine must be in [-1,1]")
    tactile_offset_gate = _positive_finite(
        tactile_max_offset_s, label="tactile_max_offset_s", allow_zero=True
    )

    tactile_supplied = (
        tactile_active is not None,
        tactile_source_valid is not None,
        tactile_timestamps_s is not None,
    )
    active: np.ndarray | None = None
    source_valid: np.ndarray | None = None
    tactile_times: np.ndarray | None = None
    if any(tactile_supplied):
        if not all(tactile_supplied):
            raise InteractionEvidenceError("tactile arrays must be supplied together")
        active = _require_bool_array(
            tactile_active, shape=(frame_count, fingertip_count), label="tactile_active"
        )
        source_valid = _require_bool_array(
            tactile_source_valid,
            shape=(frame_count, fingertip_count),
            label="tactile_source_valid",
        )
        tactile_times = np.asarray(tactile_timestamps_s, dtype=np.float64)
        if tactile_times.shape != (frame_count, fingertip_count):
            raise InteractionEvidenceError(
                "tactile_timestamps_s must have shape (T,K)"
            )
        if not np.isfinite(tactile_times[source_valid]).all():
            raise InteractionEvidenceError("valid tactile timestamps must be finite")

    pair_shape = (frame_count, object_count, fingertip_count)
    boundary_distance = np.full(pair_shape, np.nan, dtype=np.float64)
    adjacency = np.full(pair_shape, AdjacencyStatus.UNKNOWN.value, dtype="<U24")
    approach = np.full(pair_shape, ApproachStatus.UNKNOWN.value, dtype="<U24")
    co_motion = np.full(pair_shape, CoMotionStatus.UNKNOWN.value, dtype="<U24")
    tactile = np.full(pair_shape, TactileStatus.UNKNOWN.value, dtype="<U32")
    tactile_offset_ms = np.full(pair_shape, np.nan, dtype=np.float64)
    centroids = np.full((frame_count, object_count, 2), np.nan, dtype=np.float64)

    for frame in range(frame_count):
        for object_index in range(object_count):
            if mask_states[frame, object_index] == "unknown":
                continue
            mask = masks[frame, object_index]
            centroids[frame, object_index] = _mask_centroid_xy(mask)
            for fingertip in range(fingertip_count):
                # Inferred points are useful for visualization, but they are
                # not direct observations and therefore cannot support v0a.
                if tip_states[frame, fingertip] != "direct_observed":
                    continue
                distance = _point_mask_distance_px(tips[frame, fingertip], mask)
                boundary_distance[frame, object_index, fingertip] = distance
                adjacency[frame, object_index, fingertip] = (
                    AdjacencyStatus.ADJACENT_2D.value
                    if distance <= adjacency_threshold
                    else AdjacencyStatus.NOT_ADJACENT_2D.value
                )

                if frame > 0:
                    previous_inputs_direct = (
                        tip_states[frame - 1, fingertip] == "direct_observed"
                        and mask_states[frame - 1, object_index] != "unknown"
                    )
                    previous_distance = boundary_distance[
                        frame - 1, object_index, fingertip
                    ]
                    if previous_inputs_direct and np.isfinite(previous_distance):
                        distance_reduction = float(previous_distance - distance)
                        approach[frame, object_index, fingertip] = (
                            ApproachStatus.APPROACHING_2D.value
                            if distance_reduction >= approach_delta
                            else ApproachStatus.NOT_APPROACHING_2D.value
                        )

                        fingertip_motion = tips[frame, fingertip] - tips[frame - 1, fingertip]
                        object_motion = (
                            centroids[frame, object_index]
                            - centroids[frame - 1, object_index]
                        )
                        fingertip_norm = float(np.linalg.norm(fingertip_motion))
                        object_norm = float(np.linalg.norm(object_motion))
                        if fingertip_norm >= motion_min and object_norm >= motion_min:
                            cosine = float(
                                np.dot(fingertip_motion, object_motion)
                                / (fingertip_norm * object_norm)
                            )
                            residual = float(np.linalg.norm(fingertip_motion - object_motion))
                            co_motion[frame, object_index, fingertip] = (
                                CoMotionStatus.CO_MOVING_2D.value
                                if cosine >= cosine_gate and residual <= motion_residual
                                else CoMotionStatus.NOT_CO_MOVING_2D.value
                            )
                        else:
                            co_motion[frame, object_index, fingertip] = (
                                CoMotionStatus.NOT_CO_MOVING_2D.value
                            )

                if active is None or source_valid is None or tactile_times is None:
                    continue
                if not source_valid[frame, fingertip]:
                    continue
                offset = abs(float(tactile_times[frame, fingertip] - times[frame]))
                tactile_offset_ms[frame, object_index, fingertip] = offset * 1000.0
                if offset > tactile_offset_gate:
                    continue
                tactile[frame, object_index, fingertip] = (
                    TactileStatus.TACTILE_SUPPORTED_HYPOTHESIS.value
                    if active[frame, fingertip]
                    and adjacency[frame, object_index, fingertip]
                    == AdjacencyStatus.ADJACENT_2D.value
                    else TactileStatus.NOT_TACTILE_SUPPORTED.value
                )

    return InteractionEvidenceV0AResult(
        frame_timestamps_s=times.copy(),
        object_mask_state=mask_states.copy(),
        fingertip_observation_state=tip_states.copy(),
        boundary_distance_px=boundary_distance,
        adjacency_status=adjacency,
        approach_status=approach,
        co_motion_status=co_motion,
        tactile_status=tactile,
        tactile_time_offset_ms=tactile_offset_ms,
    )


def interaction_evidence_v0a_to_record(
    result: InteractionEvidenceV0AResult,
    *,
    object_ids: Sequence[str],
    fingertip_ids: Sequence[str],
) -> dict[str, Any]:
    """Serialize a result into the strict JSON contract representation."""

    shape = result.adjacency_status.shape
    if len(shape) != 3:
        raise InteractionEvidenceError("result arrays must have shape (T,O,K)")
    frame_count, object_count, fingertip_count = shape
    if (
        len(object_ids) != object_count
        or len(set(object_ids)) != object_count
        or not all(isinstance(value, str) and value for value in object_ids)
    ):
        raise InteractionEvidenceError("object_ids must be unique and match object axis")
    if (
        len(fingertip_ids) != fingertip_count
        or len(set(fingertip_ids)) != fingertip_count
        or not all(isinstance(value, str) and value for value in fingertip_ids)
    ):
        raise InteractionEvidenceError(
            "fingertip_ids must be unique and match fingertip axis"
        )
    expected_pair_shape = (frame_count, object_count, fingertip_count)
    for array, label in (
        (result.boundary_distance_px, "boundary_distance_px"),
        (result.approach_status, "approach_status"),
        (result.co_motion_status, "co_motion_status"),
        (result.tactile_status, "tactile_status"),
        (result.tactile_time_offset_ms, "tactile_time_offset_ms"),
    ):
        if np.asarray(array).shape != expected_pair_shape:
            raise InteractionEvidenceError(f"{label} shape mismatch")
    if result.frame_timestamps_s.shape != (frame_count,):
        raise InteractionEvidenceError("frame timestamp shape mismatch")
    if result.object_mask_state.shape != (frame_count, object_count):
        raise InteractionEvidenceError("mask-state shape mismatch")
    if result.fingertip_observation_state.shape != (frame_count, fingertip_count):
        raise InteractionEvidenceError("HaWoR-state shape mismatch")

    observations: list[dict[str, Any]] = []
    for frame in range(frame_count):
        for object_index, object_id in enumerate(object_ids):
            for fingertip_index, fingertip_id in enumerate(fingertip_ids):
                distance = float(result.boundary_distance_px[frame, object_index, fingertip_index])
                tactile_offset = float(
                    result.tactile_time_offset_ms[frame, object_index, fingertip_index]
                )
                observations.append(
                    {
                        "frame_index": frame,
                        "frame_timestamp_s": float(result.frame_timestamps_s[frame]),
                        "object_instance_id": object_id,
                        "fingertip_id": fingertip_id,
                        "mask_track_state": str(
                            result.object_mask_state[frame, object_index]
                        ),
                        "hawor_observation_state": str(
                            result.fingertip_observation_state[frame, fingertip_index]
                        ),
                        "boundary_distance_px": distance if np.isfinite(distance) else None,
                        "adjacency": str(
                            result.adjacency_status[frame, object_index, fingertip_index]
                        ),
                        "approach": str(
                            result.approach_status[frame, object_index, fingertip_index]
                        ),
                        "co_motion": str(
                            result.co_motion_status[frame, object_index, fingertip_index]
                        ),
                        "tactile": str(
                            result.tactile_status[frame, object_index, fingertip_index]
                        ),
                        "tactile_time_offset_ms": (
                            tactile_offset if np.isfinite(tactile_offset) else None
                        ),
                    }
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "coordinate_domain": COORDINATE_DOMAIN,
        "authority": AUTHORITY,
        "observations": observations,
        "prohibited_claims": {
            "relative_z": False,
            "occlusion_order": False,
            "force": False,
            "contact_ground_truth": False,
            "object6d": False,
            "robot_control": False,
        },
        "claim_limit": (
            "Direct image-plane adjacency, approach, co-motion, and optionally "
            "aligned tactile-supported hypotheses only; not depth, occlusion "
            "order, force, contact truth, Object6D, or Robot authority."
        ),
    }
