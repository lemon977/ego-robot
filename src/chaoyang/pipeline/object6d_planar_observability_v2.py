"""Observed-only planar Object6D contract with explicit entity semantics.

V2 keeps the proven V1 visible-surface estimator but changes the publication
model.  Three playing cards remain separate physical instances, the black card
tray is a support entity with absent mask evidence, and ``card_set`` is only a
semantic membership group.  Geometry is reported per field; no unified
``valid`` flag or group-level geometry is emitted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from chaoyang.pipeline.object6d_planar_observability_v1 import (
    DEPTH_REFERENCE as V1_DEPTH_REFERENCE,
    DIRECT_VISIBILITY,
    MASK_REGISTRATION_AUTHORITY,
    PlanarFrameInput as V1PlanarFrameInput,
    PlanarObservabilityError,
    estimate_planar_frame as estimate_planar_frame_v1,
)


SCHEMA_VERSION = "object6d-planar-observability-v2"
DEPTH_REFERENCE = "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
OBJECT_INSTANCE_IDS = (
    "playing_card_00",
    "playing_card_01",
    "playing_card_02",
)
SUPPORT_ENTITY_ID = "black_card_tray"
SEMANTIC_GROUP_ID = "card_set"
COMPONENT_NAMES = (
    "center_xyz",
    "plane_normal",
    "inplane_rotation",
    "full_extent",
)


@dataclass(frozen=True)
class PlanarFrameInput(V1PlanarFrameInput):
    """V2 input defaults to the admitted encoded physical-left Depth domain."""

    depth_reference: str = DEPTH_REFERENCE


def _unobservable_full_extent() -> dict[str, Any]:
    return {
        "observability": "UNOBSERVABLE",
        "estimate": None,
        "residual": None,
        "reason": "FULL_OBJECT_BOUNDARY_VISIBILITY_UNPROVEN",
        "semantics": (
            "FULL_PHYSICAL_OBJECT_EXTENT_REQUIRES_COMPLETE_BOUNDARY_EVIDENCE"
        ),
    }


def estimate_planar_frame(value: PlanarFrameInput) -> dict[str, Any]:
    """Publish four independently observable fields for one card/frame.

    ``center_xyz`` intentionally retains V1's conservative visible-surface
    centroid semantics.  It must not be interpreted as the hidden full-object
    centre.  ``full_extent`` remains unobservable until a future bounded method
    proves that the complete physical boundary is visible.
    """

    if value.depth_reference != DEPTH_REFERENCE:
        raise PlanarObservabilityError(
            "v2 requires encoded physical-left optical-Z input"
        )
    # V1's numerical estimator is image-domain agnostic once pixels, K and the
    # explicit depth-to-mask map are supplied.  Adapt only its historical
    # string guard; the arrays and intrinsics stay in the encoded physical-left
    # domain and V2 publishes that domain explicitly.
    observed = estimate_planar_frame_v1(
        replace(value, depth_reference=V1_DEPTH_REFERENCE)
    )
    return {
        "frame_index": observed["frame_index"],
        "mask_state": observed["mask_state"],
        "visibility_state": observed["visibility_state"],
        "mask_pixel_count": observed["mask_pixel_count"],
        "registered_valid_depth_count": observed["registered_valid_depth_count"],
        "registered_valid_depth_fraction": observed[
            "registered_valid_depth_fraction"
        ],
        "center_xyz": observed["translation"],
        "plane_normal": observed["plane_normal"],
        "inplane_rotation": observed["in_plane_rotation"],
        "full_extent": _unobservable_full_extent(),
        "hidden_geometry_inferred": False,
    }


def build_observability_document(
    *,
    session_id: str,
    object_frames: dict[str, list[dict[str, Any]]],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Build the V2 document without group or support-entity geometry."""

    if tuple(object_frames) != OBJECT_INSTANCE_IDS:
        raise PlanarObservabilityError(
            "the bounded v2 canary requires playing_card_00/01/02 in order"
        )
    counts: dict[str, dict[str, int]] = {}
    frame_count: int | None = None
    for instance_id, frames in object_frames.items():
        if frame_count is None:
            frame_count = len(frames)
        elif frame_count != len(frames):
            raise PlanarObservabilityError("object frame axes differ")
        counts[instance_id] = {
            component: sum(
                row[component]["observability"] != "UNOBSERVABLE"
                for row in frames
            )
            for component in COMPONENT_NAMES
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED_DEVELOPMENT_OBSERVABILITY",
        "session_id": session_id,
        "frame_count": int(frame_count or 0),
        "coordinate_domain": DEPTH_REFERENCE,
        "center_xyz_semantics": (
            "DIRECT_VISIBLE_SURFACE_CENTROID_NOT_FULL_OBJECT_CENTER"
        ),
        "external_accuracy": "UNVERIFIED",
        "hidden_geometry_inferred": False,
        "unified_confidence_emitted": False,
        "contact_authority": "NONE",
        "robot_authority": "NONE",
        "inputs": inputs,
        "objects": [
            {
                "instance_id": instance_id,
                "entity_role": "PHYSICAL_TASK_OBJECT",
                "geometry_class": "PLAYING_CARD_VISIBLE_PLANE",
                "frames": frames,
            }
            for instance_id, frames in object_frames.items()
        ],
        "support_entities": [
            {
                "instance_id": SUPPORT_ENTITY_ID,
                "entity_role": "SUPPORT_ENTITY",
                "mask_evidence": "ABSENT",
                "observability": "UNKNOWN",
                "geometry_authority": "NONE",
                "reason": "INDEPENDENT_SUPPORT_MASK_NOT_AVAILABLE",
            }
        ],
        "semantic_groups": [
            {
                "group_id": SEMANTIC_GROUP_ID,
                "group_type": "SEMANTIC_MEMBERSHIP_ONLY",
                "member_instance_ids": list(OBJECT_INSTANCE_IDS),
                "mask_authority": "NONE",
                "geometry_authority": "NONE",
                "pose_authority": "NONE",
            }
        ],
        "observable_frame_counts": counts,
        "claim_limit": (
            "Directly visible encoded physical-left optical-Z evidence only. center_xyz "
            "is not the hidden full-object centre; full_extent stays unobservable "
            "without complete boundary evidence. black_card_tray has no mask or "
            "geometry authority, and card_set has semantic membership only. No "
            "unified validity, external accuracy, Contact truth or Robot authority."
        ),
    }


__all__ = [
    "COMPONENT_NAMES",
    "DEPTH_REFERENCE",
    "DIRECT_VISIBILITY",
    "MASK_REGISTRATION_AUTHORITY",
    "OBJECT_INSTANCE_IDS",
    "PlanarFrameInput",
    "PlanarObservabilityError",
    "SCHEMA_VERSION",
    "SEMANTIC_GROUP_ID",
    "SUPPORT_ENTITY_ID",
    "build_observability_document",
    "estimate_planar_frame",
]
