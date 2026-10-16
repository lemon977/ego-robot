"""Pixel-local candidate occlusion: raw support, explicit geometry and display.

CPU functions only. They do not recover hidden geometry, infer contact or claim
physical correctness. Missing scene depth keeps geometry UNKNOWN. An object-
first display fallback is limited to trusted, actually visible raw support.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np

from .clean_references import validate_clean_preservation


class GeometryDecision(IntEnum):
    UNKNOWN = 0
    SCENE_FRONT = 1
    ROBOT_FRONT = 2


class DisplayPolicy(IntEnum):
    KEEP_CLEAN = 0
    DEPTH_SCENE_FRONT = 1
    DEPTH_ROBOT_FRONT = 2
    OBJECT_FIRST_FALLBACK = 3
    ROBOT_WITH_UNKNOWN_ORDER = 4


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"MASK_DOMAIN_OR_TYPE:{name}")
    return array


def resolve_robot_depth_chain(
    depth_by_link: Mapping[str, np.ndarray],
    support_by_link: Mapping[str, np.ndarray],
    *,
    required_links: Sequence[str],
    kind_by_link: Mapping[str, str],
) -> dict:
    """Resolve nearest *per-pixel* depth across the complete frozen asset links.

    The caller must supply the asset's complete required-link manifest, not just
    visible fingertips. Empty raster masks are allowed for off-screen links.
    Arm, flange, palm and fingers must all occur in that manifest. Tie ownership
    follows manifest order; no distance/contact threshold is introduced.
    """
    names = tuple(required_links)
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("ROBOT_REQUIRED_LINK_MANIFEST")
    if set(depth_by_link) != set(names) or set(support_by_link) != set(names) or set(kind_by_link) != set(names):
        raise ValueError("ROBOT_DEPTH_CHAIN_LINK_CLOSURE_MISMATCH")
    if not {"ARM", "FLANGE", "PALM", "FINGER"}.issubset(set(kind_by_link.values())):
        raise ValueError("ROBOT_DEPTH_CHAIN_MISSING_COMPLETE_COMPONENTS")
    first = np.asarray(depth_by_link[names[0]])
    if first.ndim != 2 or not first.size:
        raise ValueError("ROBOT_DEPTH_DOMAIN")
    shape = first.shape
    nearest = np.full(shape, np.inf, np.float64)
    owner = np.full(shape, -1, np.int32)
    for code, name in enumerate(names):
        support = _mask(support_by_link[name], shape, name)
        depth = np.asarray(depth_by_link[name])
        if depth.shape != shape or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"ROBOT_DEPTH_DOMAIN_OR_TYPE:{name}")
        if np.any(support & (~np.isfinite(depth) | (depth <= 0))):
            raise ValueError(f"ROBOT_VISIBLE_LINK_DEPTH_INVALID:{name}")
        nearer = support & (depth < nearest)
        nearest[nearer] = depth[nearer]
        owner[nearer] = code
    return {
        "robot_depth": nearest,
        "robot_link_index": owner,
        "link_names": names,
        "support": owner >= 0,
        "component_kinds": dict(kind_by_link),
        "complete_required_link_closure": True,
    }


def resolve_pixel_occlusion(
    robot_alpha: np.ndarray,
    depth_chain: Mapping[str, object],
    trusted_visible_object: np.ndarray,
    raw_human_support: np.ndarray,
    raw_device_support: np.ndarray,
    effective_delete_mask: np.ndarray,
    *,
    object_support_source: str,
    scene_depth: np.ndarray | None = None,
    scene_depth_valid: np.ndarray | None = None,
    scene_depth_source: str | None = None,
    depths_comparable: bool = False,
) -> dict:
    """Geometry is measured separately from the deliberately chosen display.

    ``depths_comparable`` must come from the frozen camera/unit/registration
    contract, never from a good-looking composite. Equal or absent depth keeps
    order UNKNOWN. Original human/device/deletion support cannot occlude Robot
    even if a raw depth image still contains it. Clean is never a depth source.
    """
    if object_support_source != "RAW_VISIBLE_OBJECT_SUPPORT":
        raise ValueError("SCENE_FOREGROUND_REQUIRES_VISIBLE_RAW_EVIDENCE")
    if not isinstance(depths_comparable, bool):
        raise ValueError("EXPLICIT_DEPTH_COMPARABILITY_REQUIRED")
    depth = np.asarray(depth_chain["robot_depth"])
    owner = np.asarray(depth_chain["robot_link_index"])
    if depth.ndim != 2 or not depth.size or owner.shape != depth.shape or owner.dtype != np.int32:
        raise ValueError("ROBOT_DEPTH_CHAIN_DOMAIN")
    shape = depth.shape
    alpha = np.asarray(robot_alpha)
    if (alpha.shape != shape or not (alpha.dtype == np.bool_ or np.issubdtype(alpha.dtype, np.floating))
            or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1))):
        raise ValueError("ROBOT_ALPHA_DOMAIN_OR_TYPE")
    robot = alpha > 0
    if (not depth_chain.get("complete_required_link_closure", False)
            or np.any(robot & ((owner < 0) | ~np.isfinite(depth) | (depth <= 0)))):
        raise ValueError("ROBOT_ALPHA_OUTSIDE_COMPLETE_DEPTH_CHAIN")
    visible = _mask(trusted_visible_object, shape, "trusted_visible_object")
    human = _mask(raw_human_support, shape, "raw_human_support")
    equipment = _mask(raw_device_support, shape, "raw_device_support")
    delete = _mask(effective_delete_mask, shape, "effective_delete_mask")
    eligible = visible & ~human & ~equipment & ~delete
    overlap = robot & eligible
    geometry = np.full(shape, GeometryDecision.UNKNOWN, np.uint8)
    policy = np.full(shape, DisplayPolicy.KEEP_CLEAN, np.uint8)
    policy[robot] = DisplayPolicy.ROBOT_WITH_UNKNOWN_ORDER
    valid = np.zeros(shape, bool)
    if scene_depth is not None:
        if scene_depth_source != "RAW_REGISTERED_SCENE_DEPTH":
            raise ValueError("SCENE_DEPTH_MUST_NOT_COME_FROM_CLEAN")
        scene = np.asarray(scene_depth)
        if scene.shape != shape or not np.issubdtype(scene.dtype, np.floating):
            raise ValueError("SCENE_DEPTH_DOMAIN_OR_TYPE")
        if scene_depth_valid is None:
            raise ValueError("SCENE_DEPTH_VALIDITY_REQUIRED")
        declared = _mask(scene_depth_valid, shape, "scene_depth_valid")
        if depths_comparable:
            valid = overlap & declared & np.isfinite(scene) & (scene > 0)
            front = valid & (scene < depth)
            behind = valid & (scene > depth)
            geometry[front] = GeometryDecision.SCENE_FRONT
            geometry[behind] = GeometryDecision.ROBOT_FRONT
            policy[front] = DisplayPolicy.DEPTH_SCENE_FRONT
            policy[behind] = DisplayPolicy.DEPTH_ROBOT_FRONT
    elif scene_depth_valid is not None or scene_depth_source is not None or depths_comparable:
        raise ValueError("SCENE_DEPTH_EVIDENCE_ABSENT")
    fallback = overlap & (geometry == GeometryDecision.UNKNOWN)
    policy[fallback] = DisplayPolicy.OBJECT_FIRST_FALLBACK
    names = tuple(depth_chain["link_names"])
    if np.any(robot & (owner >= len(names))):
        raise ValueError("ROBOT_LINK_INDEX_OUT_OF_MANIFEST")
    summary = {}
    for code, name in enumerate(names):
        region = robot & (owner == code)
        summary[name] = {
            "visible_robot_pixels": int(region.sum()),
            "trusted_object_overlap_pixels": int((region & overlap).sum()),
            "scene_front_pixels": int((region & (geometry == GeometryDecision.SCENE_FRONT)).sum()),
            "robot_front_pixels": int((region & (geometry == GeometryDecision.ROBOT_FRONT)).sum()),
            "unknown_pixels": int((region & (geometry == GeometryDecision.UNKNOWN)).sum()),
            "object_first_fallback_pixels": int((region & fallback).sum()),
        }
    return {
        "robot_support": robot.copy(),
        "geometry_decision": geometry,
        "display_policy": policy,
        "trusted_object_overlap": overlap,
        "excluded_raw_foreground": visible & (human | equipment | delete),
        "object_first_fallback": fallback,
        "link_summary": summary,
        "geometry_labels": {member.name: int(member) for member in GeometryDecision},
        "display_labels": {member.name: int(member) for member in DisplayPolicy},
        "depths_comparable": depths_comparable,
        "physical_contact_ground_truth": False,
        "control_ground_truth": False,
        "training_eligible": False,
    }


def compose_with_occlusion(
    raw_bgr: np.ndarray,
    clean_bgr: np.ndarray,
    robot_bgr: np.ndarray,
    robot_alpha: np.ndarray,
    occlusion: Mapping[str, object],
    effective_delete_mask: np.ndarray,
    protected_visible_mask: np.ndarray,
) -> dict:
    """Compose without changing the separate Clean verdict or whole-link masks."""
    validation = validate_clean_preservation(raw_bgr, clean_bgr, effective_delete_mask,
                                           protected_visible_mask)
    if validation["status"] != "PASS":
        raise ValueError("CLEAN_EXISTING_ZERO_CHANGE_INVARIANT_FAILED")
    clean, raw, robot = np.asarray(clean_bgr), np.asarray(raw_bgr), np.asarray(robot_bgr)
    if clean.ndim != 3 or robot.shape != clean.shape or robot.dtype != np.uint8:
        raise ValueError("ROBOT_RGB_DOMAIN_OR_TYPE")
    shape = clean.shape[:2]
    alpha = np.asarray(robot_alpha)
    if (alpha.shape != shape or not (alpha.dtype == np.bool_ or np.issubdtype(alpha.dtype, np.floating))
            or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1))):
        raise ValueError("ROBOT_ALPHA_DOMAIN_OR_TYPE")
    geometry = np.asarray(occlusion["geometry_decision"])
    policy = np.asarray(occlusion["display_policy"])
    if geometry.shape != shape or policy.shape != shape or geometry.dtype != np.uint8 or policy.dtype != np.uint8:
        raise ValueError("OCCLUSION_RESULT_DOMAIN_OR_TYPE")
    if (not np.isin(geometry, [int(value) for value in GeometryDecision]).all()
            or not np.isin(policy, [int(value) for value in DisplayPolicy]).all()):
        raise ValueError("OCCLUSION_RESULT_UNKNOWN_LABEL")
    robot_support = _mask(occlusion["robot_support"], shape, "robot_support")
    if not np.array_equal(robot_support, alpha > 0):
        raise ValueError("OCCLUSION_ROBOT_SUPPORT_CHANGED")
    if np.any((policy != DisplayPolicy.KEEP_CLEAN) & ~robot_support):
        raise ValueError("OCCLUSION_POLICY_OUTSIDE_ROBOT_SUPPORT")
    if (np.any((geometry == GeometryDecision.SCENE_FRONT) != (policy == DisplayPolicy.DEPTH_SCENE_FRONT))
            or np.any((geometry == GeometryDecision.ROBOT_FRONT) != (policy == DisplayPolicy.DEPTH_ROBOT_FRONT))):
        raise ValueError("OCCLUSION_GEOMETRY_DISPLAY_MISMATCH")
    object_front = ((policy == DisplayPolicy.DEPTH_SCENE_FRONT)
                    | (policy == DisplayPolicy.OBJECT_FIRST_FALLBACK))
    overlap = _mask(occlusion["trusted_object_overlap"], shape, "trusted_object_overlap")
    if np.any(object_front & ~overlap) or np.any(overlap & ~robot_support):
        raise ValueError("OBJECT_FIRST_OUTSIDE_TRUSTED_RAW_OVERLAP")
    draw = (alpha > 0) & ~object_front
    weight = np.where(draw, alpha, 0).astype(np.float64)[..., None]
    final = np.rint(clean.astype(np.float64) * (1 - weight) + robot.astype(np.float64) * weight).astype(np.uint8)
    # Only trusted, actually visible raw object pixels can be restored here.
    final[object_front] = raw[object_front]
    return {
        "image": final,
        "draw": draw,
        "object_front": object_front,
        "geometry_decision": geometry.copy(),
        "display_policy": policy.copy(),
        "clean_validation": validation,
        "clean_visual_quality": "NOT_EVALUATED_BY_ROBOT_OVERLAY",
        "physical_contact_ground_truth": False,
        "control_ground_truth": False,
        "training_eligible": False,
    }
