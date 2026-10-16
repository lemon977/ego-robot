"""Batch-specific visible-cup protection and visual contact candidates.

Frozen parent model objects are supplied by the registered runner. This module
does not own a GPU lease, writer, video encoder, motion solver, or output paths.
Clean and contact candidates are offline visual evidence, never physical truth.
"""

from __future__ import annotations

from typing import Any, Mapping
import copy

import cv2
import numpy as np

from chaoyang.exact78_baseline import mask as parent_mask
from chaoyang.exact78_baseline.clean import SETTINGS, _get_ref_index
from chaoyang.pipeline.exact78_visual_compositor_v1 import compose_visual_frame

PROMPTS = {
    "human": parent_mask.HUMAN_PROMPT,
    "glove": "a MANUS motion capture glove worn on a person's hand",
    "controller": "a white VR controller held by or attached to a person's hand",
    "attachment": "a device or connector attached to a person's hand or wrist",
    "cable": "a cable connected to a person's motion capture glove or wrist device",
    "object": "paper cup",
}
REFERENCE_SIZE = (640, 480)
CONTACT_PX = 12.0
WINDOW_PAD_S = 0.5
WINDOW_MERGE_GAP_S = 0.1
HEAD_HINT_SOURCES = frozenset({"RAW_POSE", "INTERP_SHORT", "EDGE_HOLD_SHORT"})


def _bool_masks(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"MASK_DOMAIN_OR_TYPE:{name}")
    return value


def _rgb_frames(value: Any) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8 or not len(value):
        raise ValueError("RGB_FRAMES_DOMAIN_OR_TYPE")
    return value


def _contact_scale(shape: tuple[int, int]) -> float:
    sx, sy = shape[1] / REFERENCE_SIZE[0], shape[0] / REFERENCE_SIZE[1]
    if not np.isclose(sx, sy):
        raise ValueError("CONTACT_RULE_REQUIRES_ISOTROPIC_DOMAIN")
    return sx


def _near(mask: np.ndarray, radius: float) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius


def _track_wearer_with_semantics(sam: Any, rgb: np.ndarray, geometry: dict,
                                anchor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Keep parent selection/retry behavior while retaining observed SAM support."""
    count, shape = len(rgb), rgb.shape[1:3]
    tracks = sam.track_text(list(rgb), PROMPTS["human"], anchor)
    aborted = list(sam.aborted)
    first_pass = set(tracks)
    core, removal, info = parent_mask.build_masks(tracks, geometry, count, shape, frames_rgb=rgb)
    retries = []
    for side_index, side in enumerate(("left", "right")):
        rate = info["sam_forearm_hit_rate"][side]
        points = info["sam_forearm_probe_points"][side]
        if rate is None or points < parent_mask.FOREARM_RETRY_PROBES_MIN or rate >= parent_mask.FOREARM_RETRY_RATE:
            continue
        retry_anchor = parent_mask.side_forearm_anchor(geometry, shape, side_index)
        if retry_anchor == anchor or any(row["anchor_frame"] == retry_anchor for row in retries):
            continue
        extra = sam.track_text(list(rgb), PROMPTS["human"], retry_anchor)
        offset = max(tracks, default=-1) + 1
        tracks.update({offset + obj: frames for obj, frames in extra.items()})
        retries.append({"side": side, "anchor_frame": retry_anchor, "hit_rate_before": rate,
                        "tracks": len(extra), "propagation_aborted": list(sam.aborted)})
    if retries:
        core, removal, info = parent_mask.build_masks(
            tracks, geometry, count, shape, frames_rgb=rgb, provider_tracks=first_pass)
    semantic = np.stack([
        parent_mask._wearer_sam(tracks, info["kept_obj_ids"], frame, shape)
        for frame in range(count)
    ])
    info["propagation_aborted"], info["forearm_retry"] = aborted, retries
    return core, removal, semantic, info


def _owned_equipment(tracks: dict, wearer: np.ndarray,
                     near_px: float) -> tuple[np.ndarray, list[dict]]:
    """Only connected components supported by wearer proximity are removed."""
    count, shape = wearer.shape[0], wearer.shape[1:]
    selected = np.zeros_like(wearer)
    near = [_near(wearer[frame], near_px) for frame in range(count)]
    rows = []
    for obj_id, frames in tracks.items():
        hits = sum(bool((parent_mask.unpack(packed, shape) & near[frame]).any())
                   for frame, packed in frames.items())
        rate = hits / max(1, len(frames))
        keep = rate >= parent_mask.TRACK_HIT_RATE_MIN
        rows.append({"obj_id": int(obj_id), "frames": len(frames),
                     "wearer_hit_rate": float(rate), "kept": keep})
        if not keep:
            continue
        for frame, packed in frames.items():
            mask = parent_mask.unpack(packed, shape)
            components, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            touched = np.unique(labels[near[frame] & mask])
            for component in touched:
                if 0 < component < components:
                    selected[frame] |= labels == component
    return selected, rows


def head_hint_valid(session: Any) -> np.ndarray:
    """Only declared raw or bounded short-fill camera poses support projection."""
    labels = getattr(session, "native_motion", {}).get("head_source")
    if labels is None:
        return np.zeros(session.frame_count, bool)
    labels = np.asarray(labels).astype(str)
    if labels.shape != (session.frame_count,):
        raise ValueError("MASK_HEAD_SOURCE_DOMAIN")
    return np.isin(labels, list(HEAD_HINT_SOURCES))


def _gate_diagnostic_geometry(geometry: dict, session: Any) -> tuple[dict, list]:
    """Diagnostic fills can be displayed, but cannot support mask ownership."""
    valid = getattr(session, "motion_valid", {}).get("hand_display_valid")
    if valid is None and hasattr(session, "hand_source"):
        labels = np.asarray(session.hand_source).astype(str)
        valid = (labels != "NONE") & ~np.char.startswith(labels, "DIAGNOSTIC")
    if valid is None:
        valid = np.ones((2, session.frame_count), bool)
    valid = np.asarray(valid, bool)
    if valid.shape != (2, session.frame_count):
        raise ValueError("MASK_HAND_DISPLAY_VALID_DOMAIN")
    valid = valid & head_hint_valid(session)[None]
    output = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in geometry.items()}
    invalid = ~valid
    for name in ("hands", "forearm", "forearm_radius", "tracker", "wrist_px_per_m"):
        if name in output:
            output[name][invalid] = np.nan
    if "hands_inside" in output:
        output["hands_inside"][invalid] = False
    return output, np.argwhere(invalid).tolist()



VISIBLE_POINT_ROLES = frozenset({"glove", "controller", "mount", "forearm", "attachment", "cable", "human"})
HUMAN_POINT_ROLES = frozenset({"glove", "forearm", "human"})
CUP_SUPPORT_DROP_RATIO = .5
CUP_PROTECTED_RETAIN_RATIO_MIN = .9


def _point_tracks(sam, rgb, seeds, anchor):
    from .sam_point_adapter import track_visible_points
    return track_visible_points(sam, rgb, seeds, anchor)


def _visible_seed_objects(visible_seeds, count, shape):
    if isinstance(visible_seeds, Mapping):
        default = visible_seeds.get("anchor_frame", visible_seeds.get("raw_frame_index", 0))
        seeds = visible_seeds.get("objects")
    else:
        seeds = visible_seeds
        default = seeds[0].get("frame_idx", seeds[0].get("frame_index", 0)) if seeds else 0
    if (isinstance(default, bool) or not isinstance(default, (int, np.integer))
            or not isinstance(seeds, (list, tuple)) or not seeds or len(seeds) > 16
            or not 0 <= default < count):
        raise ValueError("VISIBLE_SEED_LIST_OR_ANCHOR_DOMAIN")
    selected, ids, anchors = [], set(), []
    for seed in seeds:
        if not isinstance(seed, Mapping):
            raise ValueError("VISIBLE_SEED_OBJECT_MAPPING")
        obj_id = seed.get("obj_id")
        if (isinstance(obj_id, bool) or not isinstance(obj_id, (int, np.integer))
                or not 1 <= int(obj_id) <= 16 or int(obj_id) in ids):
            raise ValueError("VISIBLE_SEED_OBJECT_ID_DOMAIN")
        if seed.get("role") not in VISIBLE_POINT_ROLES or seed.get("side") not in ("left", "right"):
            raise ValueError("VISIBLE_SEED_ROLE_OR_SIDE_DOMAIN")
        own_values = [seed[key] for key in ("frame_idx", "frame_index", "raw_frame_index") if key in seed]
        if (any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in own_values)
                or len(set(own_values)) > 1):
            raise ValueError("VISIBLE_SEED_OBJECT_FRAME_ALIASES")
        own = int(own_values[0]) if own_values else int(default)
        if not 0 <= own < count:
            raise ValueError("VISIBLE_SEED_OBJECT_ANCHOR_DOMAIN")
        points = np.asarray(seed.get("points_px"), float)
        labels = np.asarray(seed.get("point_labels"))
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or len(points) > 32
                or labels.shape != (len(points),) or labels.dtype.kind not in "iu"
                or not np.isfinite(points).all()
                or not np.isin(labels, [0, 1]).all() or not (labels == 1).any()
                or not ((points[:, 0] >= 0) & (points[:, 0] < shape[1])
                        & (points[:, 1] >= 0) & (points[:, 1] < shape[0])).all()):
            raise ValueError("VISIBLE_SEED_POINT_DOMAIN")
        value = dict(seed)
        value["frame_idx"] = own
        selected.append(value)
        anchors.append(own)
        ids.add(int(obj_id))
    return selected, min(anchors)


def _validated_unseeded_declarations(visible_seeds, seeds, count):
    if not isinstance(visible_seeds, Mapping) or not visible_seeds.get("unseeded_roles"):
        return []
    declarations = visible_seeds["unseeded_roles"]
    validation = "NATIVE_QPC_AND_ORIGINAL_RGB_EXACT_VERIFIED"
    if (visible_seeds.get("unseeded_roles_validation") != validation
            or visible_seeds.get("seed_schema") not in {"pick_place_visible_mask_seeds_v2", "pick_place_visible_mask_seeds_v3"}
            or not isinstance(visible_seeds.get("seed_reference"), Mapping)
            or not isinstance(declarations, list)):
        raise ValueError("VISIBLE_UNSEEDED_DECLARATION_NOT_VALIDATED")
    seeded = {(seed["side"], seed["role"]) for seed in seeds}
    seen, accepted = set(), []
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise ValueError("VISIBLE_UNSEEDED_DECLARATION_DOMAIN")
        key = (declaration.get("side"), declaration.get("role"))
        frames = declaration.get("inspected_frames", [])
        if (key in seeded or key in seen or key[0] not in ("left", "right")
                or key[1] not in VISIBLE_POINT_ROLES
                or declaration.get("visibility") != "NOT_VISIBLE_IN_INSPECTED_NATIVE_FRAMES"
                or declaration.get("review") != "AI_CONFIRMED_NOT_VISIBLE"
                or declaration.get("validation") != validation
                or declaration.get("unreviewed_visibility_status") != "UNKNOWN"
                or declaration.get("later_visibility_review_needed") is not True
                or declaration.get("global_all_frames_invisible") is not False
                or declaration.get("physical_ground_truth", False) is not False
                or not isinstance(frames, list) or len(frames) < 3):
            raise ValueError("VISIBLE_UNSEEDED_DECLARATION_DOMAIN")
        indices = []
        for frame in frames:
            if (not isinstance(frame, Mapping) or type(frame.get("raw_frame_index")) is not int
                    or not 0 <= frame["raw_frame_index"] < count
                    or type(frame.get("qpc_ns")) is not int
                    or not isinstance(frame.get("image"), Mapping)
                    or not {"path", "bytes", "sha256"} <= set(frame["image"])):
                raise ValueError("VISIBLE_UNSEEDED_DECLARATION_DOMAIN")
            indices.append(frame["raw_frame_index"])
        if (len(set(indices)) != len(indices) or 0 not in indices or count - 1 not in indices
                or not ({(count - 1) // 2, count // 2} & set(indices))):
            raise ValueError("VISIBLE_UNSEEDED_ZERO_MID_LAST_REQUIRED")
        accepted.append(copy.deepcopy(dict(declaration)))
        seen.add(key)
    return accepted


def _cup_support_review(cup, protected, conflict):
    areas = {name: value.sum(axis=(1, 2)).astype(np.int64)
             for name, value in (("cup_observed", cup), ("protected_visible", protected),
                                 ("semantic_conflict", conflict))}
    cup_area, protected_area = areas["cup_observed"], areas["protected_visible"]
    empty = np.flatnonzero(cup_area == 0).tolist()
    lost = np.flatnonzero((cup_area > 0) & (protected_area == 0)).tolist()
    shrink = np.flatnonzero((cup_area > 0) & (protected_area < CUP_PROTECTED_RETAIN_RATIO_MIN * cup_area)).tolist()
    drop = (np.flatnonzero((cup_area[:-1] > 0)
                           & (cup_area[1:] < CUP_SUPPORT_DROP_RATIO * cup_area[:-1])) + 1).tolist()
    protected_drop = (np.flatnonzero((protected_area[:-1] > 0)
                                     & (protected_area[1:] < CUP_SUPPORT_DROP_RATIO * protected_area[:-1])) + 1).tolist()
    problems = []
    for code, missing in (
        ("CUP_ENTIRE_SESSION_SUPPORT_MISSING", not cup_area.any()),
        ("CUP_ENTIRE_SESSION_PROTECTED_SUPPORT_MISSING", not protected_area.any()),
    ):
        if missing:
            problems.append({"code": code, "severity": "QUALITY_FAILURE",
                             "frames": list(range(len(cup_area))),
                             "authority": "ENTIRE_SESSION_HAS_NO_USABLE_VISIBLE_CUP_SUPPORT"})
    for code, frames in (
        ("CUP_CURRENT_FRAME_SUPPORT_EMPTY_OR_OCCLUDED", empty),
        ("CUP_ALL_OBSERVED_SUPPORT_VETOED_BY_FOREGROUND", lost),
        ("CUP_PROTECTED_SUPPORT_SHRUNK_BY_FOREGROUND", shrink),
        ("CUP_CURRENT_FRAME_SUPPORT_SUDDEN_DROP", drop),
        ("CUP_PROTECTED_SUPPORT_SUDDEN_DROP", protected_drop),
    ):
        if frames:
            problems.append({"code": code, "severity": "REVIEW_REQUIRED", "frames": frames,
                             "authority": "REQUIRES_ACTUAL_RAW_VISIBILITY_REVIEW_NOT_HIDDEN_OBJECT_INFERENCE"})
    if conflict.any():
        problems.append({"code": "PAPER_CUP_WEARER_SEMANTIC_CONFLICT", "severity": "REVIEW_REQUIRED",
                         "frames": np.flatnonzero(conflict.any(axis=(1, 2))).tolist()})
    return {**{name + "_pixels_per_frame": value.tolist() for name, value in areas.items()},
            "sudden_drop_ratio": CUP_SUPPORT_DROP_RATIO,
            "protected_retain_ratio_min": CUP_PROTECTED_RETAIN_RATIO_MIN,
            "whole_video_has_cup_support": bool(cup_area.any()),
            "whole_video_has_protected_visible_support": bool(protected_area.any()),
            "empty_support_is_not_claim_of_visible_cup_erasure": True}, problems


def _point_geometry_diagnostics(geometry, side_masks, shape):
    points = np.asarray(geometry["hands"])
    count = points.shape[1]
    inside = np.zeros((count, 2), np.int64)
    supported = np.zeros((count, 2), np.int64)
    for frame in range(count):
        for side in range(2):
            value = points[side, frame]
            valid = np.isfinite(value).all(-1)
            value = np.round(value[valid]).astype(int)
            ok = ((value[:, 0] >= 0) & (value[:, 0] < shape[1])
                  & (value[:, 1] >= 0) & (value[:, 1] < shape[0]))
            value = value[ok]
            inside[frame, side] = len(value)
            if len(value):
                near = _near(side_masks[side, frame], CONTACT_PX * _contact_scale(shape))
                supported[frame, side] = int(near[value[:, 1], value[:, 0]].sum())
    return {"projected_hand_points_in_view": inside.tolist(),
            "projected_hand_points_near_same_side_observed_segmentation": supported.tolist(),
            "geometry_deletion_authority": False, "cross_side_registration": False,
            "unsupported_hand_hull_deletion": False}



TINY_COMPONENT_MAX_PIXELS = 16
SAME_SIDE_LARGE_COMPONENT_MIN_PIXELS = 32
TINY_COMPONENT_MIN_SEPARATION_PX = 64.0


def _tiny_component_veto(tracks, seed_by_id, point_info, count, shape):
    """Remove detached tiny observed components; never add deletion pixels.

    Support comes only from current-frame same-side SAM observations. A large
    connected component of their raw union must exist; geometry, cup pixels,
    other frames and the other hand cannot supply support or removal authority.
    Raw packed tracks are retained unchanged for the per-object evidence sidecar.
    """
    if set(tracks) - set(seed_by_id):
        raise ValueError("VISIBLE_POINT_TRACK_UNKNOWN_OBJECT_ID")
    filtered = {obj_id: dict(tracks.get(obj_id, {})) for obj_id in seed_by_id}
    diagnostics = {}
    supported = {}
    for obj_id, seed in seed_by_id.items():
        frames = tracks.get(obj_id, {})
        for frame in frames:
            if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))
                    or not 0 <= frame < count):
                raise ValueError("VISIBLE_POINT_TRACK_FRAME_DOMAIN")
        supported[obj_id] = bool(point_info.get("objects", {}).get(str(obj_id), {})
                                 .get("anchor_mask_present"))
        diagnostics[str(obj_id)] = {
            "role": seed["role"], "side": seed["side"],
            "original_pixels_per_frame": [0] * count,
            "filtered_pixels_per_frame": [0] * count,
            "removed_pixels_per_frame": [0] * count,
            "removed_components": [],
        }
    no_large_support = {"left": [], "right": []}
    for frame in range(count):
        masks = {}
        side_unions = {side: np.zeros(shape, bool) for side in ("left", "right")}
        for obj_id, seed in seed_by_id.items():
            packed = tracks.get(obj_id, {}).get(frame)
            value = (parent_mask.unpack(packed, shape) if packed is not None
                     else np.zeros(shape, bool))
            masks[obj_id] = value
            diagnostics[str(obj_id)]["original_pixels_per_frame"][frame] = int(value.sum())
            if supported[obj_id]:
                side_unions[seed["side"]] |= value
        distance_by_side = {}
        for side, union in side_unions.items():
            components, labels, stats, _ = cv2.connectedComponentsWithStats(
                union.astype(np.uint8), connectivity=8)
            large = np.zeros(shape, bool)
            for component in range(1, components):
                if stats[component, cv2.CC_STAT_AREA] >= SAME_SIDE_LARGE_COMPONENT_MIN_PIXELS:
                    large |= labels == component
            if not large.any():
                no_large_support[side].append(frame)
                distance_by_side[side] = None
            else:
                distance_by_side[side] = cv2.distanceTransform(
                    (~large).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for obj_id, seed in seed_by_id.items():
            value = masks[obj_id]
            info = diagnostics[str(obj_id)]
            result = value
            distance = distance_by_side[seed["side"]]
            if supported[obj_id] and value.any() and distance is not None:
                components, labels, stats, _ = cv2.connectedComponentsWithStats(
                    value.astype(np.uint8), connectivity=8)
                for component in range(1, components):
                    area = int(stats[component, cv2.CC_STAT_AREA])
                    if area > TINY_COMPONENT_MAX_PIXELS:
                        continue
                    selected = labels == component
                    nearest = float(distance[selected].min())
                    if nearest <= TINY_COMPONENT_MIN_SEPARATION_PX:
                        continue
                    if result is value:
                        result = value.copy()
                    result[selected] = False
                    info["removed_components"].append({
                        "raw_frame_index": frame, "area": area,
                        "bbox_xywh": stats[component, :4].astype(int).tolist(),
                        "minimum_same_side_large_component_distance_px": nearest,
                    })
                if result is not value:
                    filtered[obj_id][frame] = np.packbits(result)
            after = int(result.sum())
            info["filtered_pixels_per_frame"][frame] = after
            info["removed_pixels_per_frame"][frame] = (
                info["original_pixels_per_frame"][frame] - after)
    return filtered, {
        "recipe": "CURRENT_RAW_SAME_SIDE_OBSERVED_UNION_DETACHED_TINY_COMPONENT_VETO",
        "status": "CANDIDATE_REQUIRES_ACTUAL_GPU_AND_IMAGE_REVIEW",
        "small_component_max_pixels": TINY_COMPONENT_MAX_PIXELS,
        "same_side_union_large_component_min_pixels": SAME_SIDE_LARGE_COMPONENT_MIN_PIXELS,
        "minimum_separation_px_exclusive": TINY_COMPONENT_MIN_SEPARATION_PX,
        "frame_image_size": [shape[1], shape[0]],
        "distance_metric": "MINIMUM_EXACT_EUCLIDEAN_PIXEL_CENTER_DISTANCE",
        "only_removes_observed_deletion_pixels": True,
        "uses_current_frame_only": True,
        "geometry_authority": False, "cup_authority": False,
        "cross_side_authority": False, "adds_foreground_pixels": False,
        "raw_tracks_mutated": False,
        "no_same_side_large_component_raw_frames": no_large_support,
        "removed_pixels_total": sum(sum(value["removed_pixels_per_frame"])
                                    for value in diagnostics.values()),
        "objects": diagnostics,
    }


def _build_visible_point_masks(sam, rgb, session, visible_seeds):
    count, height, width = rgb.shape[:3]
    shape, scale = (height, width), _contact_scale((height, width))
    seeds, anchor = _visible_seed_objects(visible_seeds, count, shape)
    unseeded = _validated_unseeded_declarations(visible_seeds, seeds, count)
    exempt_roles = {(value["side"], value["role"]) for value in unseeded}
    tracks, point_info = _point_tracks(sam, rgb, seeds, anchor)
    seed_by_id = {int(seed["obj_id"]): seed for seed in seeds}
    if not isinstance(tracks, dict) or not isinstance(point_info, Mapping):
        raise ValueError("VISIBLE_POINT_ADAPTER_RESULT_DOMAIN")
    raw_point_tracks = tracks
    tracks, tiny_veto = _tiny_component_veto(tracks, seed_by_id, point_info, count, shape)
    observed = np.zeros((count, *shape), bool)
    sam_core = np.zeros_like(observed)
    equipment = np.zeros_like(observed)
    side_masks = np.zeros((2, count, *shape), bool)
    equipment_roles = {}
    foreground_ids = [[] for _ in range(count)]
    role_areas = {}
    problems = []
    if point_info.get("status") != "COMPLETE":
        problems.append({"code": "POINT_TRACK_SUPPORT_OR_RUNTIME_INCOMPLETE", "severity": "QUALITY_FAILURE",
                         "adapter_status": point_info.get("status"), "errors": point_info.get("errors", []),
                         "aborted": point_info.get("aborted", [])})
    for obj_id, seed in seed_by_id.items():
        frames = tracks.get(obj_id, {})
        declared = point_info.get("objects", {}).get(str(obj_id), {})
        present = bool(declared.get("anchor_mask_present"))
        per_frame = np.zeros_like(observed)
        # Anchor confirmation is mandatory even if a later hallucinated ID has a mask.
        if present:
            for frame, packed in frames.items():
                if not isinstance(frame, (int, np.integer)) or not 0 <= frame < count:
                    raise ValueError("VISIBLE_POINT_TRACK_FRAME_DOMAIN")
                per_frame[int(frame)] = parent_mask.unpack(packed, shape)
        else:
            problems.append({"code": "VISIBLE_ROLE_ANCHOR_SUPPORT_MISSING", "severity": "QUALITY_FAILURE",
                             "obj_id": obj_id, "role": seed["role"], "side": seed["side"]})
        areas = per_frame.sum(axis=(1, 2)).astype(np.int64)
        original_areas = tiny_veto["objects"][str(obj_id)]["original_pixels_per_frame"]
        filtered_anchor_present = bool(areas[seed["frame_idx"]])
        if present and original_areas[seed["frame_idx"]] > 0 and not filtered_anchor_present:
            problems.append({"code": "VISIBLE_ROLE_ANCHOR_SUPPORT_REMOVED_BY_TINY_COMPONENT_VETO",
                             "severity": "QUALITY_FAILURE", "obj_id": obj_id,
                             "role": seed["role"], "side": seed["side"],
                             "raw_frame_index": seed["frame_idx"]})
        for correction in seed.get("correction_points", []):
            correction_frame = next(correction[key] for key in ("frame_idx", "frame_index", "raw_frame_index") if key in correction)
            if present and original_areas[correction_frame] > 0 and not areas[correction_frame]:
                problems.append({"code": "VISIBLE_CORRECTION_SUPPORT_REMOVED_BY_TINY_COMPONENT_VETO",
                                 "severity": "QUALITY_FAILURE", "obj_id": obj_id,
                                 "role": seed["role"], "side": seed["side"],
                                 "raw_frame_index": correction_frame})
        role_areas[str(obj_id)] = {"role": seed["role"], "side": seed["side"],
                                   "pixels_per_frame": areas.tolist(),
                                   "original_pixels_per_frame": original_areas,
                                   "tiny_veto_removed_pixels_per_frame": tiny_veto["objects"][str(obj_id)]["removed_pixels_per_frame"],
                                   "anchor_mask_present": present,
                                   "anchor_filtered_mask_present": filtered_anchor_present,
                                   "anchor_raw_frame_index": seed["frame_idx"],
                                   "anchor_qpc_ns": seed.get("qpc_ns"),
                                   "anchor_image": seed.get("image")}
        for frame in np.flatnonzero(areas):
            foreground_ids[int(frame)].append(obj_id)
        if np.any(areas > .8 * height * width):
            problems.append({"code": "OBSERVED_ROLE_MASK_COVERS_MOST_IMAGE", "severity": "QUALITY_FAILURE",
                             "obj_id": obj_id, "frames": np.flatnonzero(areas > .8 * height * width).tolist()})
        observed |= per_frame
        side_masks[0 if seed["side"] == "left" else 1] |= per_frame
        if seed["role"] in HUMAN_POINT_ROLES:
            sam_core |= per_frame
        else:
            equipment |= per_frame
            equipment_roles.setdefault(seed["role"], np.zeros_like(observed))
            equipment_roles[seed["role"]] |= per_frame
    if set(tracks) - set(seed_by_id):
        raise ValueError("VISIBLE_POINT_TRACK_UNKNOWN_OBJECT_ID")
    for side in ("left", "right"):
        for role in ("glove", "controller"):
            ids = [obj_id for obj_id, seed in seed_by_id.items() if seed["side"] == side and seed["role"] == role]
            if not ids and (side, role) in exempt_roles:
                continue
            if not ids or not any(role_areas[str(obj_id)]["anchor_mask_present"] for obj_id in ids):
                problems.append({"code": "REQUIRED_VISIBLE_GLOVE_OR_CONTROLLER_SUPPORT_MISSING",
                                 "severity": "QUALITY_FAILURE", "side": side, "role": role})
    for declaration in unseeded:
        problems.append({"code": "UNSEEDED_ROLE_LATER_VISIBILITY_REVIEW_REQUIRED",
                         "severity": "REVIEW_REQUIRED", "side": declaration["side"],
                         "role": declaration["role"],
                         "inspected_raw_frame_indices": [item["raw_frame_index"] for item in declaration["inspected_frames"]],
                         "unreviewed_visibility_status": "UNKNOWN",
                         "global_all_frames_invisible": False})
    # The point state has been cleared by the adapter before this independent text state.
    # Cup has its own state and native anchor; wearer seed times cannot select it.
    cup_anchor = 0
    cup_attempts = []
    cup_tracks = sam.track_text(list(rgb), PROMPTS["object"], cup_anchor)
    cup_aborted = copy.deepcopy(list(sam.aborted))
    has_proposal = any(np.asarray(packed).any() for frames in cup_tracks.values() for packed in frames.values())
    cup_attempts.append({"anchor_frame": cup_anchor, "tracks": len(cup_tracks),
                         "has_mask_proposal": bool(has_proposal), "propagation_aborted": cup_aborted})
    if not has_proposal and cup_aborted and count > 1:
        cup_anchor = count // 2
        cup_tracks = sam.track_text(list(rgb), PROMPTS["object"], cup_anchor)
        cup_aborted = copy.deepcopy(list(sam.aborted))
        has_proposal = any(np.asarray(packed).any() for frames in cup_tracks.values() for packed in frames.values())
        cup_attempts.append({"anchor_frame": cup_anchor, "tracks": len(cup_tracks),
                             "has_mask_proposal": bool(has_proposal), "propagation_aborted": cup_aborted})
    cup = np.zeros_like(observed)
    rejected = []
    for obj_id, frames in cup_tracks.items():
        for frame, packed in frames.items():
            if not isinstance(frame, (int, np.integer)) or not 0 <= frame < count:
                raise ValueError("CUP_TEXT_TRACK_FRAME_DOMAIN")
            mask = parent_mask.unpack(packed, shape)
            if parent_mask._object_like(mask):
                cup[int(frame)] |= mask
            else:
                rejected.append({"obj_id": int(obj_id), "frame_index": int(frame),
                                 "code": "CUP_MASK_NOT_COMPACT_OR_AREA_LIMIT"})
    if cup_aborted:
        problems.append({"code": "CUP_TEXT_PROPAGATION_ABORTED", "severity": "QUALITY_FAILURE",
                         "directions": cup_aborted})
    conflict = cup & observed
    protected = cup & ~observed
    support_info, cup_problems = _cup_support_review(cup, protected, conflict)
    problems.extend(cup_problems)
    dilation_px = max(1, int(round(parent_mask.HUMAN_DILATE_PX * scale)))
    dilated = np.stack([parent_mask.dilate(mask, dilation_px) for mask in observed])
    bracket = parent_mask.bracket_fill(dilated, parent_mask.REMOVAL_BRACKET_FRAMES)
    removal = bracket & ~protected
    held = np.stack([protected[i] & _near(observed[i], CONTACT_PX * scale) for i in range(count)])
    geometry = parent_mask.pico_geometry(session, width / session.width)
    geometry, diagnostic_sides = _gate_diagnostic_geometry(geometry, session)
    geometry_info = _point_geometry_diagnostics(geometry, side_masks, shape)
    return {
        "core": observed, "sam_core": sam_core, "equipment": equipment,
        "equipment_roles": equipment_roles, "removal": removal,
        "cup_observed": cup, "semantic_conflict": conflict, "protected_visible": protected,
        "held": held, "unknown": protected.copy(),
        "point_object_tracks": raw_point_tracks,
        "metadata": {
            "schema_version": "pick-place-visual-masks-observed-point-v2",
            "tiny_component_veto": tiny_veto,
            "recipe": "VALIDATED_VISIBLE_PIXEL_SEEDS_SAM31_OBSERVED_SEGMENTATION",
            "anchor_frame": anchor, "point_tracking": dict(point_info),
            "unseeded_role_declarations": unseeded,
            "unreviewed_role_visibility_status": "UNKNOWN",
            "later_role_visibility_review_needed": bool(unseeded),
            "not_visible_declaration_is_not_global_visibility_claim": True,
            "visible_seed_evidence": {key: copy.deepcopy(visible_seeds.get(key))
                for key in ("seed_reference", "seed_index_reference", "image", "source_images")
                if isinstance(visible_seeds, Mapping) and visible_seeds.get(key) is not None},
            "foreground_ids_per_frame": foreground_ids, "foreground_role_support": role_areas,
            "cup_support": support_info, "geometry_diagnostics": geometry_info,
            "invalid_geometry_frame_sides": diagnostic_sides,
            "human": {"provider": "OBSERVED_POINT_SEGMENTATION", "pico_shift_px": [],
                      "geometry_deletion_authority": False},
            "equipment": {"provider": "OBSERVED_POINT_SEGMENTATION",
                          "seed_roles": sorted(equipment_roles)},
            "objects": {"prompt": PROMPTS["object"], "anchor_frame": cup_anchor,
                        "primary_anchor_frame": 0, "attempts": cup_attempts,
                        "fallback_retry_limit": 1,
                        "fallback_authority": "EMPTY_INITIAL_PROPOSALS_AND_ABORT_ONLY_FIXED_NATIVE_MID",
                        "tracks": len(cup_tracks), "propagation_aborted": cup_aborted,
                        "rejected_masks": rejected},
            "removal": {"observed_segmentation_dilation_px": dilation_px,
                        "temporal_bracket_frames": parent_mask.REMOVAL_BRACKET_FRAMES,
                        "bracket_added_pixels_per_frame": (bracket & ~dilated).sum((1, 2)).tolist()},
            "protected_source": "CURRENT_FRAME_SAM31_CUP_MINUS_OBSERVED_WEARER",
            "hidden_object_protection": False, "geometry_vetoes_cup_protection": False,
            "occlusion_order": "UNKNOWN_VISIBLE_CUP_OBJECT_FIRST", "quality_problems": problems,
            "human_review": "PENDING", "physical_contact_ground_truth": False,
            "training_eligible": False, "control_ground_truth": False,
        },
    }

def build_visual_masks(sam: Any, frames_rgb: np.ndarray, session: Any,
                       visible_seeds: Any = None) -> dict[str, Any]:
    """Run frozen SAM tracking with the fixed pick_place role recipe.

    Geometry can supplement deletion support, but cannot veto a cup pixel.
    Cup/human semantic conflict and unknown occlusion order are separate fields.
    Each SAM track invocation owns and clears its own video state.
    """
    rgb = _rgb_frames(frames_rgb)
    count, height, width = rgb.shape[:3]
    if count != session.frame_count:
        raise ValueError("MASK_SESSION_FRAME_COUNT")
    if not np.isclose(width / session.width, height / session.height):
        raise ValueError("MASK_SESSION_IMAGE_DOMAIN")
    task = getattr(session, "task", "pick_place")
    if task not in {"pick_place", "pick_place_1008"}:
        raise ValueError(f"UNKNOWN_TASK_CONFIGURATION:{task}")
    shape = (height, width)
    if visible_seeds is not None:
        return _build_visible_point_masks(sam, rgb, session, visible_seeds)
    scale = _contact_scale(shape)
    geometry = parent_mask.pico_geometry(session, width / session.width)
    geometry, diagnostic_sides = _gate_diagnostic_geometry(geometry, session)
    anchor = parent_mask.anchor_frame(geometry, shape)
    core, removal, sam_core, human_info = _track_wearer_with_semantics(sam, rgb, geometry, anchor)
    equipment = np.zeros_like(core)
    equipment_roles, equipment_info = {}, {}
    for role in ("glove", "controller", "attachment", "cable"):
        tracks = sam.track_text(list(rgb), PROMPTS[role], anchor)
        selected, track_rows = _owned_equipment(tracks, core | equipment, CONTACT_PX * scale)
        equipment_roles[role] = selected
        equipment |= selected
        equipment_info[role] = {"prompt": PROMPTS[role], "tracks": track_rows,
                                "propagation_aborted": list(sam.aborted)}
    object_anchor = anchor
    object_tracks = sam.track_text(list(rgb), PROMPTS["object"], object_anchor)
    object_aborted = list(sam.aborted)
    if not object_tracks and object_aborted and count // 2 != object_anchor:
        object_anchor = count // 2
        object_tracks = sam.track_text(list(rgb), PROMPTS["object"], object_anchor)
        object_aborted = list(sam.aborted)
    cup = np.zeros_like(core)
    rejected = []
    for obj_id, frames in object_tracks.items():
        for frame, packed in frames.items():
            mask = parent_mask.unpack(packed, shape)
            if parent_mask._object_like(mask):
                cup[frame] |= mask
            else:
                rejected.append({"obj_id": int(obj_id), "frame_index": int(frame),
                                 "code": "CUP_MASK_NOT_COMPACT_OR_AREA_LIMIT"})
    semantic_wearer = sam_core | equipment
    semantic_conflict = cup & semantic_wearer
    protected = cup & ~semantic_wearer
    # Never subtract the geometric hull from protection. No hidden-frame carry.
    core = core | equipment
    equipment_removal = np.stack([
        parent_mask.dilate(mask, max(1, int(round(parent_mask.HUMAN_DILATE_PX * scale))))
        for mask in equipment
    ])
    removal = (removal | parent_mask.bracket_fill(
        equipment_removal, parent_mask.REMOVAL_BRACKET_FRAMES)) & ~protected
    held = np.stack([
        protected[frame] & _near(semantic_wearer[frame], CONTACT_PX * scale)
        for frame in range(count)
    ])
    problems = []
    if not protected.any():
        problems.append({"code": "PAPER_CUP_VISIBLE_SUPPORT_MISSING", "severity": "QUALITY_FAILURE"})
    if semantic_conflict.any():
        problems.append({"code": "PAPER_CUP_WEARER_SEMANTIC_CONFLICT", "severity": "REVIEW_REQUIRED",
                         "frames": np.flatnonzero(semantic_conflict.any(axis=(1, 2))).tolist()})
    return {
        "core": core, "sam_core": sam_core, "equipment": equipment,
        "equipment_roles": equipment_roles, "removal": removal,
        "cup_observed": cup, "semantic_conflict": semantic_conflict,
        "protected_visible": protected, "held": held,
        # Ordering is UNKNOWN independently of the cup's semantic identity.
        "unknown": protected.copy(),
        "metadata": {
            "schema_version": "pick-place-visual-masks-v1", "anchor_frame": anchor,
            "prompts": dict(PROMPTS), "human": human_info, "equipment": equipment_info,
            "invalid_geometry_frame_sides": diagnostic_sides,
            "objects": {"prompt": PROMPTS["object"], "anchor_frame": object_anchor,
                        "tracks": len(object_tracks), "propagation_aborted": object_aborted,
                        "rejected_masks": rejected},
            "protected_source": "CURRENT_FRAME_SAM31_CUP_MINUS_OBSERVED_WEARER",
            "hidden_object_protection": False, "geometry_vetoes_cup_protection": False,
            "occlusion_order": "UNKNOWN_VISIBLE_CUP_OBJECT_FIRST",
            "quality_problems": problems, "human_review": "PENDING",
            "physical_contact_ground_truth": False, "training_eligible": False,
            "control_ground_truth": False,
        },
    }


def effective_clean_masks(masks: np.ndarray, protected_visible_masks: np.ndarray,
                          settings: Mapping[str, Any] = SETTINGS) -> tuple[np.ndarray, dict]:
    masks = np.asarray(masks)
    if masks.ndim != 3 or masks.dtype != np.bool_ or not len(masks):
        raise ValueError("REMOVAL_MASK_DOMAIN_OR_TYPE")
    protected = _bool_masks(protected_visible_masks, masks.shape, "protected_visible_masks")
    if dict(settings) != SETTINGS:
        raise ValueError("PROPAINTER_FROZEN_SETTINGS_DRIFT")
    dilated = np.stack([
        cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                   iterations=settings["mask_dilation"]) > 0 for mask in masks
    ])
    effective = dilated & ~protected
    overlap = np.count_nonzero(effective & protected, axis=(1, 2))
    if overlap.any():
        raise RuntimeError("PROTECTED_MASK_ENTERED_MODEL_HOLE")
    validation = {
        "protected_pixels_per_frame": np.count_nonzero(protected, axis=(1, 2)).tolist(),
        "dilated_protected_overlap_per_frame": np.count_nonzero(dilated & protected, axis=(1, 2)).tolist(),
        "effective_protected_overlap_per_frame": overlap.tolist(),
        "effective_delete_pixels_per_frame": np.count_nonzero(effective, axis=(1, 2)).tolist(),
        "model_mask_excludes_protection": True,
    }
    return effective, validation


class ProtectedProPainter:
    """Protect before every model path, using already-loaded frozen model objects."""

    def __init__(self, parent: Any):
        if dict(parent.settings) != SETTINGS:
            raise ValueError("PROPAINTER_FROZEN_SETTINGS_DRIFT")
        self.parent = parent
        self.settings = dict(parent.settings)
        self.last_validation: dict[str, Any] = {}
        self.effective_masks: np.ndarray | None = None

    def inpaint(self, frames_rgb: np.ndarray, masks: np.ndarray,
                protected_visible_masks: np.ndarray) -> np.ndarray:
        frames_rgb = _rgb_frames(frames_rgb)
        if masks.shape != frames_rgb.shape[:3]:
            raise ValueError("PROPAINTER_FRAME_MASK_DOMAIN")
        if dict(self.parent.settings) != SETTINGS or self.settings != SETTINGS:
            raise ValueError("PROPAINTER_FROZEN_SETTINGS_DRIFT")
        effective, validation = effective_clean_masks(masks, protected_visible_masks, self.settings)
        self.effective_masks = effective
        self.last_validation = validation
        torch, s = self.parent.torch, self.settings
        length, height, width = masks.shape
        if height % 8 or width % 8:
            raise ValueError("PROPAINTER_SIZE_NOT_MULTIPLE_OF_8")
        if length < 2:
            raise ValueError("PROPAINTER_SINGLE_FRAME_UNSUPPORTED")
        dilated = effective.astype(np.float32)
        # This is the frozen parent inference path; only mask gating and checks differ.
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            frames = torch.from_numpy(frames_rgb).to(self.parent.device).permute(0, 3, 1, 2).float()[None] / 127.5 - 1
            mask_t = torch.from_numpy(dilated).to(self.parent.device)[None, :, None]
            protected_t = torch.from_numpy(protected_visible_masks).to(self.parent.device)[None, :, None]
            if torch.any(mask_t[protected_t] != 0).item():
                raise RuntimeError("PROTECTED_MASK_TENSOR_NOT_ZERO")
            short = 12 if width <= 640 else (8 if width <= 720 else 4)
            forward, backward = [], []
            for f in range(0, length, short):
                end = min(length, f + short)
                chunk = frames[:, f:end] if f == 0 else frames[:, f - 1:end]
                if chunk.shape[1] < 2:
                    continue
                flow_f, flow_b = self.parent.raft(chunk, iters=s["raft_iter"])
                forward.append(flow_f)
                backward.append(flow_b)
            gt = (torch.cat(forward, 1), torch.cat(backward, 1))
            torch.cuda.empty_cache()
            if s["fp16"]:
                frames, mask_t = frames.half(), mask_t.half()
                gt = (gt[0].half(), gt[1].half())
            flow_length = gt[0].shape[1]
            sub = s["subvideo_length"]
            if flow_length > sub:
                pred_f, pred_b, pad = [], [], 5
                for f in range(0, flow_length, sub):
                    s_f, e_f = max(0, f - pad), min(flow_length, f + sub + pad)
                    pad_s, pad_e = f - s_f, e_f - min(flow_length, f + sub)
                    part = (gt[0][:, s_f:e_f], gt[1][:, s_f:e_f])
                    pred, _ = self.parent.flow.forward_bidirect_flow(part, mask_t[:, s_f:e_f + 1])
                    pred = self.parent.flow.combine_flow(part, pred, mask_t[:, s_f:e_f + 1])
                    pred_f.append(pred[0][:, pad_s:e_f - s_f - pad_e])
                    pred_b.append(pred[1][:, pad_s:e_f - s_f - pad_e])
                    torch.cuda.empty_cache()
                flows = (torch.cat(pred_f, 1), torch.cat(pred_b, 1))
            else:
                pred, _ = self.parent.flow.forward_bidirect_flow(gt, mask_t)
                flows = self.parent.flow.combine_flow(gt, pred, mask_t)
            masked = frames * (1 - mask_t)
            support = protected_t.expand(-1, -1, 3, -1, -1)
            if not torch.equal(masked[support], frames[support]):
                raise RuntimeError("PROTECTED_RGB_MASKED_BEFORE_MODEL")
            validation["model_protected_rgb_unchanged"] = True
            prop_len = min(100, sub)
            if length > prop_len:
                updated, updated_masks, pad = [], [], 10
                for f in range(0, length, prop_len):
                    s_f, e_f = max(0, f - pad), min(length, f + prop_len + pad)
                    pad_s, pad_e = f - s_f, e_f - min(length, f + prop_len)
                    b, t = mask_t[:, s_f:e_f].shape[:2]
                    imgs, local = self.parent.model.img_propagation(
                        masked[:, s_f:e_f], (flows[0][:, s_f:e_f - 1], flows[1][:, s_f:e_f - 1]),
                        mask_t[:, s_f:e_f], "nearest")
                    part = frames[:, s_f:e_f] * (1 - mask_t[:, s_f:e_f]) + imgs.view(b, t, 3, height, width) * mask_t[:, s_f:e_f]
                    updated.append(part[:, pad_s:e_f - s_f - pad_e])
                    updated_masks.append(local.view(b, t, 1, height, width)[:, pad_s:e_f - s_f - pad_e])
                    torch.cuda.empty_cache()
                updated_frames, updated_mask = torch.cat(updated, 1), torch.cat(updated_masks, 1)
            else:
                b, t = mask_t.shape[:2]
                imgs, local = self.parent.model.img_propagation(masked, flows, mask_t, "nearest")
                updated_frames = frames * (1 - mask_t) + imgs.view(b, t, 3, height, width) * mask_t
                updated_mask = local.view(b, t, 1, height, width)
            stride = s["neighbor_length"] // 2
            ref_num = sub // s["ref_stride"] if length > sub else -1
            result = np.zeros((length, height, width, 3), np.uint8)
            visits = np.zeros(length, np.int32)
            binary = dilated[..., None]
            for f in range(0, length, stride):
                neighbors = list(range(max(0, f - stride), min(length, f + stride + 1)))
                refs = _get_ref_index(f, neighbors, length, s["ref_stride"], ref_num)
                ids = neighbors + refs
                pred = self.parent.model(updated_frames[:, ids], (flows[0][:, neighbors[:-1]], flows[1][:, neighbors[:-1]]),
                                         mask_t[:, ids], updated_mask[:, ids], len(neighbors))
                pred = ((pred.view(-1, 3, height, width).float() + 1) / 2).cpu().permute(0, 2, 3, 1).numpy() * 255
                for i, index in enumerate(neighbors):
                    image = pred[i].astype(np.uint8) * binary[index] + frames_rgb[index] * (1 - binary[index])
                    if visits[index]:
                        image = result[index].astype(np.float32) * 0.5 + image * 0.5
                    result[index] = image.astype(np.uint8)
                    visits[index] += 1
            torch.cuda.empty_cache()
        if not (visits > 0).all():
            raise RuntimeError("PROPAINTER_OUTPUT_FRAME_NOT_VISITED")
        keep = ~effective
        result[keep] = frames_rgb[keep]
        result[protected_visible_masks] = frames_rgb[protected_visible_masks]
        changed = np.any(result != frames_rgb, axis=-1)
        outside = np.count_nonzero(changed & ~effective, axis=(1, 2))
        protected_changed = np.count_nonzero(changed & protected_visible_masks, axis=(1, 2))
        validation["outside_delete_changed_pixels_per_frame"] = outside.tolist()
        validation["protected_changed_pixels_per_frame"] = protected_changed.tolist()
        validation["output_frame_visits"] = visits.tolist()
        validation["status"] = "PASS" if not outside.any() and not protected_changed.any() else "FAIL"
        if validation["status"] != "PASS":
            raise RuntimeError("CLEAN_ORIGINAL_PIXEL_PROTECTION_FAILED")
        return result


def compose_frame(raw_bgr: np.ndarray, clean_bgr: np.ndarray, robot_bgr: np.ndarray,
                  robot_alpha: np.ndarray, masks: Mapping[str, np.ndarray], *,
                  robot_observed: bool, masks_available: bool = True) -> dict[str, Any]:
    """Keep parent behavior, then put trusted UNKNOWN cup pixels in front."""
    shape = np.asarray(clean_bgr).shape[:2]
    raw_bgr = np.asarray(raw_bgr)
    if raw_bgr.shape != np.asarray(clean_bgr).shape or raw_bgr.dtype != np.uint8:
        raise ValueError("RAW_COMPOSE_DOMAIN_OR_TYPE")
    protected = _bool_masks(masks["protected_visible"], shape, "protected_visible")
    unknown = _bool_masks(masks["unknown"], shape, "unknown")
    human = _bool_masks(masks["core"], shape, "core")
    if np.any(raw_bgr[protected] != clean_bgr[protected]):
        raise RuntimeError("CLEAN_PROTECTED_PIXELS_DRIFT")
    result = compose_visual_frame(clean_bgr, robot_bgr, robot_alpha, protected, human, unknown,
                                  robot_observed=robot_observed, masks_available=masks_available)
    override = result["draw"] & protected & unknown
    result["image"][override] = raw_bgr[override]
    result["draw"][override] = False
    result["object_front"][override] = True
    # Original unknown evidence remains untouched; display is not physical authority.
    result["unknown_visible_object_first"] = override
    result["unknown_visible_object_first_pixels"] = int(override.sum())
    result["physical_contact_ground_truth"] = False
    result["control_ground_truth"] = False
    return result


def review_windows_from_qpc(qpc: np.ndarray, qpc_frequency: float,
                            trigger_frames: list[int], *,
                            trigger_key: str = "candidate_frames") -> list[dict]:
    """The same declared-QPC padding/merge rule for contact and anomaly review."""
    qpc = np.asarray(qpc)
    if qpc.ndim != 1 or not len(qpc) or not np.isfinite(qpc).all() or np.any(qpc[1:] <= qpc[:-1]):
        raise ValueError("CONTACT_QPC_NOT_STRICTLY_INCREASING")
    if not np.isfinite(qpc_frequency) or qpc_frequency <= 0:
        raise ValueError("CONTACT_QPC_FREQUENCY")
    if trigger_key not in {"candidate_frames", "abnormal_frames"}:
        raise ValueError("REVIEW_WINDOW_TRIGGER_ROLE")
    if any(isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer))
           or not 0 <= frame < len(qpc) for frame in trigger_frames):
        raise ValueError("REVIEW_WINDOW_TRIGGER_FRAME_DOMAIN")
    times = (qpc - qpc[0]).astype(np.float64) / qpc_frequency
    spans = []
    for frame in sorted(set(int(value) for value in trigger_frames)):
        start = max(0.0, float(times[frame]) - WINDOW_PAD_S)
        end = min(float(times[-1]), float(times[frame]) + WINDOW_PAD_S)
        if spans and start <= spans[-1]["end_time_s"] + WINDOW_MERGE_GAP_S + 1e-12:
            spans[-1]["end_time_s"] = max(end, spans[-1]["end_time_s"])
            spans[-1][trigger_key].append(frame)
        else:
            spans.append({"start_time_s": start, "end_time_s": end, trigger_key: [frame]})
    for span in spans:
        span["start_frame"] = int(np.searchsorted(times, span["start_time_s"], side="left"))
        span["end_frame_inclusive"] = int(np.searchsorted(times, span["end_time_s"], side="right") - 1)
    return spans


def build_contact_candidates(qpc: np.ndarray, qpc_frequency: float,
                             human: np.ndarray, protected_visible: np.ndarray, *,
                             hand_points_px: np.ndarray | None = None,
                             robot_hand_masks: np.ndarray | None = None) -> dict[str, Any]:
    """QPC-based 2-D candidates; missing visible evidence stays UNKNOWN."""
    qpc = np.asarray(qpc)
    if qpc.ndim != 1 or not len(qpc) or not np.isfinite(qpc).all() or np.any(qpc[1:] <= qpc[:-1]):
        raise ValueError("CONTACT_QPC_NOT_STRICTLY_INCREASING")
    if not np.isfinite(qpc_frequency) or qpc_frequency <= 0:
        raise ValueError("CONTACT_QPC_FREQUENCY")
    protected = np.asarray(protected_visible)
    if protected.ndim != 3 or len(protected) != len(qpc):
        raise ValueError("CONTACT_FRAME_MASK_DOMAIN")
    protected = _bool_masks(protected, protected.shape, "protected_visible")
    human = _bool_masks(human, protected.shape, "human")
    robot = (np.zeros_like(human) if robot_hand_masks is None else
             _bool_masks(robot_hand_masks, human.shape, "robot_hand_masks"))
    points = None
    if hand_points_px is not None:
        points = np.asarray(hand_points_px, dtype=float)
        if points.ndim != 4 or points.shape[:2] != (2, len(qpc)) or points.shape[-1] != 2:
            raise ValueError("CONTACT_PROJECTED_HAND_DOMAIN")
        if points.shape[2] == 21:
            points = points[:, :, [4, 8, 12, 16, 20]]
    times = (qpc - qpc[0]).astype(np.float64) / qpc_frequency
    radius = CONTACT_PX * _contact_scale(protected.shape[1:])
    rows, triggers = [], []
    height, width = protected.shape[1:]
    for frame, cup in enumerate(protected):
        reasons, distance = [], None
        projected_valid = None if points is None else False
        if cup.any():
            near = _near(cup, radius)
            if (human[frame] & cup).any():
                reasons.append("HUMAN_CUP_IMAGE_OVERLAP")
            elif (human[frame] & near).any():
                reasons.append("HUMAN_CUP_IMAGE_PROXIMITY")
            if (robot[frame] & cup).any():
                reasons.append("ROBOT_HAND_CUP_IMAGE_OVERLAP")
            if points is not None:
                xy = points[:, frame].reshape(-1, 2)
                valid = np.isfinite(xy).all(-1)
                xy = xy[valid]
                ij = np.rint(xy).astype(int)
                inside = ((ij[:, 0] >= 0) & (ij[:, 0] < width) &
                          (ij[:, 1] >= 0) & (ij[:, 1] < height))
                ij = ij[inside]
                if len(ij):
                    projected_valid = True
                    distances = cv2.distanceTransform((~cup).astype(np.uint8), cv2.DIST_L2, 5)
                    distance = float(distances[ij[:, 1], ij[:, 0]].min())
                    if distance <= radius:
                        reasons.append("PROJECTED_FINGERTIP_CUP_IMAGE_PROXIMITY")
        row = {"frame_index": frame, "qpc": qpc[frame].item(), "time_s": float(times[frame]),
               "cup_visible": bool(cup.any()), "candidate": bool(reasons),
               "status": "CANDIDATE" if reasons else (
                   "NO_2D_CANDIDATE" if cup.any() and projected_valid is not False else "UNKNOWN"),
               "projected_hand_valid": projected_valid, "reasons": reasons, "fingertip_distance_px": distance,
               "physical_contact": "UNKNOWN"}
        rows.append(row)
        if reasons:
            triggers.append(frame)
    spans = review_windows_from_qpc(qpc, qpc_frequency, triggers)
    return {"schema_version": "pick-place-visual-contact-candidates-v1",
            "qpc_frequency": float(qpc_frequency), "near_px": radius, "frames": rows,
            "windows": spans, "window_pad_s": WINDOW_PAD_S, "window_merge_gap_s": WINDOW_MERGE_GAP_S,
            "physical_contact_ground_truth": False, "training_eligible": False,
            "control_ground_truth": False}
