"""Candidate-only RGB-supported repairs, retaining the original V1 masks.

No model, file, session-ID or manual-point interface is exposed. Same-capture
projection hints may veto contradictory identity; they never supply pixels.
The caller verifies sealed input files and supplies accepted original identities.
Automatic points are model proposals, never manually confirmed observations.
"""
from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np


REPAIR_AUTHORITY = "MODEL_RGB_PROPOSAL_POINT"
IDENTITY_QUALIFICATION_RULE = "EXCLUDE_COMPONENT_OPPOSITE_ONLY_CURRENT_SAME_CAPTURE_SEARCH_HIT_V1"
SINGLE_MAX_GAP = "SINGLE_MAX_GAP"
SPREAD_MAX_GAP_THREE = "SPREAD_MAX_GAP_THREE"
SPREAD_POINT_RULE = "L2_BOUNDARY_DISTANCE_TIMES_NEAREST_SELECTED_EUCLIDEAN_DISTANCE_SQUARED_ROW_MAJOR_Y_X"
WEARER_ROLES = frozenset({"human", "glove"})
SIDES = ("left", "right")
BINDING_KEYS = ("frames_sha256", "raw_source_sha256", "frame_map_sha256", "model_sha256")
REQUIRED_ARRAYS = ("core", "sam_core", "equipment", "removal", "protected_visible")


class V1MaskRepairContractError(ValueError):
    """Repair cannot prove its domain, identity or source support."""


def _error(message: str) -> None:
    raise V1MaskRepairContractError(message)


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _bool(value: Any, shape: tuple[int, ...], code: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_ or array.shape != shape:
        _error(code)
    return array


def _bindings(original: Mapping[str, Any], supplied: Mapping[str, str] | None) -> dict:
    value = supplied if supplied is not None else original.get("input_bindings")
    if not isinstance(value, Mapping):
        _error("REPAIR_CURRENT_SOURCE_BINDINGS_REQUIRED")
    result = {}
    for key in BINDING_KEYS:
        sha = value.get(key)
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            _error("REPAIR_FOUR_CURRENT_SOURCE_SHA256_REQUIRED")
        result[key] = sha
    return result


def _prepare(original: Mapping[str, Any], proposals: Sequence[Any], assignments: Sequence[Mapping],
             bindings: Mapping[str, str] | None) -> tuple:
    if not isinstance(original, Mapping):
        _error("REPAIR_ORIGINAL_MASK_MAPPING_REQUIRED")
    core = np.asarray(original.get("core"))
    if core.dtype != np.bool_ or core.ndim != 3 or not len(core) or min(core.shape) < 1:
        _error("REPAIR_ORIGINAL_NATIVE_BOOL_N_H_W_REQUIRED")
    shape = core.shape
    for key in REQUIRED_ARRAYS:
        _bool(original.get(key), shape, "REPAIR_ORIGINAL_MASK_DOMAIN:" + key)
    if np.any(original["removal"] & original["protected_visible"]):
        _error("REPAIR_ORIGINAL_REMOVAL_PROTECTION_CONFLICT")
    source = _bindings(original, bindings)
    if not isinstance(proposals, (list, tuple)) or not isinstance(assignments, (list, tuple)):
        _error("REPAIR_TRACKS_AND_ACCEPTED_ASSIGNMENTS_SEQUENCE_REQUIRED")
    by_key = {}
    for proposal in proposals:
        role, track = _get(proposal, "role"), _get(proposal, "track_id")
        if not isinstance(role, str) or isinstance(track, bool) or not isinstance(track, (int, np.integer)) or track < 0:
            _error("REPAIR_RGB_PROPOSAL_IDENTITY_REQUIRED")
        key = (role, int(track))
        if key in by_key:
            _error("REPAIR_DUPLICATE_ROLE_TRACK")
        # Non-wearer tracks may be supplied for audit, never become repair input.
        if role not in WEARER_ROLES:
            by_key[key] = None
            continue
        if _get(proposal, "source") != "SAM31_RGB_TEXT":
            _error("REPAIR_ONLY_SAM31_RGB_TEXT_WEARER_SUPPORT")
        if any(_get(proposal, key) != sha for key, sha in source.items()):
            _error("REPAIR_CURRENT_SOURCE_SIGNATURE_MISMATCH")
        masks = _bool(_get(proposal, "masks"), shape, "REPAIR_RGB_PROPOSAL_NATIVE_DOMAIN")
        unsupported = _get(proposal, "unsupported_frames")
        if unsupported is not None:
            unsupported = _bool(unsupported, (shape[0],), "REPAIR_UNSUPPORTED_FRAME_DOMAIN")
            masks = masks.copy()
            masks[unsupported] = False
        by_key[key] = masks
    prepared, seen = [], set()
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            _error("REPAIR_ACCEPTED_IDENTITY_MAPPING_REQUIRED")
        role, track = assignment.get("role"), assignment.get("track_id")
        if role not in WEARER_ROLES:
            continue
        if isinstance(track, bool) or not isinstance(track, (int, np.integer)) or track < 0:
            _error("REPAIR_ACCEPTED_RGB_TRACK_ID_REQUIRED")
        key = (role, int(track))
        side, obj_id = assignment.get("side"), assignment.get("obj_id")
        original_role = assignment.get("original_role", role)
        reason = assignment.get("association")
        if (side not in SIDES or isinstance(obj_id, bool) or not isinstance(obj_id, (int, np.integer))
                or not 1 <= obj_id <= 16 or not isinstance(reason, str) or not reason
                or "UNKNOWN" in reason.upper() or assignment.get("accepted") is False
                or original_role not in WEARER_ROLES):
            _error("REPAIR_LEGAL_ORIGINAL_OBJECT_AND_ACCEPTED_SIDE_REQUIRED")
        if key in seen or key not in by_key or by_key[key] is None:
            _error("REPAIR_ACCEPTED_TRACK_MISSING_OR_DUPLICATE")
        seen.add(key)
        prepared.append({"role": role, "original_role": original_role, "track_id": int(track), "side": side,
                         "obj_id": int(obj_id), "association": reason,
                         "identity_unique": assignment.get("identity_unique") is True,
                         "masks": by_key[key]})
    prepared.sort(key=lambda row: (row["obj_id"], row["role"], row["track_id"]))
    objects = {}
    for row in prepared:
        previous = objects.setdefault(row["obj_id"], [])
        if previous and (any((other["original_role"], other["side"]) != (row["original_role"], row["side"])
                             for other in previous)
                         or not row["identity_unique"] or not all(other["identity_unique"] for other in previous)):
            _error("REPAIR_AMBIGUOUS_MULTIPLE_TEXT_TRACKS_FOR_ORIGINAL_OBJECT")
        previous.append(row)
    side_support = np.zeros((2, *shape), bool)
    if "side_support" in original:
        side_support |= _bool(original["side_support"], side_support.shape,
                              "REPAIR_ORIGINAL_SIDE_SUPPORT_DOMAIN")
    for row in prepared:
        side_support[SIDES.index(row["side"])] |= row["masks"]
    covered = np.zeros(shape, bool)
    for key in ("core", "sam_core", "equipment", "removal", "effective_removal"):
        if key in original:
            covered |= _bool(original[key], shape, "REPAIR_ORIGINAL_MASK_DOMAIN:" + key)
    return shape, source, prepared, side_support, covered


def _gap(original: Mapping, row: Mapping, side_support: np.ndarray, covered: np.ndarray) -> np.ndarray:
    return (row["masks"] & ~covered & ~original["protected_visible"]
            & ~side_support[1 - SIDES.index(row["side"])])


def _point(region: np.ndarray) -> list[float]:
    distance = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    # np.argmax is row-major: equal maxima select smallest y, then x.
    y, x = np.unravel_index(int(np.argmax(distance)), region.shape)
    if not region[y, x] or distance[y, x] <= 0:
        _error("REPAIR_POSITIVE_POINT_WITHOUT_RGB_INTERIOR")
    return [float(x), float(y)]


def _prompt_policy(value: str) -> str:
    if not isinstance(value, str) or value not in (SINGLE_MAX_GAP, SPREAD_MAX_GAP_THREE):
        _error("REPAIR_UNKNOWN_PROMPT_POLICY")
    return value


def _spread_points(region: np.ndarray) -> list[list[float]]:
    """Keep the legacy first point; spread up to two more inside the same gap.

    Scores are padded L2 boundary distance (OpenCV mask size 5) multiplied
    by squared Euclidean distance to the nearest selected point. Equal
    scores retain row-major y/x ordering. No area or separation threshold.
    """
    points = [_point(region)]
    maximum = min(3, int(np.count_nonzero(region)))
    if maximum == 1:
        return points
    boundary = cv2.distanceTransform(np.pad(region.astype(np.uint8), 1),
                                    cv2.DIST_L2, 5)[1:-1, 1:-1].astype(np.float64)
    yy, xx = np.indices(region.shape, dtype=np.float64)
    nearest_squared = np.full(region.shape, np.inf, np.float64)
    while len(points) < maximum:
        x, y = points[-1]
        nearest_squared = np.minimum(nearest_squared, (xx - x) ** 2 + (yy - y) ** 2)
        score = np.where(region, boundary * nearest_squared, 0.0)
        next_y, next_x = np.unravel_index(int(np.argmax(score)), region.shape)
        if not region[next_y, next_x] or score[next_y, next_x] <= 0:
            _error("REPAIR_SPREAD_POINT_WITHOUT_DISTINCT_RGB_INTERIOR")
        points.append([float(next_x), float(next_y)])
    return points


def qualify_identity_support(proposals: Sequence[Any], assignments: Sequence[Mapping],
                             identity_hints: np.ndarray,
                             input_bindings: Mapping[str, str]) -> tuple[list[Any], dict]:
    """Veto explicitly contradictory RGB components; never project pixels.

    The existing eight-pixel identity search is a *negative* authority only.
    Global identity remains unchanged for both-side/no-side hits, with explicit
    uncertainty records. The same returned subset must feed prompts and merge.
    """
    from chaoyang.inpainting_candidate.automatic_v1_seeds import _hits
    from chaoyang.exact78_baseline.mask import HIT_RADIUS_PX

    bindings = _bindings({}, input_bindings)
    if not isinstance(proposals, (list, tuple)) or not isinstance(assignments, (list, tuple)):
        _error("REPAIR_IDENTITY_TRACKS_AND_ASSIGNMENTS_SEQUENCE_REQUIRED")
    by_key = {}
    for value in assignments:
        if not isinstance(value, Mapping):
            _error("REPAIR_ACCEPTED_IDENTITY_MAPPING_REQUIRED")
        if value.get("role") not in WEARER_ROLES:
            continue
        track = value.get("track_id")
        if isinstance(track, bool) or not isinstance(track, (int, np.integer)) or track < 0:
            _error("REPAIR_ACCEPTED_RGB_TRACK_ID_REQUIRED")
        key = (value["role"], int(track))
        if key in by_key or value.get("side") not in SIDES or value.get("accepted") is False:
            _error("REPAIR_IDENTITY_ASSIGNMENT_DUPLICATE_OR_UNKNOWN")
        by_key[key] = value
    hints = np.asarray(identity_hints)
    if (hints.ndim != 4 or hints.shape[0] != 2 or hints.shape[-1] != 2
            or hints.dtype.kind != 'f' or np.isinf(hints).any()):
        _error("REPAIR_SAME_CAPTURE_FLOAT_SIDE_N_K_XY_HINTS_REQUIRED")
    result, objects, seen = [], [], set()
    for proposal in proposals:
        role, track = _get(proposal, "role"), _get(proposal, "track_id")
        if role not in WEARER_ROLES:
            result.append(proposal)
            continue
        if isinstance(track, bool) or not isinstance(track, (int, np.integer)) or track < 0:
            _error("REPAIR_RGB_PROPOSAL_IDENTITY_REQUIRED")
        key = (role, int(track))
        if key in seen:
            _error("REPAIR_DUPLICATE_ROLE_TRACK")
        seen.add(key)
        if _get(proposal, "source") != 'SAM31_RGB_TEXT' or any(_get(proposal, key) != sha for key, sha in bindings.items()):
            _error("REPAIR_CURRENT_SOURCE_SIGNATURE_MISMATCH")
        masks = np.asarray(_get(proposal, "masks"))
        if masks.dtype != np.bool_ or masks.ndim != 3 or not len(masks) or min(masks.shape) < 1 or masks.shape[0] != hints.shape[1]:
            _error("REPAIR_IDENTITY_NATIVE_MASK_AND_HINT_FRAME_DOMAIN")
        unsupported = _get(proposal, "unsupported_frames")
        if unsupported is not None:
            unsupported = _bool(unsupported, (len(masks),), "REPAIR_UNSUPPORTED_FRAME_DOMAIN")
        if key not in by_key:
            # Unknown tracks remain unknown; they cannot acquire assignments.
            result.append(proposal)
            continue
        accepted = by_key[key]
        assigned_side = SIDES.index(accepted['side'])
        qualified = masks.copy()
        removed_pixels = np.zeros(len(masks), np.int64)
        conflicts, uncertainty = [], []
        total_components = 0
        for frame in range(len(masks)):
            if unsupported is not None and unsupported[frame]:
                continue
            count, labels, stats, _ = cv2.connectedComponentsWithStats(masks[frame].astype(np.uint8), connectivity=8)
            total_components += count - 1
            for component_id in range(1, count):
                component = labels == component_id
                hits = _hits(component[None], hints[:, frame:frame+1], HIT_RADIUS_PX)[:, 0]
                contradiction = bool(hits[1-assigned_side] and not hits[assigned_side])
                detail = {'native_frame': frame, 'component_id': component_id,
                          'area': int(stats[component_id, cv2.CC_STAT_AREA]),
                          'bbox_xywh': stats[component_id, :4].astype(int).tolist(),
                          'search_hits_left_right': hits.tolist()}
                if contradiction:
                    qualified[frame, component] = False
                    removed_pixels[frame] += detail['area']
                    conflicts.append({**detail, 'decision': 'UNKNOWN_IDENTITY_OPPOSITE_ONLY_COMPONENT_EXCLUDED'})
                elif bool(hits.all()) or not bool(hits.any()):
                    uncertainty.append({**detail, 'decision': 'BOTH_SIDES_HIT_EXISTING_GLOBAL_IDENTITY_ONLY'
                                         if hits.all() else 'NO_LOCAL_HINT_HIT_EXISTING_GLOBAL_IDENTITY_ONLY',
                                         'new_identity_promotion': False})
        if isinstance(proposal, Mapping):
            replacement = {**proposal, 'masks': qualified}
        elif is_dataclass(proposal):
            replacement = replace(proposal, masks=qualified)
        else:
            replacement = copy.copy(proposal)
            setattr(replacement, 'masks', qualified)
        result.append(replacement)
        objects.append({'role': role, 'track_id': int(track), 'assigned_side': accepted['side'],
                        'original_obj_id': accepted.get('obj_id'), 'components_examined': total_components,
                        'pixels_excluded_per_native_frame': removed_pixels.tolist(),
                        'pixels_excluded_total': int(removed_pixels.sum()),
                        'explicit_opposite_identity_conflicts': conflicts,
                        'retained_identity_uncertainty': uncertainty})
        if np.any(qualified & ~masks):
            _error("REPAIR_IDENTITY_QUALIFICATION_ADDED_PIXELS")
    if any(key not in seen for key in by_key):
        _error("REPAIR_ACCEPTED_TRACK_MISSING_FOR_IDENTITY_QUALIFICATION")
    return result, {'schema': 'inpainting-v1-rgb-frame-component-identity-qualification-v1',
                    'rule': IDENTITY_QUALIFICATION_RULE, 'input_bindings': bindings,
                    'existing_search_radius_px': HIT_RADIUS_PX, 'new_numeric_thresholds': 0,
                    'objects': objects, 'pixels_excluded_total': sum(obj['pixels_excluded_total'] for obj in objects),
                    'output_is_original_rgb_mask_subset': True, 'foreground_pixels_added': 0,
                    'projection_is_pixel_evidence': False, 'projection_role': 'IDENTITY_CONTRADICTION_NEGATIVE_ONLY',
                    'manual_prompt_removal': False, 'session_id_branches': 0,
                    'visual_quality': 'NOT_EVALUATED', 'human_review': 'PENDING',
                    'training_eligible': False, 'control_ground_truth': False}


def make_repair_prompts(original_masks: Mapping[str, Any], proposal_tracks: Sequence[Any],
                        accepted_assignments: Sequence[Mapping], *,
                        input_bindings: Mapping[str, str] | None = None,
                        prompt_policy: str = SINGLE_MAX_GAP) -> tuple[list[dict], dict]:
    """One maximum-gap automatic anchor per unambiguous original object.

    Returns original ``obj_id``/role/side seeds accepted as *anchors* by the
    unchanged ``track_visible_points`` adapter. Do not put these automatic
    points in its manual ``correction_points`` list. Caller overwrites the
    adapter authority metadata with REPAIR_AUTHORITY before publishing it.
    """
    policy = _prompt_policy(prompt_policy)
    shape, source, rows, sides, covered = _prepare(original_masks, proposal_tracks, accepted_assignments, input_bindings)
    candidates, diagnostics = {}, []
    for row in rows:
        missing = _gap(original_masks, row, sides, covered)
        areas = missing.sum(axis=(1, 2))
        frame = int(np.argmax(areas))
        diagnostic = {key: row[key] for key in ("role", "original_role", "track_id", "side", "obj_id", "association")}
        diagnostic["gap_pixels_per_native_frame"] = areas.astype(int).tolist()
        diagnostic["status"] = "NO_UNCOVERED_RGB_SUPPORT" if not areas[frame] else "AUTOMATIC_POINT_PROPOSED"
        diagnostics.append(diagnostic)
        if not areas[frame]:
            continue
        points = [_point(missing[frame])] if policy == SINGLE_MAX_GAP else _spread_points(missing[frame])
        prompt = {**{key: row[key] for key in ("track_id", "side", "obj_id", "association")},
                  "role": row["original_role"], "proposal_role": row["role"],
                  "frame_idx": frame, "points_px": points, "point_labels": [1] * len(points),
                  "gap_pixels_at_anchor": int(areas[frame]), "seed_authority": REPAIR_AUTHORITY,
                  "human_visible_confirmed": False, "physical_segmentation_ground_truth": False,
                  "input_bindings": dict(source)}
        incumbent = candidates.get(row["obj_id"])
        # Rows are already identity-sorted; equal area keeps earliest frame,
        # then lexical role/track as deterministic tie breakers.
        rank = (-int(areas[frame]), frame, row["role"], row["track_id"])
        if incumbent is None or rank < incumbent[0]:
            candidates[row["obj_id"]] = (rank, prompt)
    prompts = [candidates[obj][1] for obj in sorted(candidates)]
    metadata = {
        "schema": "inpainting-v1-rgb-mask-repair-prompts-v1", "status": "READY_MODEL_CANDIDATE" if prompts else "NO_RGB_GAP",
        "input_bindings": source, "native_shape": list(shape), "point_authority": REPAIR_AUTHORITY,
        "automatic_points": len(prompts), "manual_points_added": 0, "maximum_points_per_original_object": 1,
        "projection_is_pixel_evidence": False, "generic_object_promoted": False,
        "diagnostics": diagnostics, "visual_quality": "NOT_EVALUATED", "human_review": "PENDING",
        "training_eligible": False, "control_ground_truth": False,
    }
    if policy == SPREAD_MAX_GAP_THREE:
        metadata.update(prompt_policy=policy, spread_point_rule=SPREAD_POINT_RULE,
                        automatic_points=sum(len(prompt["points_px"]) for prompt in prompts),
                        prompted_objects=len(prompts), maximum_points_per_original_object=3)
    return prompts, metadata


def _point_array(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    if isinstance(value, Mapping):
        result = np.zeros(shape, bool)
        for frame, mask in value.items():
            if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < shape[0]:
                _error("REPAIR_POINT_NATIVE_FRAME_DOMAIN")
            array = np.asarray(mask)
            if array.dtype == np.bool_ and array.shape == shape[1:]:
                result[int(frame)] = array
            elif array.dtype == np.uint8 and array.shape == ((shape[1] * shape[2] + 7) // 8,):
                result[int(frame)] = np.unpackbits(array, count=shape[1] * shape[2]).reshape(shape[1:]).astype(bool)
            else:
                _error("REPAIR_POINT_PACKED_OR_BOOL_NATIVE_DOMAIN")
        return result
    return _bool(value, shape, "REPAIR_POINT_NATIVE_BOOL_N_H_W_REQUIRED")


def merge_repair_support(original_masks: Mapping[str, Any], point_tracks: Mapping[int, Any],
                         proposal_tracks: Sequence[Any], prompts: Sequence[Mapping],
                         accepted_assignments: Sequence[Mapping], *,
                         input_bindings: Mapping[str, str] | None = None,
                         prompt_policy: str = SINGLE_MAX_GAP) -> tuple[dict, dict]:
    """Union POINT∩same-track TEXT only, preserving every original support.

    No anchor output means that object's additions are rejected. Frozen V1
    dilation/bracket/effective-mask helpers are reused without new thresholds.
    """
    policy = _prompt_policy(prompt_policy)
    shape, source, rows, sides, covered = _prepare(original_masks, proposal_tracks, accepted_assignments, input_bindings)
    expected, _ = make_repair_prompts(original_masks, proposal_tracks, accepted_assignments,
                                     input_bindings=source, prompt_policy=policy)
    if list(prompts) != expected:
        _error("REPAIR_PROMPTS_NOT_DETERMINISTIC_CURRENT_RGB_PROPOSALS")
    if not isinstance(point_tracks, Mapping) or any(type(key) is not int or key not in {p["obj_id"] for p in prompts} for key in point_tracks):
        _error("REPAIR_POINT_UNPROMPTED_OR_INVALID_OBJECT_ID")
    by_key = {(row["role"], row["track_id"]): row for row in rows}
    addition, diagnostics = np.zeros(shape, bool), []
    for prompt in prompts:
        row = by_key[(prompt["proposal_role"], prompt["track_id"])]
        points = _point_array(point_tracks.get(prompt["obj_id"], {}), shape)
        x, y = map(int, prompt["points_px"][0])
        anchor_supported = bool(points[prompt["frame_idx"], y, x])
        supported = (points & row["masks"] & ~original_masks["protected_visible"]
                     & ~sides[1 - SIDES.index(row["side"])] & ~covered)
        if not anchor_supported:
            supported[:] = False
        addition |= supported
        diagnostics.append({"obj_id": prompt["obj_id"], "role": row["original_role"], "proposal_role": row["role"], "track_id": row["track_id"],
                            "side": row["side"], "anchor_point_supported": anchor_supported,
                            "added_pixels_per_native_frame": supported.sum(axis=(1, 2)).astype(int).tolist()})
        if policy == SPREAD_MAX_GAP_THREE:
            diagnostics[-1]["automatic_point_support_at_anchor"] = [
                {"points_px": [float(px), float(py)],
                 "point_supported": bool(points[prompt["frame_idx"], int(py), int(px)])}
                for px, py in prompt["points_px"]]
    arrays = {key: (value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value))
              for key, value in original_masks.items()}
    arrays["core"] |= addition
    arrays["sam_core"] |= addition
    metadata = original_masks.get("metadata", {})
    if "metadata" in metadata:
        metadata = metadata["metadata"]
    rule = metadata.get("removal", {})
    radius, temporal = rule.get("observed_segmentation_dilation_px"), rule.get("temporal_bracket_frames")
    if type(radius) is not int or radius < 0 or type(temporal) is not int or temporal < 0:
        _error("REPAIR_FROZEN_ORIGINAL_DILATION_AND_BRACKET_REQUIRED")
    from chaoyang.exact78_baseline.mask import bracket_fill, dilate
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    dilated = np.stack([dilate(frame, radius) for frame in arrays["core"]])
    candidate_removal = bracket_fill(dilated, temporal) & ~arrays["protected_visible"]
    arrays["removal"] |= candidate_removal
    effective, clean_gate = effective_clean_masks(arrays["removal"], arrays["protected_visible"])
    arrays["effective_removal"] = effective
    if np.any(original_masks["core"] & ~arrays["core"]) or not np.array_equal(arrays["equipment"], original_masks["equipment"]):
        _error("REPAIR_ORIGINAL_SUPPORT_NOT_PRESERVED")
    if not np.array_equal(arrays["protected_visible"], original_masks["protected_visible"]):
        _error("REPAIR_ORIGINAL_PROTECTION_CHANGED")
    report = {"schema": "inpainting-v1-rgb-mask-repair-merge-v1", "status": "MASK_CANDIDATE_NOT_VISUALLY_ADMITTED",
              "input_bindings": source, "native_shape": list(shape), "point_authority": REPAIR_AUTHORITY,
              "objects": diagnostics, "new_direct_support_pixels_per_native_frame": addition.sum(axis=(1, 2)).astype(int).tolist(),
              "direct_support_rule": "CORRECTION_POINT_INTERSECT_SAME_TRACK_TEXT_EXCLUDE_PROTECTION_AND_OPPOSITE_SIDE",
              "removal_expansion_rule": copy.deepcopy(rule), "effective_clean_gate": clean_gate,
              "original_core_pixels_removed": 0, "original_equipment_changed_pixels": 0,
              "original_protection_changed_pixels": 0, "manual_points_added": 0,
              "projection_is_pixel_evidence": False, "generic_object_promoted": False,
              "visual_quality": "NOT_EVALUATED", "human_review": "PENDING",
              "training_eligible": False, "control_ground_truth": False}
    if policy == SPREAD_MAX_GAP_THREE:
        report.update(prompt_policy=policy, spread_point_rule=SPREAD_POINT_RULE,
                      anchor_support_rule="LEGACY_FIRST_POINT_ONLY_UNCHANGED")
    arrays["metadata"]["candidate_v1_rgb_mask_repair"] = copy.deepcopy(report)
    return arrays, report
