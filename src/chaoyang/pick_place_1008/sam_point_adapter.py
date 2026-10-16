"""Visible-point instance tracking over the supplied frozen SAM3.1 model.

The caller verifies raw-image seed evidence and owns model loading. Point labels
are image annotations; returned masks are model proposals, never observations
of hidden pixels or physical segmentation/contact ground truth.
"""
from __future__ import annotations

from typing import Any

import numpy as np

ROLES = frozenset({"human", "glove", "controller", "mount", "forearm", "attachment", "cable"})
SIDES = frozenset({"left", "right"})
MAX_OBJECTS = 16
NO_POINTS_MESSAGE = "No points are provided"
CORRECTION_VALIDATION = "NATIVE_QPC_ORIGINAL_RGB_AND_SOURCE_IMAGE_ROW_EXACT_VERIFIED"
CORRECTION_SEMANTICS = "VISIBLE_FOREGROUND_PIXEL_NOT_JOINT_OR_CAD_POINT"
MAX_CORRECTIONS_PER_OBJECT = 16


class SamPointContractError(ValueError):
    pass


def _validated_inputs(rgb, seeds, anchor):
    frames = np.asarray(rgb)
    if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8
            or not len(frames) or min(frames.shape[1:3]) <= 0):
        raise SamPointContractError("RAW_RGB_UINT8_N_H_W_3_REQUIRED")
    count, height, width = frames.shape[:3]
    if type(anchor) is not int or not 0 <= anchor < count:
        raise SamPointContractError("POINT_ANCHOR_OUTSIDE_RAW_VIDEO")
    if not isinstance(seeds, (list, tuple)) or len(seeds) > MAX_OBJECTS:
        raise SamPointContractError("VISIBLE_SEEDS_LIST_WITH_AT_MOST_16_OBJECTS_REQUIRED")
    prepared, ids = [], set()
    for seed in seeds:
        if not isinstance(seed, dict):
            raise SamPointContractError("VISIBLE_SEED_OBJECT_REQUIRED")
        obj_id = seed.get("obj_id")
        if (type(obj_id) is not int or obj_id < 0 or obj_id in ids
                or seed.get("role") not in ROLES or seed.get("side") not in SIDES):
            raise SamPointContractError("UNIQUE_OBJECT_ID_AND_EXPLICIT_ROLE_SIDE_REQUIRED")
        ids.add(obj_id)
        seed_frames = [seed[key] for key in ("frame_idx", "frame_index", "raw_frame_index")
                       if key in seed]
        if (any(type(value) is not int or not 0 <= value < count for value in seed_frames)
                or len(set(seed_frames)) > 1):
            raise SamPointContractError("VISIBLE_SEED_NATIVE_ANCHOR_FIELDS_MUST_AGREE_AND_BE_BOUNDED")
        seed_frame = seed_frames[0] if seed_frames else anchor
        if ("visibility" in seed and seed["visibility"] != "VISIBLE") \
                or seed.get("raw_observed") is False:
            raise SamPointContractError("HIDDEN_OR_NONOBSERVED_POINT_SEEDS_REJECTED")
        try:
            points = np.asarray(seed.get("points_px"), dtype=np.float64)
            labels = np.asarray(seed.get("point_labels"))
        except (TypeError, ValueError) as error:
            raise SamPointContractError("VISIBLE_POINT_ARRAY_INVALID") from error
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                or not np.isfinite(points).all()
                or np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)
                or labels.shape != (len(points),) or labels.dtype.kind not in "iu"
                or not np.isin(labels, [0, 1]).all() or not np.any(labels == 1)):
            raise SamPointContractError("BOUNDED_VISIBLE_POINTS_WITH_FG_BG_LABELS_REQUIRED")
        correction_values = seed.get("correction_points", [])
        if not isinstance(correction_values, list) or len(correction_values) > MAX_CORRECTIONS_PER_OBJECT:
            raise SamPointContractError("VISIBLE_CORRECTION_LIST_WITH_AT_MOST_16_FRAMES_REQUIRED")
        corrections, used_frames = [], {seed_frame}
        for correction in correction_values:
            if (not isinstance(correction, dict)
                    or correction.get("visible_confirmed") is not True
                    or correction.get("native_evidence_validation") != CORRECTION_VALIDATION
                    or correction.get("seed_semantics") != CORRECTION_SEMANTICS
                    or type(correction.get("qpc_ns")) is not int
                    or not isinstance(correction.get("image"), dict)
                    or not isinstance(correction.get("session_id"), str)
                    or not correction["session_id"]
                    or not isinstance(correction.get("device_serial"), str)
                    or not correction["device_serial"]
                    or any(correction.get(key, seed[key]) != seed[key]
                           for key in ("obj_id", "side", "role"))):
                raise SamPointContractError("VISIBLE_CORRECTION_VERIFIED_NATIVE_EVIDENCE_REQUIRED")
            aliases = [correction[key] for key in ("frame_idx", "frame_index", "raw_frame_index")
                       if key in correction]
            if (not aliases or any(type(value) is not int or not 0 <= value < count for value in aliases)
                    or len(set(aliases)) != 1 or aliases[0] in used_frames):
                raise SamPointContractError("VISIBLE_CORRECTION_DISTINCT_NATIVE_FRAME_FIELDS_REQUIRED")
            correction_frame = aliases[0]
            try:
                correction_points = np.asarray(correction.get("points_px"), dtype=float)
                correction_labels = np.asarray(correction.get("point_labels"))
            except (TypeError, ValueError) as error:
                raise SamPointContractError("VISIBLE_CORRECTION_POINT_DOMAIN") from error
            if (correction_points.ndim != 2 or correction_points.shape[1] != 2
                    or not 1 <= len(correction_points) <= 32
                    or correction_labels.shape != (len(correction_points),)
                    or correction_labels.dtype.kind not in "iu"
                    or not np.isfinite(correction_points).all()
                    or not np.isin(correction_labels, [0, 1]).all()
                    or not np.any(correction_labels == 1)
                    or np.any(correction_points[:, 0] < 0) or np.any(correction_points[:, 0] >= width)
                    or np.any(correction_points[:, 1] < 0) or np.any(correction_points[:, 1] >= height)):
                raise SamPointContractError("VISIBLE_CORRECTION_POINT_DOMAIN")
            corrections.append({**correction, "frame_idx": correction_frame,
                "points_px": correction_points.tolist(),
                "point_labels": correction_labels.astype(int).tolist(),
                "points_relative": (correction_points / [width, height]).tolist()})
            used_frames.add(correction_frame)
        prepared.append({"obj_id": obj_id, "role": seed["role"], "side": seed["side"],
                         "frame_idx": seed_frame,
                         "points_px": points.tolist(), "point_labels": labels.astype(int).tolist(),
                         "points_relative": (points / [width, height]).tolist(),
                         "correction_points": sorted(corrections, key=lambda value: value["frame_idx"])})
    return frames, prepared


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.stack([_numpy(item) for item in value]) if value else np.asarray([])
    return np.asarray(value)


def _normalized_outputs(outputs, height, width):
    if outputs is None:
        return np.zeros((0, height, width), bool), np.zeros(0, np.int64)
    if not isinstance(outputs, dict):
        raise SamPointContractError("MODEL_OUTPUT_MAPPING_REQUIRED")
    raw_masks, raw_ids = outputs.get("out_binary_masks"), outputs.get("out_obj_ids")
    masks = np.zeros((0, height, width), bool) if raw_masks is None else _numpy(raw_masks)
    ids = np.zeros(0, np.int64) if raw_ids is None else _numpy(raw_ids).reshape(-1)
    if not masks.size:
        masks = np.zeros((0, height, width), bool)
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if (masks.ndim != 3 or masks.shape[1:] != (height, width) or masks.dtype != np.bool_
            or ids.dtype.kind not in "iu" or len(ids) != len(masks)
            or len(set(ids.tolist())) != len(ids)):
        raise SamPointContractError("MODEL_MASK_DOMAIN_OR_OBJECT_ID_BINDING_INVALID")
    return masks, ids.astype(np.int64, copy=False)


def track_visible_points(sam: Any, rgb, seeds, anchor: int):
    """Return (obj_id -> frame -> packed bool masks, explicit proposal metadata).

    Each object binds its own recorded native anchor; all are added before
    propagation in one state. Empty frame-cache entries satisfy the pinned
    vendor's point-only merge precondition without adding masks or observations.
    Coordinates use the RGB domain and the vendor's rel_coordinates=True.
    Invalid input raises before allocation. Model errors return explicit error
    metadata with any already-supported proposals, so callers can fail closed.
    """
    frames, prepared = _validated_inputs(rgb, seeds, anchor)
    count, height, width = frames.shape[:3]
    prompt_frames = [frame for seed in prepared
                     for frame in [seed["frame_idx"], *[value["frame_idx"] for value in seed["correction_points"]]]]
    propagation_anchor = min(prompt_frames, default=anchor)
    metadata = {
        "schema": "pick_place_sam31_visible_point_proposals_v1",
        "status": "NO_VISIBLE_SEEDS" if not prepared else "PENDING",
        "anchor_raw_frame_index": propagation_anchor, "caller_default_anchor_raw_frame_index": anchor,
        "raw_frame_count": count, "seed_anchors_may_differ": True,
        "same_object_native_corrections": True,
        "correction_count": sum(len(seed["correction_points"]) for seed in prepared),
        "cache_bootstrap_policy": "EMPTY_CACHE_BOOTSTRAP_NO_MASKS",
        "cache_bootstrap_contains_observations": False,
        "cache_bootstrap_native_frame_indices": [],
        "rgb_domain": [width, height], "seed_count": len(prepared),
        "max_objects": MAX_OBJECTS, "one_video_state": True,
        "coordinates": "RAW_RGB_NORMALIZED_X_OVER_WIDTH_Y_OVER_HEIGHT",
        "rel_coordinates": True, "point_branch": "SUPPLIED_SAM31_OFFICIAL_INSTANCE_INTERACTIVITY",
        "seed_authority": "CALLER_DECLARED_ACTUAL_VISIBLE_RAW_IMAGE_POINTS",
        "model_weight_loading_owned_by_caller": True, "sam_weights_modified": False,
        "physical_segmentation_ground_truth": False, "physical_contact_ground_truth": False,
        "hidden_pixel_observation_claim": False, "training_eligible": False,
        "control_ground_truth": False, "errors": [], "aborted": [],
        "unexpected_obj_ids": [], "propagation": [],
        "explicit_empty_frame_retractions": [], "lost_required_native_support": [],
        "objects": {str(seed["obj_id"]): {
            "obj_id": seed["obj_id"], "role": seed["role"], "side": seed["side"],
            "points_px": seed["points_px"], "point_labels": seed["point_labels"],
            "points_relative": seed["points_relative"], "anchor_raw_frame_index": seed["frame_idx"],
            "anchor_mask_present": False, "initial_anchor_mask_present": False,
            "track_frame_indices": [], "status": "PENDING",
            "correction_points": seed["correction_points"],
            "required_correction_raw_frame_indices": [value["frame_idx"] for value in seed["correction_points"]],
            "supported_correction_raw_frame_indices": [],
            "initial_supported_correction_raw_frame_indices": [],
        } for seed in prepared},
    }
    tracks = {}
    if not prepared:
        return tracks, metadata
    import torch
    from PIL import Image

    allowed_ids = {seed["obj_id"] for seed in prepared}
    supported_ids = set()
    state = None
    phase = "init_state"
    def store(frame, outputs):
        if not isinstance(frame, (int, np.integer)) or not 0 <= frame < count:
            raise SamPointContractError("MODEL_FRAME_OUTSIDE_NATIVE_RAW_VIDEO")
        masks, ids = _normalized_outputs(outputs, height, width)
        nonempty = set()
        for mask, obj in zip(masks, ids):
            obj = int(obj)
            if obj not in allowed_ids:
                metadata["unexpected_obj_ids"].append(obj)
                raise SamPointContractError("MODEL_RETURNED_UNSEEDED_OBJECT_ID")
            if mask.any():
                # np.packbits owns independent bytes; vendor CPU buffers are reused.
                tracks.setdefault(obj, {})[int(frame)] = np.packbits(mask.reshape(-1))
                nonempty.add(obj)
            elif int(frame) in tracks.get(obj, {}):
                # An explicit returned-empty mask retracts this ID/frame's older
                # proposal. An ID absent from a partial output has no such meaning.
                del tracks[obj][int(frame)]
                metadata["explicit_empty_frame_retractions"].append(
                    {"obj_id": obj, "raw_frame_index": int(frame), "phase": phase})
        return nonempty
    def refresh_required_support():
        lost = []
        for seed in prepared:
            obj = seed["obj_id"]
            record = metadata["objects"][str(obj)]
            frames_for_object = tracks.get(obj, {})
            record["anchor_mask_present"] = (record["initial_anchor_mask_present"]
                                             and seed["frame_idx"] in frames_for_object)
            if record["initial_anchor_mask_present"] and not record["anchor_mask_present"]:
                lost.append({"obj_id": obj, "raw_frame_index": seed["frame_idx"], "kind": "ANCHOR"})
            initial_corrections = record["initial_supported_correction_raw_frame_indices"]
            record["supported_correction_raw_frame_indices"] = [
                frame for frame in initial_corrections if frame in frames_for_object]
            lost.extend({"obj_id": obj, "raw_frame_index": frame, "kind": "CORRECTION"}
                        for frame in initial_corrections if frame not in frames_for_object)
        metadata["lost_required_native_support"] = lost
        return lost
    try:
        state = sam.model.init_state(resource_path=[Image.fromarray(frame) for frame in frames],
                                     offload_video_to_cpu=True, async_loading_frames=False)
        if not isinstance(state, dict):
            raise SamPointContractError("MODEL_INFERENCE_STATE_MAPPING_REQUIRED")
        phase = "empty_cache_bootstrap"
        cache_writer = getattr(sam.model, "_cache_frame_outputs", None)
        if not callable(cache_writer):
            raise SamPointContractError("OFFICIAL_EMPTY_FRAME_CACHE_WRITER_REQUIRED")
        if "cached_frame_outputs" in state and not isinstance(state["cached_frame_outputs"], dict):
            raise SamPointContractError("MODEL_FRAME_CACHE_MAPPING_REQUIRED")
        for native_frame in range(count):
            if native_frame not in state.get("cached_frame_outputs", {}):
                # Pinned vendor _build_sam2_output returns early for uncached
                # frames, before merging real point/tracker output. Empty shells
                # enable its official merge; existing predictions are untouched.
                cache_writer(state, native_frame, {})
                if state.get("cached_frame_outputs", {}).get(native_frame) != {}:
                    raise SamPointContractError("EMPTY_FRAME_CACHE_BOOTSTRAP_MUST_CONTAIN_NO_MASKS")
                metadata["cache_bootstrap_native_frame_indices"].append(native_frame)
        for seed in prepared:
            phase = f"add_prompt:{seed['obj_id']}"
            seed_anchor = seed["frame_idx"]
            frame, outputs = sam.model.add_prompt(
                inference_state=state, frame_idx=seed_anchor, text_str=None,
                points=torch.tensor(seed["points_relative"], dtype=torch.float32),
                point_labels=torch.tensor(seed["point_labels"], dtype=torch.int64),
                obj_id=seed["obj_id"], clear_old_points=True, rel_coordinates=True,
                output_prob_thresh=.5)
            if frame != seed_anchor:
                raise SamPointContractError("MODEL_SEED_ANCHOR_BINDING_MISMATCH")
            found = store(frame, outputs)
            if seed["obj_id"] in found:
                supported_ids.add(seed["obj_id"])
                metadata["objects"][str(seed["obj_id"])]["anchor_mask_present"] = True
                metadata["objects"][str(seed["obj_id"])]["initial_anchor_mask_present"] = True
        for seed in prepared:
            for correction in seed["correction_points"]:
                correction_frame = correction["frame_idx"]
                phase = f"add_correction:{seed['obj_id']}:{correction_frame}"
                frame, outputs = sam.model.add_prompt(
                    inference_state=state, frame_idx=correction_frame, text_str=None,
                    points=torch.tensor(correction["points_relative"], dtype=torch.float32),
                    point_labels=torch.tensor(correction["point_labels"], dtype=torch.int64),
                    obj_id=seed["obj_id"], clear_old_points=True, rel_coordinates=True,
                    output_prob_thresh=.5)
                if frame != correction_frame:
                    raise SamPointContractError("MODEL_CORRECTION_NATIVE_FRAME_BINDING_MISMATCH")
                found = store(frame, outputs)
                if seed["obj_id"] not in found:
                    raise SamPointContractError("MODEL_CORRECTION_TARGET_MASK_SUPPORT_REQUIRED")
                metadata["objects"][str(seed["obj_id"])]["supported_correction_raw_frame_indices"].append(correction_frame)
                metadata["objects"][str(seed["obj_id"])]["initial_supported_correction_raw_frame_indices"].append(correction_frame)
        phase = "required_support_after_prompts"
        if refresh_required_support():
            raise SamPointContractError("REQUIRED_NATIVE_PROMPT_SUPPORT_RETRACTED")
        if supported_ids:
            directions = ([(False, count - propagation_anchor)]
                          + ([(True, propagation_anchor)] if propagation_anchor else []))
            for reverse, maximum in directions:
                direction = "reverse" if reverse else "forward"
                phase = f"propagate:{direction}"
                yielded = set()
                expected = (set(range(propagation_anchor, count)) if not reverse
                            else set(range(0, propagation_anchor)))
                try:
                    for frame, outputs in sam.model.propagate_in_video(
                            inference_state=state, start_frame_idx=propagation_anchor,
                            max_frame_num_to_track=maximum, reverse=reverse,
                            output_prob_thresh=.5):
                        store(frame, outputs)
                        yielded.add(int(frame))
                except RuntimeError as error:
                    if NO_POINTS_MESSAGE not in str(error):
                        raise
                    metadata["aborted"].append({"direction": direction,
                                                "reason": NO_POINTS_MESSAGE})
                metadata["propagation"].append({
                    "direction": direction, "start_raw_frame_index": propagation_anchor,
                    "max_frame_num_to_track": maximum,
                    "expected_raw_frame_indices": sorted(expected),
                    "yielded_raw_frame_indices": sorted(yielded),
                    "missing_raw_frame_indices": sorted(expected - yielded),
                })
                if not metadata["aborted"] and yielded != expected:
                    raise SamPointContractError("INCOMPLETE_NATIVE_PROPAGATION_FRAME_COVERAGE")
        phase = "required_support_after_propagation"
        if refresh_required_support():
            raise SamPointContractError("REQUIRED_NATIVE_PROMPT_SUPPORT_RETRACTED")
    except Exception as error:
        metadata["errors"].append({"phase": phase, "exception_type": type(error).__name__,
                                   "message": str(error)[:2048]})
    finally:
        if state is not None:
            try:
                state.clear()
            except Exception as error:
                metadata["errors"].append({"phase": "state_clear",
                                          "exception_type": type(error).__name__,
                                          "message": str(error)[:2048]})
    refresh_required_support()
    supported_ids = {obj for obj in supported_ids
                     if metadata["objects"][str(obj)]["anchor_mask_present"]}
    tracks = {obj: values for obj, values in tracks.items() if obj in supported_ids}
    for obj, record in metadata["objects"].items():
        values = tracks.get(int(obj), {})
        record["track_frame_indices"] = sorted(values)
        record["tracked_frame_count"] = len(values)
        record["raw_frame_coverage"] = len(values) / count
        record["status"] = ("TRACKED_MODEL_PROPOSAL" if record["anchor_mask_present"] and values
                            else "NO_ANCHOR_MASK_SUPPORT")
    if metadata["errors"]:
        metadata["status"] = "TRACKING_ERROR"
    elif metadata["aborted"]:
        metadata["status"] = "PROPAGATION_ABORTED"
    elif not supported_ids:
        metadata["status"] = "NO_SUPPORTED_TRACKS"
    elif supported_ids != allowed_ids:
        metadata["status"] = "PARTIAL_SUPPORT"
    else:
        metadata["status"] = "COMPLETE"
    return tracks, metadata
