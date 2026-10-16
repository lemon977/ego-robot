"""Candidate-only, RGB-supported automatic masks; no model is loaded here.

``collect_sam_rgb_proposals`` bridges the installed ``Sam3Video.track_text`` API.
The generic ``object`` concept is a first engineering proposal, not an exhaustive
instance segmenter. A registered class-agnostic provider may supply the same
``RGBProposal`` contract. Every proposal is bound to the exact decoded RGB bytes,
native frame mapping, raw source and model signatures supplied by the caller.

Model masks are *visible-support proposals*, not segmentation truth. Hardware
projection can select an already proposed human instance or assign its side; it
never creates foreground pixels. Automatic positive points come from current RGB
proposal interiors, never an old/manual seed file. No historical visual module,
task configuration, session ID or external point list is accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from chaoyang.exact78_baseline import mask as parent_mask
from chaoyang.exact78_baseline.clean import SETTINGS
from chaoyang.pick_place_1008.sam_point_adapter import MAX_OBJECTS, _normalized_outputs


PROMPTS = {
    "human": parent_mask.HUMAN_PROMPT,
    "glove": "a glove worn on a person's hand",
    "controller": "a handheld or hand-mounted VR controller",
    "attachment": "a device or connector attached to a person's hand or wrist",
    "cable": "a cable connected to a person's glove or wearable wrist device",
    "object": "object",
}
EQUIPMENT_ROLES = ("glove", "controller", "attachment", "cable")
PROPOSAL_SOURCES = frozenset({
    "SAM31_RGB_TEXT", "RGB_CLASS_AGNOSTIC_INSTANCE", "SAM31_RGB_POINT_AUTO",
    "SAM31_RGB_TEXT_POINT_WEARER_UNION",
})
TEXT_WEARER_RETENTION = "PRESERVE_TEXT_WEARER_RGB_SUPPORT_V1"
SOURCE_AWARE_RETENTION_VETO = "SOURCE_AWARE_POINT_WEARER_V1"
OBJECT_POINT_TEXT_IDENTITY = "OBJECT_POINT_TEXT_IDENTITY_V1"
TEXT_POINT_WEARER_UNION = "SAM31_RGB_TEXT_POINT_WEARER_UNION"
REFERENCE_SIZE = (640, 480)


class AutomaticMaskContractError(ValueError):
    """The candidate cannot substantiate or interpret this input contract."""


@dataclass(frozen=True)
class RGBProposal:
    """A single stable native-frame track, supplied by an RGB model adapter.

    ``unsupported_frames`` reports explicit adapter failure/low support; there
    is deliberately no new score threshold. Missing native frames stay empty.
    SHA declarations must be checked against frozen files by the task runner.
    """

    role: str
    track_id: int
    masks: np.ndarray
    frames_sha256: str
    raw_source_sha256: str
    frame_map_sha256: str
    model_sha256: str
    source: str = "SAM31_RGB_TEXT"
    unsupported_frames: np.ndarray | None = None
    prior_unsupported_frames: np.ndarray | None = None
    refinement_receipt_payload_sha256: str | None = None
    refinement_status: str | None = None
    retained_text_masks: np.ndarray | None = None
    point_masks: np.ndarray | None = None
    retention_conflict_masks: np.ndarray | None = None


def _rgb(value: Any) -> np.ndarray:
    frames = np.asarray(value)
    if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8
            or not len(frames) or min(frames.shape[1:3]) < 1):
        raise AutomaticMaskContractError("RAW_RGB_UINT8_NATIVE_N_H_W_3_REQUIRED")
    return frames


def frames_sha256(frames_rgb: np.ndarray) -> str:
    """Hash array domain plus actual decoded RGB, without a temporary file."""
    frames = _rgb(frames_rgb)
    digest = hashlib.sha256()
    digest.update(str(frames.shape).encode("ascii"))
    digest.update(frames.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(frames).data)
    return digest.hexdigest()


def _sha(value: Any, code: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise AutomaticMaskContractError(code)
    return value


def _signature_values(raw_source_sha256: str, frame_map_sha256: str,
                      model_sha256: str) -> tuple[str, str, str]:
    return (_sha(raw_source_sha256, "RAW_SOURCE_SHA_REQUIRED"),
            _sha(frame_map_sha256, "NATIVE_FRAME_MAP_SHA_REQUIRED"),
            _sha(model_sha256, "RGB_MODEL_SHA_REQUIRED"))


def resolve_controller_text_prompt(value: str | None) -> str:
    """Permit one separately frozen candidate; defaults remain untouched."""
    if value is None:
        return PROMPTS["controller"]
    if type(value) is not str or value != "VR controller":
        raise AutomaticMaskContractError("CONTROLLER_TEXT_PROMPT_NONE_OR_VR_CONTROLLER_REQUIRED")
    return value


def resolve_glove_text_prompt(value: str | None) -> str:
    """One explicit whole-gloved-hand concept; the old default is untouched."""
    if value is None:
        return PROMPTS["glove"]
    if type(value) is not str or value != "a gloved hand":
        raise TypeError("GLOVE_TEXT_PROMPT_NONE_OR_A_GLOVED_HAND_REQUIRED: unexpected keyword override value")
    return value


def resolve_text_wearer_retention(value: str | None) -> str | None:
    """One explicit post-G1 recipe; no default or model-call changes."""
    if value is None:
        return None
    if type(value) is not str or value != TEXT_WEARER_RETENTION:
        raise AutomaticMaskContractError("TEXT_WEARER_RETENTION_NONE_OR_FROZEN_POLICY_REQUIRED")
    return value


def resolve_source_aware_retention_veto(value: str | None) -> str | None:
    """Explicit post-G3 policy; None keeps the frozen global-veto behavior."""
    if value is None:
        return None
    if type(value) is not str or value != SOURCE_AWARE_RETENTION_VETO:
        raise AutomaticMaskContractError("SOURCE_AWARE_RETENTION_VETO_NONE_OR_FROZEN_POLICY_REQUIRED")
    return value


def collect_sam_rgb_proposals(sam: Any, frames_rgb: np.ndarray, *,
                              raw_source_sha256: str, frame_map_sha256: str,
                              model_sha256: str,
                              controller_text_prompt: str | None = None,
                              glove_text_prompt: str | None = None) -> tuple[list[RGBProposal], dict]:
    """Call the existing model API, using a fixed, non-task-specific recipe.

    Actual invocation requires a separately registered model task/GPU lease.
    Native frame zero is the fixed primary anchor. This bridge does not claim
    to discover all objects, does not synthesize absent frames, and never reads
    seeds. Each ``track_text`` invocation owns its normal SAM video state.
    """
    controller_prompt = resolve_controller_text_prompt(controller_text_prompt)
    glove_prompt = resolve_glove_text_prompt(glove_text_prompt)
    frames = _rgb(frames_rgb)
    raw_sha, mapping_sha, model_sha = _signature_values(
        raw_source_sha256, frame_map_sha256, model_sha256)
    frame_sha = frames_sha256(frames)
    if not callable(getattr(sam, "track_text", None)):
        raise AutomaticMaskContractError("SAM31_RGB_TEXT_PROVIDER_REQUIRED")
    count, height, width = frames.shape[:3]
    proposals, calls = [], []
    prompts = dict(PROMPTS)
    prompts["controller"] = controller_prompt
    prompts["glove"] = glove_prompt
    for role, prompt in prompts.items():
        tracks = sam.track_text(list(frames), prompt, 0)
        if not isinstance(tracks, Mapping):
            raise AutomaticMaskContractError("SAM_TRACK_MAPPING_REQUIRED")
        for track_id, native_masks in tracks.items():
            if (isinstance(track_id, bool) or not isinstance(track_id, (int, np.integer))
                    or int(track_id) < 0 or not isinstance(native_masks, Mapping)):
                raise AutomaticMaskContractError("SAM_NATIVE_TRACK_ID_OR_MAPPING_INVALID")
            masks = np.zeros((count, height, width), bool)
            for frame_index, packed in native_masks.items():
                if (isinstance(frame_index, bool)
                        or not isinstance(frame_index, (int, np.integer))
                        or not 0 <= int(frame_index) < count):
                    raise AutomaticMaskContractError("SAM_NATIVE_FRAME_OUT_OF_DOMAIN")
                packed = np.asarray(packed)
                if (packed.ndim != 1 or packed.dtype != np.uint8
                        or packed.size != (height * width + 7) // 8):
                    raise AutomaticMaskContractError("SAM_PACKED_VISIBLE_MASK_INVALID")
                masks[int(frame_index)] = parent_mask.unpack(packed, (height, width))
            proposals.append(RGBProposal(
                role, int(track_id), masks, frame_sha, raw_sha, mapping_sha, model_sha))
        calls.append({"role": role, "prompt": prompt, "anchor_native_frame": 0,
                      "tracks": len(tracks),
                      "propagation_aborted": list(getattr(sam, "aborted", []))})
    receipt = {
        "calls": calls,
        "object_provider": "SAM31_GENERIC_OBJECT_CONCEPT_NOT_ALL_INSTANCES",
        "model_execution_performed_by_caller": True,
        "raw_source_sha256": raw_sha,
        "frame_map_sha256": mapping_sha,
        "model_sha256": model_sha,
        "frames_sha256": frame_sha,
    }
    if glove_text_prompt is not None:
        receipt["glove_text_prompt"] = glove_text_prompt
    return proposals, receipt


def _near(mask: np.ndarray, radius: float) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius


def _projection(value: Any, count: int) -> np.ndarray | None:
    if value is None:
        return None
    points = np.asarray(value)
    if (points.ndim != 4 or points.shape[:2] != (2, count)
            or points.shape[-1] != 2 or points.dtype.kind not in "fiu"
            or np.isinf(points).any()):
        raise AutomaticMaskContractError("PROJECTION_HINTS_REQUIRE_2_N_K_XY_WITH_NAN_MISSING")
    return points.astype(float)


def _projection_association(masks: np.ndarray, hints: np.ndarray | None) -> tuple[bool, str, dict]:
    if hints is None:
        return True, "UNKNOWN", {"projection_available": False}
    count, height, width = masks.shape
    hit_frames = np.zeros((2, count), bool)
    available_frames = np.zeros(count, bool)
    for frame in range(count):
        near = parent_mask.dilate(masks[frame], parent_mask.HIT_RADIUS_PX)
        for side in range(2):
            points = hints[side, frame]
            points = points[np.isfinite(points).all(axis=1)]
            points = points[(points[:, 0] >= 0) & (points[:, 0] < width)
                            & (points[:, 1] >= 0) & (points[:, 1] < height)]
            if not len(points):
                continue
            available_frames[frame] = True
            # Pixel-centre rounding can hit the far edge; clip after filtering.
            pixels = np.rint(points).astype(int)
            pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
            pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
            hit_frames[side, frame] = near[pixels[:, 1], pixels[:, 0]].any()
    present = masks.any(axis=(1, 2))
    if not (present & available_frames).any():
        return True, "UNKNOWN", {"projection_available": False}
    rate = float(hit_frames.any(axis=0).sum() / max(1, int(present.sum())))
    both = hit_frames.any(axis=1)
    side = "BOTH" if both.all() else ("LEFT" if both[0] else ("RIGHT" if both[1] else "UNKNOWN"))
    return rate >= parent_mask.TRACK_HIT_RATE_MIN, side, {
        "projection_available": True, "projected_hit_rate": rate,
        "projection_is_pixel_evidence": False,
        "hit_native_frames_by_side": [np.flatnonzero(row).tolist() for row in hit_frames],
    }


def _owned_components(masks: np.ndarray, wearer: np.ndarray,
                      radius: float) -> tuple[np.ndarray, dict]:
    near = [_near(frame, radius) for frame in wearer]
    present = masks.any(axis=(1, 2))
    hits = np.asarray([bool((frame & close).any()) for frame, close in zip(masks, near)])
    rate = float(hits.sum() / max(1, int(present.sum())))
    kept = rate >= parent_mask.TRACK_HIT_RATE_MIN
    selected = np.zeros_like(masks)
    if kept:
        for frame, mask in enumerate(masks):
            components, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            for component in np.unique(labels[near[frame] & mask]):
                if 0 < component < components:
                    selected[frame] |= labels == component
    return selected, {"wearer_hit_rate": rate, "kept": kept}


def _positive_point(masks: np.ndarray) -> tuple[int, list[float]] | None:
    areas = masks.sum(axis=(1, 2))
    if not areas.any():
        return None
    anchor = int(np.argmax(areas))
    mask = masks[anchor]
    # Zero padding makes the interior argmax well-defined even for full masks.
    depth = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    row, column = np.unravel_index(int(np.argmax(depth)), mask.shape)
    if not mask[row, column]:
        raise AutomaticMaskContractError("AUTO_POINT_NOT_INSIDE_CURRENT_RGB_PROPOSAL")
    return anchor, [float(column), float(row)]


def _build_automatic_masks(frames_rgb: np.ndarray, proposals: Sequence[RGBProposal], *,
                           raw_source_sha256: str, frame_map_sha256: str,
                           model_sha256: str, projection_hints: np.ndarray | None = None,
                           source_aware_retention_veto: str | None = None) -> dict:
    """Build separated candidate masks from exact current-RGB model proposals.

    Frozen V1 hit, proximity, dilation, temporal bracket, ProPainter dilation
    and compact-object constants are reused unchanged. Native masks unsupported
    by the adapter remain UNKNOWN; projection hulls and hidden-object carries
    are never added. The returned internal points are ready for an independently
    registered RGB refinement adapter but are not secretly fed into the old
    manually-annotated seed runner.
    """
    veto_policy = resolve_source_aware_retention_veto(source_aware_retention_veto)
    frames = _rgb(frames_rgb)
    raw_sha, map_sha, model_sha = _signature_values(
        raw_source_sha256, frame_map_sha256, model_sha256)
    sha = frames_sha256(frames)
    count, height, width = frames.shape[:3]
    if not np.isclose(width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1]):
        raise AutomaticMaskContractError("FROZEN_PROXIMITY_REQUIRES_ISOTROPIC_640_480_DOMAIN")
    scale = width / REFERENCE_SIZE[0]
    hints = _projection(projection_hints, count)
    if not isinstance(proposals, (list, tuple)):
        raise AutomaticMaskContractError("RGB_PROPOSAL_SEQUENCE_REQUIRED")
    prepared, seen, unknown = [], set(), np.zeros((count, height, width), bool)
    retention_unknown = np.zeros_like(unknown)
    for proposal in proposals:
        if not isinstance(proposal, RGBProposal):
            raise AutomaticMaskContractError("RGB_MODEL_PROPOSAL_NOT_MANUAL_SEED_REQUIRED")
        if (proposal.role not in PROMPTS or proposal.source not in PROPOSAL_SOURCES
                or (proposal.source == "RGB_CLASS_AGNOSTIC_INSTANCE" and proposal.role != "object")
                or isinstance(proposal.track_id, bool)
                or not isinstance(proposal.track_id, (int, np.integer)) or proposal.track_id < 0):
            raise AutomaticMaskContractError("RGB_PROPOSAL_ROLE_SOURCE_OR_ID_INVALID")
        key = (proposal.role, int(proposal.track_id))
        if key in seen:
            raise AutomaticMaskContractError("DUPLICATE_ROLE_TRACK_ID")
        seen.add(key)
        if (proposal.frames_sha256 != sha or proposal.raw_source_sha256 != raw_sha
                or proposal.frame_map_sha256 != map_sha or proposal.model_sha256 != model_sha):
            raise AutomaticMaskContractError("CURRENT_RGB_SOURCE_FRAME_MAP_OR_MODEL_SHA_MISMATCH")
        masks = np.asarray(proposal.masks)
        if masks.shape != unknown.shape or masks.dtype != np.bool_:
            raise AutomaticMaskContractError("NATIVE_RGB_PROPOSAL_MASK_DOMAIN_OR_TYPE")
        origins = (proposal.retained_text_masks, proposal.point_masks, proposal.retention_conflict_masks)
        if proposal.source == TEXT_POINT_WEARER_UNION:
            if (proposal.role not in {"human", "glove"}
                    or any(not isinstance(item, np.ndarray) or item.shape != masks.shape
                           or item.dtype != np.bool_ for item in origins)
                    or not np.array_equal(masks, origins[0] | origins[1])
                    or np.any(origins[0] & origins[1]) or np.any(origins[0] & origins[2])
                    or proposal.refinement_receipt_payload_sha256 is None):
                raise AutomaticMaskContractError("TEXT_POINT_WEARER_ORIGIN_OR_UNION_INVALID")
            unknown |= origins[2]
            retention_unknown |= origins[2]
        elif any(item is not None for item in origins):
            raise AutomaticMaskContractError("TEXT_POINT_ORIGINS_REQUIRE_EXPLICIT_UNION_SOURCE")
        unsupported = proposal.unsupported_frames
        if unsupported is None:
            unsupported = np.zeros(count, bool)
        unsupported = np.asarray(unsupported)
        if unsupported.shape != (count,) or unsupported.dtype != np.bool_:
            raise AutomaticMaskContractError("NATIVE_UNSUPPORTED_FRAME_FLAGS_INVALID")
        prior = proposal.prior_unsupported_frames
        if prior is not None:
            prior = np.asarray(prior)
            if prior.shape != (count,) or prior.dtype != np.bool_:
                raise AutomaticMaskContractError("PRIOR_UNSUPPORTED_FRAME_FLAGS_INVALID")
        if proposal.refinement_receipt_payload_sha256 is not None:
            _sha(proposal.refinement_receipt_payload_sha256, "REFINEMENT_RECEIPT_PAYLOAD_SHA_INVALID")
        valid = masks.copy()
        unknown[unsupported] |= masks[unsupported]
        valid[unsupported] = False
        prepared.append((proposal, valid))

    human = np.zeros_like(unknown)
    equipment = np.zeros_like(unknown)
    equipment_roles = {role: np.zeros_like(unknown) for role in EQUIPMENT_ROLES}
    rows, prompts = [], []
    accepted_point_wearer = np.zeros_like(unknown)
    accepted_point_sources = []

    def record(proposal: RGBProposal, selected: np.ndarray, detail: dict, side: str = "UNKNOWN") -> None:
        unsupported = proposal.unsupported_frames
        unsupported_native_frames = ([] if unsupported is None
                                     else np.flatnonzero(unsupported).tolist())
        rows.append({"role": proposal.role, "track_id": int(proposal.track_id),
                     "source": proposal.source, "side": side,
                     "unsupported_native_frames": unsupported_native_frames,
                     "prior_stage_unsupported_native_frames": (
                         [] if proposal.prior_unsupported_frames is None
                         else np.flatnonzero(proposal.prior_unsupported_frames).tolist()),
                     "refinement_receipt_payload_sha256": proposal.refinement_receipt_payload_sha256,
                     "refinement_status": proposal.refinement_status,
                     "selected_pixels_per_native_frame": selected.sum(axis=(1, 2)).tolist(), **detail})
        if proposal.source == TEXT_POINT_WEARER_UNION:
            rows[-1].update(
                text_wearer_retention=TEXT_WEARER_RETENTION,
                retained_text_only_pixels_per_native_frame=(selected & proposal.retained_text_masks).sum(axis=(1, 2)).tolist(),
                point_model_pixels_per_native_frame=(selected & proposal.point_masks).sum(axis=(1, 2)).tolist(),
                point_stage_unsupported_native_frames=np.flatnonzero(~proposal.point_masks.any(axis=(1, 2))).tolist(),
                retention_conflict_pixels_per_native_frame=proposal.retention_conflict_masks.sum(axis=(1, 2)).tolist(),
                retained_text_is_point_observation=False,
                origins_signature={key: hashlib.sha256(np.ascontiguousarray(value).data).hexdigest()
                    for key, value in zip(("retained_text_masks", "point_masks", "retention_conflict_masks"),
                                          (proposal.retained_text_masks, proposal.point_masks, proposal.retention_conflict_masks))})
        if (veto_policy is not None and proposal.role in {"human", "glove"}
                and proposal.source in {"SAM31_RGB_POINT_AUTO", TEXT_POINT_WEARER_UNION}
                and proposal.refinement_receipt_payload_sha256 is not None
                and proposal.refinement_status in {
                    "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE",
                    "INCOMPLETE_RGB_PROMPT_SUPPORT", "PROPAGATION_ABORTED"}):
            # Use this proposal's already-valid selected pixels, not an OR of
            # another track's support. Rejected associations and unsupported
            # native frames have already been removed from `selected`.
            point = (selected & proposal.point_masks
                     if proposal.source == TEXT_POINT_WEARER_UNION else selected)
            accepted_point_wearer[:] |= point
            accepted_point_sources.append({
                "role": proposal.role, "track_id": int(proposal.track_id),
                "source": proposal.source,
                "refinement_receipt_payload_sha256": proposal.refinement_receipt_payload_sha256,
                "refinement_status": proposal.refinement_status,
                "selected_point_pixels_per_native_frame": point.sum(axis=(1, 2)).tolist()})
        point = _positive_point(selected)
        if point is not None:
            anchor, pixel = point
            prompts.append({"role": proposal.role, "track_id": int(proposal.track_id),
                            "side": side, "native_frame_index": anchor,
                            "points_px": [pixel], "point_labels": [1],
                            "source": "CURRENT_RGB_MODEL_MASK_INTERIOR_AUTO",
                            "frames_sha256": sha, "raw_source_sha256": raw_sha,
                            "frame_map_sha256": map_sha, "model_sha256": model_sha})

    for proposal, masks in prepared:
        if proposal.role != "human":
            continue
        kept, side, detail = _projection_association(masks, hints)
        selected = masks if kept else np.zeros_like(masks)
        human |= selected
        unknown |= masks & ~selected
        record(proposal, selected, {"kept": kept, **detail}, side)

    # One recipe change: valid RGB glove proposals independently seed wearer
    # support. Do not rename them human, bootstrap them from other equipment,
    # or reconstruct their absent/unsupported native-frame pixels.
    for proposal, masks in prepared:
        if proposal.role != "glove":
            continue
        equipment_roles["glove"] |= masks
        equipment |= masks
        record(proposal, masks, {"kept": bool(masks.any()),
            "wearer_support_rule": "INDEPENDENT_CURRENT_RGB_GLOVE_PROPOSAL",
            "wearer_support_is_human_measurement": False})
    wearer_support = human | equipment_roles["glove"]

    for role in EQUIPMENT_ROLES:
        if role == "glove":
            continue
        for proposal, masks in prepared:
            if proposal.role != role:
                continue
            selected, detail = _owned_components(masks, wearer_support | equipment, parent_mask.CONTACT_PX * scale)
            equipment_roles[role] |= selected
            equipment |= selected
            unknown |= masks & ~selected
            record(proposal, selected, detail)

    core = human | equipment
    objects = np.zeros_like(core)
    for proposal, masks in prepared:
        if proposal.role != "object":
            continue
        selected, detail = _owned_components(masks, core, parent_mask.CONTACT_PX * scale)
        rejected_frames = []
        for frame, mask in enumerate(selected):
            if mask.any() and not parent_mask._object_like(mask):
                unknown[frame] |= mask
                selected[frame] = False
                rejected_frames.append(frame)
        objects |= selected
        record(proposal, selected, {**detail, "frozen_compact_object_rejected_native_frames": rejected_frames})

    conflict = (objects & core) | retention_unknown
    protected = objects & ~core & ~retention_unknown
    unknown |= conflict
    # A TEXT-only conflict remains UNKNOWN, but cannot cancel a separately
    # bound and already-selected wearer POINT prediction. This changes only
    # removal eligibility; it does not promote that prediction to pixel truth
    # or change the original protection / semantic-conflict bookkeeping.
    retention_veto = (retention_unknown if veto_policy is None else
                      retention_unknown & ~accepted_point_wearer)
    radius = max(1, int(round(parent_mask.HUMAN_DILATE_PX * scale)))
    dilated = np.stack([parent_mask.dilate(mask, radius) for mask in core])
    removal = parent_mask.bracket_fill(dilated, parent_mask.REMOVAL_BRACKET_FRAMES) & ~protected & ~retention_veto
    effective = np.stack([
        cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                   iterations=SETTINGS["mask_dilation"]) > 0 for mask in removal
    ]) & ~protected & ~retention_veto
    held = np.stack([protected[index] & _near(core[index], parent_mask.CONTACT_PX * scale)
                     for index in range(count)])
    problems = []
    wearer_missing = np.flatnonzero(~wearer_support.any(axis=(1, 2))).tolist()
    if not wearer_support.any():
        problems.append({"code": "CURRENT_RGB_WEARER_SUPPORT_MISSING", "status": "UNKNOWN"})
    elif wearer_missing:
        problems.append({"code": "CURRENT_RGB_WEARER_NATIVE_FRAMES_WITHOUT_SUPPORT",
                         "status": "UNKNOWN", "native_frames": wearer_missing})
    if unknown.any():
        problems.append({"code": "RGB_PROPOSAL_SUPPORT_OR_SEMANTIC_CONFLICT_UNKNOWN",
                         "native_frames": np.flatnonzero(unknown.any(axis=(1, 2))).tolist()})
    point_proposals = [proposal for proposal in proposals if proposal.source in
                       {"SAM31_RGB_POINT_AUTO", TEXT_POINT_WEARER_UNION}]
    bound_point_receipts = bool(point_proposals) and all(
        proposal.refinement_receipt_payload_sha256 is not None
        and proposal.refinement_status in {
            "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE", "INCOMPLETE_RGB_PROMPT_SUPPORT",
            "PROPAGATION_ABORTED",
        } for proposal in point_proposals)
    if point_proposals and not bound_point_receipts:
        problems.append({"code": "POINT_REFINEMENT_RECEIPT_UNBOUND", "status": "UNKNOWN"})
    result = {
        "core": core, "sam_core": human, "human": human, "equipment": equipment,
        "wearer_support": wearer_support,
        "equipment_roles": equipment_roles, "object_observed": objects,
        "protected_visible": protected, "semantic_conflict": conflict,
        "removal": removal, "effective_removal": effective, "held": held,
        "unknown": unknown, "geometry_unknown": protected.copy(),
        "automatic_points": prompts,
        "metadata": {
            "schema_version": "inpainting-candidate-automatic-masks-v1",
            "candidate_only": True, "frames_sha256": sha, "raw_source_sha256": raw_sha,
            "frame_map_sha256": map_sha, "model_sha256": model_sha,
            "prompts": dict(PROMPTS), "proposal_tracks": rows,
            "producer_manual_seed_reads": 0, "per_session_overrides": 0,
            "read_proof_scope": "THIS_MODULE_NO_FILE_IO_UPSTREAM_READS_REQUIRE_TASK_AUDIT",
            "projection_is_pixel_evidence": False, "hidden_object_protection": False,
            "object_provider_coverage": "GENERIC_RGB_PROPOSALS_NOT_ALL_INSTANCES",
            "object_shape_rule": "UNCHANGED_V1_COMPACT_AREA_FILTER_LARGE_OBJECTS_UNKNOWN",
            "conflict_is_protection": False, "semantic_conflicts_removed_as_wearer_proposal_not_truth": True,
            "protected_effective_removal_overlap": int((protected & effective).sum()),
            "producer_scope": "BUILD_ONLY_NO_MODEL_EXECUTION",
            "automatic_point_refinement_executed_by_this_builder": False,
            "automatic_point_refinement_executed": (
                True if bound_point_receipts else (None if point_proposals else False)),
            "upstream_point_refinement_status": (
                "RECEIPT_PAYLOAD_BOUND" if bound_point_receipts
                else ("UNBOUND_POINT_SOURCE_DECLARATION" if point_proposals else "NOT_PRESENT")),
            "upstream_point_refinement_receipts": [
                {"role": proposal.role, "track_id": int(proposal.track_id),
                 "payload_sha256": proposal.refinement_receipt_payload_sha256,
                 "status": proposal.refinement_status} for proposal in point_proposals],
            "point_execution_proof": "PAYLOAD_SIGNATURE_PRESENT_TASK_MUST_BIND_ACTUAL_IMMUTABLE_RECEIPT",
            "wearer_support_status_by_native_frame": [
                "RGB_MODEL_PROPOSAL" if present else "UNKNOWN"
                for present in wearer_support.any(axis=(1, 2))],
            "wearer_support_sources_by_native_frame": [
                [role for role, masks in (("human", human), ("glove", equipment_roles["glove"]))
                 if masks[index].any()] for index in range(count)],
            "human_support_status_by_native_frame": [
                "RGB_MODEL_PROPOSAL" if present else "UNKNOWN"
                for present in human.any(axis=(1, 2))],
            "glove_support_rule": "INDEPENDENT_CURRENT_RGB_PROPOSAL_NOT_HUMAN_MEASUREMENT",
            "wearer_support_is_direct_3d_measurement": False,
            "time_mode": "OFFLINE_BIDIRECTIONAL_SAM_TRACKS_AND_TEMPORAL_BRACKET",
            "quality_problems": problems, "human_review": "PENDING",
            "visual_quality": "NOT_EVALUATED", "training_eligible": False,
            "control_ground_truth": False, "physical_contact_ground_truth": False,
        },
    }
    unions = [proposal for proposal in proposals if proposal.source == TEXT_POINT_WEARER_UNION]
    if unions:
        for role in ("human", "glove"):
            selected = human if role == "human" else equipment_roles["glove"]
            for field in ("retained_text_masks", "point_masks", "retention_conflict_masks"):
                origin = np.zeros_like(unknown)
                for proposal in unions:
                    if proposal.role == role:
                        origin |= getattr(proposal, field)
                result["wearer_origin__" + role + "__" + field] = (
                    origin if field == "retention_conflict_masks" else origin & selected)
        result["metadata"]["text_wearer_retention"] = TEXT_WEARER_RETENTION
        result["metadata"]["retained_text_is_point_observation"] = False
        result["metadata"]["wearer_origin_arrays"] = [key for key in result if key.startswith("wearer_origin__")]
    if veto_policy is not None:
        result["accepted_point_wearer_support"] = accepted_point_wearer
        result["retention_veto"] = retention_veto
        result["metadata"].update({
            "source_aware_retention_veto": veto_policy,
            "retention_veto_rule": "RETENTION_UNKNOWN_AND_NOT_ACCEPTED_SELECTED_BOUND_WEARER_POINT",
            "accepted_point_wearer_sources": accepted_point_sources,
            "accepted_point_wearer_pixels_per_native_frame": accepted_point_wearer.sum(axis=(1, 2)).tolist(),
            "retention_veto_pixels_per_native_frame": retention_veto.sum(axis=(1, 2)).tolist(),
            "retention_conflict_with_accepted_point_pixels_per_native_frame": (
                retention_unknown & accepted_point_wearer).sum(axis=(1, 2)).tolist(),
            "POINT_is_pixel_truth": False,
            "retention_unknown_bookkeeping_preserved": True})
    return result


def build_automatic_masks(frames_rgb: np.ndarray, proposals: Sequence[RGBProposal], *,
                          raw_source_sha256: str, frame_map_sha256: str,
                          model_sha256: str, projection_hints: np.ndarray | None = None) -> dict:
    """Frozen default ABI; source-aware candidate rules require the wrapper."""
    return _build_automatic_masks(frames_rgb, proposals,
        raw_source_sha256=raw_source_sha256, frame_map_sha256=frame_map_sha256,
        model_sha256=model_sha256, projection_hints=projection_hints)


def build_automatic_masks_with_source_aware_veto(
        frames_rgb: np.ndarray, proposals: Sequence[RGBProposal], *,
        raw_source_sha256: str, frame_map_sha256: str, model_sha256: str,
        projection_hints: np.ndarray | None = None,
        source_aware_retention_veto: str | None = None) -> dict:
    """One explicit candidate policy; None is an exact default no-op."""
    return _build_automatic_masks(frames_rgb, proposals,
        raw_source_sha256=raw_source_sha256, frame_map_sha256=frame_map_sha256,
        model_sha256=model_sha256, projection_hints=projection_hints,
        source_aware_retention_veto=source_aware_retention_veto)


def _refine_sam_rgb_proposals(sam: Any, frames_rgb: np.ndarray,
                             proposals: Sequence[RGBProposal], *,
                             raw_source_sha256: str, frame_map_sha256: str,
                             model_sha256: str,
                             projection_hints: np.ndarray | None = None,
                             text_wearer_retention: str | None = None) -> tuple[list[RGBProposal], dict]:
    """Feed internally generated current-RGB points to the installed SAM model.

    This is an explicit invocation bridge, not a new model or a manual-seed
    evidence adapter. It uses the existing official ``add_prompt``/propagation
    APIs and unchanged 0.5 model output threshold. For each selected RGB track,
    correction points come only from its first/last supported RGB frame and
    maximum-area frame; object identity is unchanged. UNKNOWN side is retained
    as metadata and is not invented as left/right to satisfy an old seed schema.

    The caller must own the registered GPU lease. G0 unit tests use only a fake
    provider; no model weights are loaded by this module. Every call is bounded
    by the input's native frame count and existing MAX_OBJECTS capacity.
    """
    retention = resolve_text_wearer_retention(text_wearer_retention)
    frames = _rgb(frames_rgb)
    # Validation and track selection occur before any model/state allocation.
    built = build_automatic_masks(
        frames, proposals, raw_source_sha256=raw_source_sha256,
        frame_map_sha256=frame_map_sha256, model_sha256=model_sha256,
        projection_hints=projection_hints)
    by_key = {(item.role, int(item.track_id)): item for item in proposals}
    prepared = []
    for point in built["automatic_points"]:
        proposal = by_key[(point["role"], point["track_id"])]
        if proposal.role == "human":
            selected = proposal.masks & built["human"]
        elif proposal.role == "object":
            selected = proposal.masks & built["object_observed"]
        else:
            selected = proposal.masks & built["equipment_roles"][proposal.role]
        if proposal.unsupported_frames is not None:
            selected = selected.copy()
            selected[proposal.unsupported_frames] = False
        visible = np.flatnonzero(selected.any(axis=(1, 2)))
        if not len(visible):
            continue
        anchors = sorted({int(visible[0]), int(visible[-1]), point["native_frame_index"]})
        corrections = []
        for native_frame in anchors:
            result = _positive_point(selected[native_frame:native_frame + 1])
            if result is None:
                raise AutomaticMaskContractError("AUTO_CORRECTION_REQUIRES_CURRENT_RGB_MASK_SUPPORT")
            _, pixel = result
            corrections.append({"native_frame_index": native_frame, "points_px": [pixel],
                                "point_labels": [1],
                                "source": "CURRENT_RGB_MODEL_MASK_INTERIOR_AUTO"})
        prepared.append({"proposal": proposal, "side": point["side"],
                         "model_obj_id": len(prepared) + 1, "corrections": corrections})
    if len(prepared) > MAX_OBJECTS:
        raise AutomaticMaskContractError("AUTO_POINT_EXISTING_MODEL_OBJECT_CAPACITY_EXCEEDED")
    metadata = {
        "schema_version": "inpainting-candidate-auto-point-refinement-v1",
        "status": "NO_RGB_SUPPORTED_AUTOMATIC_POINTS" if not prepared else "PENDING",
        "automatic_points_submitted": 0, "manual_points_submitted": 0,
        "manual_visible_confirmation_claim": False,
        "model_execution_performed": False,
        "native_frame_count": len(frames), "raw_source_sha256": raw_source_sha256,
        "frame_map_sha256": frame_map_sha256, "model_sha256": model_sha256,
        "frames_sha256": frames_sha256(frames),
        "point_origin": "CURRENT_RGB_MODEL_MASK_NOT_MANUAL_CONFIRMATION_OR_HARDWARE",
        "side_unknown_is_preserved": True, "aborted": [], "required_support_missing": [],
        "objects": [{"role": row["proposal"].role,
                     "track_id": int(row["proposal"].track_id), "side": row["side"],
                     "model_obj_id": row["model_obj_id"], "corrections": row["corrections"]}
                    for row in prepared],
        "visual_quality": "NOT_EVALUATED", "human_review": "PENDING",
        "training_eligible": False, "control_ground_truth": False,
    }
    if not prepared:
        if retention is not None:
            metadata["text_wearer_retention"] = retention
        return list(proposals), metadata
    model = getattr(sam, "model", None)
    if any(not callable(getattr(model, method, None)) for method in
           ("init_state", "add_prompt", "propagate_in_video", "_cache_frame_outputs")):
        raise AutomaticMaskContractError("OFFICIAL_SAM_AUTOMATIC_POINT_MODEL_API_REQUIRED")

    # Lazy imports: importing the candidate package itself loads no Torch/model.
    import torch
    from PIL import Image

    count, height, width = frames.shape[:3]
    tracks = {row["model_obj_id"]: {} for row in prepared}
    allowed_ids = set(tracks)

    def store(frame: Any, outputs: Any) -> None:
        if (isinstance(frame, bool) or not isinstance(frame, (int, np.integer))
                or not 0 <= int(frame) < count):
            raise AutomaticMaskContractError("AUTO_POINT_MODEL_NATIVE_FRAME_OUT_OF_DOMAIN")
        masks, ids = _normalized_outputs(outputs, height, width)
        for mask, object_id in zip(masks, ids):
            object_id = int(object_id)
            if object_id not in allowed_ids:
                raise AutomaticMaskContractError("AUTO_POINT_MODEL_UNPROMPTED_ID")
            if mask.any():
                tracks[object_id][int(frame)] = mask.copy()
            else:
                # Explicit empty output retracts earlier support at this frame.
                tracks[object_id].pop(int(frame), None)

    state = None
    try:
        state = model.init_state(
            resource_path=[Image.fromarray(frame) for frame in frames],
            offload_video_to_cpu=True, async_loading_frames=False)
        if not isinstance(state, dict):
            raise AutomaticMaskContractError("AUTO_POINT_MODEL_STATE_MAPPING_REQUIRED")
        # The pinned model requires cached entries for official point-output
        # merging. Empty shells contain no masks and are never RGB observations.
        for native_frame in range(count):
            if native_frame not in state.get("cached_frame_outputs", {}):
                model._cache_frame_outputs(state, native_frame, {})
                if state.get("cached_frame_outputs", {}).get(native_frame) != {}:
                    raise AutomaticMaskContractError("AUTO_POINT_CACHE_BOOTSTRAP_MUST_BE_EMPTY")
        for row in prepared:
            for correction in row["corrections"]:
                native_frame = correction["native_frame_index"]
                relative = np.asarray(correction["points_px"], float) / [width, height]
                frame, outputs = model.add_prompt(
                    inference_state=state, frame_idx=native_frame, text_str=None,
                    points=torch.tensor(relative, dtype=torch.float32),
                    point_labels=torch.tensor(correction["point_labels"], dtype=torch.int64),
                    obj_id=row["model_obj_id"], clear_old_points=True,
                    rel_coordinates=True, output_prob_thresh=.5)
                if frame != native_frame:
                    raise AutomaticMaskContractError("AUTO_POINT_NATIVE_ANCHOR_BINDING_MISMATCH")
                store(frame, outputs)
                metadata["automatic_points_submitted"] += 1
        first_anchor = min(correction["native_frame_index"] for row in prepared
                           for correction in row["corrections"])
        directions = [(False, count - first_anchor)]
        if first_anchor:
            directions.append((True, first_anchor + 1))
        for reverse, maximum in directions:
            yielded = set()
            try:
                for frame, outputs in model.propagate_in_video(
                        inference_state=state, start_frame_idx=first_anchor,
                        max_frame_num_to_track=maximum, reverse=reverse,
                        output_prob_thresh=.5):
                    if frame in yielded or len(yielded) >= maximum:
                        raise AutomaticMaskContractError("AUTO_POINT_NATIVE_PROPAGATION_DUPLICATE_OR_BUDGET")
                    yielded.add(frame)
                    store(frame, outputs)
            except RuntimeError as error:
                if parent_mask.PROPAGATION_ABORTED not in str(error):
                    raise
                metadata["aborted"].append("reverse" if reverse else "forward")
        metadata["model_execution_performed"] = True
    finally:
        if isinstance(state, dict):
            state.clear()

    refined = list(proposals)
    replacements = []
    for row in prepared:
        proposal = row["proposal"]
        masks = np.zeros((count, height, width), bool)
        for frame, mask in tracks[row["model_obj_id"]].items():
            masks[frame] = mask
        # Text-stage failure flags describe the OLD producer. A new point-stage
        # mask is independent RGB model support and must not be erased by them.
        # Missing/explicitly retracted point output remains UNKNOWN instead.
        unsupported = ~masks.any(axis=(1, 2))
        for correction in row["corrections"]:
            native_frame = correction["native_frame_index"]
            if native_frame not in tracks[row["model_obj_id"]]:
                unsupported[native_frame] = True
                metadata["required_support_missing"].append({
                    "role": proposal.role, "track_id": int(proposal.track_id),
                    "native_frame_index": native_frame})
        metadata_row = next(value for value in metadata["objects"]
                            if value["model_obj_id"] == row["model_obj_id"])
        metadata_row["prior_stage_unsupported_native_frames"] = (
            [] if proposal.unsupported_frames is None
            else np.flatnonzero(proposal.unsupported_frames).tolist())
        metadata_row["point_model_supported_native_frames"] = np.flatnonzero(~unsupported).tolist()
        metadata_row["point_stage_unsupported_native_frames"] = np.flatnonzero(unsupported).tolist()
        replacements.append((proposal, masks, unsupported))
    metadata["status"] = ("INCOMPLETE_RGB_PROMPT_SUPPORT" if metadata["required_support_missing"]
                          else ("PROPAGATION_ABORTED" if metadata["aborted"]
                                else "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE"))
    # This is a canonical PAYLOAD signature, not an on-disk receipt-file SHA.
    # The outer runner persists and signs the immutable receipt independently.
    if retention is not None:
        metadata["text_wearer_retention"] = retention
        metadata["retained_text_is_point_observation"] = False
    payload_sha = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()
    metadata["receipt_payload_sha256"] = payload_sha
    metadata["receipt_signature_scope"] = "CANONICAL_JSON_EXCLUDING_RECEIPT_SIGNATURE_FIELDS"
    for proposal, masks, unsupported in replacements:
        replacement = RGBProposal(
            proposal.role, int(proposal.track_id), masks, proposal.frames_sha256,
            proposal.raw_source_sha256, proposal.frame_map_sha256, proposal.model_sha256,
            source="SAM31_RGB_POINT_AUTO", unsupported_frames=unsupported,
            prior_unsupported_frames=(None if proposal.unsupported_frames is None
                                      else proposal.unsupported_frames.copy()),
            refinement_receipt_payload_sha256=payload_sha,
            refinement_status=metadata["status"])
        index = next(index for index, value in enumerate(refined)
                     if (value.role, value.track_id) == (proposal.role, proposal.track_id))
        refined[index] = replacement
    if retention is not None:
        refined = _merge_text_wearer_support(proposals, refined, built)
    return refined, metadata


def _merge_text_wearer_support(text_proposals, point_proposals, built_text):
    """The single post-inference merge shared by fresh and CPU audit paths."""
    if [(p.role, int(p.track_id)) for p in text_proposals] != [
            (p.role, int(p.track_id)) for p in point_proposals]:
        raise AutomaticMaskContractError("TEXT_POINT_TRACK_IDENTITIES_OR_ORDER_CHANGED")
    object_support = np.zeros_like(built_text["core"])
    for proposal in [*text_proposals, *point_proposals]:
        if proposal.role == "object":
            observed = proposal.masks.copy()
            if proposal.unsupported_frames is not None:
                observed[proposal.unsupported_frames] = False
            object_support |= observed
    merged = []
    for text, point in zip(text_proposals, point_proposals):
        if (text.role not in {"human", "glove"} or text.source != "SAM31_RGB_TEXT"
                or point.source != "SAM31_RGB_POINT_AUTO"):
            merged.append(point)
            continue
        if (point.refinement_receipt_payload_sha256 is None or point.refinement_status not in {
                "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE", "INCOMPLETE_RGB_PROMPT_SUPPORT",
                "PROPAGATION_ABORTED"}):
            raise AutomaticMaskContractError("TEXT_POINT_REPLACEMENT_RECEIPT_REQUIRED")
        selected = built_text["human"] if text.role == "human" else built_text["equipment_roles"]["glove"]
        legal_text = text.masks & selected
        point_masks = point.masks.copy()
        if point.unsupported_frames is not None:
            point_masks[point.unsupported_frames] = False
        text_only = legal_text & ~point_masks
        conflict = text_only & object_support
        retained = text_only & ~object_support
        masks = point_masks | retained
        merged.append(replace(point, source=TEXT_POINT_WEARER_UNION, masks=masks,
            unsupported_frames=~masks.any(axis=(1, 2)), retained_text_masks=retained,
            point_masks=point_masks, retention_conflict_masks=conflict))
    return merged


def merge_text_wearer_support(frames_rgb: np.ndarray, text_proposals: Sequence[RGBProposal],
                             point_proposals: Sequence[RGBProposal], *,
                             raw_source_sha256: str, frame_map_sha256: str,
                             model_sha256: str, projection_hints: np.ndarray | None = None,
                             text_wearer_retention: str | None = TEXT_WEARER_RETENTION) -> list[RGBProposal]:
    """Pure counterfactual over supplied proposals; no SAM or file writes.

    Frozen historical packs are permitted only to the separately registered CPU
    evaluator, not a cold-start producer. This returns in-memory proposals, not
    a fresh Mask receipt or any visual/geometry qualification.
    """
    retention = resolve_text_wearer_retention(text_wearer_retention)
    signatures = dict(raw_source_sha256=raw_source_sha256, frame_map_sha256=frame_map_sha256,
                      model_sha256=model_sha256, projection_hints=projection_hints)
    built = build_automatic_masks(frames_rgb, text_proposals, **signatures)
    build_automatic_masks(frames_rgb, point_proposals, **signatures)
    return list(point_proposals) if retention is None else _merge_text_wearer_support(
        text_proposals, point_proposals, built)


def merge_text_wearer_support_with_object_point_identity(
        frames_rgb: np.ndarray, text_proposals: Sequence[RGBProposal],
        point_proposals: Sequence[RGBProposal], *, raw_source_sha256: str,
        frame_map_sha256: str, model_sha256: str,
        projection_hints: np.ndarray | None = None,
        text_wearer_retention: str | None = TEXT_WEARER_RETENTION,
        object_point_identity_policy: str | None = None,
        built_text: Mapping | None = None) -> tuple[list[RGBProposal], dict]:
    """Opt-in, same-source identity check; returns the ONE final proposal list.

    An object POINT-only pixel cannot displace already selected wearer TEXT
    where *no* valid object TEXT supports that pixel. Genuine object TEXT from
    any track remains a conflict, not a universal wearer-wins rule. No pixels
    are invented: only object POINT pixels are excluded before the unchanged
    merge, and those same filtered object proposals reach the final builder.
    ``built_text`` can reuse a validated TEXT build from this same invocation.
    Returned metadata contains a boolean diagnostic array; callers seal it
    separately rather than serializing it as an ordinary JSON value.
    """
    if object_point_identity_policy is None:
        return merge_text_wearer_support(frames_rgb, text_proposals, point_proposals,
            raw_source_sha256=raw_source_sha256, frame_map_sha256=frame_map_sha256,
            model_sha256=model_sha256, projection_hints=projection_hints,
            text_wearer_retention=text_wearer_retention), {"object_point_identity_policy": None,
                "status": "DISABLED_EXACT_EXISTING_MERGE", "model_calls": 0}
    if object_point_identity_policy != OBJECT_POINT_TEXT_IDENTITY:
        raise AutomaticMaskContractError("UNKNOWN_OBJECT_POINT_IDENTITY_POLICY")
    if resolve_text_wearer_retention(text_wearer_retention) != TEXT_WEARER_RETENTION:
        raise AutomaticMaskContractError("OBJECT_POINT_IDENTITY_REQUIRES_TEXT_RETENTION")
    frames = _rgb(frames_rgb)
    shape = frames.shape[:3]
    raw_sha, map_sha, model_sha = _signature_values(raw_source_sha256, frame_map_sha256, model_sha256)
    payload_sha = frames_sha256(frames)
    identities = [(p.role, int(p.track_id)) for p in text_proposals]
    if (identities != [(p.role, int(p.track_id)) for p in point_proposals]
            or len(set(identities)) != len(identities)):
        raise AutomaticMaskContractError("OBJECT_POINT_IDENTITY_TRACK_ORDER_OR_DUPLICATE")

    def valid(proposal, *, text):
        if (not isinstance(proposal, RGBProposal) or proposal.role not in PROMPTS
                or proposal.frames_sha256 != payload_sha or proposal.raw_source_sha256 != raw_sha
                or proposal.frame_map_sha256 != map_sha or proposal.model_sha256 != model_sha
                or (text and proposal.source != "SAM31_RGB_TEXT")
                or (not text and proposal.source not in {"SAM31_RGB_TEXT", "SAM31_RGB_POINT_AUTO"})
                or proposal.masks.dtype != np.bool_ or proposal.masks.shape != shape
                or isinstance(proposal.track_id, bool) or not isinstance(proposal.track_id, (int, np.integer))):
            raise AutomaticMaskContractError("OBJECT_POINT_IDENTITY_SOURCE_OR_NATIVE_DOMAIN")
        observed = proposal.masks
        if proposal.unsupported_frames is not None:
            flags = np.asarray(proposal.unsupported_frames)
            if flags.dtype != np.bool_ or flags.shape != (shape[0],):
                raise AutomaticMaskContractError("OBJECT_POINT_IDENTITY_UNSUPPORTED_DOMAIN")
            if flags.any():
                observed = observed.copy()
                observed[flags] = False
        if not text and proposal.source == "SAM31_RGB_POINT_AUTO":
            if (proposal.refinement_receipt_payload_sha256 is None or proposal.refinement_status not in {
                    "EXECUTED_RGB_AUTOMATIC_POINT_MODEL_CANDIDATE", "INCOMPLETE_RGB_PROMPT_SUPPORT", "PROPAGATION_ABORTED"}):
                raise AutomaticMaskContractError("OBJECT_POINT_IDENTITY_BOUND_RAW_POINT_REQUIRED")
            _sha(proposal.refinement_receipt_payload_sha256, "OBJECT_POINT_IDENTITY_POINT_SIGNATURE_REQUIRED")
        return observed

    text_valid = [valid(p, text=True) for p in text_proposals]
    point_valid = [valid(p, text=False) for p in point_proposals]
    if built_text is None:
        built_text = build_automatic_masks(frames, text_proposals, raw_source_sha256=raw_sha,
            frame_map_sha256=map_sha, model_sha256=model_sha, projection_hints=projection_hints)
    metadata = built_text.get("metadata", {})
    if any(metadata.get(k) != v for k, v in (("frames_sha256", payload_sha),
            ("raw_source_sha256", raw_sha), ("frame_map_sha256", map_sha), ("model_sha256", model_sha))):
        raise AutomaticMaskContractError("OBJECT_POINT_IDENTITY_LEGAL_TEXT_BINDING_DRIFT")
    selected_human = built_text.get("human")
    selected_glove = built_text.get("equipment_roles", {}).get("glove")
    if any(not isinstance(a, np.ndarray) or a.dtype != np.bool_ or a.shape != shape
            for a in (selected_human, selected_glove)):
        raise AutomaticMaskContractError("OBJECT_POINT_IDENTITY_LEGAL_TEXT_NATIVE_DOMAIN")
    wearer, objects = np.zeros(shape, bool), np.zeros(shape, bool)
    for text, observed in zip(text_proposals, text_valid):
        if text.role == "human":
            wearer |= observed & selected_human
        elif text.role == "glove":
            wearer |= observed & selected_glove
        elif text.role == "object":
            objects |= observed
    rejected_union, filtered, rows = np.zeros(shape, bool), [], []
    for point, observed in zip(point_proposals, point_valid):
        if point.role != "object" or point.source != "SAM31_RGB_POINT_AUTO":
            filtered.append(point)
            continue
        rejected = observed & wearer & ~objects
        rejected_union |= rejected
        if rejected.any():
            masks = point.masks & ~rejected
            flags = ~masks.any(axis=(1, 2))
            if point.unsupported_frames is not None:
                flags |= point.unsupported_frames
            filtered.append(replace(point, masks=masks, unsupported_frames=flags))
        else:
            filtered.append(point)
        rows.append({"role": "object", "track_id": int(point.track_id),
            "point_pixels_before_per_native_frame": observed.sum(axis=(1, 2)).tolist(),
            "rejected_point_only_pixels_per_native_frame": rejected.sum(axis=(1, 2)).tolist(),
            "point_pixels_after_per_native_frame": (observed & ~rejected).sum(axis=(1, 2)).tolist()})
    merged = _merge_text_wearer_support(text_proposals, filtered, built_text)
    return merged, {"object_point_identity_policy": OBJECT_POINT_TEXT_IDENTITY,
        "status": "APPLIED_TO_MERGE_AND_FINAL_OBJECT_PROPOSALS",
        "identity_rejected_object_point": rejected_union, "object_tracks": rows,
        "rejected_object_point_pixels_per_native_frame": rejected_union.sum(axis=(1, 2)).tolist(),
        "valid_text_object_pixels_rejected": int((rejected_union & objects).sum()),
        "source_signatures": {"frames_sha256": payload_sha, "raw_source_sha256": raw_sha,
            "frame_map_sha256": map_sha, "model_sha256": model_sha},
        "true_text_object_conflict": "PRESERVED_UNKNOWN_UNCHANGED_RULES",
        "thresholds_changed": False, "model_calls": 0,
        "visual_quality": "NOT_EVALUATED", "human_review": "PENDING",
        "training_eligible": False, "control_ground_truth": False}


def refine_sam_rgb_proposals(sam: Any, frames_rgb: np.ndarray,
                            proposals: Sequence[RGBProposal], *,
                            raw_source_sha256: str, frame_map_sha256: str,
                            model_sha256: str,
                            projection_hints: np.ndarray | None = None) -> tuple[list[RGBProposal], dict]:
    """Original public ABI and unchanged default POINT replacement recipe."""
    return _refine_sam_rgb_proposals(sam, frames_rgb, proposals,
        raw_source_sha256=raw_source_sha256, frame_map_sha256=frame_map_sha256,
        model_sha256=model_sha256, projection_hints=projection_hints)


def refine_sam_rgb_proposals_with_text_retention(sam: Any, frames_rgb: np.ndarray,
                            proposals: Sequence[RGBProposal], *,
                            raw_source_sha256: str, frame_map_sha256: str,
                            model_sha256: str, projection_hints: np.ndarray | None = None,
                            text_wearer_retention: str | None = None) -> tuple[list[RGBProposal], dict]:
    """Explicit optional retention; shares the exact single inference path."""
    return _refine_sam_rgb_proposals(sam, frames_rgb, proposals,
        raw_source_sha256=raw_source_sha256, frame_map_sha256=frame_map_sha256,
        model_sha256=model_sha256, projection_hints=projection_hints,
        text_wearer_retention=text_wearer_retention)
