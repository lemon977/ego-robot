"""CPU-only connector for D1 -> D4 -> frozen wearable identity union.

The connector consumes one already-produced arm raw inventory.  D4 owns the
single D1 invocation, then accepted wearable raw identities may contribute
pixels only through bitwise OR with the exact accepted arm raw mask.  This
module has no inference callback, CUDA dependency, fallback, pixel fill, or
session/frame-specific selector behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from pipeline import lr_contact_phase_object6d_evidence as d4
from pipeline import lr_distributed_side_evidence as d1
from pipeline import neutral_wearable_identity_v2 as wearable


GPU_EXECUTION_AUTHORIZED = False
FROZEN_THRESHOLDS = (0.20, 0.05, 0.12)
FROZEN_WEARABLE_PROMPT_ORDER = ("W1", "W2", "W3", "W4")
WEARABLE_COMBINATION_RULE = "ALL_INDEPENDENT_ACCEPTS_RAW_BOOLEAN_OR_V1"
WEARABLE_METADATA_SOURCE_KIND = "SAM31_RAW_WEARABLE_INSTANCE_JSON"
WEARABLE_MASK_SOURCE_KIND = "SAM31_RAW_WEARABLE_MASK_PNG"
FORBIDDEN_SESSION = "grap_a_cap_025"
FORBIDDEN_PATH_COMPONENTS = frozenset(
    {"labels", "blind", "blind_test", "processed", FORBIDDEN_SESSION}
)


class UnifiedSelectorError(RuntimeError):
    """A frame identity, frozen evidence, or raw-pixel join is invalid."""


@dataclass(frozen=True)
class BoundWearableRawInstance:
    """One raw model instance bound to verified metadata and PNG bytes."""

    instance: wearable.RawWearableInstance
    metadata_ref: wearable.EvidenceRef
    mask_png_ref: wearable.EvidenceRef


@dataclass(frozen=True)
class FrozenWearableFrame:
    """Session binding around verified task32 wearable raw inputs."""

    session_id: str
    frame_index: int
    prompt_contract_ref: wearable.EvidenceRef
    wrist_authorities: Mapping[wearable.Side, wearable.WristAuthority]
    instances_by_prompt: Mapping[str, Sequence[BoundWearableRawInstance]]


@dataclass(frozen=True)
class SelectedWearableIdentity:
    prompt_id: str
    prompt_text: str
    prompt_contract_sha256: str
    raw_instance_offset: int
    instance_id: int
    raw_mask_sha256: str
    metadata_ref: wearable.EvidenceRef
    mask_png_ref: wearable.EvidenceRef
    added_pixels_against_arm: int


@dataclass(frozen=True)
class FinalSideSelection:
    side: d1.Side
    arm_status: d1.DecisionStatus
    arm_identity: wearable.AcceptedArmRawIdentity | None
    selected_wearables: tuple[SelectedWearableIdentity, ...]
    final_mask: np.ndarray | None
    final_mask_sha256: str | None
    added_pixels: int
    union_operations: int


@dataclass(frozen=True)
class UnifiedFrameSelection:
    session_id: str
    frame_index: int
    contact: d4.ContactAwareSelection
    wearable_decisions: Mapping[
        str, Mapping[wearable.Side, wearable.SideWearableDecision]
    ]
    sides: Mapping[d1.Side, FinalSideSelection]
    prompt_contract_ref: wearable.EvidenceRef
    optimized_hawor_source_ref: wearable.EvidenceRef
    raw_hawor_source_ref: wearable.EvidenceRef
    wrist_authorities: Mapping[wearable.Side, wearable.WristAuthority]
    execution_order: tuple[str, str, str] = ("D1", "D4", "WEARABLE_UNION")
    wearable_prompt_order: tuple[str, str, str, str] = FROZEN_WEARABLE_PROMPT_ORDER
    wearable_combination_rule: str = WEARABLE_COMBINATION_RULE
    wearable_prompt_invocation_counts: tuple[
        tuple[str, int], tuple[str, int], tuple[str, int], tuple[str, int]
    ] = (("W1", 1), ("W2", 1), ("W3", 1), ("W4", 1))
    frozen_thresholds: tuple[float, float, float] = FROZEN_THRESHOLDS
    semantic_change_class: str = "H5(b)_PIXEL_SEMANTIC_CRITERION_CHANGE"
    model_calls: int = 0
    fill_operations: int = 0
    morphology_operations: int = 0
    crop_operations: int = 0
    temporal_propagation_operations: int = 0
    fallback_operations: int = 0


def _ref_identity(ref: object) -> tuple[str, int, str]:
    try:
        path = ref.path  # type: ignore[attr-defined]
        size = ref.bytes  # type: ignore[attr-defined]
        sha256 = ref.sha256  # type: ignore[attr-defined]
    except AttributeError as error:
        raise UnifiedSelectorError("malformed evidence reference") from error
    if (
        type(path) is not str
        or not path
        or type(size) is not int
        or size <= 0
        or type(sha256) is not str
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise UnifiedSelectorError("malformed evidence reference")
    return path, size, sha256


def _reject_forbidden_ref(ref: object) -> None:
    path, _, _ = _ref_identity(ref)
    parts = {part.lower() for part in Path(path).parts}
    if parts & FORBIDDEN_PATH_COMPONENTS:
        raise UnifiedSelectorError("forbidden evidence path")


def _reject_mask_aliases(masks: Sequence[np.ndarray]) -> None:
    for index, left in enumerate(masks):
        for right in masks[index + 1 :]:
            if np.shares_memory(left, right):
                raise UnifiedSelectorError(
                    "raw mask alias across independent identities"
                )


def _immutable_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Copy a bool HxW mask onto an immutable ``bytes`` backing store."""

    array = np.asarray(mask)
    if array.ndim != 2 or array.dtype != np.bool_:
        raise UnifiedSelectorError("mask snapshot must be exact bool HxW")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.bool_).reshape(
        array.shape
    )
    if frozen.flags.writeable:
        raise UnifiedSelectorError("immutable mask snapshot construction failed")
    return frozen


def _mask_ref_record(ref: wearable.EvidenceRef) -> dict[str, object]:
    return {
        "path": ref.path,
        "bytes": ref.bytes,
        "sha256": ref.sha256,
        "source_kind": ref.source_kind,
    }


def _decode_verified_binary_png(
    source_ref: wearable.EvidenceRef,
    *,
    evidence_root: Path,
    image_shape: tuple[int, int],
) -> np.ndarray:
    if source_ref.source_kind != WEARABLE_MASK_SOURCE_KIND:
        raise UnifiedSelectorError("wearable mask PNG source kind mismatch")
    payload = source_ref.read_verified(evidence_root)
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                raise UnifiedSelectorError("wearable mask evidence is not one PNG")
            if image.mode not in {"1", "L"}:
                raise UnifiedSelectorError(
                    "wearable mask PNG must be one-channel binary"
                )
            image.load()
            decoded = np.asarray(image)
    except (OSError, UnidentifiedImageError) as error:
        raise UnifiedSelectorError("wearable mask PNG full decode failed") from error
    if decoded.shape != image_shape:
        raise UnifiedSelectorError("wearable mask PNG shape mismatch")
    if decoded.dtype == np.bool_:
        binary = decoded
    elif decoded.dtype == np.uint8 and np.isin(decoded, (0, 255)).all():
        binary = decoded == 255
    else:
        raise UnifiedSelectorError("wearable mask PNG is not exactly binary")
    return _immutable_bool_mask(binary)


def _verify_bound_wearable(
    bound: BoundWearableRawInstance,
    *,
    prompt_id: str,
    session_id: str,
    frame_index: int,
    prompt_contract_ref: wearable.EvidenceRef,
    evidence_root: Path,
    image_shape: tuple[int, int],
) -> BoundWearableRawInstance:
    if not isinstance(bound, BoundWearableRawInstance):
        raise UnifiedSelectorError("unbound wearable raw instance")
    instance = bound.instance
    if (
        not isinstance(instance, wearable.RawWearableInstance)
        or not isinstance(bound.metadata_ref, wearable.EvidenceRef)
        or not isinstance(bound.mask_png_ref, wearable.EvidenceRef)
    ):
        raise UnifiedSelectorError("wearable raw binding type mismatch")
    if (
        type(instance.frame_index) is not int
        or type(instance.raw_instance_offset) is not int
        or type(instance.instance_id) is not int
        or type(instance.score) is not float
        or instance.frame_index < 0
        or instance.raw_instance_offset < 0
        or instance.instance_id < 0
    ):
        raise UnifiedSelectorError("wearable raw numeric identity is invalid")
    if (
        instance.prompt_id != prompt_id
        or instance.prompt_text != wearable.FROZEN_PROMPTS[prompt_id]
        or instance.frame_index != frame_index
    ):
        raise UnifiedSelectorError("wearable prompt/frame identity mismatch")
    if instance.prompt_contract_ref != prompt_contract_ref:
        raise UnifiedSelectorError("wearable prompt reference mismatch")
    instance.validate(image_shape)
    if bound.metadata_ref.source_kind != WEARABLE_METADATA_SOURCE_KIND:
        raise UnifiedSelectorError("wearable metadata source kind mismatch")
    if _ref_identity(bound.metadata_ref) == _ref_identity(bound.mask_png_ref):
        raise UnifiedSelectorError("wearable metadata/mask evidence alias")

    mask_from_png = _decode_verified_binary_png(
        bound.mask_png_ref,
        evidence_root=evidence_root,
        image_shape=image_shape,
    )
    if not np.array_equal(mask_from_png, instance.mask):
        raise UnifiedSelectorError("wearable PNG/instance mask mismatch")
    if wearable.mask_sha256(mask_from_png) != instance.raw_mask_sha256:
        raise UnifiedSelectorError("wearable PNG/raw mask SHA mismatch")

    metadata_bytes = bound.metadata_ref.read_verified(evidence_root)
    try:
        metadata = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UnifiedSelectorError("wearable metadata JSON decode failed") from error
    expected = {
        "source_kind": WEARABLE_METADATA_SOURCE_KIND,
        "session_id": session_id,
        "frame_index": frame_index,
        "prompt_id": prompt_id,
        "prompt_text": instance.prompt_text,
        "raw_instance_offset": instance.raw_instance_offset,
        "instance_id": instance.instance_id,
        "score": instance.score,
        "raw_mask_sha256": instance.raw_mask_sha256,
        "mask_png_ref": _mask_ref_record(bound.mask_png_ref),
    }
    if not isinstance(metadata, dict) or metadata != expected:
        raise UnifiedSelectorError("wearable metadata/instance payload mismatch")
    if (
        type(metadata["frame_index"]) is not int
        or type(metadata["raw_instance_offset"]) is not int
        or type(metadata["instance_id"]) is not int
        or type(metadata["score"]) is not float
    ):
        raise UnifiedSelectorError("wearable metadata numeric schema mismatch")

    frozen_instance = replace(instance, mask=mask_from_png)
    return BoundWearableRawInstance(
        frozen_instance,
        bound.metadata_ref,
        bound.mask_png_ref,
    )


def _preflight(
    authorities: Mapping[d1.Side, d1.SideAuthority],
    arm_raws: Sequence[d1.RawInstance],
    contact_geometry: Mapping[d1.Side, d4.ContactGeometryInput],
    wearable_frame: FrozenWearableFrame,
    *,
    evidence_root: Path,
    image_shape: tuple[int, int],
    thresholds: d1.SelectorThresholds,
) -> tuple[
    str,
    int,
    wearable.VerifiedPromptContract,
    Mapping[str, tuple[BoundWearableRawInstance, ...]],
]:
    observed_thresholds = (
        thresholds.min_joint_support_ratio,
        thresholds.min_side_margin,
        thresholds.max_object_overlap_over_instance,
    )
    if observed_thresholds != FROZEN_THRESHOLDS:
        raise UnifiedSelectorError("frozen .20/.05/.12 thresholds changed")
    if set(authorities) != set(d1.SIDES) or set(contact_geometry) != set(d1.SIDES):
        raise UnifiedSelectorError("exact left/right arm evidence required")
    if any(
        type(authority.frame_index) is not int for authority in authorities.values()
    ):
        raise UnifiedSelectorError("arm authority frame identity is not exact int")
    identities = {
        (authority.session_id, authority.frame_index)
        for authority in authorities.values()
    }
    if len(identities) != 1:
        raise UnifiedSelectorError("cross-session/frame arm authorities")
    session_id, frame_index = next(iter(identities))
    if (
        type(session_id) is not str
        or type(frame_index) is not int
        or frame_index < 0
        or wearable.SESSION_ID_PATTERN.fullmatch(session_id) is None
        or session_id == FORBIDDEN_SESSION
    ):
        raise UnifiedSelectorError("forbidden or invalid session")
    if (wearable_frame.session_id, wearable_frame.frame_index) != (
        session_id,
        frame_index,
    ) or (
        type(wearable_frame.session_id) is not str
        or type(wearable_frame.frame_index) is not int
    ):
        raise UnifiedSelectorError("cross-session/frame wearable inventory")
    if set(wearable_frame.wrist_authorities) != set(wearable.SIDES):
        raise UnifiedSelectorError("exact left/right wearable authorities required")
    if set(wearable_frame.instances_by_prompt) != set(wearable.FROZEN_PROMPTS):
        raise UnifiedSelectorError("exact frozen prompt inventory required")
    if tuple(wearable.FROZEN_PROMPTS) != FROZEN_WEARABLE_PROMPT_ORDER:
        raise UnifiedSelectorError("frozen W1-W4 prompt order changed")
    _reject_mask_aliases(
        [raw.mask for raw in arm_raws]
        + [
            bound.instance.mask
            for prompt_id in FROZEN_WEARABLE_PROMPT_ORDER
            for bound in wearable_frame.instances_by_prompt[prompt_id]
            if isinstance(bound, BoundWearableRawInstance)
        ]
    )

    all_refs: list[object] = [wearable_frame.prompt_contract_ref]
    all_refs.extend(
        authority.source_frame_selection for authority in authorities.values()
    )
    all_refs.extend(raw.raw_source for raw in arm_raws)
    for authority in wearable_frame.wrist_authorities.values():
        all_refs.extend(
            (authority.optimized_source_ref, authority.raw_hawor_source_ref)
        )
    for prompt_id in FROZEN_WEARABLE_PROMPT_ORDER:
        for bound in wearable_frame.instances_by_prompt[prompt_id]:
            if not isinstance(bound, BoundWearableRawInstance):
                raise UnifiedSelectorError("unbound wearable raw instance")
            all_refs.extend((bound.metadata_ref, bound.mask_png_ref))
    for ref in all_refs:
        _reject_forbidden_ref(ref)

    left_wrist = wearable_frame.wrist_authorities["left"]
    right_wrist = wearable_frame.wrist_authorities["right"]
    for side, authority in wearable_frame.wrist_authorities.items():
        if (
            authority.side != side
            or type(authority.frame_index) is not int
            or authority.frame_index != frame_index
        ):
            raise UnifiedSelectorError("wearable wrist authority identity mismatch")
    optimized_key = _ref_identity(left_wrist.optimized_source_ref)
    raw_hawor_key = _ref_identity(left_wrist.raw_hawor_source_ref)
    if optimized_key == raw_hawor_key:
        raise UnifiedSelectorError("HaWoR optimized/raw reference alias")
    if (
        _ref_identity(right_wrist.optimized_source_ref) != optimized_key
        or _ref_identity(right_wrist.raw_hawor_source_ref) != raw_hawor_key
    ):
        raise UnifiedSelectorError("left/right HaWoR frozen reference mismatch")
    if _ref_identity(wearable_frame.prompt_contract_ref) in {
        optimized_key,
        raw_hawor_key,
    }:
        raise UnifiedSelectorError("prompt/HaWoR reference alias")

    raw_ref_keys = [_ref_identity(raw.raw_source) for raw in arm_raws]
    if len(raw_ref_keys) != len(set(raw_ref_keys)):
        raise UnifiedSelectorError("arm raw evidence reference alias")
    for raw in arm_raws:
        if (raw.session_id, raw.frame_index) != (session_id, frame_index):
            raise UnifiedSelectorError("cross-session/frame arm raw inventory")

    root_parts = {part.lower() for part in evidence_root.resolve().parts}
    if root_parts & FORBIDDEN_PATH_COMPONENTS:
        raise UnifiedSelectorError("forbidden evidence root")
    prompt_contract = wearable.verify_prompt_contract(
        wearable_frame.prompt_contract_ref, evidence_root
    )
    # Bind both HaWoR authorities to real frozen bytes before any selection.
    left_wrist.optimized_source_ref.read_verified(evidence_root)
    left_wrist.raw_hawor_source_ref.read_verified(evidence_root)

    prepared: dict[str, tuple[BoundWearableRawInstance, ...]] = {}
    evidence_keys: list[tuple[str, int, str]] = []
    all_masks: list[np.ndarray] = [raw.mask for raw in arm_raws]
    for prompt_id in FROZEN_WEARABLE_PROMPT_ORDER:
        source_records = tuple(wearable_frame.instances_by_prompt[prompt_id])
        verified_records: list[BoundWearableRawInstance] = []
        observed_offsets: set[int] = set()
        observed_instance_ids: set[int] = set()
        for bound in source_records:
            verified = _verify_bound_wearable(
                bound,
                prompt_id=prompt_id,
                session_id=session_id,
                frame_index=frame_index,
                prompt_contract_ref=wearable_frame.prompt_contract_ref,
                evidence_root=evidence_root,
                image_shape=image_shape,
            )
            instance = verified.instance
            if instance.raw_instance_offset in observed_offsets:
                raise UnifiedSelectorError("duplicate wearable raw offset")
            observed_offsets.add(instance.raw_instance_offset)
            if instance.instance_id in observed_instance_ids:
                raise UnifiedSelectorError("duplicate wearable instance id")
            observed_instance_ids.add(instance.instance_id)
            verified_records.append(verified)
            all_masks.append(instance.mask)
            evidence_keys.extend(
                (
                    _ref_identity(verified.metadata_ref),
                    _ref_identity(verified.mask_png_ref),
                )
            )
        prepared[prompt_id] = tuple(verified_records)
    if len(evidence_keys) != len(set(evidence_keys)):
        raise UnifiedSelectorError("wearable raw evidence reference alias")
    _reject_mask_aliases(all_masks)
    return (
        session_id,
        frame_index,
        prompt_contract,
        MappingProxyType(prepared),
    )


def _selected_arm_raw(
    decision: d1.SideDecision,
    arm_raws: Sequence[d1.RawInstance],
) -> d1.RawInstance:
    matches = [
        raw
        for raw in arm_raws
        if raw.raw_instance_offset == decision.raw_instance_offset
        and raw.instance_id == decision.instance_id
    ]
    if len(matches) != 1:
        raise UnifiedSelectorError("accepted arm/raw identity mismatch")
    return matches[0]


def _arm_identity(
    side: d1.Side,
    session_id: str,
    frame_index: int,
    decision: d1.SideDecision,
    raw: d1.RawInstance,
    *,
    evidence_root: Path,
) -> wearable.AcceptedArmRawIdentity:
    payload = raw.raw_source.read_verified(allowed_root=evidence_root)
    typed_ref = wearable.EvidenceRef(
        raw.raw_source.path,
        raw.raw_source.bytes,
        raw.raw_source.sha256,
        "TASK27_SELECTED_ARM_RAW_EVIDENCE_JSON",
    )
    identity = wearable.AcceptedArmRawIdentity(
        side=side,
        session_id=session_id,
        frame_id=f"{session_id}:{frame_index}",
        frame_index=frame_index,
        status="ACCEPT",
        selected_offset=raw.raw_instance_offset,
        selected_instance_id=raw.instance_id,
        selected_raw_evidence_ref=typed_ref,
        selected_raw_evidence_bytes=payload,
        selected_candidate_eligible=True,
        selected_candidate_rejection_reasons=tuple(),
        raw_mask_sha256=d1.mask_sha256(raw.mask),
    )
    if decision.mask is None:
        raise UnifiedSelectorError("accepted arm decision has no raw mask")
    identity.verify(decision.mask)
    return identity


def _freeze_side_decision(decision: d1.SideDecision) -> d1.SideDecision:
    mask = None
    if decision.mask is not None:
        mask = _immutable_bool_mask(decision.mask)
    return replace(decision, mask=mask)


def _freeze_frame_selection(selection: d1.FrameSelection) -> d1.FrameSelection:
    return replace(
        selection,
        left=_freeze_side_decision(selection.left),
        right=_freeze_side_decision(selection.right),
    )


def _freeze_contact_selection(
    contact: d4.ContactAwareSelection,
) -> d4.ContactAwareSelection:
    base = _freeze_frame_selection(contact.base_selection)
    selection = (
        base
        if contact.selection is contact.base_selection
        else _freeze_frame_selection(contact.selection)
    )
    return replace(
        contact,
        base_selection=base,
        selection=selection,
        contact_measurements=MappingProxyType(dict(contact.contact_measurements)),
        application_audits=MappingProxyType(dict(contact.application_audits)),
    )


def select_frame(
    authorities: Mapping[d1.Side, d1.SideAuthority],
    shared_arm_raw_inventory: Sequence[d1.RawInstance],
    contact_geometry: Mapping[d1.Side, d4.ContactGeometryInput],
    wearable_frame: FrozenWearableFrame,
    *,
    evidence_root: Path,
    image_shape: tuple[int, int],
    thresholds: d1.SelectorThresholds = d1.SelectorThresholds(),
) -> UnifiedFrameSelection:
    """Select one generic session frame in fixed D1 -> D4 -> union order."""

    if set(authorities) != set(d1.SIDES) or set(contact_geometry) != set(d1.SIDES):
        raise UnifiedSelectorError("exact left/right arm evidence required")
    if set(wearable_frame.wrist_authorities) != set(wearable.SIDES):
        raise UnifiedSelectorError("exact left/right wearable authorities required")
    arm_raws = tuple(shared_arm_raw_inventory)
    authority_snapshot = MappingProxyType(
        {side: authorities[side] for side in d1.SIDES}
    )
    geometry_snapshot = MappingProxyType(
        {side: contact_geometry[side] for side in d1.SIDES}
    )
    wrist_snapshot = MappingProxyType(
        {side: wearable_frame.wrist_authorities[side] for side in wearable.SIDES}
    )
    wearable_frame_snapshot = replace(
        wearable_frame,
        wrist_authorities=wrist_snapshot,
    )
    session_id, frame_index, prompt_contract, bound_by_prompt = _preflight(
        authority_snapshot,
        arm_raws,
        geometry_snapshot,
        wearable_frame_snapshot,
        evidence_root=evidence_root,
        image_shape=image_shape,
        thresholds=thresholds,
    )
    # D4 calls D1 exactly once and receives this exact sequence object.
    contact = _freeze_contact_selection(
        d4.select_frame(
            authority_snapshot,
            arm_raws,
            geometry_snapshot,
            evidence_root=evidence_root,
            image_shape=image_shape,
            thresholds=thresholds,
        )
    )

    decisions: dict[str, Mapping[wearable.Side, wearable.SideWearableDecision]] = {}
    for prompt_id in FROZEN_WEARABLE_PROMPT_ORDER:
        prompt_decisions = wearable.select_wearable_per_side(
            tuple(bound.instance for bound in bound_by_prompt[prompt_id]),
            {side: wrist_snapshot[side] for side in wearable.SIDES},
            image_shape,
            prompt_contract=prompt_contract,
        )
        if set(prompt_decisions) != set(wearable.SIDES):
            raise UnifiedSelectorError("wearable selector side result drift")
        decisions[prompt_id] = MappingProxyType(
            {side: prompt_decisions[side] for side in wearable.SIDES}
        )

    sides: dict[d1.Side, FinalSideSelection] = {}
    for side in d1.SIDES:
        arm_decision = getattr(contact.selection, side)
        if arm_decision.status != "ACCEPT":
            sides[side] = FinalSideSelection(
                side,
                arm_decision.status,
                None,
                tuple(),
                None,
                None,
                0,
                0,
            )
            continue
        arm_raw = _selected_arm_raw(arm_decision, arm_raws)
        identity = _arm_identity(
            side,
            session_id,
            frame_index,
            arm_decision,
            arm_raw,
            evidence_root=evidence_root,
        )
        arm_mask = np.asarray(arm_decision.mask, dtype=np.bool_)
        final_mask = arm_mask.copy()
        selected: list[SelectedWearableIdentity] = []
        for prompt_id in FROZEN_WEARABLE_PROMPT_ORDER:
            decision = decisions[prompt_id][side]
            if decision.status != "ACCEPT":
                continue
            candidates = [
                bound
                for bound in bound_by_prompt[prompt_id]
                if bound.instance.raw_instance_offset == decision.selected_offset
                and bound.instance.instance_id == decision.selected_instance_id
                and bound.instance.raw_mask_sha256 == decision.selected_raw_mask_sha256
            ]
            if len(candidates) != 1:
                raise UnifiedSelectorError("accepted wearable/raw identity mismatch")
            bound = candidates[0]
            instance = bound.instance
            pair_union = wearable.identity_union(
                arm_mask,
                instance.mask,
                side=side,
                arm_identity=identity,
                wearable_decision=decision,
                wearable_instance=instance,
            )
            final_mask = np.logical_or(final_mask, pair_union)
            selected.append(
                SelectedWearableIdentity(
                    prompt_id=prompt_id,
                    prompt_text=instance.prompt_text,
                    prompt_contract_sha256=instance.prompt_contract_ref.sha256,
                    raw_instance_offset=instance.raw_instance_offset,
                    instance_id=instance.instance_id,
                    raw_mask_sha256=instance.raw_mask_sha256,
                    metadata_ref=bound.metadata_ref,
                    mask_png_ref=bound.mask_png_ref,
                    added_pixels_against_arm=int(
                        np.count_nonzero(instance.mask & ~arm_mask)
                    ),
                )
            )
        allowed = arm_mask.copy()
        for item in selected:
            matches = [
                bound.instance
                for bound in bound_by_prompt[item.prompt_id]
                if bound.instance.raw_instance_offset == item.raw_instance_offset
                and bound.instance.instance_id == item.instance_id
                and bound.instance.raw_mask_sha256 == item.raw_mask_sha256
            ]
            if len(matches) != 1:
                raise UnifiedSelectorError("selected wearable identity drift")
            allowed = np.logical_or(allowed, matches[0].mask)
        if not np.array_equal(final_mask, allowed):
            raise UnifiedSelectorError("final H manufactured pixels")
        added = int(np.count_nonzero(final_mask & ~arm_mask))
        frozen_final_mask = _immutable_bool_mask(final_mask)
        sides[side] = FinalSideSelection(
            side,
            arm_decision.status,
            identity,
            tuple(selected),
            frozen_final_mask,
            d1.mask_sha256(frozen_final_mask),
            added,
            len(selected),
        )

    return UnifiedFrameSelection(
        session_id,
        frame_index,
        contact,
        MappingProxyType(decisions),
        MappingProxyType(sides),
        prompt_contract_ref=wearable_frame_snapshot.prompt_contract_ref,
        optimized_hawor_source_ref=wrist_snapshot["left"].optimized_source_ref,
        raw_hawor_source_ref=wrist_snapshot["left"].raw_hawor_source_ref,
        wrist_authorities=wrist_snapshot,
    )
