"""Causal Robotized-RGB assembly with explicit pixel provenance.

This module is deliberately smaller than a renderer.  It consumes frozen
ownership and Robot-render outputs and enforces the training-time information
boundary around object appearance donors.  In particular, Clean RGB is only a
background layer and can never be selected as object appearance.

``CAUSAL_TRAINING_INPUT`` may use the current Raw object pixels, temporal
object donors whose frame id is no later than the target frame, and a trusted
textured renderer.  ``OFFLINE_BIDIRECTIONAL_VISUALIZATION`` may inspect future
donors, but its result is never training-authorized.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Sequence

import numpy as np

from .occlusion_compositor_v1 import ObjectPixelSource, Ownership


SCHEMA_VERSION = "ROBOTIZED_COMPOSITOR_CAUSAL_V1"
UNKNOWN_SENTINEL_RGB = np.asarray([255, 0, 255], dtype=np.uint8)


class CausalCompositorError(ValueError):
    """Raised when an input could violate provenance or causality."""


class VisualInputMode(str, Enum):
    CAUSAL_TRAINING_INPUT = "CAUSAL_TRAINING_INPUT"
    OFFLINE_BIDIRECTIONAL_VISUALIZATION = "OFFLINE_BIDIRECTIONAL_VISUALIZATION"


@dataclass(frozen=True)
class TemporalObjectDonor:
    """One same-session object appearance candidate.

    ``rgb`` and ``valid_mask`` are deliberately not inspected until the donor
    passes the mode/frame-id eligibility gate.  Consequently a future frame's
    bytes cannot affect a causal target, including through validation paths.
    """

    frame_id: int
    rgb: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class CausalRobotizedFrame:
    target_frame_id: int
    mode: VisualInputMode
    rgb: np.ndarray
    effective_ownership: np.ndarray
    training_valid_mask: np.ndarray
    object_pixel_source: np.ndarray
    object_donor_frame_id: np.ndarray
    referenced_donor_frame_ids: tuple[int, ...]
    training_authorized: bool
    rgb_sha256: str
    training_valid_mask_sha256: str
    clean_pixels_may_supply_object: bool = False


def _rgb(value: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (*shape, 3) or result.dtype != np.uint8:
        raise CausalCompositorError(f"{label} must be uint8 with shape {(*shape, 3)}")
    return result


def _mask(value: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.bool_:
        raise CausalCompositorError(f"{label} must be bool with shape {shape}")
    return result


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _eligible_donors(
    donors: Sequence[TemporalObjectDonor],
    *,
    target_frame_id: int,
    mode: VisualInputMode,
) -> tuple[TemporalObjectDonor, ...]:
    ids = [int(item.frame_id) for item in donors]
    if len(ids) != len(set(ids)):
        raise CausalCompositorError("temporal object donor frame ids must be unique")
    if any(frame_id < 0 for frame_id in ids):
        raise CausalCompositorError("temporal object donor frame ids must be non-negative")
    if mode is VisualInputMode.CAUSAL_TRAINING_INPUT:
        # Most recent historical evidence wins.  Future candidates are filtered
        # using metadata only; their arrays are never read or validated.
        eligible = [item for item in donors if int(item.frame_id) <= target_frame_id]
        return tuple(sorted(eligible, key=lambda item: int(item.frame_id), reverse=True))
    # Offline visualization may use either direction.  The deterministic order
    # is nearest in time, then the earlier frame for equal distance.
    return tuple(
        sorted(
            donors,
            key=lambda item: (abs(int(item.frame_id) - target_frame_id), int(item.frame_id)),
        )
    )


def compose_robotized_frame(
    *,
    target_frame_id: int,
    mode: VisualInputMode | str,
    clean_background_rgb: np.ndarray,
    ownership: np.ndarray,
    ownership_training_valid_mask: np.ndarray,
    current_raw_rgb: np.ndarray,
    current_raw_object_visible_mask: np.ndarray,
    robot_rgb: np.ndarray,
    robot_valid_mask: np.ndarray,
    temporal_object_donors: Sequence[TemporalObjectDonor] = (),
    renderer_rgb: np.ndarray | None = None,
    renderer_valid_mask: np.ndarray | None = None,
) -> CausalRobotizedFrame:
    """Compose one Robotized frame under a frozen causal/offline mode.

    Missing legal object appearance and missing Robot render pixels are changed
    to ``TIE_UNKNOWN`` and rendered with an explicit sentinel.  They never fall
    through to the Clean background while being claimed as object/Robot pixels.
    """

    if int(target_frame_id) != target_frame_id or target_frame_id < 0:
        raise CausalCompositorError("target_frame_id must be a non-negative integer")
    target_frame_id = int(target_frame_id)
    try:
        frozen_mode = mode if isinstance(mode, VisualInputMode) else VisualInputMode(mode)
    except ValueError as exc:
        raise CausalCompositorError(f"unknown visual input mode: {mode}") from exc

    owner = np.asarray(ownership)
    if owner.ndim != 2 or not np.issubdtype(owner.dtype, np.integer):
        raise CausalCompositorError("ownership must be a 2D integer image")
    shape = owner.shape
    legal_ownership = {int(value) for value in Ownership}
    if not set(np.unique(owner).tolist()).issubset(legal_ownership):
        raise CausalCompositorError("ownership contains an unknown value")
    owner_valid = _mask(ownership_training_valid_mask, shape, "ownership_training_valid_mask")
    clean = _rgb(clean_background_rgb, shape, "clean_background_rgb")
    current_raw = _rgb(current_raw_rgb, shape, "current_raw_rgb")
    current_visible = _mask(
        current_raw_object_visible_mask, shape, "current_raw_object_visible_mask"
    )
    robot = _rgb(robot_rgb, shape, "robot_rgb")
    robot_valid = _mask(robot_valid_mask, shape, "robot_valid_mask")

    object_rgb = np.zeros_like(clean)
    object_source = np.full(shape, int(ObjectPixelSource.NONE_UNKNOWN), dtype=np.uint8)
    donor_frame_id = np.full(shape, -1, dtype=np.int64)
    object_rgb[current_visible] = current_raw[current_visible]
    object_source[current_visible] = int(ObjectPixelSource.RAW_VISIBLE)
    donor_frame_id[current_visible] = target_frame_id

    referenced: list[int] = []
    for donor in _eligible_donors(
        temporal_object_donors, target_frame_id=target_frame_id, mode=frozen_mode
    ):
        donor_id = int(donor.frame_id)
        donor_rgb = _rgb(donor.rgb, shape, f"temporal donor {donor_id} rgb")
        donor_valid = _mask(
            donor.valid_mask, shape, f"temporal donor {donor_id} valid_mask"
        )
        select = donor_valid & (object_source == int(ObjectPixelSource.NONE_UNKNOWN))
        if np.any(select):
            object_rgb[select] = donor_rgb[select]
            object_source[select] = int(ObjectPixelSource.TEMPORAL_OBJECT_DONOR)
            donor_frame_id[select] = donor_id
            referenced.append(donor_id)

    if renderer_rgb is None and renderer_valid_mask is None:
        pass
    elif renderer_rgb is None or renderer_valid_mask is None:
        raise CausalCompositorError("renderer RGB and valid mask must be supplied together")
    else:
        renderer = _rgb(renderer_rgb, shape, "renderer_rgb")
        renderer_valid = _mask(renderer_valid_mask, shape, "renderer_valid_mask")
        select = renderer_valid & (object_source == int(ObjectPixelSource.NONE_UNKNOWN))
        object_rgb[select] = renderer[select]
        object_source[select] = int(ObjectPixelSource.TEXTURED_OBJECT_RENDERER)

    effective_owner = owner.astype(np.uint8, copy=True)
    object_front = effective_owner == int(Ownership.OBJECT_FRONT)
    legal_object = object_source != int(ObjectPixelSource.NONE_UNKNOWN)
    effective_owner[object_front & ~legal_object] = int(Ownership.TIE_UNKNOWN)
    robot_front = effective_owner == int(Ownership.ROBOT_FRONT)
    effective_owner[robot_front & ~robot_valid] = int(Ownership.TIE_UNKNOWN)

    output = clean.copy()
    effective_object_front = effective_owner == int(Ownership.OBJECT_FRONT)
    effective_robot_front = effective_owner == int(Ownership.ROBOT_FRONT)
    output[effective_object_front] = object_rgb[effective_object_front]
    output[effective_robot_front] = robot[effective_robot_front]
    unknown = effective_owner == int(Ownership.TIE_UNKNOWN)
    output[unknown] = UNKNOWN_SENTINEL_RGB

    training_authorized = frozen_mode is VisualInputMode.CAUSAL_TRAINING_INPUT
    training_valid = owner_valid & ~unknown
    if not training_authorized:
        # Offline bidirectional products are QA/visualization only, even when
        # every pixel can be ordered geometrically.
        training_valid[:] = False

    causal_references = tuple(sorted(set(referenced)))
    if training_authorized and any(frame_id > target_frame_id for frame_id in causal_references):
        raise CausalCompositorError("causal output references a future donor")
    # A final defensive invariant: no object pixel can have NONE provenance.
    if np.any(effective_object_front & (object_source == int(ObjectPixelSource.NONE_UNKNOWN))):
        raise CausalCompositorError("OBJECT_FRONT lacks legal non-Clean appearance provenance")

    return CausalRobotizedFrame(
        target_frame_id=target_frame_id,
        mode=frozen_mode,
        rgb=output,
        effective_ownership=effective_owner,
        training_valid_mask=training_valid,
        object_pixel_source=object_source,
        object_donor_frame_id=donor_frame_id,
        referenced_donor_frame_ids=causal_references,
        training_authorized=training_authorized,
        rgb_sha256=_sha256_array(output),
        training_valid_mask_sha256=_sha256_array(training_valid),
    )


def assert_causal_frame_contract(frame: CausalRobotizedFrame) -> None:
    """Validate a published training frame without trusting its producer."""

    if frame.clean_pixels_may_supply_object:
        raise CausalCompositorError("Clean/background pixels may not supply object appearance")
    unknown = frame.effective_ownership == int(Ownership.TIE_UNKNOWN)
    if np.any(frame.training_valid_mask[unknown]):
        raise CausalCompositorError("UNKNOWN pixels must have training_valid_mask=false")
    if frame.mode is VisualInputMode.CAUSAL_TRAINING_INPUT:
        if not frame.training_authorized:
            raise CausalCompositorError("causal frame must be explicitly training-authorized")
        if any(item > frame.target_frame_id for item in frame.referenced_donor_frame_ids):
            raise CausalCompositorError("causal frame references a future donor")
    elif frame.training_authorized or np.any(frame.training_valid_mask):
        raise CausalCompositorError("offline bidirectional frame cannot be used for training")
