"""Offline Robot display admission with explicit source and fallback semantics.

This module decides whether each anatomical side may be replaced by a rendered
Robot.  It never edits RGB pixels.  Consumers must preserve the original pixels
for every side that is not admitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Iterable


SCHEMA_VERSION = "FINAL_FRAME_DISPOSITION_V1"
ANATOMICAL_SIDES = ("left", "right")


class MotionSource(str, Enum):
    SENSOR_OBSERVATION = "SENSOR_OBSERVATION"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    TIME_INTERPOLATION = "TIME_INTERPOLATION"
    HOLD = "HOLD"
    COPIED_OTHER_SIDE = "COPIED_OTHER_SIDE"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


class SideDisplay(str, Enum):
    ROBOT_REPLACE = "ROBOT_REPLACE"
    ORIGINAL_PIXELS = "ORIGINAL_PIXELS"
    NO_VISIBLE_TARGET = "NO_VISIBLE_TARGET"


class FrameDisplay(str, Enum):
    ROBOT = "ROBOT"
    PARTIAL_PRESERVE = "PARTIAL_PRESERVE"
    ORIGINAL_FRAME_FALLBACK = "ORIGINAL_FRAME_FALLBACK"


@dataclass(frozen=True)
class SideEvidence:
    frame_id: int
    side: str
    replacement_required: bool
    motion_source: MotionSource | str
    source_valid: bool
    quality_pass: bool
    renderer_valid: bool
    source_artifact_sha256: str | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class SideDecision:
    side: str
    display: SideDisplay
    motion_source: MotionSource
    reason: str
    source_artifact_sha256: str | None


@dataclass(frozen=True)
class FrameDecision:
    schema_version: str
    frame_id: int
    display: FrameDisplay
    sides: tuple[SideDecision, ...]
    preserve_original_pixels: bool
    robot_replaced_sides: tuple[str, ...]
    original_preserved_sides: tuple[str, ...]
    offline_visualization_only: bool = True
    training_authorized: bool = False


def _source(value: MotionSource | str) -> MotionSource:
    try:
        return value if isinstance(value, MotionSource) else MotionSource(value)
    except ValueError as exc:
        raise ValueError(f"UNKNOWN_MOTION_SOURCE:{value}") from exc


def decide_side(evidence: SideEvidence) -> SideDecision:
    if evidence.side not in ANATOMICAL_SIDES:
        raise ValueError("ANATOMICAL_SIDE_REQUIRED")
    if type(evidence.frame_id) is not int or evidence.frame_id < 0:
        raise ValueError("NONNEGATIVE_INTEGER_FRAME_REQUIRED")
    for value in (
        evidence.replacement_required,
        evidence.source_valid,
        evidence.quality_pass,
        evidence.renderer_valid,
    ):
        if type(value) is not bool:
            raise ValueError("BOOLEAN_EVIDENCE_REQUIRED")
    source = _source(evidence.motion_source)
    if not evidence.replacement_required:
        return SideDecision(
            evidence.side,
            SideDisplay.NO_VISIBLE_TARGET,
            source,
            "NO_VISIBLE_TARGET_IN_RAW_FRAME",
            evidence.source_artifact_sha256,
        )
    if not evidence.source_valid:
        reason = evidence.failure_reason or "SOURCE_INVALID_OR_MISSING"
    elif source not in {MotionSource.SENSOR_OBSERVATION, MotionSource.MODEL_INFERENCE}:
        reason = f"SOURCE_KIND_NOT_ADMITTED:{source.value}"
    elif not evidence.quality_pass:
        reason = evidence.failure_reason or "INDEPENDENT_QUALITY_NOT_PASSED"
    elif not evidence.renderer_valid:
        reason = evidence.failure_reason or "ROBOT_RENDER_NOT_VALID"
    elif not evidence.source_artifact_sha256 or re.fullmatch(
        r"[0-9a-f]{64}", evidence.source_artifact_sha256
    ) is None:
        reason = "SOURCE_ARTIFACT_SHA256_REQUIRED"
    else:
        return SideDecision(
            evidence.side,
            SideDisplay.ROBOT_REPLACE,
            source,
            "QUALIFIED_OFFLINE_ROBOT_REPLACEMENT",
            evidence.source_artifact_sha256,
        )
    return SideDecision(
        evidence.side,
        SideDisplay.ORIGINAL_PIXELS,
        source,
        reason,
        evidence.source_artifact_sha256,
    )


def decide_frame(
    evidence: Iterable[SideEvidence],
    *,
    compositor_inputs_valid: bool,
) -> FrameDecision:
    rows = tuple(evidence)
    if len(rows) != 2 or {row.side for row in rows} != set(ANATOMICAL_SIDES):
        raise ValueError("EXACTLY_ONE_ROW_PER_ANATOMICAL_SIDE_REQUIRED")
    if len({row.frame_id for row in rows}) != 1:
        raise ValueError("FRAME_ID_MISMATCH")
    if type(compositor_inputs_valid) is not bool:
        raise ValueError("BOOLEAN_COMPOSITOR_VALIDITY_REQUIRED")
    decisions = tuple(sorted((decide_side(row) for row in rows), key=lambda row: ANATOMICAL_SIDES.index(row.side)))
    frame_id = rows[0].frame_id
    if not compositor_inputs_valid:
        decisions = tuple(
            SideDecision(
                row.side,
                SideDisplay.ORIGINAL_PIXELS if row.display is SideDisplay.ROBOT_REPLACE else row.display,
                row.motion_source,
                "COMPOSITOR_INPUTS_INVALID" if row.display is SideDisplay.ROBOT_REPLACE else row.reason,
                row.source_artifact_sha256,
            )
            for row in decisions
        )
    replaced = tuple(row.side for row in decisions if row.display is SideDisplay.ROBOT_REPLACE)
    preserved = tuple(row.side for row in decisions if row.display is SideDisplay.ORIGINAL_PIXELS)
    display = (
        FrameDisplay.ORIGINAL_FRAME_FALLBACK
        if not replaced
        else FrameDisplay.PARTIAL_PRESERVE
        if preserved
        else FrameDisplay.ROBOT
    )
    return FrameDecision(
        schema_version=SCHEMA_VERSION,
        frame_id=frame_id,
        display=display,
        sides=decisions,
        preserve_original_pixels=bool(preserved),
        robot_replaced_sides=replaced,
        original_preserved_sides=preserved,
    )


def decide_timeline(
    frames: Iterable[Iterable[SideEvidence]],
    *,
    compositor_inputs_valid: Iterable[bool],
) -> tuple[FrameDecision, ...]:
    frame_rows = tuple(tuple(rows) for rows in frames)
    compositor = tuple(compositor_inputs_valid)
    if len(frame_rows) != len(compositor):
        raise ValueError("TIMELINE_LENGTH_MISMATCH")
    if not frame_rows:
        raise ValueError("EMPTY_TIMELINE")
    decisions = tuple(
        decide_frame(rows, compositor_inputs_valid=valid)
        for rows, valid in zip(frame_rows, compositor)
    )
    ids = [item.frame_id for item in decisions]
    if any(right <= left for left, right in zip(ids, ids[1:])):
        raise ValueError("FRAME_IDS_MUST_BE_STRICTLY_INCREASING")
    return decisions
