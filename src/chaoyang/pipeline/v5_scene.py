"""Pixel-domain-only masks and compositing for the V5 offline visual Scene.

Every input mask is an observation candidate from the original RGB domain.
The generated clean image has no geometry or contact authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


HUMAN_ROLES = frozenset({"human", "human_hand", "hand", "human_forearm", "forearm", "sleeve"})
DEVICE_ROLES = frozenset({"device", "capture_device", "tracker", "wristband", "glove", "cable"})
OBJECT_ROLES = frozenset({"object", "task_object", "object_visible"})


@dataclass(frozen=True)
class FrameRoles:
    human: np.ndarray
    device: np.ndarray
    object_visible: np.ndarray

    def __post_init__(self) -> None:
        shape = self.human.shape
        if len(shape) != 2 or self.device.shape != shape or self.object_visible.shape != shape:
            raise ValueError("ROLE_MASK_SHAPE")
        if any(a.dtype != np.bool_ for a in (self.human, self.device, self.object_visible)):
            raise ValueError("ROLE_MASK_DTYPE")


def merge_roles(role_masks: dict[str, np.ndarray], shape: tuple[int, int]) -> FrameRoles:
    """Merge semantic roles while keeping human, equipment and objects separate."""
    merged = [np.zeros(shape, dtype=bool) for _ in range(3)]
    for role, label_image in role_masks.items():
        value = np.asarray(label_image)
        if value.shape != shape or value.ndim != 2:
            raise ValueError(f"ROLE_MASK_DOMAIN:{role}")
        normalized = role.lower().replace("-", "_")
        if normalized in HUMAN_ROLES:
            index = 0
        elif normalized in DEVICE_ROLES:
            index = 1
        elif normalized in OBJECT_ROLES:
            index = 2
        else:
            raise ValueError(f"UNKNOWN_ROLE:{role}")
        merged[index] |= value > 0
    return FrameRoles(*merged)


def _component_hulls(mask: np.ndarray) -> np.ndarray:
    """Fill holes within each connected component without joining two hands."""
    value = np.asarray(mask, dtype=np.uint8)
    count, labels = cv2.connectedComponents(value, connectivity=8)
    out = np.zeros(value.shape, dtype=np.uint8)
    for component in range(1, count):
        yy, xx = np.where(labels == component)
        if len(xx) < 3:
            out[yy, xx] = 1
        else:
            hull = cv2.convexHull(np.column_stack((xx, yy)).astype(np.int32))
            cv2.fillConvexPoly(out, hull, 1)
    return out > 0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if not 0 <= radius <= 16:
        raise ValueError("SUPPORT_RADIUS")
    if radius == 0:
        return mask.copy()
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def build_scene_masks(
    roles: list[FrameRoles], *, temporal_radius: int = 2,
    support_margin: int = 8, trusted_object_frames: Iterable[int] = (),
) -> list[dict[str, np.ndarray | dict[str, int | bool]]]:
    """Build write, context, visible protection and uncertainty in one pixel domain.

    The temporal union is only a removal/context support. Object pixels from
    other frames never become protection in the target frame.
    """
    if not roles or not 0 <= temporal_radius <= 2:
        raise ValueError("SCENE_WINDOW")
    shape = roles[0].human.shape
    if any(r.human.shape != shape for r in roles):
        raise ValueError("SCENE_DOMAIN_MISMATCH")
    trusted = set(trusted_object_frames)
    if any(not 0 <= i < len(roles) for i in trusted):
        raise ValueError("TRUSTED_FRAME_RANGE")
    out = []
    for i in range(len(roles)):
        start = max(0, i - temporal_radius)
        stop = min(len(roles), i + temporal_radius + 1)
        out.append(build_scene_mask_window(roles[start:stop], i - start, i,
                                           i in trusted, support_margin=support_margin))
    return out


def build_scene_mask_window(window: list[FrameRoles], center: int, frame_id: int,
                            trusted_object: bool, *, support_margin: int = 8
                            ) -> dict[str, np.ndarray | dict[str, int | bool]]:
    if not window or not 0 <= center < len(window) or len(window) > 5:
        raise ValueError("SCENE_WINDOW")
    r = window[center]
    if any(item.human.shape != r.human.shape for item in window):
        raise ValueError("SCENE_DOMAIN_MISMATCH")
    support = [
        _dilate(_component_hulls(item.human) | _component_hulls(item.device), support_margin)
        for item in window
    ]
    temporal_support = np.logical_or.reduce(support)
    protected = r.object_visible.copy() if trusted_object else np.zeros(r.human.shape, dtype=bool)
    write = temporal_support & ~protected
    temporal_objects = np.logical_or.reduce([item.object_visible for item in window])
    temporal_only_object = temporal_objects & ~r.object_visible
    conflicting_visible_object = r.object_visible & (r.human | r.device)
    unknown = temporal_only_object | conflicting_visible_object
    if not trusted_object:
        unknown |= r.object_visible
    context_exclude = temporal_support.copy()
    if np.any((r.human | r.device) & ~context_exclude):
        raise AssertionError("UNCOVERED_CONTEXT_FOREGROUND")
    if np.any(write & protected):
        raise AssertionError("PROTECTED_WRITE")
    return {
        "write": write, "protect": protected,
        "context_exclude": context_exclude, "unknown": unknown,
        "stats": {
            "frame_id": frame_id,
            "human_px": int(r.human.sum()),
            "device_px": int(r.device.sum()),
            "object_visible_px": int(r.object_visible.sum()),
            "trusted_object": trusted_object,
            "write_px": int(write.sum()),
            "protect_px": int(protected.sum()),
            "context_exclude_px": int(context_exclude.sum()),
            "temporal_object_only_px": int(temporal_only_object.sum()),
            "object_foreground_conflict_px": int(conflicting_visible_object.sum()),
            "unknown_px": int(unknown.sum()),
        },
    }


def build_conservative_repair_window(
    window: list[FrameRoles], center: int, frame_id: int, *, support_margin: int = 4
) -> dict[str, np.ndarray | dict[str, int | bool]]:
    """Build a semantic-base removal candidate with bounded local repair.

    Unlike :func:`build_scene_mask_window`, this candidate never takes a
    convex hull and never unions an entire neighbouring foreground into the
    current frame.  A neighbour can only repair pixels close to the current
    semantic foreground.  This is deliberately conservative and remains a
    candidate until visible-object protection and visual quality are proven.
    """
    if not window or not 0 <= center < len(window) or len(window) > 5:
        raise ValueError("SCENE_WINDOW")
    if not 0 <= support_margin <= 8:
        raise ValueError("CONSERVATIVE_SUPPORT_MARGIN")
    current = window[center]
    if any(item.human.shape != current.human.shape for item in window):
        raise ValueError("SCENE_DOMAIN_MISMATCH")
    semantic = current.human | current.device
    base = _dilate(semantic, support_margin)
    near = _dilate(semantic, min(16, support_margin + 4))
    neighbour = np.logical_or.reduce([item.human | item.device for item in window])
    repair = neighbour & near & ~base
    write = base | repair
    # No task-object observation is promoted to trusted protection here.
    protect = np.zeros(current.human.shape, dtype=bool)
    unknown = current.object_visible | (write & current.object_visible)
    context_exclude = write.copy()
    if np.any(semantic & ~context_exclude):
        raise AssertionError("UNCOVERED_CURRENT_SEMANTIC_FOREGROUND")
    return {
        "write": write,
        "protect": protect,
        "context_exclude": context_exclude,
        "unknown": unknown,
        "base": base,
        "repair": repair,
        "stats": {
            "frame_id": frame_id,
            "human_px": int(current.human.sum()),
            "device_px": int(current.device.sum()),
            "object_visible_px": int(current.object_visible.sum()),
            "base_px": int(base.sum()),
            "repair_px": int(repair.sum()),
            "write_px": int(write.sum()),
            "object_overlap_px": int((write & current.object_visible).sum()),
            "trusted_object": False,
        },
    }


def build_object_protected_repair_window(
    window: list[FrameRoles], center: int, frame_id: int, *,
    support_margin: int = 4, conflict_margin: int = 1,
) -> dict[str, np.ndarray | dict[str, int | bool]]:
    """Conservative repair with direct-visible object interiors protected.

    The object role is allowed to *select* pixels already observed in the
    current frame, but never to synthesize an occluded object.  Pixels where
    object evidence touches current human/device evidence stay writable and
    UNKNOWN; only the directly visible object interior outside a one-pixel
    foreground conflict band is protected.  This avoids the invalid rule
    ``erase = erase - all(object)`` at true hand/object occlusions.
    """
    if not 0 <= conflict_margin <= support_margin:
        raise ValueError("OBJECT_CONFLICT_MARGIN")
    candidate = build_conservative_repair_window(
        window, center, frame_id, support_margin=support_margin,
    )
    current = window[center]
    semantic = current.human | current.device
    conflict_band = _dilate(semantic, conflict_margin)
    # Erosion removes uncertain object boundaries before any pixel receives
    # protection authority. A tiny object may therefore remain unprotected.
    kernel = np.ones((3, 3), np.uint8)
    object_interior = cv2.erode(current.object_visible.astype(np.uint8), kernel) > 0
    protect = object_interior & ~conflict_band
    write = np.asarray(candidate["write"], bool) & ~protect
    conflict = current.object_visible & conflict_band
    unknown = np.asarray(candidate["unknown"], bool) | conflict
    context_exclude = write.copy()
    if np.any(write & protect):
        raise AssertionError("PROTECTED_WRITE")
    if np.any(semantic & ~context_exclude & ~protect):
        raise AssertionError("UNCOVERED_CURRENT_SEMANTIC_FOREGROUND")
    stats = dict(candidate["stats"])
    stats.update({
        "protect_px": int(protect.sum()),
        "write_px": int(write.sum()),
        "object_conflict_px": int(conflict.sum()),
        "protected_object_write_reduction_px": int(
            (np.asarray(candidate["write"], bool) & protect).sum()
        ),
        "trusted_object": "CURRENT_FRAME_ERODED_INTERIOR_ONLY",
    })
    return {
        **candidate,
        "write": write,
        "protect": protect,
        "context_exclude": context_exclude,
        "unknown": unknown,
        "stats": stats,
    }


def composite_clean(raw: np.ndarray, generated: np.ndarray,
                    write: np.ndarray, protect: np.ndarray) -> np.ndarray:
    """Hard paste only within M_write in the final output pixel domain."""
    if raw.shape != generated.shape or raw.shape[:2] != write.shape or write.shape != protect.shape:
        raise ValueError("CLEAN_DOMAIN_MISMATCH")
    if np.any(write & protect):
        raise ValueError("PROTECTED_WRITE")
    out = raw.copy()
    out[write] = generated[write]
    if not np.array_equal(out[~write], raw[~write]):
        raise AssertionError("OUTSIDE_WRITE_CHANGED")
    if not np.array_equal(out[protect], raw[protect]):
        raise AssertionError("PROTECTED_CHANGED")
    return out
