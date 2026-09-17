"""Geometry-only adapter from HaWoR and Object6D to contact hypotheses.

The adapter consumes camera-frame MANO joints and a *separate* Object6D pose
hypothesis timeline.  It does not read Clean, Robot state, or rendered pixels.
Only a fingertip-to-oriented-box digital distance is computed; this is
development evidence and is not physical contact truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import numpy as np


class ContactInputAdapterError(ValueError):
    """Raised when frame, side, object, or coordinate axes do not close."""


class ContactState(IntEnum):
    UNKNOWN = 0
    APPROACH = 1
    TOUCH = 2
    GRASP = 3
    RELEASE = 4


@dataclass(frozen=True)
class DirectTouchEvidence:
    tip_signed_distance_m: np.ndarray
    finger_contact: np.ndarray
    contact_state: np.ndarray
    contact_confidence: np.ndarray


def signed_distance_points_to_oriented_box(
    points_camera: np.ndarray,
    object_to_camera: np.ndarray,
    object_size_m: np.ndarray,
) -> np.ndarray:
    """Return exact point-to-box SDF in metres for one oriented cuboid."""

    points = np.asarray(points_camera, dtype=np.float64)
    transform = np.asarray(object_to_camera, dtype=np.float64)
    size = np.asarray(object_size_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ContactInputAdapterError("points_camera must be finite (N,3)")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ContactInputAdapterError("object_to_camera must be finite (4,4)")
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0.0):
        raise ContactInputAdapterError("object_size_m must be positive finite (3,)")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ContactInputAdapterError("object rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ContactInputAdapterError("object rotation must be proper")
    local = (points - transform[:3, 3]) @ rotation
    delta = np.abs(local) - size[None] * 0.5
    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
    inside = np.minimum(np.max(delta, axis=1), 0.0)
    return outside + inside


def build_direct_touch_evidence(
    joints_camera: np.ndarray,
    hand_observed: np.ndarray,
    pose_object_to_camera: np.ndarray,
    pose_valid: np.ndarray,
    object_sizes_m: np.ndarray,
    *,
    tip_joint_indices: Sequence[int],
    touch_distance_m: float = 0.010,
) -> DirectTouchEvidence:
    """Build conservative direct digital-touch evidence.

    Frames outside the touch band remain ``UNKNOWN`` rather than being called
    no-contact, approach, grasp, or release.  Those semantic states need
    additional temporal/visibility evidence that this geometric adapter does
    not possess.
    """

    joints = np.asarray(joints_camera, dtype=np.float64)
    observed = np.asarray(hand_observed)
    poses = np.asarray(pose_object_to_camera, dtype=np.float64)
    valid = np.asarray(pose_valid)
    sizes = np.asarray(object_sizes_m, dtype=np.float64)
    tips = np.asarray(tuple(tip_joint_indices), dtype=np.int64)
    if joints.ndim != 4 or joints.shape[-1] != 3:
        raise ContactInputAdapterError("joints_camera must have shape (H,T,J,3)")
    hand_count, frame_count, joint_count, _ = joints.shape
    if observed.shape != (hand_count, frame_count) or observed.dtype != np.bool_:
        raise ContactInputAdapterError("hand_observed must be bool (H,T)")
    if valid.ndim != 2 or valid.shape[0] != frame_count or valid.dtype != np.bool_:
        raise ContactInputAdapterError("pose_valid must be bool (T,O)")
    object_count = valid.shape[1]
    if poses.shape != (frame_count, object_count, 4, 4):
        raise ContactInputAdapterError("pose_object_to_camera must be (T,O,4,4)")
    if sizes.shape != (object_count, 3):
        raise ContactInputAdapterError("object_sizes_m must be (O,3)")
    if tips.ndim != 1 or tips.size == 0 or np.any(tips < 0) or np.any(tips >= joint_count):
        raise ContactInputAdapterError("tip_joint_indices are invalid")
    threshold = float(touch_distance_m)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ContactInputAdapterError("touch_distance_m must be positive and finite")

    distances = np.full((hand_count, frame_count, tips.size, object_count), np.nan, dtype=np.float32)
    finger_contact = np.zeros_like(distances, dtype=np.bool_)
    state = np.full((hand_count, frame_count, object_count), int(ContactState.UNKNOWN), dtype=np.uint8)
    confidence = np.zeros((hand_count, frame_count, object_count), dtype=np.float32)
    for frame in range(frame_count):
        for obj in range(object_count):
            if not valid[frame, obj]:
                continue
            for hand in range(hand_count):
                if not observed[hand, frame]:
                    continue
                points = joints[hand, frame, tips]
                if not np.isfinite(points).all():
                    continue
                signed = signed_distance_points_to_oriented_box(
                    points, poses[frame, obj], sizes[obj]
                )
                distances[hand, frame, :, obj] = signed.astype(np.float32)
                touching = np.abs(signed) <= threshold
                finger_contact[hand, frame, :, obj] = touching
                if np.any(touching):
                    minimum = float(np.min(np.abs(signed[touching])))
                    state[hand, frame, obj] = int(ContactState.TOUCH)
                    confidence[hand, frame, obj] = np.float32(
                        np.clip(1.0 - minimum / threshold, 0.0, 1.0)
                    )
    return DirectTouchEvidence(distances, finger_contact, state, confidence)


def freeze_four_touch_diagnostic_frames(
    contact_state: np.ndarray,
    contact_confidence: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """Freeze four deterministic frames spanning observed touch segments.

    Preference is the first segment start, its strongest contiguous evidence
    frame, its end, then the strongest frame of a later segment.  With state
    alone, segment midpoints are used.  Fewer than four unique touch frames is
    a fail-closed error.
    """

    state = np.asarray(contact_state)
    if state.ndim != 3:
        raise ContactInputAdapterError("contact_state must be (H,T,O)")
    score: np.ndarray | None = None
    if contact_confidence is not None:
        confidence = np.asarray(contact_confidence, dtype=np.float64)
        if confidence.shape != state.shape or not np.isfinite(confidence).all():
            raise ContactInputAdapterError("contact_confidence must be finite and match state")
        score = np.max(confidence, axis=(0, 2))
    active = np.any(state == int(ContactState.TOUCH), axis=(0, 2))
    frames = np.flatnonzero(active).tolist()
    if len(frames) < 4:
        raise ContactInputAdapterError("fewer than four unique direct-touch frames")
    segments: list[list[int]] = []
    for frame in frames:
        if not segments or frame != segments[-1][-1] + 1:
            segments.append([frame])
        else:
            segments[-1].append(frame)
    first = segments[0]
    first_best = first[len(first) // 2] if score is None else max(first, key=lambda item: score[item])
    candidates = [first[0], first_best, first[-1]]
    for segment in segments[1:]:
        best = segment[len(segment) // 2] if score is None else max(segment, key=lambda item: score[item])
        candidates.extend((best, segment[0], segment[-1]))
    candidates.extend(frames)
    unique: list[int] = []
    for frame in candidates:
        if frame not in unique:
            unique.append(frame)
        if len(unique) == 4:
            return tuple(unique)  # type: ignore[return-value]
    raise ContactInputAdapterError("cannot freeze four unique direct-touch frames")
