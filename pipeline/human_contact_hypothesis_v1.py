"""Independent human-contact hypotheses for temporarily unobserved objects.

This module is intentionally upstream of every Robot solver and renderer.  It
never edits, fills, or republishes formal Object6D observations.  Its output is
a separate hypothesis tensor whose source is explicit for every object/frame.

The implementation is deliberately conservative: only short, bounded gaps
with two direct observations can be filled.  Callers may supply an independently
estimated hand/object attachment trajectory, but it is accepted only when the
entry distance and transform drift gates pass.  Everything else remains
``UNKNOWN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from typing import Sequence

import numpy as np


SCHEMA_VERSION = "HUMAN_CONTACT_HYPOTHESIS_V1"
MAX_GAP_FRAMES = 15
MAX_GAP_SECONDS = 0.5
ENDPOINT_TRANSLATION_M = 0.015
ENDPOINT_ROTATION_DEG = 10.0
ATTACHMENT_ENTRY_DISTANCE_M = 0.010
ATTACHMENT_TRANSLATION_DRIFT_M = 0.015
ATTACHMENT_ROTATION_DRIFT_DEG = 10.0


class ContactHypothesisError(ValueError):
    """Raised when input authority or hypothesis closure is invalid."""


class PoseSource(IntEnum):
    UNKNOWN = 0
    DIRECT_OBJECT6D = 1
    BIDIRECTIONAL_RIGID_HYPOTHESIS = 2
    HAND_OBJECT_ATTACHMENT_HYPOTHESIS = 3


@dataclass(frozen=True)
class ContactHypothesisResult:
    pose_object_to_camera: np.ndarray
    pose_valid: np.ndarray
    pose_source: np.ndarray
    hypothesis: np.ndarray
    confidence: np.ndarray
    formal_object6d_sha256_before: str
    formal_object6d_sha256_after: str


def _array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _validate_pose_array(poses: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(poses, dtype=np.float64)
    v = np.asarray(valid)
    if p.ndim != 4 or p.shape[-2:] != (4, 4):
        raise ContactHypothesisError("Object6D poses must have shape (T,O,4,4)")
    if v.shape != p.shape[:2] or v.dtype != np.bool_:
        raise ContactHypothesisError("Object6D valid must be bool with shape (T,O)")
    if not np.isfinite(p[v]).all():
        raise ContactHypothesisError("direct Object6D poses must be finite")
    expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0])
    for matrix in p[v]:
        if not np.allclose(matrix[3], expected_bottom, atol=1e-8):
            raise ContactHypothesisError("Object6D transform bottom row is invalid")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ContactHypothesisError("Object6D rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ContactHypothesisError("Object6D rotation is not proper")
    return p, v


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _so3_interpolate(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return a.copy()
    axis_skew = (relative - relative.T) / (2.0 * np.sin(angle))
    scaled = fraction * angle
    delta = np.eye(3) + np.sin(scaled) * axis_skew + (1.0 - np.cos(scaled)) * (axis_skew @ axis_skew)
    result = a @ delta
    u, _, vh = np.linalg.svd(result)
    result = u @ vh
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def _interpolate_pose(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _so3_interpolate(a[:3, :3], b[:3, :3], fraction)
    result[:3, 3] = (1.0 - fraction) * a[:3, 3] + fraction * b[:3, 3]
    return result


def _contiguous_false_runs(values: np.ndarray) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, item in enumerate(values.tolist() + [True]):
        if not item and start is None:
            start = index
        elif item and start is not None:
            result.append((start, index))
            start = None
    return result


def build_contact_pose_hypotheses(
    formal_object6d_poses: np.ndarray,
    formal_object6d_valid: np.ndarray,
    *,
    object_ids: Sequence[str],
    fps: float,
    attachment_poses: np.ndarray | None = None,
    attachment_valid: np.ndarray | None = None,
    attachment_entry_distance_m: np.ndarray | None = None,
    attachment_relative_translation_drift_m: np.ndarray | None = None,
    attachment_relative_rotation_drift_deg: np.ndarray | None = None,
) -> ContactHypothesisResult:
    """Create a sidecar pose timeline without mutating formal Object6D.

    ``attachment_poses`` is an independent estimate in the same
    object-to-camera convention.  It is never copied into formal Object6D and
    is accepted only for an otherwise unknown, bounded gap.
    """

    raw_poses = np.asarray(formal_object6d_poses)
    before = _array_sha(raw_poses)
    poses, valid = _validate_pose_array(raw_poses, np.asarray(formal_object6d_valid))
    frame_count, object_count = valid.shape
    if len(object_ids) != object_count or len(set(object_ids)) != object_count:
        raise ContactHypothesisError("object_ids must be unique and match Object6D object axis")
    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ContactHypothesisError("fps must be finite and positive")
    max_gap = min(MAX_GAP_FRAMES, int(np.floor(MAX_GAP_SECONDS * rate + 1e-12)))

    output = np.full_like(poses, np.nan, dtype=np.float64)
    output_valid = np.zeros_like(valid)
    source = np.full(valid.shape, int(PoseSource.UNKNOWN), dtype=np.uint8)
    hypothesis = np.zeros_like(valid)
    confidence = np.zeros(valid.shape, dtype=np.float32)
    output[valid] = poses[valid]
    output_valid[valid] = True
    source[valid] = int(PoseSource.DIRECT_OBJECT6D)
    confidence[valid] = 1.0

    attach_p: np.ndarray | None = None
    attach_v: np.ndarray | None = None
    attach_d: np.ndarray | None = None
    attach_translation_drift: np.ndarray | None = None
    attach_rotation_drift: np.ndarray | None = None
    attachment_inputs = (
        attachment_poses,
        attachment_valid,
        attachment_entry_distance_m,
        attachment_relative_translation_drift_m,
        attachment_relative_rotation_drift_deg,
    )
    if any(value is not None for value in attachment_inputs):
        if any(value is None for value in attachment_inputs):
            raise ContactHypothesisError("attachment inputs must be supplied together")
        assert attachment_poses is not None
        assert attachment_valid is not None
        assert attachment_entry_distance_m is not None
        assert attachment_relative_translation_drift_m is not None
        assert attachment_relative_rotation_drift_deg is not None
        attach_p, attach_v = _validate_pose_array(attachment_poses, np.asarray(attachment_valid))
        attach_d = np.asarray(attachment_entry_distance_m, dtype=np.float64)
        attach_translation_drift = np.asarray(
            attachment_relative_translation_drift_m, dtype=np.float64
        )
        attach_rotation_drift = np.asarray(
            attachment_relative_rotation_drift_deg, dtype=np.float64
        )
        if (
            attach_p.shape != poses.shape
            or attach_v.shape != valid.shape
            or attach_d.shape != valid.shape
            or attach_translation_drift.shape != valid.shape
            or attach_rotation_drift.shape != valid.shape
        ):
            raise ContactHypothesisError("attachment inputs must match Object6D axes")
        for values, label in (
            (attach_d, "attachment entry distances"),
            (attach_translation_drift, "attachment relative translation drift"),
            (attach_rotation_drift, "attachment relative rotation drift"),
        ):
            if not np.isfinite(values[attach_v]).all() or np.any(values[attach_v] < 0.0):
                raise ContactHypothesisError(f"{label} must be finite and non-negative")

    for object_index in range(object_count):
        for start, stop in _contiguous_false_runs(valid[:, object_index]):
            gap = stop - start
            if gap > max_gap or start == 0 or stop == frame_count:
                continue
            left = poses[start - 1, object_index]
            right = poses[stop, object_index]
            translation = float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
            rotation = _rotation_angle_deg(left[:3, :3], right[:3, :3])
            if translation <= ENDPOINT_TRANSLATION_M and rotation <= ENDPOINT_ROTATION_DEG:
                for frame in range(start, stop):
                    fraction = (frame - start + 1) / (gap + 1)
                    output[frame, object_index] = _interpolate_pose(left, right, fraction)
                    output_valid[frame, object_index] = True
                    source[frame, object_index] = int(PoseSource.BIDIRECTIONAL_RIGID_HYPOTHESIS)
                    hypothesis[frame, object_index] = True
                    confidence[frame, object_index] = np.float32(np.exp(-gap / max(1, max_gap)))
                continue

            if (
                attach_p is None
                or attach_v is None
                or attach_d is None
                or attach_translation_drift is None
                or attach_rotation_drift is None
            ):
                continue
            indices = np.arange(start, stop)
            if not np.all(attach_v[indices, object_index]):
                continue
            if float(attach_d[start, object_index]) > ATTACHMENT_ENTRY_DISTANCE_M:
                continue
            # These are hand/object *relative-transform* drift metrics from an
            # independent attachment estimator.  Global object travel is not
            # drift and must not be rejected here.
            if (
                float(np.max(attach_translation_drift[indices, object_index]))
                > ATTACHMENT_TRANSLATION_DRIFT_M
                or float(np.max(attach_rotation_drift[indices, object_index]))
                > ATTACHMENT_ROTATION_DRIFT_DEG
            ):
                continue
            output[indices, object_index] = attach_p[indices, object_index]
            output_valid[indices, object_index] = True
            source[indices, object_index] = int(PoseSource.HAND_OBJECT_ATTACHMENT_HYPOTHESIS)
            hypothesis[indices, object_index] = True
            confidence[indices, object_index] = np.float32(0.75 * np.exp(-gap / max(1, max_gap)))

    after = _array_sha(raw_poses)
    if before != after:
        raise ContactHypothesisError("formal Object6D mutated while building hypotheses")
    return ContactHypothesisResult(
        pose_object_to_camera=output,
        pose_valid=output_valid,
        pose_source=source,
        hypothesis=hypothesis,
        confidence=confidence,
        formal_object6d_sha256_before=before,
        formal_object6d_sha256_after=after,
    )
