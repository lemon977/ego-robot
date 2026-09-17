"""Bounded V7.1 object-pose hypotheses without mutating formal Object6D.

Formal Object6D remains ``DIRECT_OBSERVED_ONLY`` with ``KEEP_INVALID``.  This
module emits a separate development-only sidecar.  Every inferred pose keeps
its evidence parents and the combined graph is validated by
``object_contact_evidence_v1`` before it is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np

from chaoyang.pipeline.object_contact_evidence_v1 import (
    EvidenceGraph,
    EvidenceGraphError,
    validate_evidence_dag,
)


SCHEMA_VERSION = "OBJECT_POSE_HYPOTHESIS_V2"
MAX_GAP_FRAMES = 15
MAX_GAP_SECONDS = 0.5
MAX_ENDPOINT_TRANSLATION_M = 0.015
MAX_ENDPOINT_ROTATION_DEG = 10.0
MAX_ATTACHMENT_ENTRY_DISTANCE_M = 0.010
MAX_ATTACHMENT_TRANSLATION_DRIFT_M = 0.015
MAX_ATTACHMENT_ROTATION_DRIFT_DEG = 10.0


class ObjectPoseHypothesisError(ValueError):
    """Raised when inputs or evidence violate the fail-closed contract."""


class PoseHypothesisSource(str, Enum):
    DIRECT_OBJECT6D = "DIRECT_OBJECT6D"
    BIDIRECTIONAL_TRACKED = "BIDIRECTIONAL_TRACKED"
    HAND_OBJECT_ATTACHMENT = "HAND_OBJECT_ATTACHMENT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PoseEvidenceRecord:
    frame_index: int
    object_id: str
    source: PoseHypothesisSource
    evidence_id: str
    parent_evidence_ids: tuple[str, ...]
    confidence: float
    reason: str


@dataclass(frozen=True)
class ObjectPoseHypothesisResult:
    pose_object_to_camera: np.ndarray
    pose_valid: np.ndarray
    source: np.ndarray
    confidence: np.ndarray
    records: tuple[PoseEvidenceRecord, ...]
    evidence_graph: EvidenceGraph
    formal_object6d_sha256_before: str
    formal_object6d_sha256_after: str
    formal_object6d_mutated: bool
    authority_promotable: bool = False


def _arrays_sha(poses: np.ndarray, valid: np.ndarray) -> str:
    digest = sha256()
    for value in (np.ascontiguousarray(poses), np.ascontiguousarray(valid)):
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _validate_poses(poses: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_array = np.asarray(poses, dtype=np.float64)
    valid_array = np.asarray(valid)
    if pose_array.ndim != 4 or pose_array.shape[-2:] != (4, 4):
        raise ObjectPoseHypothesisError("poses must have shape (T,O,4,4)")
    if valid_array.dtype != np.bool_ or valid_array.shape != pose_array.shape[:2]:
        raise ObjectPoseHypothesisError("valid must be bool with shape (T,O)")
    for matrix in pose_array[valid_array]:
        if not np.isfinite(matrix).all():
            raise ObjectPoseHypothesisError("valid poses must be finite")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ObjectPoseHypothesisError("invalid transform bottom row")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ObjectPoseHypothesisError("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ObjectPoseHypothesisError("rotation must be proper")
    return pose_array, valid_array


def _rotation_delta_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _interpolate_rotation(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-10:
        return left.copy()
    skew = (relative - relative.T) / (2.0 * np.sin(angle))
    scaled = alpha * angle
    estimate = left @ (
        np.eye(3) + np.sin(scaled) * skew + (1.0 - np.cos(scaled)) * (skew @ skew)
    )
    u, _, vh = np.linalg.svd(estimate)
    result = u @ vh
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def _interpolate_pose(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _interpolate_rotation(left[:3, :3], right[:3, :3], alpha)
    result[:3, 3] = (1.0 - alpha) * left[:3, 3] + alpha * right[:3, 3]
    return result


def _false_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for frame, item in enumerate(valid.tolist() + [True]):
        if not item and start is None:
            start = frame
        elif item and start is not None:
            runs.append((start, frame))
            start = None
    return runs


def _node(
    evidence_id: str,
    evidence_type: str,
    parents: Sequence[str],
    depth: int,
) -> dict[str, Any]:
    direct_or_tracked = evidence_type in {"DIRECT_OBJECT6D", "BIDIRECTIONAL_TRACKED"}
    contact = evidence_type in {"DIRECT_OBJECT6D", "BIDIRECTIONAL_TRACKED", "CONTACT_SEED"}
    return {
        "evidence_id": evidence_id,
        "evidence_type": evidence_type,
        "parent_evidence_ids": list(parents),
        "evidence_depth": depth,
        "may_support_contact_authority": contact,
        "may_support_object6d_authority": direct_or_tracked,
        "may_support_gold_contact_accuracy": False,
        "may_upgrade_tactile_supported_contact": False,
    }


def _normalize_optional_attachment(
    poses: np.ndarray,
    valid: np.ndarray,
    values: tuple[np.ndarray | None, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if all(item is None for item in values):
        return None
    if any(item is None for item in values):
        raise ObjectPoseHypothesisError("attachment inputs must be supplied together")
    attachment_poses, attachment_valid, entry, translation_drift, rotation_drift = values
    assert attachment_poses is not None and attachment_valid is not None
    attach_poses, attach_valid = _validate_poses(attachment_poses, np.asarray(attachment_valid))
    metrics = tuple(np.asarray(item, dtype=np.float64) for item in (entry, translation_drift, rotation_drift))
    if attach_poses.shape != poses.shape or attach_valid.shape != valid.shape:
        raise ObjectPoseHypothesisError("attachment pose axes must match formal Object6D")
    if any(item.shape != valid.shape for item in metrics):
        raise ObjectPoseHypothesisError("attachment metrics must have shape (T,O)")
    for item in metrics:
        if not np.isfinite(item[attach_valid]).all() or np.any(item[attach_valid] < 0.0):
            raise ObjectPoseHypothesisError("attachment metrics must be finite and non-negative")
    return attach_poses, attach_valid, metrics[0], metrics[1], metrics[2]


def build_object_pose_hypotheses(
    formal_object6d_poses: np.ndarray,
    formal_object6d_valid: np.ndarray,
    *,
    task_id: str,
    object_ids: Sequence[str],
    fps: float,
    formal_evidence_ids: np.ndarray,
    visibly_deformed: np.ndarray | None = None,
    poker_dimensions_m: Sequence[float] | None = None,
    contact_seed_nodes: Sequence[Mapping[str, Any]] = (),
    attachment_seed_by_object: Mapping[str, str] | None = None,
    attachment_poses: np.ndarray | None = None,
    attachment_valid: np.ndarray | None = None,
    attachment_entry_distance_m: np.ndarray | None = None,
    attachment_relative_translation_drift_m: np.ndarray | None = None,
    attachment_relative_rotation_drift_deg: np.ndarray | None = None,
) -> ObjectPoseHypothesisResult:
    """Build a bounded sidecar and validate its complete evidence DAG.

    Missing a future direct observation, exceeding any gap/endpoint gate, or
    encountering Chips deformation yields ``UNKNOWN``.  Attachment candidates
    require an already-valid ``CONTACT_SEED``; their output can never support
    Contact, Object6D, gold accuracy, or tactile authority.
    """

    raw_poses = np.asarray(formal_object6d_poses)
    raw_valid = np.asarray(formal_object6d_valid)
    before = _arrays_sha(raw_poses, raw_valid)
    poses, valid = _validate_poses(raw_poses, raw_valid)
    frame_count, object_count = valid.shape
    ids = tuple(object_ids)
    if len(ids) != object_count or len(set(ids)) != object_count:
        raise ObjectPoseHypothesisError("object_ids must be unique and match the object axis")
    if task_id not in {"poker", "chips"}:
        raise ObjectPoseHypothesisError("task_id must be poker or chips")
    if task_id == "chips" and object_count != 3:
        raise ObjectPoseHypothesisError("Chips requires exactly three independent instances")
    if any("union" in item.lower() for item in ids):
        raise ObjectPoseHypothesisError("union object identity is forbidden")
    if task_id == "poker":
        if poker_dimensions_m is None:
            raise ObjectPoseHypothesisError("Poker requires thin two-sided rigid dimensions")
        dimensions = np.asarray(poker_dimensions_m, dtype=np.float64)
        if dimensions.shape != (3,) or np.any(~np.isfinite(dimensions)) or np.any(dimensions <= 0):
            raise ObjectPoseHypothesisError("Poker dimensions must be positive finite (3,)")
        if dimensions[2] >= min(dimensions[:2]) * 0.05:
            raise ObjectPoseHypothesisError("Poker must use a thin two-sided rigid model")

    rate = float(fps)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ObjectPoseHypothesisError("fps must be finite and positive")
    max_gap = min(MAX_GAP_FRAMES, int(np.floor(MAX_GAP_SECONDS * rate + 1e-12)))

    evidence_ids = np.asarray(formal_evidence_ids, dtype=object)
    if evidence_ids.shape != valid.shape:
        raise ObjectPoseHypothesisError("formal_evidence_ids must have shape (T,O)")
    direct_ids: list[str] = []
    for frame, object_index in zip(*np.nonzero(valid), strict=True):
        evidence_id = evidence_ids[frame, object_index]
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ObjectPoseHypothesisError("every direct pose needs a formal evidence id")
        direct_ids.append(evidence_id)
    if len(direct_ids) != len(set(direct_ids)):
        raise ObjectPoseHypothesisError("formal evidence ids must be unique per observation")

    deformed = np.zeros(valid.shape, dtype=np.bool_)
    if visibly_deformed is not None:
        deformed = np.asarray(visibly_deformed)
        if deformed.dtype != np.bool_ or deformed.shape != valid.shape:
            raise ObjectPoseHypothesisError("visibly_deformed must be bool with shape (T,O)")
    if task_id == "poker" and np.any(deformed):
        raise ObjectPoseHypothesisError("Poker fixture is rigid; deformation is not permitted")

    attachment = _normalize_optional_attachment(
        poses,
        valid,
        (
            attachment_poses,
            attachment_valid,
            attachment_entry_distance_m,
            attachment_relative_translation_drift_m,
            attachment_relative_rotation_drift_deg,
        ),
    )
    seed_by_object = dict(attachment_seed_by_object or {})
    if attachment is not None and set(seed_by_object) != set(ids):
        raise ObjectPoseHypothesisError("attachment requires one declared contact seed per object")

    output = np.full_like(poses, np.nan)
    output_valid = np.zeros_like(valid)
    sources = np.full(valid.shape, PoseHypothesisSource.UNKNOWN.value, dtype=object)
    confidence = np.zeros(valid.shape, dtype=np.float32)
    records: list[PoseEvidenceRecord] = []
    nodes: list[dict[str, Any]] = [dict(item) for item in contact_seed_nodes]

    for frame in range(frame_count):
        for object_index, object_id in enumerate(ids):
            if valid[frame, object_index] and not (task_id == "chips" and deformed[frame, object_index]):
                evidence_id = str(evidence_ids[frame, object_index])
                output[frame, object_index] = poses[frame, object_index]
                output_valid[frame, object_index] = True
                sources[frame, object_index] = PoseHypothesisSource.DIRECT_OBJECT6D.value
                confidence[frame, object_index] = 1.0
                nodes.append(_node(evidence_id, "DIRECT_OBJECT6D", (), 0))
                records.append(PoseEvidenceRecord(frame, object_id, PoseHypothesisSource.DIRECT_OBJECT6D, evidence_id, (), 1.0, "DIRECT_OBSERVED"))
            else:
                evidence_id = f"objpose-unknown-{object_index}-{frame}"
                reason = "CHIPS_DEFORMED_RIGID_POSE_FORBIDDEN" if deformed[frame, object_index] else "NO_LEGAL_BOUNDED_HYPOTHESIS"
                records.append(PoseEvidenceRecord(frame, object_id, PoseHypothesisSource.UNKNOWN, evidence_id, (), 0.0, reason))

    for object_index, object_id in enumerate(ids):
        for start, stop in _false_runs(valid[:, object_index]):
            gap = stop - start
            # Both bounding direct observations are mandatory; no future
            # endpoint means UNKNOWN rather than causal-looking extrapolation.
            if gap > max_gap or start == 0 or stop == frame_count:
                continue
            if task_id == "chips" and np.any(deformed[start - 1 : stop + 1, object_index]):
                continue
            left = poses[start - 1, object_index]
            right = poses[stop, object_index]
            endpoint_translation = float(np.linalg.norm(right[:3, 3] - left[:3, 3]))
            endpoint_rotation = _rotation_delta_deg(left[:3, :3], right[:3, :3])
            left_id = str(evidence_ids[start - 1, object_index])
            right_id = str(evidence_ids[stop, object_index])
            if endpoint_translation <= MAX_ENDPOINT_TRANSLATION_M and endpoint_rotation <= MAX_ENDPOINT_ROTATION_DEG:
                for frame in range(start, stop):
                    alpha = (frame - start + 1) / (gap + 1)
                    evidence_id = f"objpose-tracked-{object_index}-{frame}"
                    output[frame, object_index] = _interpolate_pose(left, right, alpha)
                    output_valid[frame, object_index] = True
                    sources[frame, object_index] = PoseHypothesisSource.BIDIRECTIONAL_TRACKED.value
                    score = float(np.exp(-gap / max(1, max_gap)))
                    confidence[frame, object_index] = score
                    parents = (left_id, right_id)
                    nodes.append(_node(evidence_id, "BIDIRECTIONAL_TRACKED", parents, 1))
                    records[frame * object_count + object_index] = PoseEvidenceRecord(frame, object_id, PoseHypothesisSource.BIDIRECTIONAL_TRACKED, evidence_id, parents, score, "BOUNDED_TWO_ENDPOINT_TRACK")
                continue

            if attachment is None:
                continue
            attach_poses, attach_valid, entry, translation_drift, rotation_drift = attachment
            frames = np.arange(start, stop)
            if not np.all(attach_valid[frames, object_index]):
                continue
            if float(entry[start, object_index]) > MAX_ATTACHMENT_ENTRY_DISTANCE_M:
                continue
            if float(np.max(translation_drift[frames, object_index])) > MAX_ATTACHMENT_TRANSLATION_DRIFT_M:
                continue
            if float(np.max(rotation_drift[frames, object_index])) > MAX_ATTACHMENT_ROTATION_DRIFT_DEG:
                continue
            seed_id = seed_by_object[object_id]
            for frame in range(start, stop):
                evidence_id = f"objpose-attachment-{object_index}-{frame}"
                output[frame, object_index] = attach_poses[frame, object_index]
                output_valid[frame, object_index] = True
                sources[frame, object_index] = PoseHypothesisSource.HAND_OBJECT_ATTACHMENT.value
                score = float(0.75 * np.exp(-gap / max(1, max_gap)))
                confidence[frame, object_index] = score
                parents = (seed_id,)
                seed_depth = next((int(item["evidence_depth"]) for item in nodes if item.get("evidence_id") == seed_id), -1)
                if seed_depth < 0:
                    raise ObjectPoseHypothesisError(f"missing contact seed evidence: {seed_id}")
                nodes.append(_node(evidence_id, "HAND_OBJECT_ATTACHMENT", parents, seed_depth + 1))
                records[frame * object_count + object_index] = PoseEvidenceRecord(frame, object_id, PoseHypothesisSource.HAND_OBJECT_ATTACHMENT, evidence_id, parents, score, "CONTACT_SEEDED_ATTACHMENT")

    for record in records:
        if record.source is PoseHypothesisSource.UNKNOWN:
            nodes.append(_node(record.evidence_id, "UNKNOWN", (), 0))

    try:
        graph = validate_evidence_dag(nodes)
    except EvidenceGraphError as error:
        raise ObjectPoseHypothesisError(f"evidence DAG rejected: {error}") from error
    after = _arrays_sha(raw_poses, raw_valid)
    if before != after:
        raise ObjectPoseHypothesisError("formal Object6D was mutated")
    return ObjectPoseHypothesisResult(
        pose_object_to_camera=output,
        pose_valid=output_valid,
        source=sources,
        confidence=confidence,
        records=tuple(records),
        evidence_graph=graph,
        formal_object6d_sha256_before=before,
        formal_object6d_sha256_after=after,
        formal_object6d_mutated=False,
    )
