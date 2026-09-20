"""Full-frame Kai22 FK sidecar and fail-closed V3.2 tier accounting.

The sidecar consumes an already existing q22 trajectory.  It does not
retarget, smooth, clip, interpolate, or forward-fill it.  Every valid frame is
checked against the pinned URDF limits, expanded to all link transforms, and
passed through a non-parent self-collision checker.  Missing collision or
clip diagnostics stay unknown and cannot receive KINEMATIC_ONLY admission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol
import uuid

import numpy as np

from chaoyang.pipeline.robot_renderer_cycles import (
    UrdfModel,
    forward_kinematics,
    parse_urdf,
)
from chaoyang.pipeline.temporal_authority_v1 import validate_online_current_inputs


SCHEMA_VERSION = "KAI22_FULL_FK_SIDECAR_V1"
TIER_SCHEMA_VERSION = "KAI22_R0_TIERED_ADMISSION_V32"
PHYSICAL_SIDES = ("left", "right")
DEFAULT_ANATOMICAL_SIDE_BY_PHYSICAL = ("right", "left")
LEVELS = ("KINEMATIC_ONLY", "DEVELOPMENT_R0", "H50_READY")


class Kai22SidecarError(ValueError):
    """Raised when the q22/URDF contract is incomplete or ambiguous."""


@dataclass(frozen=True)
class CollisionDiagnostic:
    known: bool
    non_adjacent_collision_pass: bool
    illegal_contact_count: int
    max_penetration_m: float


class CollisionChecker(Protocol):
    def check(self, physical_side: int, q22: np.ndarray) -> CollisionDiagnostic: ...

    def close(self) -> None: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Kai22SidecarError(f"{name} must be a lowercase SHA-256")
    return value


def load_pinned_kaihand_models(repo_root: Path) -> tuple[tuple[UrdfModel, UrdfModel], dict[str, Any]]:
    """Load the two KaiHand URDFs only after verifying the repository asset pin."""

    root = repo_root.resolve(strict=True)
    pin_path = root / "assets/robot/ROBOT_ASSET_PIN.json"
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    if pin.get("status") != "T0_IDENTITY_PIN_NOT_EXECUTION_AUTHORIZATION":
        raise Kai22SidecarError("unexpected robot asset pin status")
    files = {str(row["path"]): row for row in pin.get("files", [])}
    hand_rows: dict[str, Mapping[str, Any]] = {}
    for row in pin.get("urdfs", []):
        path = str(row.get("path", ""))
        if "KaiBot-Dexhand shell-URDF-L-" in path:
            hand_rows["left"] = row
        elif "KaiBot-Dexhand shell-URDF-R-" in path:
            hand_rows["right"] = row
    if set(hand_rows) != set(PHYSICAL_SIDES):
        raise Kai22SidecarError("asset pin does not contain exactly two KaiHand URDFs")
    models: list[UrdfModel] = []
    refs: dict[str, Any] = {}
    for side in PHYSICAL_SIDES:
        relative = str(hand_rows[side]["path"])
        file_row = files.get(relative)
        if file_row is None:
            raise Kai22SidecarError(f"pinned URDF is absent from file closure: {relative}")
        path = (root / relative).resolve(strict=True)
        if root != path and root not in path.parents:
            raise Kai22SidecarError("pinned URDF escapes repository root")
        observed_sha = _sha256(path)
        if path.stat().st_size != int(file_row["bytes"]) or observed_sha != file_row["sha256"]:
            raise Kai22SidecarError(f"pinned URDF drift: {relative}")
        model = parse_urdf(path)
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        if len(model.links) != 23 or len(moving) != 22:
            raise Kai22SidecarError(f"Kai22 closure mismatch for {side}")
        models.append(model)
        refs[side] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": observed_sha,
        }
    return (models[0], models[1]), {
        "asset_pin": {
            "path": "assets/robot/ROBOT_ASSET_PIN.json",
            "bytes": pin_path.stat().st_size,
            "sha256": _sha256(pin_path),
        },
        "urdf": refs,
    }


class PyBulletNonAdjacentSelfCollisionChecker:
    """CPU-only collision checker using pinned URDF collision geometry."""

    def __init__(self, models: Sequence[UrdfModel], *, penetration_tolerance_m: float = 0.0):
        if len(models) != 2:
            raise Kai22SidecarError("exactly two physical-side URDFs are required")
        if penetration_tolerance_m < 0 or not np.isfinite(penetration_tolerance_m):
            raise Kai22SidecarError("penetration tolerance must be finite and non-negative")
        try:
            import pybullet as bullet
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise Kai22SidecarError("pybullet is required for self-collision diagnostics") from exc
        self._bullet = bullet
        self._tolerance = float(penetration_tolerance_m)
        self._client = bullet.connect(bullet.DIRECT)
        if self._client < 0:
            raise Kai22SidecarError("pybullet DIRECT connection failed")
        self._bodies: list[int] = []
        self._joint_indices: list[tuple[int, ...]] = []
        try:
            for model in models:
                body = bullet.loadURDF(
                    str(model.path),
                    useFixedBase=True,
                    flags=bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                    physicsClientId=self._client,
                )
                expected = tuple(
                    joint.name for joint in model.joints if joint.joint_type != "fixed"
                )
                indices_by_name: dict[str, int] = {}
                for index in range(bullet.getNumJoints(body, physicsClientId=self._client)):
                    info = bullet.getJointInfo(body, index, physicsClientId=self._client)
                    if int(info[3]) >= 0:
                        indices_by_name[info[1].decode("utf-8")] = index
                if set(indices_by_name) != set(expected):
                    raise Kai22SidecarError("pybullet and parsed URDF joint identities differ")
                self._bodies.append(body)
                self._joint_indices.append(tuple(indices_by_name[name] for name in expected))
        except Exception:
            self.close()
            raise

    def check(self, physical_side: int, q22: np.ndarray) -> CollisionDiagnostic:
        if physical_side not in (0, 1):
            raise Kai22SidecarError("physical_side must be 0 or 1")
        q = np.asarray(q22, dtype=np.float64)
        if q.shape != (22,) or not np.isfinite(q).all():
            raise Kai22SidecarError("collision checker requires one finite q22 row")
        bullet = self._bullet
        body = self._bodies[physical_side]
        for index, value in zip(self._joint_indices[physical_side], q, strict=True):
            bullet.resetJointState(
                body, index, float(value), physicsClientId=self._client
            )
        bullet.performCollisionDetection(physicsClientId=self._client)
        penetrations = [
            max(0.0, -float(point[8]))
            for point in bullet.getContactPoints(body, body, physicsClientId=self._client)
            if max(0.0, -float(point[8])) > self._tolerance
        ]
        return CollisionDiagnostic(
            known=True,
            non_adjacent_collision_pass=not penetrations,
            illegal_contact_count=len(penetrations),
            max_penetration_m=max(penetrations, default=0.0),
        )

    def close(self) -> None:
        if getattr(self, "_client", -1) >= 0:
            self._bullet.disconnect(self._client)
            self._client = -1


def _model_vectors(model: UrdfModel) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
    names = tuple(joint.name for joint in moving)
    lower = np.asarray([joint.lower for joint in moving], dtype=np.float64)
    upper = np.asarray([joint.upper for joint in moving], dtype=np.float64)
    if len(names) != 22 or lower.shape != (22,) or upper.shape != (22,):
        raise Kai22SidecarError("URDF does not define an exact 22-joint bounded hand")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise Kai22SidecarError("URDF joint limits are incomplete or invalid")
    return names, lower, upper


def build_kai22_full_fk_sidecar(
    *,
    session_id: str,
    frame_ids: np.ndarray,
    timestamps_s: np.ndarray,
    q22: np.ndarray,
    q22_valid: np.ndarray,
    clip_delta: np.ndarray,
    models: Sequence[UrdfModel],
    collision_checker: CollisionChecker,
    anatomical_side_by_physical: Sequence[str] = DEFAULT_ANATOMICAL_SIDE_BY_PHYSICAL,
    source_observed: np.ndarray | None = None,
    source_temporal_authority: str,
    producer_sha256: str,
    config_sha256: str,
    input_sha256: str,
    urdf_refs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Expand an existing physical-side q22 sequence to complete FK evidence."""

    if not isinstance(session_id, str) or not session_id:
        raise Kai22SidecarError("session_id must be non-empty")
    if len(models) != 2 or tuple(anatomical_side_by_physical) not in {
        ("left", "right"),
        ("right", "left"),
    }:
        raise Kai22SidecarError("physical/anatomical side mapping must be an explicit bijection")
    ids = np.asarray(frame_ids, dtype=np.int64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    values = np.asarray(q22, dtype=np.float64)
    valid = np.asarray(q22_valid, dtype=bool)
    deltas = np.asarray(clip_delta, dtype=np.float64)
    if ids.ndim != 1 or ids.size == 0 or times.shape != ids.shape:
        raise Kai22SidecarError("frame_ids and timestamps_s must be non-empty [T]")
    expected_q_shape = (ids.size, 2, 22)
    if values.shape != expected_q_shape or deltas.shape != expected_q_shape or valid.shape != (ids.size, 2):
        raise Kai22SidecarError("q22/clip_delta must be [T,2,22] and validity [T,2]")
    if np.any(np.diff(ids) <= 0) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise Kai22SidecarError("frame and timestamp axes must be strictly increasing")
    observed = valid.copy() if source_observed is None else np.asarray(source_observed, dtype=bool)
    if observed.shape != valid.shape or np.any(valid & ~observed):
        raise Kai22SidecarError("valid q22 cannot be invented outside source_observed")
    for name, sha in (
        ("producer_sha256", producer_sha256),
        ("config_sha256", config_sha256),
        ("input_sha256", input_sha256),
    ):
        _require_sha(sha, name=name)
    if not isinstance(source_temporal_authority, str) or not source_temporal_authority:
        raise Kai22SidecarError("source temporal authority must be explicit")

    names: list[tuple[str, ...]] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    link_names = [tuple(model.links) for model in models]
    if any(len(side_links) != 23 for side_links in link_names):
        raise Kai22SidecarError("each KaiHand URDF must have exactly 23 links")
    for model in models:
        side_names, lower, upper = _model_vectors(model)
        names.append(side_names)
        lowers.append(lower)
        uppers.append(upper)

    frames = ids.size
    q_finite = np.isfinite(values).all(axis=2)
    clip_known = np.isfinite(deltas).all(axis=2)
    clip_linf = np.full((frames, 2), np.nan, dtype=np.float64)
    clip_linf[clip_known] = np.max(np.abs(deltas), axis=2)[clip_known]
    limit_pass = np.zeros((frames, 2), dtype=bool)
    max_limit_violation = np.full((frames, 2), np.nan, dtype=np.float64)
    fk = np.full((frames, 2, 23, 4, 4), np.nan, dtype=np.float64)
    fk_finite = np.zeros((frames, 2), dtype=bool)
    collision_known = np.zeros((frames, 2), dtype=bool)
    collision_pass = np.zeros((frames, 2), dtype=bool)
    illegal_contacts = np.full((frames, 2), -1, dtype=np.int32)
    max_penetration = np.full((frames, 2), np.nan, dtype=np.float64)
    failure_records: list[dict[str, Any]] = []

    for frame in range(frames):
        for side in range(2):
            if not valid[frame, side]:
                continue
            if not q_finite[frame, side]:
                failure_records.append(
                    {"source_frame": int(ids[frame]), "physical_robot_side": PHYSICAL_SIDES[side], "gate": "Q22_FINITE"}
                )
                continue
            q = values[frame, side]
            below = np.maximum(lowers[side] - q, 0.0)
            above = np.maximum(q - uppers[side], 0.0)
            violation = float(np.max(np.maximum(below, above)))
            max_limit_violation[frame, side] = violation
            limit_pass[frame, side] = violation <= 1e-12
            if not limit_pass[frame, side]:
                failure_records.append(
                    {"source_frame": int(ids[frame]), "physical_robot_side": PHYSICAL_SIDES[side], "gate": "HARD_URDF_LIMIT", "max_violation_rad": violation}
                )
                continue
            try:
                transforms = forward_kinematics(
                    models[side], dict(zip(names[side], q.tolist(), strict=True))
                )
                fk[frame, side] = np.stack([transforms[link] for link in link_names[side]])
                fk_finite[frame, side] = bool(np.isfinite(fk[frame, side]).all())
            except Exception as exc:
                failure_records.append(
                    {"source_frame": int(ids[frame]), "physical_robot_side": PHYSICAL_SIDES[side], "gate": "FULL_FK", "reason": f"{type(exc).__name__}:{str(exc)[:240]}"}
                )
                continue
            diagnostic = collision_checker.check(side, q)
            collision_known[frame, side] = bool(diagnostic.known)
            collision_pass[frame, side] = bool(
                diagnostic.known and diagnostic.non_adjacent_collision_pass
            )
            illegal_contacts[frame, side] = int(diagnostic.illegal_contact_count)
            max_penetration[frame, side] = float(diagnostic.max_penetration_m)
            if not collision_pass[frame, side]:
                failure_records.append(
                    {"source_frame": int(ids[frame]), "physical_robot_side": PHYSICAL_SIDES[side], "gate": "NON_ADJACENT_SELF_COLLISION", "diagnostic_known": bool(diagnostic.known), "illegal_contact_count": int(diagnostic.illegal_contact_count), "max_penetration_m": float(diagnostic.max_penetration_m)}
                )

    static_pass = (
        valid
        & q_finite
        & clip_known
        & limit_pass
        & fk_finite
        & collision_known
        & collision_pass
    )
    side_rows = []
    for side, physical_name in enumerate(PHYSICAL_SIDES):
        side_rows.append(
            {
                "physical_robot_side": physical_name,
                "anatomical_side": str(anatomical_side_by_physical[side]),
                "valid_frames": int(valid[:, side].sum()),
                "finite_q22_frames": int((valid[:, side] & q_finite[:, side]).sum()),
                "hard_limit_pass_frames": int((valid[:, side] & limit_pass[:, side]).sum()),
                "full_fk_pass_frames": int((valid[:, side] & fk_finite[:, side]).sum()),
                "collision_diagnostic_frames": int((valid[:, side] & collision_known[:, side]).sum()),
                "non_adjacent_self_collision_pass_frames": int((valid[:, side] & collision_pass[:, side]).sum()),
                "zero_clip_delta_frames": int((valid[:, side] & clip_known[:, side] & (clip_linf[:, side] <= 1e-12)).sum()),
                "static_gate_pass_frames": int(static_pass[:, side].sum()),
            }
        )
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "session_id": np.asarray(session_id),
        "source_frame": ids,
        "timestamp_s": times,
        "physical_robot_side": np.asarray(PHYSICAL_SIDES),
        "anatomical_side": np.asarray(tuple(anatomical_side_by_physical)),
        "joint_names": np.asarray(names),
        "link_names": np.asarray(link_names),
        "q22": values,
        "q22_valid": valid,
        "source_observed": observed,
        "clip_delta": deltas,
        "clip_delta_linf": clip_linf,
        "q22_finite": q_finite,
        "hard_limit_pass": limit_pass,
        "max_limit_violation_rad": max_limit_violation,
        "fk_root_relative": fk,
        "fk_finite": fk_finite,
        "collision_diagnostic_known": collision_known,
        "non_adjacent_self_collision_pass": collision_pass,
        "illegal_contact_count": illegal_contacts,
        "max_penetration_m": max_penetration,
        "static_gate_pass": static_pass,
        "joint_lower_rad": np.stack(lowers),
        "joint_upper_rad": np.stack(uppers),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETED_FULL_FRAME_DIAGNOSTICS",
        "session_id": session_id,
        "frame_count": int(frames),
        "q22_units": "radians",
        "fk_frame": "KAIHAND_ROOT_RELATIVE",
        "side_axis": side_rows,
        "joint_names": {PHYSICAL_SIDES[index]: list(names[index]) for index in range(2)},
        "link_names": {PHYSICAL_SIDES[index]: list(link_names[index]) for index in range(2)},
        "urdf_refs": dict(urdf_refs or {}),
        "diagnostics": {
            "failure_count": len(failure_records),
            "failures": failure_records,
            "clip_delta_missing_valid_side_frames": int((valid & ~clip_known).sum()),
            "collision_unknown_valid_side_frames": int((valid & ~collision_known).sum()),
        },
        "provenance": {
            "source_frame": "source_frame",
            "timestamp": "timestamp_s",
            "image_domain": "NOT_CONSUMED_BY_FK_SIDECAR",
            "units": {"q22": "radians", "FK_translation": "meters", "timestamp": "seconds"},
            "side": "EXPLICIT_PHYSICAL_AND_ANATOMICAL_AXES",
            "valid": "q22_valid_AND_NO_FILL",
            "observed_or_inferred": "source_observed_PRESERVED",
            "temporal_authority": source_temporal_authority,
            "producer_sha256": producer_sha256,
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
        },
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    return manifest, arrays


def evaluate_kai22_r0_tiers_v32(
    *,
    frame_ids: np.ndarray,
    q22_valid: np.ndarray,
    fk_finite: np.ndarray,
    hard_limit_pass: np.ndarray,
    collision_known: np.ndarray,
    non_adjacent_self_collision_pass: np.ndarray,
    clip_delta_linf: np.ndarray,
    mapping_semantics_pass: bool,
    clip_masks_mapping_error: bool,
    independent_window_mask: np.ndarray,
    stable_coverage_pass: bool,
    full_replay_reviewable: bool,
    development_threshold_authority: Mapping[str, Any] | None,
    temporal_authority_audit: Mapping[str, Any],
    h50_current_input_fields: Sequence[str],
    h50_validity_authority_pass: bool,
    h50_suffix_invariance_pass: bool,
    h50_minimum_contiguous_frames: int = 51,
) -> dict[str, Any]:
    """Admit no more than the evidence and frozen threshold authority support."""

    ids = np.asarray(frame_ids, dtype=np.int64)
    valid = np.asarray(q22_valid, dtype=bool)
    fk = np.asarray(fk_finite, dtype=bool)
    limits = np.asarray(hard_limit_pass, dtype=bool)
    collision_is_known = np.asarray(collision_known, dtype=bool)
    collision = np.asarray(non_adjacent_self_collision_pass, dtype=bool)
    clip = np.asarray(clip_delta_linf, dtype=np.float64)
    window = np.asarray(independent_window_mask, dtype=bool)
    if ids.ndim != 1 or ids.size == 0 or np.any(np.diff(ids) <= 0):
        raise Kai22SidecarError("frame_ids must be non-empty and strictly increasing")
    expected = (ids.size, 2)
    if any(value.shape != expected for value in (valid, fk, limits, collision_is_known, collision, clip, window)):
        raise Kai22SidecarError("all tier diagnostics must have shape [T,2]")
    boolean_flags = (
        mapping_semantics_pass,
        clip_masks_mapping_error,
        stable_coverage_pass,
        full_replay_reviewable,
        h50_validity_authority_pass,
        h50_suffix_invariance_pass,
    )
    if any(not isinstance(value, bool) for value in boolean_flags):
        raise Kai22SidecarError("tier flags must be booleans")
    if (
        not isinstance(h50_minimum_contiguous_frames, int)
        or isinstance(h50_minimum_contiguous_frames, bool)
        or h50_minimum_contiguous_frames <= 0
    ):
        raise Kai22SidecarError("H50 minimum contiguous frames must be positive")

    kinematic_reasons: list[str] = []
    if not mapping_semantics_pass:
        kinematic_reasons.append("SIDE_AXIS_UNIT_ZERO_MAPPING_NOT_VERIFIED")
    if clip_masks_mapping_error:
        kinematic_reasons.append("CLIP_MASKS_MAPPING_ERROR")
    if not valid.any():
        kinematic_reasons.append("NO_VALID_Q22_FRAME")
    if np.any(valid & ~fk):
        kinematic_reasons.append("FULL_FK_NONFINITE_FOR_VALID_Q22")
    if np.any(valid & ~limits):
        kinematic_reasons.append("HARD_URDF_LIMIT_FAILED")
    if np.any(valid & ~collision_is_known):
        kinematic_reasons.append("SELF_COLLISION_DIAGNOSTIC_UNKNOWN")
    if np.any(valid & ~collision):
        kinematic_reasons.append("NON_ADJACENT_SELF_COLLISION_FAILED")
    if np.any(valid & ~np.isfinite(clip)):
        kinematic_reasons.append("CLIP_DELTA_UNKNOWN")
    kinematic_pass = not kinematic_reasons

    development_reasons: list[str] = []
    if not kinematic_pass:
        development_reasons.append("KINEMATIC_ONLY_PREREQUISITE_FAILED")
    if development_threshold_authority is None:
        development_reasons.append("MISSING_FROZEN_DEVELOPMENT_THRESHOLD_AUTHORITY")
    else:
        if development_threshold_authority.get("status") != "FROZEN_NUMERIC_THRESHOLDS":
            development_reasons.append("DEVELOPMENT_THRESHOLD_AUTHORITY_NOT_FROZEN")
        for field in ("minimum_contiguous_frames", "minimum_coverage_fraction", "authority_sha256"):
            if field not in development_threshold_authority:
                development_reasons.append(f"DEVELOPMENT_THRESHOLD_AUTHORITY_MISSING:{field}")
        if "authority_sha256" in development_threshold_authority:
            _require_sha(
                str(development_threshold_authority["authority_sha256"]),
                name="development_threshold_authority.authority_sha256",
            )
    if np.any(valid & (clip > 1e-12)):
        development_reasons.append("NONZERO_CLIP_DELTA")
    if not stable_coverage_pass:
        development_reasons.append("STABLE_COVERAGE_FAILED")
    if not full_replay_reviewable:
        development_reasons.append("FULL_REPLAY_NOT_REVIEWABLE")

    admitted_window = window & valid & fk & limits & collision
    longest = 0
    for side in range(2):
        current = 0
        previous: int | None = None
        for selected, frame_id in zip(admitted_window[:, side].tolist(), ids.tolist(), strict=True):
            if selected and (previous is None or frame_id == previous + 1):
                current += 1
            elif selected:
                current = 1
            else:
                current = 0
            longest = max(longest, current)
            previous = frame_id
    if development_threshold_authority is not None and not any(
        reason.startswith("DEVELOPMENT_THRESHOLD_AUTHORITY") for reason in development_reasons
    ):
        minimum = int(development_threshold_authority["minimum_contiguous_frames"])
        coverage_minimum = float(development_threshold_authority["minimum_coverage_fraction"])
        if minimum <= 0 or not 0.0 <= coverage_minimum <= 1.0:
            raise Kai22SidecarError("frozen development thresholds are invalid")
        if longest < minimum:
            development_reasons.append("INDEPENDENT_CONTIGUOUS_WINDOW_TOO_SHORT")
        coverage = float(admitted_window.sum() / max(valid.sum(), 1))
        if coverage < coverage_minimum:
            development_reasons.append("INDEPENDENT_WINDOW_COVERAGE_TOO_LOW")
    else:
        minimum = None
        coverage_minimum = None
        coverage = float(admitted_window.sum() / max(valid.sum(), 1))
    development_pass = not development_reasons

    h50_reasons: list[str] = []
    if not development_pass:
        h50_reasons.append("DEVELOPMENT_R0_PREREQUISITE_FAILED")
    if not h50_validity_authority_pass:
        h50_reasons.append("H50_VALID_CURRENT_FUTURE_AUTHORITY_FAILED")
    if not h50_suffix_invariance_pass:
        h50_reasons.append("H50_SUFFIX_INVARIANCE_FAILED")
    if longest < h50_minimum_contiguous_frames:
        h50_reasons.append("H50_CONTIGUOUS_WINDOW_TOO_SHORT")
    h50_reasons.extend(
        f"TEMPORAL_INPUT:{reason}"
        for reason in validate_online_current_inputs(
            temporal_authority_audit, h50_current_input_fields
        )
    )
    h50_pass = not h50_reasons
    highest = (
        "H50_READY"
        if h50_pass
        else "DEVELOPMENT_R0"
        if development_pass
        else "KINEMATIC_ONLY"
        if kinematic_pass
        else "NONE"
    )
    return {
        "schema_version": TIER_SCHEMA_VERSION,
        "status": "PASSED_FAIL_CLOSED_TIER_ACCOUNTING",
        "highest_admitted_level": highest,
        "levels": {
            "KINEMATIC_ONLY": {
                "status": "PASS" if kinematic_pass else "BLOCKED",
                "reason_codes": sorted(set(kinematic_reasons)),
                "valid_side_frames": int(valid.sum()),
            },
            "DEVELOPMENT_R0": {
                "status": "PASS" if development_pass else "BLOCKED",
                "reason_codes": sorted(set(development_reasons)),
                "longest_independent_contiguous_frames": longest,
                "independent_window_coverage_fraction": coverage,
                "minimum_contiguous_frames": minimum,
                "minimum_coverage_fraction": coverage_minimum,
            },
            "H50_READY": {
                "status": "PASS" if h50_pass else "BLOCKED",
                "reason_codes": sorted(set(h50_reasons)),
                "required_current_input_fields": list(h50_current_input_fields),
                "minimum_contiguous_frames": h50_minimum_contiguous_frames,
                "longest_contiguous_frames": longest,
            },
        },
        "control_ground_truth": False,
        "training_eligible": h50_pass,
        "physical_deployable": False,
        "external_metric_authority": False,
    }


def write_kai22_full_fk_sidecar(
    *,
    manifest_path: Path,
    arrays_path: Path,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Write immutable NPZ+JSON artifacts and bind JSON to the NPZ bytes."""

    if manifest_path.exists() or arrays_path.exists():
        raise FileExistsError("Kai22 sidecar outputs are immutable")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.parent.resolve() != arrays_path.parent.resolve():
        raise Kai22SidecarError("manifest and arrays must share one isolated output directory")
    temporary_npz = arrays_path.with_name(f".{arrays_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    temporary_json = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        np.savez_compressed(temporary_npz, **arrays)
        payload = dict(manifest)
        payload["arrays_artifact"] = {
            "path": arrays_path.name,
            "bytes": temporary_npz.stat().st_size,
            "sha256": _sha256(temporary_npz),
        }
        with temporary_json.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_npz, arrays_path)
        os.replace(temporary_json, manifest_path)
        return payload
    finally:
        temporary_npz.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)


def reload_kai22_full_fk_sidecar(
    manifest_path: Path, arrays_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise Kai22SidecarError("sidecar manifest schema mismatch")
    ref = manifest.get("arrays_artifact", {})
    if (
        ref.get("path") != arrays_path.name
        or int(ref.get("bytes", -1)) != arrays_path.stat().st_size
        or ref.get("sha256") != _sha256(arrays_path)
    ):
        raise Kai22SidecarError("sidecar NPZ exact reference mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "schema_version",
        "source_frame",
        "timestamp_s",
        "physical_robot_side",
        "anatomical_side",
        "joint_names",
        "q22",
        "q22_valid",
        "clip_delta",
        "fk_root_relative",
        "fk_finite",
        "hard_limit_pass",
        "collision_diagnostic_known",
        "non_adjacent_self_collision_pass",
    }
    if not required.issubset(arrays):
        raise Kai22SidecarError(f"sidecar NPZ fields missing: {sorted(required - set(arrays))}")
    if str(arrays["schema_version"].item()) != SCHEMA_VERSION:
        raise Kai22SidecarError("sidecar NPZ schema mismatch")
    return manifest, arrays
