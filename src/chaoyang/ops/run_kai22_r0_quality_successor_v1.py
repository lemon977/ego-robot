#!/usr/bin/env python3
"""CPU-only Kai22 R0 own-gate evaluator for frozen W1 packages.

This successor deliberately does not equate raw HaWoR structural eligibility
with R0 quality.  It recomputes the structural mask, retargets every eligible
side-frame, runs pinned-URDF limit/FK and intra-hand collision checks on every
retargeted frame, and computes q22 motion only inside contiguous windows using
the source video timestamps.

The current V2.1 contracts do not freeze a q22 velocity/acceleration threshold
or a minimum quality-window length.  ``DIAGNOSTIC_ONLY_NO_QUALITY_UPGRADE`` is
therefore the only mode usable by the supplied A1 example.  A future governed
contract may use ``FROZEN_QUALITY_GATE`` only when all three missing thresholds
and their exact authority reference are supplied.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

import numpy as np


SCHEMA_VERSION = "0915-kai22-r0-quality-successor-config-v1"
RESULT_SCHEMA = "0915-kai22-r0-quality-successor-result-v1"
DETAIL_SCHEMA = "0915-kai22-r0-quality-propagation-v1"
SIDES = ("left", "right")
HUMAN_TO_PHYSICAL = np.asarray((1, 0), dtype=np.int64)
MANO_JOINT_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
COHORTS: dict[str, tuple[tuple[str, str, int], ...]] = {
    "W1_DIAG": (
        ("play_cards_0915_044", "playing_cards", 166),
        ("get_potato_chips_0915_097", "potato_chips", 394),
    ),
    "W1_ADOPTION": (
        ("play_cards_0915_106", "playing_cards", 170),
        ("get_potato_chips_0915_029", "potato_chips", 234),
    ),
}


class ContractError(RuntimeError):
    """The exact input or quality contract is not satisfied."""


class RuntimeGateError(RuntimeError):
    """The CPU evaluator could not execute a required own gate."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def exact_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def verify_ref(value: Mapping[str, Any], *, label: str) -> Path:
    if not {"path", "bytes", "sha256"}.issubset(value):
        raise ContractError(f"{label} must be an exact path/bytes/sha256 reference")
    path = Path(str(value["path"])).resolve(strict=True)
    if path.stat().st_size != int(value["bytes"]) or sha256(path) != str(value["sha256"]):
        raise ContractError(f"{label} exact reference drift: {path}")
    return path


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def maximal_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    ids = np.flatnonzero(np.asarray(mask, dtype=bool))
    if ids.size == 0:
        return []
    chunks = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if chunk.size]


def structural_mask(hawor: Mapping[str, np.ndarray], width: int, height: int) -> np.ndarray:
    """Recompute the A1/A2 structural mask; this is evaluation scope, not quality."""

    observed = np.asarray(hawor["observed"], dtype=bool)
    joints_2d = np.asarray(hawor["joints_2d"], dtype=np.float64)
    joints_3d = np.asarray(hawor["joints_3d_camera"], dtype=np.float64)
    roots = np.asarray(hawor["root_orient_camera"], dtype=np.float64)
    provenance = np.asarray(hawor["provenance"]).astype(str)
    if observed.ndim != 2 or observed.shape[0] != 2:
        raise ContractError(f"observed must be [2,T], got {observed.shape}")
    frames = observed.shape[1]
    if joints_2d.shape != (2, frames, 21, 2):
        raise ContractError("joints_2d shape mismatch")
    if joints_3d.shape != (2, frames, 21, 3):
        raise ContractError("joints_3d_camera shape mismatch")
    if roots.shape != (2, frames, 3, 3) or provenance.shape != (2, frames):
        raise ContractError("root/provenance shape mismatch")
    finite_2d = np.isfinite(joints_2d).all(axis=(2, 3))
    finite_3d = np.isfinite(joints_3d).all(axis=(2, 3))
    with np.errstate(invalid="ignore"):
        in_frame = (
            (joints_2d[..., 0] >= 0).all(axis=2)
            & (joints_2d[..., 0] < width).all(axis=2)
            & (joints_2d[..., 1] >= 0).all(axis=2)
            & (joints_2d[..., 1] < height).all(axis=2)
        )
        positive_depth = (joints_3d[..., 2] > 0).all(axis=2)
    finite_root = np.isfinite(roots).all(axis=(2, 3))
    safe_roots = np.where(finite_root[..., None, None], roots, np.eye(3))
    determinant = np.linalg.det(safe_roots)
    orthogonality = np.max(
        np.abs(np.transpose(safe_roots, (0, 1, 3, 2)) @ safe_roots - np.eye(3)),
        axis=(2, 3),
    )
    root_valid = finite_root & (determinant > 0) & (orthogonality <= 0.0001)
    return (
        observed
        & finite_2d
        & in_frame
        & finite_3d
        & positive_depth
        & root_valid
        & (provenance == "OBSERVED")
    )


def windows_to_mask(row: Mapping[str, Any], frame_count: int) -> np.ndarray:
    mask = np.zeros(frame_count, dtype=bool)
    for index, window in enumerate(row.get("windows", [])):
        start = int(window["start_frame_inclusive"])
        end = int(window["end_frame_inclusive"])
        if start < 0 or end < start or end >= frame_count:
            raise ContractError(f"invalid structural window {index}: {start}..{end}")
        if int(window["frame_count"]) != end - start + 1 or mask[start : end + 1].any():
            raise ContractError("overlapping or inconsistent structural window")
        if window.get("kai22_r0_evaluation_allowed") is not True:
            raise ContractError("window is not explicitly allowed for R0 evaluation")
        if window.get("kai22_r0_quality_admitted") != "PENDING_R0_OWN_GATES":
            raise ContractError("source window already claims unexpected R0 quality")
        mask[start : end + 1] = True
    if int(row["structural_admitted_frames"]) != int(mask.sum()):
        raise ContractError("structural window count mismatch")
    return mask


def motion_diagnostics(q22: np.ndarray, timestamps_s: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Use real dt and never form derivatives across invalid/gap boundaries."""

    q = np.asarray(q22, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if q.ndim != 2 or q.shape[1] != 22 or q.shape[0] != times.size or mask.shape != (times.size,):
        raise ContractError("q22 motion axis mismatch")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ContractError("source video_time_s must be finite and strictly increasing")
    velocity_values: list[float] = []
    acceleration_values: list[float] = []
    velocity_by_frame = np.full((times.size, 22), np.nan, dtype=np.float64)
    acceleration_by_frame = np.full((times.size, 22), np.nan, dtype=np.float64)
    motion_context = np.zeros(times.size, dtype=bool)
    for start, end in maximal_runs(mask & np.isfinite(q).all(axis=1)):
        ids = np.arange(start, end + 1, dtype=np.int64)
        if ids.size < 2:
            continue
        dt = np.diff(times[ids])
        velocity = np.diff(q[ids], axis=0) / dt[:, None]
        velocity_by_frame[ids[1:]] = velocity
        velocity_values.extend(np.abs(velocity).ravel().tolist())
        if ids.size < 3:
            continue
        middle_dt = 0.5 * (dt[:-1] + dt[1:])
        acceleration = np.diff(velocity, axis=0) / middle_dt[:, None]
        acceleration_by_frame[ids[1:-1]] = acceleration
        acceleration_values.extend(np.abs(acceleration).ravel().tolist())
        # Both outer frames are explicit boundary-invalid for the combined
        # velocity+acceleration motion gate.
        motion_context[ids[1:-1]] = True
    return {
        "velocity_by_frame": velocity_by_frame,
        "acceleration_by_frame": acceleration_by_frame,
        "motion_context": motion_context,
        "summary": {
            "max_abs_velocity_rad_s": max(velocity_values, default=None),
            "p95_abs_velocity_rad_s": (
                float(np.percentile(velocity_values, 95)) if velocity_values else None
            ),
            "max_abs_acceleration_rad_s2": max(acceleration_values, default=None),
            "p95_abs_acceleration_rad_s2": (
                float(np.percentile(acceleration_values, 95)) if acceleration_values else None
            ),
            "actual_dt_used": True,
            "cross_gap_derivatives": 0,
            "boundary_context_frames_each_side": 1,
        },
    }


def numeric_threshold_vector(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = np.full(22, float(value), dtype=np.float64)
    else:
        result = np.asarray(value, dtype=np.float64)
    if result.shape != (22,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ContractError(f"{name} must be a positive scalar or 22-vector")
    return result


def evaluate_motion_gate(
    diagnostics: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    velocity = np.asarray(diagnostics["velocity_by_frame"], dtype=np.float64)
    acceleration = np.asarray(diagnostics["acceleration_by_frame"], dtype=np.float64)
    context = np.asarray(diagnostics["motion_context"], dtype=bool)
    if policy.get("status") != "FROZEN_NUMERIC_THRESHOLDS":
        return np.zeros(context.shape, dtype=bool), {
            "status": "PENDING_MISSING_FROZEN_Q22_MOTION_THRESHOLDS",
            "gate_executed": False,
            "pass_frames": 0,
        }
    velocity_limit = numeric_threshold_vector(
        policy.get("max_abs_velocity_rad_s"), name="max_abs_velocity_rad_s"
    )
    acceleration_limit = numeric_threshold_vector(
        policy.get("max_abs_acceleration_rad_s2"), name="max_abs_acceleration_rad_s2"
    )
    with np.errstate(invalid="ignore"):
        passed = (
            context
            & np.isfinite(velocity).all(axis=1)
            & np.isfinite(acceleration).all(axis=1)
            & (np.abs(velocity) <= velocity_limit[None, :] + 1e-12).all(axis=1)
            & (np.abs(acceleration) <= acceleration_limit[None, :] + 1e-12).all(axis=1)
        )
    return passed, {
        "status": "PASS_OR_REJECT_PER_FRAME",
        "gate_executed": True,
        "pass_frames": int(passed.sum()),
        "velocity_threshold_rad_s": velocity_limit.tolist(),
        "acceleration_threshold_rad_s2": acceleration_limit.tolist(),
    }


class RobotBackend(Protocol):
    joint_names: tuple[tuple[str, ...], tuple[str, ...]]
    lower: tuple[np.ndarray, np.ndarray]
    upper: tuple[np.ndarray, np.ndarray]
    collision_shape_counts: tuple[int, int]

    def retarget(self, points: np.ndarray, physical_side: int) -> tuple[np.ndarray, float]: ...

    def fk(self, q22: np.ndarray, physical_side: int) -> tuple[bool, str | None]: ...

    def collision(
        self, q22: np.ndarray, physical_side: int, tolerance_m: float
    ) -> tuple[bool, int, float, list[dict[str, Any]]]: ...

    def close(self) -> None: ...


class PinnedRobotBackend:
    """Thin adapter around the current pinned Chaoyang URDF implementation."""

    def __init__(self, repo_root: Path) -> None:
        sys.path.insert(0, str((repo_root / "src").resolve()))
        from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
        from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
        from chaoyang.pipeline.robot_visual_relative_v1 import hand_limits, retarget_kaihand_frame

        try:
            import pybullet as bullet
        except ModuleNotFoundError as exc:  # pragma: no cover - local minimal host
            raise RuntimeGateError("pybullet is required for full-frame collision") from exc
        self._bullet = bullet
        self._forward_kinematics = forward_kinematics
        self._retarget_kaihand_frame = retarget_kaihand_frame
        self._assets = load_pinned_robot_assets(repo_root)
        self._models = (self._assets.left_hand, self._assets.right_hand)
        self._limits = hand_limits(self._assets)
        moving = tuple(
            tuple(joint for joint in model.joints if joint.joint_type != "fixed")
            for model in self._models
        )
        self.joint_names = tuple(tuple(joint.name for joint in side) for side in moving)  # type: ignore[assignment]
        if any(len(names) != 22 for names in self.joint_names):
            raise RuntimeGateError("pinned KaiHand moving-joint closure is not 22")
        self.lower = tuple(np.asarray(item.lower, dtype=np.float64) for item in self._limits)  # type: ignore[assignment]
        self.upper = tuple(np.asarray(item.upper, dtype=np.float64) for item in self._limits)  # type: ignore[assignment]
        self._client = bullet.connect(bullet.DIRECT)
        if self._client < 0:
            raise RuntimeGateError("pybullet DIRECT connection failed")
        flags = bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self._bodies: list[int] = []
        self._joint_index_by_name: list[dict[str, int]] = []
        self._link_names: list[dict[int, str]] = []
        urdf_paths = tuple(model.path.resolve(strict=True) for model in self._models)
        for side, path in enumerate(urdf_paths):
            body = bullet.loadURDF(
                str(path), useFixedBase=True, flags=flags, physicsClientId=self._client
            )
            self._bodies.append(body)
            indices: dict[str, int] = {}
            links = {-1: bullet.getBodyInfo(body, physicsClientId=self._client)[0].decode("utf-8")}
            for index in range(bullet.getNumJoints(body, physicsClientId=self._client)):
                info = bullet.getJointInfo(body, index, physicsClientId=self._client)
                name = info[1].decode("utf-8")
                links[index] = info[12].decode("utf-8")
                if int(info[3]) >= 0:
                    indices[name] = index
            if set(indices) != set(self.joint_names[side]):
                raise RuntimeGateError("pybullet/repository KaiHand joint identity drift")
            self._joint_index_by_name.append(indices)
            self._link_names.append(links)
        counts = []
        for body in self._bodies:
            count = sum(
                bool(bullet.getCollisionShapeData(body, link, physicsClientId=self._client))
                for link in range(-1, bullet.getNumJoints(body, physicsClientId=self._client))
            )
            counts.append(count)
        self.collision_shape_counts = (counts[0], counts[1])
        # Both current KaiHand URDFs have base + all 22 moving links covered.
        if any(count != 23 for count in self.collision_shape_counts):
            raise RuntimeGateError(
                f"incomplete KaiHand collision geometry: {self.collision_shape_counts}"
            )

    def retarget(self, points: np.ndarray, physical_side: int) -> tuple[np.ndarray, float]:
        return self._retarget_kaihand_frame(points, self._limits[physical_side])

    def fk(self, q22: np.ndarray, physical_side: int) -> tuple[bool, str | None]:
        try:
            transforms = self._forward_kinematics(
                self._models[physical_side],
                dict(zip(self.joint_names[physical_side], np.asarray(q22, float), strict=True)),
            )
        except Exception as exc:  # output is evidence, so preserve a bounded reason
            return False, f"{type(exc).__name__}:{str(exc)[:240]}"
        passed = (
            set(transforms) == set(self._models[physical_side].links)
            and all(np.asarray(value).shape == (4, 4) for value in transforms.values())
            and all(np.isfinite(value).all() for value in transforms.values())
        )
        return passed, None if passed else "NONFINITE_OR_INCOMPLETE_LINK_TRANSFORMS"

    def collision(
        self, q22: np.ndarray, physical_side: int, tolerance_m: float
    ) -> tuple[bool, int, float, list[dict[str, Any]]]:
        bullet = self._bullet
        body = self._bodies[physical_side]
        for name, value in zip(self.joint_names[physical_side], q22, strict=True):
            bullet.resetJointState(
                body,
                self._joint_index_by_name[physical_side][name],
                float(value),
                physicsClientId=self._client,
            )
        bullet.performCollisionDetection(physicsClientId=self._client)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int, float]] = set()
        for point in bullet.getContactPoints(body, body, physicsClientId=self._client):
            distance = float(point[8])
            penetration = max(0.0, -distance)
            key = (int(point[3]), int(point[4]), round(distance, 12))
            if penetration <= tolerance_m or key in seen:
                continue
            seen.add(key)
            rows.append({
                "link_a": self._link_names[physical_side].get(int(point[3]), str(point[3])),
                "link_b": self._link_names[physical_side].get(int(point[4]), str(point[4])),
                "contact_distance_m": distance,
                "penetration_depth_m": penetration,
            })
        deepest = max((row["penetration_depth_m"] for row in rows), default=0.0)
        return not rows, len(rows), float(deepest), rows[:20]

    def close(self) -> None:
        if getattr(self, "_client", -1) >= 0:
            self._bullet.disconnect(self._client)
            self._client = -1


def timestamps_from_manifest(manifest_path: Path, expected_frames: int) -> tuple[np.ndarray, np.ndarray]:
    manifest = load_json(manifest_path)
    files = manifest.get("files")
    if manifest.get("count") != expected_frames or not isinstance(files, list) or len(files) != expected_frames:
        raise ContractError("source metadata manifest frame count mismatch")
    frame_ids: list[int] = []
    timestamp_ns: list[int] = []
    video_time_s: list[float] = []
    for index, item in enumerate(files):
        path = verify_ref(item, label=f"source metadata frame {index}")
        value = load_json(path).get("metadata", {})
        frame_ids.append(int(value.get("frame_index", index)))
        timestamp_ns.append(int(value["ts"]))
        video_time_s.append(float(value["video_time_s"]))
    if frame_ids != list(range(expected_frames)):
        raise ContractError("source metadata frame axis is not exact 0..T-1")
    ns = np.asarray(timestamp_ns, dtype=np.int64)
    seconds = np.asarray(video_time_s, dtype=np.float64)
    if np.any(np.diff(ns) <= 0) or not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0):
        raise ContractError("source timestamps are not strictly increasing")
    return ns, seconds


def validate_motion_policy(config: Mapping[str, Any]) -> bool:
    mode = config.get("run_mode")
    policy = config.get("gate_policy", {}).get("motion", {})
    if mode == "DIAGNOSTIC_ONLY_NO_QUALITY_UPGRADE":
        if policy.get("status") != "MISSING_FROZEN_THRESHOLDS":
            raise ContractError("diagnostic mode must state MISSING_FROZEN_THRESHOLDS")
        return False
    if mode != "FROZEN_QUALITY_GATE":
        raise ContractError("unknown run_mode")
    if policy.get("status") != "FROZEN_NUMERIC_THRESHOLDS":
        raise ContractError("quality mode requires frozen numeric motion thresholds")
    numeric_threshold_vector(policy.get("max_abs_velocity_rad_s"), name="velocity")
    numeric_threshold_vector(policy.get("max_abs_acceleration_rad_s2"), name="acceleration")
    if not isinstance(policy.get("minimum_window_frames"), int) or policy["minimum_window_frames"] < 3:
        raise ContractError("quality mode requires minimum_window_frames >= 3")
    if policy.get("numeric_thresholds_changed") is not False:
        raise ContractError("successor may not silently change numeric thresholds")
    verify_ref(policy.get("threshold_authority_ref", {}), label="motion threshold authority")
    return True


def load_source_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    role = str(config.get("cohort_role"))
    if role not in COHORTS:
        raise ContractError("cohort_role must be W1_DIAG or W1_ADOPTION")
    source = config.get("source_package")
    if not isinstance(source, Mapping):
        raise ContractError("source_package object required")
    result_path = verify_ref(source.get("result", {}), label="source package RESULT")
    prepared_path = verify_ref(source.get("prepared_manifest", {}), label="prepared manifest")
    admission_path = verify_ref(source.get("consumer_admission", {}), label="consumer admission")
    windows_path = verify_ref(source.get("window_validity", {}), label="window validity")
    result = load_json(result_path)
    prepared = load_json(prepared_path)
    admission = load_json(admission_path)
    windows = load_json(windows_path)
    source_signature = str(config.get("source_signature_sha256"))
    parent_candidate = str(config.get("parent_candidate_signature_sha256"))
    if role == "W1_DIAG":
        signature_fields = [
            result.get("candidate_signature_sha256"),
            admission.get("candidate_signature_sha256"),
            windows.get("candidate_signature_sha256"),
            prepared.get("candidate_signature_sha256"),
        ]
    else:
        signature_fields = [
            result.get("adoption_adapter_or_run_signature_sha256"),
            admission.get("adoption_adapter_or_run_signature_sha256"),
            windows.get("adoption_adapter_or_run_signature_sha256"),
            prepared.get("adoption_adapter_or_run_signature_sha256"),
        ]
        parent_fields = [
            admission.get("parent_candidate_signature_sha256"),
            windows.get("parent_candidate_signature_sha256"),
            prepared.get("parent_candidate_signature_sha256"),
        ]
        if len(set(parent_fields)) != 1 or parent_fields[0] != parent_candidate:
            raise ContractError(f"W1-ADOPTION parent candidate lineage drift: {parent_fields}")
    if role == "W1_DIAG" and parent_candidate != source_signature:
        raise ContractError("W1-DIAG source signature must equal its frozen parent candidate")
    if any(value != source_signature for value in signature_fields):
        raise ContractError(f"candidate/run signature split or drift: {signature_fields}")
    if prepared.get("cohort_role") != role or int(prepared.get("session_count", -1)) != 2:
        raise ContractError("prepared manifest cohort mismatch")
    expected = COHORTS[role]
    prepared_rows = prepared.get("results", [])
    if [row.get("session_id") for row in prepared_rows] != [row[0] for row in expected]:
        raise ContractError("fixed cohort order mismatch")
    admission_by_id = {row["session_id"]: row for row in admission.get("sessions", [])}
    window_by_key = {
        (row["session_id"], row["side"]): row for row in windows.get("rows", [])
    }
    return {
        "result": result,
        "prepared": prepared,
        "admission": admission,
        "windows": windows,
        "prepared_rows": prepared_rows,
        "admission_by_id": admission_by_id,
        "window_by_key": window_by_key,
        "paths": [result_path, prepared_path, admission_path, windows_path],
    }


def evaluate_session(
    *,
    session: Mapping[str, Any],
    admission: Mapping[str, Any],
    window_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    hawor: Mapping[str, np.ndarray],
    timestamps_s: np.ndarray,
    backend: RobotBackend,
    gate_policy: Mapping[str, Any],
    quality_mode: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    session_id = str(session["session_id"])
    frame_count = int(session["frame_count"])
    if tuple(str(value) for value in np.asarray(hawor["mano_joint_names"]).tolist()) != MANO_JOINT_NAMES:
        raise ContractError("MANO21 joint identity mismatch")
    if tuple(str(value) for value in np.asarray(hawor["anatomical_side_names"]).tolist()) != SIDES:
        raise ContractError("anatomical side identity mismatch")
    if not np.array_equal(np.asarray(hawor["original_frame_indices"]), np.arange(frame_count)):
        raise ContractError("HaWoR frame axis mismatch")
    domain = session.get("input_domain", {})
    output_size = domain.get("output_size", [1280, 960])
    width, height = int(output_size[0]), int(output_size[1])
    recomputed = structural_mask(hawor, width, height)
    source_window_mask = np.zeros_like(recomputed)
    for anatomical_side, name in enumerate(SIDES):
        row = window_by_key.get((session_id, name))
        if row is None:
            raise ContractError(f"missing structural window row: {session_id}/{name}")
        source_window_mask[anatomical_side] = windows_to_mask(row, frame_count)
    if not np.array_equal(source_window_mask, recomputed):
        raise ContractError("source structural windows differ from raw HaWoR recomputation")
    source_sides = {row["side"]: row for row in admission.get("sides", [])}
    if set(source_sides) != set(SIDES):
        raise ContractError("consumer admission must contain both anatomical sides")
    for side, name in enumerate(SIDES):
        if source_sides[name].get("r0_quality_admitted") != "PENDING_R0_OWN_GATES":
            raise ContractError("source package does not leave R0 quality pending")
        if int(source_sides[name]["r0_structural_eligible_frames"]) != int(recomputed[side].sum()):
            raise ContractError("consumer admission structural count mismatch")

    joints = np.asarray(hawor["joints_3d_camera"], dtype=np.float64)
    q22 = np.full((frame_count, 2, 22), np.nan, dtype=np.float64)
    retarget_loss = np.full((frame_count, 2), np.nan, dtype=np.float64)
    q22_computed = np.zeros((frame_count, 2), dtype=bool)
    limit_pass = np.zeros((frame_count, 2), dtype=bool)
    limit_saturated = np.zeros((frame_count, 2), dtype=bool)
    fk_pass = np.zeros((frame_count, 2), dtype=bool)
    collision_pass = np.zeros((frame_count, 2), dtype=bool)
    illegal_contacts = np.zeros((frame_count, 2), dtype=np.int32)
    max_penetration_m = np.full((frame_count, 2), np.nan, dtype=np.float64)
    failures: list[dict[str, Any]] = []
    tolerance = float(gate_policy["collision"]["penetration_tolerance_m"])
    for anatomical_side in range(2):
        physical_side = int(HUMAN_TO_PHYSICAL[anatomical_side])
        for frame in np.flatnonzero(recomputed[anatomical_side]):
            try:
                q, loss = backend.retarget(joints[anatomical_side, frame], physical_side)
            except Exception as exc:
                failures.append({
                    "frame_id": int(frame), "anatomical_side": SIDES[anatomical_side],
                    "physical_kaihand": SIDES[physical_side], "gate": "RETARGET",
                    "reason": f"{type(exc).__name__}:{str(exc)[:240]}",
                })
                continue
            q = np.asarray(q, dtype=np.float64)
            if q.shape != (22,) or not np.isfinite(q).all():
                failures.append({
                    "frame_id": int(frame), "anatomical_side": SIDES[anatomical_side],
                    "physical_kaihand": SIDES[physical_side], "gate": "RETARGET",
                    "reason": "NONFINITE_OR_WRONG_SHAPE_Q22",
                })
                continue
            q22[frame, physical_side] = q
            retarget_loss[frame, physical_side] = float(loss)
            q22_computed[frame, physical_side] = True
            lower, upper = backend.lower[physical_side], backend.upper[physical_side]
            limit_pass[frame, physical_side] = bool(
                (q >= lower - 1e-9).all() and (q <= upper + 1e-9).all()
            )
            limit_saturated[frame, physical_side] = bool(
                np.isclose(q, lower, atol=1e-9).any() or np.isclose(q, upper, atol=1e-9).any()
            )
            fk_ok, fk_reason = backend.fk(q, physical_side)
            fk_pass[frame, physical_side] = fk_ok
            if not fk_ok:
                failures.append({
                    "frame_id": int(frame), "anatomical_side": SIDES[anatomical_side],
                    "physical_kaihand": SIDES[physical_side], "gate": "FK",
                    "reason": fk_reason,
                })
            collision_ok, count, deepest, contacts = backend.collision(q, physical_side, tolerance)
            collision_pass[frame, physical_side] = collision_ok
            illegal_contacts[frame, physical_side] = count
            max_penetration_m[frame, physical_side] = deepest
            if not collision_ok:
                failures.append({
                    "frame_id": int(frame), "anatomical_side": SIDES[anatomical_side],
                    "physical_kaihand": SIDES[physical_side], "gate": "INTRA_HAND_COLLISION",
                    "reason": "PENETRATION_ABOVE_TOLERANCE", "contacts": contacts,
                })

    static_pass = q22_computed & limit_pass & fk_pass & collision_pass
    motion_context = np.zeros((frame_count, 2), dtype=bool)
    motion_pass = np.zeros((frame_count, 2), dtype=bool)
    velocity = np.full((frame_count, 2, 22), np.nan, dtype=np.float64)
    acceleration = np.full((frame_count, 2, 22), np.nan, dtype=np.float64)
    motion_rows: dict[str, Any] = {}
    for physical_side, name in enumerate(SIDES):
        diag = motion_diagnostics(q22[:, physical_side], timestamps_s, q22_computed[:, physical_side])
        motion_context[:, physical_side] = np.asarray(diag["motion_context"], dtype=bool)
        velocity[:, physical_side] = np.asarray(diag["velocity_by_frame"])
        acceleration[:, physical_side] = np.asarray(diag["acceleration_by_frame"])
        passed, gate = evaluate_motion_gate(diag, gate_policy["motion"])
        motion_pass[:, physical_side] = passed
        motion_rows[name] = {**diag["summary"], **gate}

    own_gate_pass = static_pass & motion_pass if quality_mode else np.zeros_like(static_pass)
    minimum = gate_policy["motion"].get("minimum_window_frames") if quality_mode else None
    final_pass = np.zeros_like(own_gate_pass)
    quality_windows: list[dict[str, Any]] = []
    if quality_mode:
        assert isinstance(minimum, int)
        for physical_side, physical_name in enumerate(SIDES):
            anatomical_side = int(HUMAN_TO_PHYSICAL[physical_side])
            for start, end in maximal_runs(own_gate_pass[:, physical_side]):
                count = end - start + 1
                admitted = count >= minimum
                if admitted:
                    final_pass[start : end + 1, physical_side] = True
                quality_windows.append({
                    "anatomical_side": SIDES[anatomical_side],
                    "physical_kaihand": physical_name,
                    "start_frame_inclusive": start,
                    "end_frame_inclusive": end,
                    "frame_count": count,
                    "minimum_window_frames": minimum,
                    "r0_own_gates": "PASS" if admitted else "REJECTED_MINIMUM_WINDOW",
                    "r0_quality_admitted": admitted,
                })
    else:
        for physical_side, physical_name in enumerate(SIDES):
            anatomical_side = int(HUMAN_TO_PHYSICAL[physical_side])
            for start, end in maximal_runs(static_pass[:, physical_side] & motion_context[:, physical_side]):
                quality_windows.append({
                    "anatomical_side": SIDES[anatomical_side],
                    "physical_kaihand": physical_name,
                    "start_frame_inclusive": start,
                    "end_frame_inclusive": end,
                    "frame_count": end - start + 1,
                    "r0_own_gates": "PENDING_MOTION_THRESHOLDS_AND_MINIMUM_WINDOW",
                    "r0_quality_admitted": False,
                })

    side_rows = []
    for anatomical_side, anatomical_name in enumerate(SIDES):
        physical_side = int(HUMAN_TO_PHYSICAL[anatomical_side])
        source = source_sides[anatomical_name]
        side_rows.append({
            "anatomical_side": anatomical_name,
            "physical_kaihand": SIDES[physical_side],
            "hawor_session_strict": source["hawor_session_strict"],
            "hawor_side_strict": source["hawor_side_strict"],
            "structural_eligible_frames": int(recomputed[anatomical_side].sum()),
            "q22_computed_frames": int(q22_computed[:, physical_side].sum()),
            "joint_limit_pass_frames": int(limit_pass[:, physical_side].sum()),
            "joint_limit_saturated_frames": int(limit_saturated[:, physical_side].sum()),
            "fk_pass_frames": int(fk_pass[:, physical_side].sum()),
            "collision_pass_frames": int(collision_pass[:, physical_side].sum()),
            "motion_context_frames": int(motion_context[:, physical_side].sum()),
            "motion_gate_pass_frames": int(motion_pass[:, physical_side].sum()),
            "r0_quality_admitted_frames": int(final_pass[:, physical_side].sum()),
            "r0_quality_admitted": bool(final_pass[:, physical_side].any()),
            "r0_quality_status": (
                "PASS_DEVELOPMENT_R0_OWN_GATES"
                if final_pass[:, physical_side].any()
                else (
                    "REJECTED_R0_OWN_GATES"
                    if quality_mode
                    else "PENDING_MISSING_FROZEN_MOTION_AND_WINDOW_THRESHOLDS"
                )
            ),
            "motion": motion_rows[SIDES[physical_side]],
        })

    result = {
        "schema_version": DETAIL_SCHEMA,
        "session_id": session_id,
        "task": session["task"],
        "frame_count": frame_count,
        "status": (
            "COMPLETED_R0_OWN_GATES"
            if quality_mode
            else "COMPLETED_DIAGNOSTIC_NO_R0_QUALITY_UPGRADE"
        ),
        "quality_mode": quality_mode,
        "source_structural_is_quality": False,
        "q22_units": "radians",
        "side_contract": {
            "source_axis": ["anatomical_left", "anatomical_right"],
            "q22_axis": ["kaihand_left", "kaihand_right"],
            "human_to_physical": {"left": "right", "right": "left"},
        },
        "joint_order": {name: list(backend.joint_names[index]) for index, name in enumerate(SIDES)},
        "collision_scope": {
            "evaluated": "PINNED_KAIHAND_INTRA_HAND_NON_PARENT_SELF_COLLISION_EVERY_Q22_FRAME",
            "cross_hand": "UNVERIFIED",
            "tianji_arm": "UNVERIFIED",
            "object": "UNVERIFIED",
            "environment": "UNVERIFIED",
            "collision_shape_counts_base_plus_links": {
                "left": backend.collision_shape_counts[0], "right": backend.collision_shape_counts[1]
            },
            "penetration_tolerance_m": tolerance,
        },
        "sides": side_rows,
        "quality_windows": quality_windows,
        "failure_count": len(failures),
        "failures": failures,
        "control_ground_truth": False,
        "training_eligible": False,
        "physical_deployment_authorized": False,
        "claim_limit": (
            "Development-relative Kai22 q22 own-gate evidence only. Collision is intra-hand "
            "against pinned digital URDF geometry; cross-hand, arm, object, environment, "
            "control, external metric, and physical deployment remain unverified."
        ),
    }
    arrays = {
        "q22": q22,
        "retarget_loss": retarget_loss,
        "frame_id": np.arange(frame_count, dtype=np.int64),
        "timestamps_s": np.asarray(timestamps_s, dtype=np.float64),
        "structural_eligible_anatomical": recomputed,
        "q22_computed": q22_computed,
        "joint_limit_pass": limit_pass,
        "joint_limit_saturated": limit_saturated,
        "fk_pass": fk_pass,
        "intra_hand_collision_pass": collision_pass,
        "illegal_contact_count": illegal_contacts,
        "max_penetration_m": max_penetration_m,
        "motion_context_valid": motion_context,
        "velocity_rad_s": velocity,
        "acceleration_rad_s2": acceleration,
        "motion_gate_pass": motion_pass,
        "r0_static_gate_pass": static_pass,
        "r0_quality_admitted": final_pass,
        "human_to_physical": HUMAN_TO_PHYSICAL,
    }
    return result, arrays


def execute(
    *,
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    backend_factory: Callable[[Path], RobotBackend] = PinnedRobotBackend,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    config = load_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ContractError("unexpected config schema_version")
    quality_mode = validate_motion_policy(config)
    for label, value in config.get("code_refs", {}).items():
        verify_ref(value, label=f"code ref {label}")
    for label, value in config.get("robot_asset_refs", {}).items():
        verify_ref(value, label=f"robot asset ref {label}")
    source = load_source_contract(config)
    run_signature_payload = {
        "schema_version": "0915-kai22-r0-quality-successor-signature-v1",
        "implementation": exact_ref(Path(__file__)),
        "config": exact_ref(config_path),
        "source_package": config["source_package"],
        "source_signature_sha256": config["source_signature_sha256"],
        "parent_candidate_signature_sha256": config["parent_candidate_signature_sha256"],
        "cohort_role": config["cohort_role"],
        "code_refs": config["code_refs"],
        "robot_asset_refs": config["robot_asset_refs"],
        "gate_policy": config["gate_policy"],
    }
    run_signature = canonical_sha(run_signature_payload)
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    backend: RobotBackend | None = None
    try:
        backend = backend_factory(repo_root)
        rows = []
        counts = {
            "sessions": 0, "timeline_frames": 0, "structural_side_frames": 0,
            "q22_side_frames": 0, "fk_pass_side_frames": 0,
            "collision_pass_side_frames": 0, "quality_admitted_side_frames": 0,
        }
        expected = COHORTS[str(config["cohort_role"])]
        prepared_by_id = {row["session_id"]: row for row in source["prepared_rows"]}
        for session_id, task, frame_count in expected:
            prepared = prepared_by_id[session_id]
            if prepared.get("task") != task or int(prepared.get("frame_count", -1)) != frame_count:
                raise ContractError(f"fixed cohort metadata drift: {session_id}")
            admission = source["admission_by_id"].get(session_id)
            if admission is None:
                raise ContractError(f"missing consumer admission: {session_id}")
            npz_path = verify_ref(admission["npz"], label=f"{session_id} HaWoR NPZ")
            metadata_manifest = verify_ref(
                prepared["source_frame_metadata_manifest"],
                label=f"{session_id} source metadata manifest",
            )
            timestamp_ns, timestamps_s = timestamps_from_manifest(metadata_manifest, frame_count)
            with np.load(npz_path, allow_pickle=False) as archive:
                hawor = {key: np.asarray(archive[key]) for key in archive.files}
            detail, arrays = evaluate_session(
                session=prepared,
                admission=admission,
                window_by_key=source["window_by_key"],
                hawor=hawor,
                timestamps_s=timestamps_s,
                backend=backend,
                gate_policy=config["gate_policy"],
                quality_mode=quality_mode,
            )
            arrays["timestamp_ns"] = timestamp_ns
            target = staging / "sessions" / task / session_id
            target.mkdir(parents=True)
            array_path = target / "KAI22_R0_QUALITY_PROPAGATION_V1.npz"
            atomic_npz(array_path, **arrays)
            detail["inputs"] = {
                "hawor_npz": exact_ref(npz_path),
                "source_frame_metadata_manifest": exact_ref(metadata_manifest),
            }
            # Publish-path references must point at the final immutable directory.
            final_array = output_dir / "sessions" / task / session_id / array_path.name
            detail["arrays"] = {
                "path": str(final_array.resolve()),
                "bytes": array_path.stat().st_size,
                "sha256": sha256(array_path),
            }
            atomic_json(target / "RESULT.json", detail)
            row = {
                "session_id": session_id,
                "task": task,
                "status": detail["status"],
                "structural_side_frames": sum(item["structural_eligible_frames"] for item in detail["sides"]),
                "q22_side_frames": sum(item["q22_computed_frames"] for item in detail["sides"]),
                "fk_pass_side_frames": sum(item["fk_pass_frames"] for item in detail["sides"]),
                "collision_pass_side_frames": sum(item["collision_pass_frames"] for item in detail["sides"]),
                "quality_admitted_side_frames": sum(item["r0_quality_admitted_frames"] for item in detail["sides"]),
                "result": {
                    "path": str((output_dir / "sessions" / task / session_id / "RESULT.json").resolve()),
                    "bytes": (target / "RESULT.json").stat().st_size,
                    "sha256": sha256(target / "RESULT.json"),
                },
            }
            rows.append(row)
            counts["sessions"] += 1
            counts["timeline_frames"] += frame_count
            for key in (
                "structural_side_frames", "q22_side_frames", "fk_pass_side_frames",
                "collision_pass_side_frames", "quality_admitted_side_frames",
            ):
                counts[key] += int(row[key])
        status = (
            "COMPLETED_R0_OWN_GATES"
            if quality_mode
            else "COMPLETED_DIAGNOSTIC_NO_R0_QUALITY_UPGRADE"
        )
        propagation = {
            "schema_version": DETAIL_SCHEMA,
            "status": status,
            "run_signature_sha256": run_signature,
            "source_signature_sha256": config["source_signature_sha256"],
            "parent_candidate_signature_sha256": config["parent_candidate_signature_sha256"],
            "cohort_role": config["cohort_role"],
            "quality_mode": quality_mode,
            "source_structural_is_quality": False,
            "counts": counts,
            "sessions": rows,
            "r0_quality_upgraded": bool(quality_mode and counts["quality_admitted_side_frames"] > 0),
            "control_ground_truth": False,
            "training_eligible": False,
            "physical_deployment_authorized": False,
        }
        atomic_json(staging / "R0_QUALITY_PROPAGATION_V1.json", propagation)
        atomic_json(staging / "RUN_SIGNATURE.json", {
            **run_signature_payload, "run_signature_sha256": run_signature,
        })
        propagation_ref = {
            "path": str((output_dir / "R0_QUALITY_PROPAGATION_V1.json").resolve()),
            "bytes": (staging / "R0_QUALITY_PROPAGATION_V1.json").stat().st_size,
            "sha256": sha256(staging / "R0_QUALITY_PROPAGATION_V1.json"),
        }
        signature_ref = {
            "path": str((output_dir / "RUN_SIGNATURE.json").resolve()),
            "bytes": (staging / "RUN_SIGNATURE.json").stat().st_size,
            "sha256": sha256(staging / "RUN_SIGNATURE.json"),
        }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": status,
            "run_signature_sha256": run_signature,
            "source_signature_sha256": config["source_signature_sha256"],
            "parent_candidate_signature_sha256": config["parent_candidate_signature_sha256"],
            "cohort_role": config["cohort_role"],
            "counts": counts,
            "r0_quality_upgraded": propagation["r0_quality_upgraded"],
            "first_blocker": (
                None if quality_mode else "MISSING_FROZEN_Q22_MOTION_THRESHOLDS_AND_MINIMUM_WINDOW"
            ),
            "r0_quality_propagation": propagation_ref,
            "run_signature": signature_ref,
            "source_mutated": False,
            "processed_mutated": False,
            "control_ground_truth": False,
            "training_eligible": False,
            "physical_deployment_authorized": False,
            "claim_limit": (
                "CPU development evidence only. Raw HaWoR structural eligibility is not R0 "
                "quality; no control, training, external metric, or deployment authority."
            ),
        }
        atomic_json(staging / "RESULT.json", result)
        result_ref = {
            "path": str((output_dir / "RESULT.json").resolve()),
            "bytes": (staging / "RESULT.json").stat().st_size,
            "sha256": sha256(staging / "RESULT.json"),
        }
        atomic_json(staging / "RUN_RECEIPT.json", {
            "schema_version": "0915-kai22-r0-quality-successor-run-receipt-v1",
            "status": status,
            "run_signature_sha256": run_signature,
            "result": result_ref,
            "r0_quality_propagation": propagation_ref,
            "run_signature": signature_ref,
            "source_mutated": False,
            "processed_mutated": False,
            "r0_quality_upgraded": propagation["r0_quality_upgraded"],
        })
        os.replace(staging, output_dir)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if backend is not None:
            backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = execute(
            repo_root=args.repo_root,
            config_path=args.config,
            output_dir=args.output_dir,
        )
    except (ContractError, RuntimeGateError, FileExistsError, KeyError, ValueError) as exc:
        print(json.dumps({
            "status": "BLOCKED_INPUT_FINAL" if isinstance(exc, ContractError) else "FAILED_RUNTIME_FINAL",
            "error_type": type(exc).__name__,
            "first_blocker": str(exc),
            "r0_quality_upgraded": False,
        }, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
