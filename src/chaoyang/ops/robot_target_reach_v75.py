"""Shared, authority-neutral Robot target/reach diagnostics for V7.1-R3."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SIDES = ("left", "right")


class RobotTargetContractError(RuntimeError):
    """Raised when a target/reach input cannot be proven from its contract."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RobotTargetContractError(f"JSON object required: {path}")
    return value


def verify_ref(reference: dict[str, Any], label: str) -> Path:
    required = {"path", "bytes", "sha256"}
    if not isinstance(reference, dict) or not required.issubset(reference):
        raise RobotTargetContractError(f"{label}: complete path/bytes/SHA reference required")
    path = Path(str(reference["path"])).resolve(strict=True)
    if path.stat().st_size != int(reference["bytes"]) or sha256_file(path) != reference["sha256"]:
        raise RobotTargetContractError(f"{label}: bytes/SHA mismatch")
    return path


def atomic_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def status_specific_artifacts(
    row: dict[str, Any], *, kind: str, session: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = row.get("status")
    statuses = {
        "arm": {
            "PASS_ARM_METHOD1_CARRIED_NO_ROUND2": ("method1_result", "method1_states"),
            "PASS_ARM_ROUND2_BIDIRECTIONAL": ("result", "states"),
            "HOLD_ARM_AFTER_TWO_METHODS_FAILED_QUALITY_C": ("result", "states"),
        },
        "hand": {
            "PASS_HAND_METHOD1_CARRIED_NO_ROUND2": ("method1_result", "method1_states"),
            "PASS_HAND_ROUND2_BIDIRECTIONAL": ("result", "states"),
            "HOLD_HAND_AFTER_TWO_METHODS_FAILED_QUALITY_C": ("result", "states"),
        },
    }
    if kind not in statuses or status not in statuses[kind]:
        raise RobotTargetContractError(f"{session}: unsupported {kind} terminal status {status!r}")
    result_key, states_key = statuses[kind][status]
    result, states = row.get(result_key), row.get(states_key)
    if not isinstance(result, dict) or not isinstance(states, dict):
        raise RobotTargetContractError(
            f"{session}: {kind} row {status!r} lacks {result_key}/{states_key} artifact references"
        )
    return result, states


def proper_se3_mask(transforms: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if transforms.ndim != 4 or transforms.shape[-2:] != (4, 4):
        raise RobotTargetContractError(f"expected [T,S,4,4] transforms, got {transforms.shape}")
    if valid.shape != transforms.shape[:2]:
        raise RobotTargetContractError(f"valid/transform shape mismatch: {valid.shape}/{transforms.shape}")
    selected = transforms[valid]
    if not len(selected):
        return np.zeros(valid.shape, dtype=bool)
    finite = np.isfinite(selected).all(axis=(1, 2))
    rotation = selected[:, :3, :3]
    orthogonal = np.max(np.abs(np.swapaxes(rotation, 1, 2) @ rotation - np.eye(3)), axis=(1, 2)) <= 1e-4
    determinant = np.abs(np.linalg.det(rotation) - 1.0) <= 1e-4
    bottom = np.max(np.abs(selected[:, 3] - np.asarray([0.0, 0.0, 0.0, 1.0])), axis=1) <= 1e-8
    output = np.zeros(valid.shape, dtype=bool)
    output[valid] = finite & orthogonal & determinant & bottom
    return output


def consecutive_path_length(points: np.ndarray, frames: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    consecutive = np.diff(frames) == 1
    return float(np.linalg.norm(np.diff(points, axis=0)[consecutive], axis=1).sum())


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator <= 1e-9 else float(numerator / denominator)


def side_target_metrics(
    human_wrist: np.ndarray,
    target: np.ndarray,
    actual: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    frames = np.flatnonzero(valid)
    if not len(frames):
        return {"valid_frames": 0, "status": "UNKNOWN_NO_OBSERVATION"}
    anchor = int(frames[0])
    human_points = human_wrist[frames]
    target_points = target[frames, :3, 3]
    actual_points = actual[frames, :3, 3]
    human_delta = human_points - human_wrist[anchor]
    target_delta = target_points - target[anchor, :3, 3]
    delta_error_mm = 1000.0 * np.linalg.norm(target_delta - human_delta, axis=1)
    target_actual_error_mm = 1000.0 * np.linalg.norm(actual_points - target_points, axis=1)
    target_extent = float(np.percentile(np.linalg.norm(target_delta, axis=1), 95))
    actual_delta = actual_points - actual[anchor, :3, 3]
    actual_extent = float(np.percentile(np.linalg.norm(actual_delta, axis=1), 95))
    active = np.linalg.norm(target_delta, axis=1) >= 0.005
    dot = np.sum(target_delta[active] * actual_delta[active], axis=1)
    norms = np.linalg.norm(target_delta[active], axis=1) * np.linalg.norm(actual_delta[active], axis=1)
    directional = norms > 1e-12
    cosines = dot[directional] / norms[directional]
    away_ratio = None if not len(cosines) else float(np.mean(cosines < 0.0))
    large_tracking_error_ratio = float(np.mean(target_actual_error_mm > 10.0))
    return {
        "status": "MEASURED",
        "valid_frames": int(len(frames)),
        "anchor_frame": anchor,
        "human_to_target_delta_error_mm_p95": float(np.percentile(delta_error_mm, 95)),
        "human_path_length_m": consecutive_path_length(human_points, frames),
        "target_path_length_m": consecutive_path_length(target_points, frames),
        "actual_path_length_m": consecutive_path_length(actual_points, frames),
        "target_motion_extent_m_p95": target_extent,
        "actual_motion_extent_m_p95": actual_extent,
        "actual_target_motion_retention": safe_ratio(actual_extent, target_extent),
        "target_actual_position_error_mm_p50": float(np.percentile(target_actual_error_mm, 50)),
        "target_actual_position_error_mm_p95": float(np.percentile(target_actual_error_mm, 95)),
        "target_tracking_large_error_ratio": large_tracking_error_ratio,
        "persistent_away_frame_ratio": away_ratio,
        "world_axis_target_extent_m": (np.max(target_points, axis=0) - np.min(target_points, axis=0)).tolist(),
        "world_axis_actual_extent_m": (np.max(actual_points, axis=0) - np.min(actual_points, axis=0)).tolist(),
    }


def percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    return None if not array.size else float(np.percentile(array, q))
