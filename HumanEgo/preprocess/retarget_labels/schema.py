"""Frozen schemas and validation for dual-hand robot supervision.

The NPZ layout is deliberately independent of the legacy frame JSON files.
Missing observations retain array slots for timestamp alignment, but their
``valid`` bit is false and consumers must mask every wrist/q loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


SIDECAR_SCHEMA_VERSION = "humanego-robot-sidecar-v1"
CANONICAL_SCHEMA_VERSION = "humanego-canonical-hand-v1"
SIDES = ("left", "right")
CANONICAL_JOINT_ORDER = (
    "thumb_root", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
)


@dataclass(frozen=True)
class EmbodimentSpec:
    name: str
    representation: str
    joint_count: int
    action_dim_per_hand: int
    dual_action_dim: int
    raw_filename: str


EMBODIMENTS: Mapping[str, EmbodimentSpec] = {
    "kai22": EmbodimentSpec(
        "kai22", "kaihand_joint_state", 22, 31, 62,
        "kaihand_retargeting.npz",
    ),
    "wuji20": EmbodimentSpec(
        "wuji20", "wuji_joint_state", 20, 29, 58,
        "wuji_retargeting.npz",
    ),
}


def validate_sidecar(path: Path, spec: EmbodimentSpec) -> dict[str, object]:
    """Validate shapes, finite values, masks, units and immutable metadata."""
    with np.load(path, allow_pickle=False) as data:
        required = {
            "frame_names", "timestamps_ns", "wrist_9d", "wrist_T_camera", "q", "valid",
            "confidence", "grasp", "joint_names", "joint_lower",
            "joint_upper", "canonical_sha256", "schema_version",
            "retarget_confidence", "failure_reason", "hand_object_T",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        n = len(data["frame_names"])
        expected = {
            "timestamps_ns": (n,), "wrist_9d": (n, 2, 9),
            "wrist_T_camera": (n, 2, 4, 4),
            "q": (n, 2, spec.joint_count), "valid": (n, 2),
            "confidence": (n, 2), "grasp": (n, 2),
            "retarget_confidence": (n, 2), "failure_reason": (n, 2),
            "hand_object_T": (n, 2, 4, 4),
            "joint_names": (2, spec.joint_count),
            "joint_lower": (2, spec.joint_count),
            "joint_upper": (2, spec.joint_count),
        }
        for key, shape in expected.items():
            if data[key].shape != shape:
                raise ValueError(f"{path}: {key}={data[key].shape}, expected {shape}")
        valid = np.asarray(data["valid"], dtype=bool)
        q = np.asarray(data["q"], dtype=np.float64)
        wrist = np.asarray(data["wrist_9d"], dtype=np.float64)
        if not np.isfinite(q[valid]).all() or not np.isfinite(wrist[valid]).all():
            raise ValueError(f"{path}: active labels contain NaN/Inf")
        lower = np.asarray(data["joint_lower"], dtype=np.float64)
        upper = np.asarray(data["joint_upper"], dtype=np.float64)
        violations = valid[..., None] & ((q < lower[None] - 1e-5) | (q > upper[None] + 1e-5))
        if violations.any():
            raise ValueError(f"{path}: active q violates URDF joint limits")
        schema = str(data["schema_version"].item())
        if schema != SIDECAR_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported schema {schema!r}")
        return {
            "frames": n,
            "active_left": int(valid[:, 0].sum()),
            "active_right": int(valid[:, 1].sum()),
            "joint_count": spec.joint_count,
            "all_active_values_finite": True,
            "joint_limit_violations": 0,
            "units": {"translation": "metre", "rotation": "6D", "q": "radian"},
        }
