from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "HumanEgo"))

from utils.visual_aux_contract import (  # noqa: E402
    VisualAuxContractError,
    validate_future_2d_npz,
    validate_real_robot_manifest,
)


def reference(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_future_2d_shape_and_range(tmp_path: Path) -> None:
    path = tmp_path / "labels.npz"
    np.savez(path, future_2d_xy_original=np.zeros((2, 50, 2, 2)), future_2d_xy_normalized=np.zeros((2, 50, 2, 2)), future_2d_valid=np.ones((2, 50, 2), dtype=bool))
    assert validate_future_2d_npz(path, 2)["valid_targets"] == 200


def test_future_2d_rejects_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "labels.npz"
    normalized = np.zeros((1, 50, 2, 2))
    normalized[0, 0, 0, 0] = 2
    np.savez(path, future_2d_xy_original=np.zeros_like(normalized), future_2d_xy_normalized=normalized, future_2d_valid=np.ones((1, 50, 2), dtype=bool))
    with pytest.raises(VisualAuxContractError, match="escape"):
        validate_future_2d_npz(path, 1)


def test_real_robot_manifest_must_be_pass(tmp_path: Path) -> None:
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"x")
    ref = reference(artifact)
    manifest = {
        "schema_version": "real-robot-policy-manifest-v1",
        "status": "BLOCKED_EXTERNAL",
        "action_contract": {"source": "REAL_ROBOT_DEMONSTRATION", "synchronized": True, "required_arrays": ["timestamp_s", "action", "robot_state", "frame_index"], "human_video_action_forbidden": True},
        "sessions": [{"session": "s", "task": "chips", "split": "train", "rgb": ref, "action": ref, "robot_state": ref, "timestamp": ref, "frame_ledger": ref}],
        "claim_limit": "test"
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VisualAuxContractError, match="not PASS"):
        validate_real_robot_manifest(path, ROOT / "contracts" / "real_robot_policy_manifest_v1.schema.json")
