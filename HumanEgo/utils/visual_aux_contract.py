from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import jsonschema
import numpy as np


HORIZON = 50
END_EFFECTORS = 2
XY = 2


class VisualAuxContractError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_ref(reference: Mapping[str, Any]) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute() or not path.is_file():
        raise VisualAuxContractError(f"artifact missing or non-absolute: {path}")
    if path.stat().st_size != reference.get("bytes") or sha256(path) != reference.get("sha256"):
        raise VisualAuxContractError(f"artifact bytes/SHA mismatch: {path}")
    return path


def validate_manifest(manifest: Mapping[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(manifest)


def validate_future_2d_npz(path: Path, frame_count: int) -> dict[str, int]:
    with np.load(path, allow_pickle=False) as data:
        required = {"future_2d_xy_original", "future_2d_xy_normalized", "future_2d_valid"}
        missing = sorted(required - set(data.files))
        if missing:
            raise VisualAuxContractError(f"future 2D arrays missing: {missing}")
        original = np.asarray(data["future_2d_xy_original"])
        normalized = np.asarray(data["future_2d_xy_normalized"])
        valid = np.asarray(data["future_2d_valid"])
    expected_xy = (frame_count, HORIZON, END_EFFECTORS, XY)
    expected_valid = (frame_count, HORIZON, END_EFFECTORS)
    if original.shape != expected_xy or normalized.shape != expected_xy or valid.shape != expected_valid:
        raise VisualAuxContractError(
            f"future 2D shape mismatch: original={original.shape} normalized={normalized.shape} valid={valid.shape}"
        )
    valid = valid.astype(bool)
    if not np.isfinite(original[valid]).all() or not np.isfinite(normalized[valid]).all():
        raise VisualAuxContractError("valid future 2D labels contain non-finite values")
    if valid.any() and ((normalized[valid] < 0).any() or (normalized[valid] > 1).any()):
        raise VisualAuxContractError("valid normalized future 2D labels escape [0,1]")
    return {"frame_count": frame_count, "valid_targets": int(valid.sum()), "total_targets": int(valid.size)}


def validate_visual_aux_manifest(manifest_path: Path, schema_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, schema_path)
    summaries = []
    for row in manifest["sessions"]:
        for key in ("human_raw_rgb", "robotized_rgb", "future_2d_labels", "paired_frame_ledger"):
            verify_ref(row[key])
        summaries.append(validate_future_2d_npz(Path(row["future_2d_labels"]["path"]), int(row["frame_count"])))
    return {"status": "PASS", "sessions": len(summaries), "valid_targets": sum(item["valid_targets"] for item in summaries)}


def validate_real_robot_manifest(manifest_path: Path, schema_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, schema_path)
    for row in manifest["sessions"]:
        for key in ("rgb", "action", "robot_state", "timestamp", "frame_ledger"):
            verify_ref(row[key])
    if manifest["status"] != "PASS_REAL_ROBOT_ACTION":
        raise VisualAuxContractError("real Robot action manifest is not PASS_REAL_ROBOT_ACTION")
    return {"status": "PASS", "sessions": len(manifest["sessions"])}
