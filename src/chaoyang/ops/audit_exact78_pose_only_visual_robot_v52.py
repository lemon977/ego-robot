#!/usr/bin/env python3
"""Audit V5.2 POSE_ONLY_VISUAL_ROBOT candidates and publish a current index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "exact78-pose-only-visual-robot-candidate-index-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_ref(root: Path, item: dict[str, Any], label: str) -> Path:
    p = Path(item["path"])
    if not p.is_absolute():
        p = root / p
    if not p.is_file() or p.is_symlink():
        raise RuntimeError(f"{label}: regular non-symlink file required: {p}")
    if p.stat().st_size != int(item["bytes"]) or sha256(p) != item["sha256"]:
        raise RuntimeError(f"{label}: byte/SHA mismatch: {p}")
    return p


def decoded_frames(video: Path) -> int:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of", "json", str(video),
    ], text=True))
    return int(data["streams"][0]["nb_read_frames"])


def audit_session(root: Path) -> dict[str, Any]:
    result_path = root / "RESULT.json"
    result = json.loads(result_path.read_text())
    required_false = ("authority", "metric_object_geometry", "contact_frame_valid", "control_ground_truth", "action_sidecar_published")
    if result.get("status") != "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY" or result.get("terminal_mode") != "POSE_ONLY_VISUAL_ROBOT":
        raise RuntimeError(f"unexpected pose-only status: {root}")
    for key in required_false:
        if result.get(key) is not False:
            raise RuntimeError(f"{key} must be false: {root}")
    if result.get("contact_state") != "UNKNOWN" or result.get("physical_deployment", {}).get("status") != "BLOCKED_EXTERNAL":
        raise RuntimeError(f"claim boundary violation: {root}")
    if not result.get("clean_join_ready") or result.get("base_motion_during_session") is not False:
        raise RuntimeError(f"Clean/base contract violation: {root}")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise RuntimeError(f"symlink forbidden: {root}")

    outputs = result["outputs"]
    paths = {name: check_ref(root, item, name) for name, item in outputs.items()}
    count = int(result["frame_count"])
    if decoded_frames(paths["review_video"]) != count:
        raise RuntimeError(f"decoded frame mismatch: {root}")
    if len(paths["frame_manifest"].read_text().splitlines()) != count:
        raise RuntimeError(f"manifest frame mismatch: {root}")
    z = np.load(paths["trajectory_sidecar"], allow_pickle=False)
    required = {
        "source_frame_ids", "q_arm_rad", "q_hand_rad", "valid_side_frame", "contact_state",
        "contact_frame_valid", "metric_object_geometry", "control_ground_truth",
    }
    if not required.issubset(z.files):
        raise RuntimeError(f"sidecar keys missing: {root}")
    expected = np.arange(count, dtype=np.int64)
    if z["source_frame_ids"].dtype != np.int64 or not np.array_equal(z["source_frame_ids"], expected):
        raise RuntimeError(f"source frame identity mismatch: {root}")
    valid = np.asarray(z["valid_side_frame"], dtype=bool)
    if valid.shape != (count, 2):
        raise RuntimeError(f"validity shape mismatch: {root}")
    for key in ("q_arm_rad", "q_hand_rad"):
        finite = np.isfinite(z[key]).all(axis=-1)
        if np.any(valid != finite):
            raise RuntimeError(f"UNKNOWN/NaN invariant mismatch for {key}: {root}")
    if np.any(z["contact_state"] != 0) or np.any(z["contact_frame_valid"]):
        raise RuntimeError(f"contact must remain UNKNOWN: {root}")
    if bool(z["metric_object_geometry"]) or bool(z["control_ground_truth"]):
        raise RuntimeError(f"forbidden truth bit enabled: {root}")
    return {
        "task": result["task"],
        "session_id": result["session"],
        "mode": result["terminal_mode"],
        "status": result["status"],
        "frame_count": count,
        "valid_side_rows": int(valid.sum()),
        "unknown_side_rows": int((~valid).sum()),
        "result": ref(result_path),
        "trajectory_sidecar": ref(paths["trajectory_sidecar"]),
        "review_video": ref(paths["review_video"]),
        "run_signature": result["run_signature"],
        "physical_deployment": "BLOCKED_EXTERNAL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    rows = [audit_session(p) for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if len({row["session_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate session candidates")
    result = {
        "schema_version": SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS_CURRENT_CANDIDATES_REVIEW_READY_NOT_AUTHORITY",
        "authority": False,
        "counts": {
            "sessions": len(rows),
            "by_task": dict(sorted(Counter(row["task"] for row in rows).items())),
            "valid_side_rows": sum(row["valid_side_rows"] for row in rows),
            "unknown_side_rows": sum(row["unknown_side_rows"] for row in rows),
        },
        "rows": rows,
        "claim_limit": "Current pose-only visual candidates only; not final 156-row Robot authority and not action/contact/deployment truth.",
    }
    target = root / "CURRENT_CANDIDATE_INDEX.json"
    fd, tmp = tempfile.mkstemp(prefix=".CURRENT_CANDIDATE_INDEX.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical_json(result))
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(json.dumps({"status": "PASS", "sessions": len(rows), "output": str(target), "sha256": sha256(target)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
