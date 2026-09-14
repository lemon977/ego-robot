#!/usr/bin/env python3
"""Fail-closed preflight for Clean-joined Exact78 Robot candidates.

This command performs no IK, rendering, contact inference or authority publish.
It freezes the exact input bytes for a later bounded two-method Robot run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
REQUIRED_PROGRAMS = (
    "run_hawor_temporal_jerk_successor.py",
    "run_robot_motion_transfer_arm_canary_v2.py",
    "run_robot_arm_segment_bidirectional_v3.py",
    "run_robot_hand_fullsession_v2.py",
    "run_robot_hand_segment_bidirectional_v3.py",
    "select_robot_fixed_placement_v1.py",
    "render_robot_motion_transfer_fullsession_v2.py",
    "adopt_exact78_pose_only_visual_robot_v52.py",
    "publish_exact78_robot_failed_quality_terminal_v52.py",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    if not candidate.is_file():
        raise ValueError(f"not a regular file: {candidate}")
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256(candidate)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def require_ref(expected: dict[str, Any], label: str) -> dict[str, Any]:
    if set(expected) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label}: exact artifact ref required")
    actual = ref(Path(str(expected["path"])))
    if actual != expected:
        raise ValueError(f"{label}: artifact ref mismatch expected={expected} actual={actual}")
    return actual


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_session(spec: dict[str, Any], matrix_row: dict[str, Any]) -> dict[str, Any]:
    session = str(spec.get("session"))
    if matrix_row.get("robot_current_state") != "READY_FOR_ROBOT_CURRENT_DRAFT":
        raise ValueError(f"{session}: matrix row is no longer READY_FOR_ROBOT_CURRENT_DRAFT")
    for field in ("task", "session", "frame_count", "split"):
        matrix_value = matrix_row.get("session_id") if field == "session" else matrix_row.get(field)
        if spec.get(field) != matrix_value:
            raise ValueError(f"{session}: {field} mismatch contract={spec.get(field)} matrix={matrix_value}")
    clean_ref = require_ref(spec["clean_result"], f"{session}:clean_result")
    if clean_ref != matrix_row.get("clean_result"):
        raise ValueError(f"{session}: clean result does not match matrix")
    clean = load(Path(clean_ref["path"]))
    if not (
        clean.get("grade") == "B"
        and clean.get("downstream_authorized") is True
        and clean.get("session") in {None, session}
        and clean.get("session_id") in {None, session}
    ):
        raise ValueError(f"{session}: Clean join gate failed")
    bounded_result_ref = require_ref(spec["bounded_result"], f"{session}:bounded_result")
    if bounded_result_ref != matrix_row.get("hawor", {}).get("result"):
        raise ValueError(f"{session}: bounded HaWoR result does not match matrix")
    bounded = load(Path(bounded_result_ref["path"]))
    if bounded.get("numeric_gate_pass") is not True or not str(bounded.get("status", "")).startswith("PASS"):
        raise ValueError(f"{session}: bounded HaWoR numeric gate failed")
    bounded_npz_ref = require_ref(spec["bounded_npz"], f"{session}:bounded_npz")
    if bounded.get("outputs", {}).get("npz") != bounded_npz_ref:
        raise ValueError(f"{session}: bounded NPZ lineage mismatch")
    source_video_ref = require_ref(spec["source_video"], f"{session}:source_video")
    if bounded.get("inputs", {}).get("source_video") != source_video_ref:
        raise ValueError(f"{session}: source video lineage mismatch")
    return {
        "task": spec["task"],
        "session": session,
        "frame_count": spec["frame_count"],
        "split": spec["split"],
        "status": "READY_FOR_BOUNDED_TWO_METHOD_ROBOT",
        "clean_result": clean_ref,
        "bounded_result": bounded_result_ref,
        "bounded_npz": bounded_npz_ref,
        "source_video": source_video_ref,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = load(args.contract)
    matrix = load(args.matrix)
    if contract.get("schema_version") != "exact78-robot-ready-input-v52-v1":
        raise SystemExit("unsupported contract schema")
    if contract.get("status") != "FROZEN_PREFLIGHT_PENDING":
        raise SystemExit("contract is not frozen for preflight")
    matrix_ref = require_ref(contract["matrix_snapshot"], "matrix_snapshot")
    if matrix_ref != ref(args.matrix):
        raise SystemExit("--matrix does not match contract matrix_snapshot")
    sessions = contract.get("sessions", [])
    if len(sessions) != 3 or len({row.get("session") for row in sessions}) != 3:
        raise SystemExit("contract must contain exactly three unique sessions")
    matrix_by_session = {row["session_id"]: row for row in matrix.get("rows", [])}
    rows = [validate_session(row, matrix_by_session.get(row.get("session"), {})) for row in sessions]
    templates = {
        task: {name: require_ref(value, f"accepted_templates:{task}:{name}") for name, value in group.items()}
        for task, group in contract.get("accepted_templates", {}).items()
    }
    if set(templates) != {"chips", "poker"} or any(set(group) != {"states", "hawor"} for group in templates.values()):
        raise SystemExit("both exact accepted template pairs are required")
    programs = {name: ref(PROJECT / "tools" / name) for name in REQUIRED_PROGRAMS}
    result = {
        "schema_version": "exact78-robot-ready-preflight-v52-v1",
        "created_at": now(),
        "status": "PASS_READY_FOR_BOUNDED_TWO_METHOD_ROBOT",
        "contract": ref(args.contract),
        "matrix_snapshot": matrix_ref,
        "counts": {"sessions": 3, "chips": 2, "poker": 1},
        "sessions": rows,
        "accepted_templates": templates,
        "placement_policy": contract["placement_policy"],
        "method_budget": contract["method_budget"],
        "programs": programs,
        "gpu_calls": 0,
        "claim_limit": (
            "Input and code closure only; no IK/render was run, no method round was consumed, "
            "and no Robot/contact/action/deployment authority is published."
        ),
    }
    atomic_new(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
