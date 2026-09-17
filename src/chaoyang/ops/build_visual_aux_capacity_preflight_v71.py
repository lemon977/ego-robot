#!/usr/bin/env python3
"""Build a bounded capacity forecast for the two V7.1 Visual Aux pairs.

This is a routing proof, not a data or checkpoint authority.  It keeps already
eligible legacy sessions, the immutable metric Robot expansion selection, and
the calibration-missing Visual Tier selection separate so failures cannot be
silently counted as completed training inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REQUIREMENTS = {
    "train": {"sessions": 16, "windows": 256},
    "validation": {"sessions": 3, "windows": 48},
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def selection_sessions(payload: dict[str, Any]) -> list[str]:
    rows = payload.get("sessions")
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        raise RuntimeError("metric Robot selection.sessions must be a string list")
    return rows


def summarize(
    matrix: dict[str, Any],
    robot_selection: dict[str, Any],
    visual_tier: dict[str, Any],
    legacy_eligibility: dict[str, Any],
) -> dict[str, Any]:
    matrix_rows = matrix.get("rows")
    if not isinstance(matrix_rows, list):
        raise RuntimeError("matrix.rows missing")
    indexed = {str(row["session_id"]): row for row in matrix_rows}
    if len(indexed) != len(matrix_rows):
        raise RuntimeError("duplicate matrix session_id")

    legacy_rows = [
        row for row in legacy_eligibility.get("rows", [])
        if row.get("status") == "READY" and row.get("split") in REQUIREMENTS
    ]
    robot_rows = [indexed[session] for session in selection_sessions(robot_selection)]
    visual_rows = visual_tier.get("sessions")
    if not isinstance(visual_rows, list):
        raise RuntimeError("visual tier sessions missing")

    sources = {
        "already_eligible_legacy": legacy_rows,
        "pending_metric_robot_expansion": robot_rows,
        "pending_visual_tier": visual_rows,
    }
    session_sources: dict[str, str] = {}
    for source, rows in sources.items():
        for row in rows:
            session = str(row.get("session") or row.get("session_id"))
            if session in session_sources and session_sources[session] != source:
                raise RuntimeError(f"session appears in multiple capacity sources: {session}")
            session_sources[session] = source

    result: dict[str, Any] = {}
    for task in ("chips", "poker"):
        task_result: dict[str, Any] = {}
        for split, required in REQUIREMENTS.items():
            by_source: dict[str, Any] = {}
            potential_sessions: set[str] = set()
            observed_windows = 0
            for source, rows in sources.items():
                selected = [
                    row for row in rows
                    if row.get("task") == task and row.get("split") == split
                ]
                sessions = sorted(str(row.get("session") or row.get("session_id")) for row in selected)
                windows = sum(int(row.get("eligible_h50_window_count", 0)) for row in selected)
                by_source[source] = {"sessions": sessions, "count": len(sessions), "observed_windows": windows}
                potential_sessions.update(sessions)
                observed_windows += windows
            session_capacity_ok = len(potential_sessions) >= required["sessions"]
            task_result[split] = {
                "sources": by_source,
                "unique_potential_sessions": len(potential_sessions),
                "observed_ready_windows_lower_bound": observed_windows,
                "requirements": required,
                "session_capacity_ok": session_capacity_ok,
                "actual_training_ready": False,
                "claim_limit": "Pending sources are capacity only until their immutable Clean/Robot/bundle receipts pass.",
            }
        task_result["capacity_status"] = (
            "PASS_CAPACITY_ROUTE_EXISTS_PENDING_ARTIFACTS"
            if all(task_result[split]["session_capacity_ok"] for split in REQUIREMENTS)
            else "BLOCKED_DATA_VOLUME_CAPACITY"
        )
        result[task] = task_result
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--robot-selection", type=Path, required=True)
    parser.add_argument("--visual-tier-selection", type=Path, required=True)
    parser.add_argument("--legacy-eligibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"immutable output exists: {args.output}")

    inputs = {
        "matrix": ref(args.matrix),
        "robot_selection": ref(args.robot_selection),
        "visual_tier_selection": ref(args.visual_tier_selection),
        "legacy_eligibility": ref(args.legacy_eligibility),
    }
    tasks = summarize(
        load(args.matrix), load(args.robot_selection),
        load(args.visual_tier_selection), load(args.legacy_eligibility),
    )
    status = (
        "PASS_CAPACITY_ROUTE_EXISTS_PENDING_ARTIFACTS"
        if all(row["capacity_status"] == "PASS_CAPACITY_ROUTE_EXISTS_PENDING_ARTIFACTS" for row in tasks.values())
        else "BLOCKED_DATA_VOLUME_CAPACITY"
    )
    output = {
        "schema_version": "visual-aux-capacity-preflight-v71-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "tasks": tasks,
        "inputs": inputs,
        "control_ground_truth": False,
        "checkpoint_authority": False,
        "claim_limit": "Routing capacity forecast only; pending sessions, windows and checkpoints are not claimed complete.",
    }
    atomic_json(args.output.resolve(), output)
    print(json.dumps({"status": status, "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if status.startswith("PASS_") else 3


if __name__ == "__main__":
    raise SystemExit(main())
