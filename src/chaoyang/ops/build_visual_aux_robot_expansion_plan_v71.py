#!/usr/bin/env python3
"""Build a deterministic, split-preserving Robot expansion plan for Visual Aux.

This is a planning artifact only.  It never promotes a Robot candidate, moves a
split, treats a visual trajectory as an action, or claims that Robotized RGB is
causal.  Metric-ready rows are preferred; calibration-missing triple-A/B rows
are used only when the frozen split cannot meet the session minimum otherwise.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/CONVERSION_CAUSE_LEDGER_V2.json"
DEFAULT_MATRIX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
DEFAULT_ELIGIBILITY = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_visual_aux_eligibility_index_v56/ELIGIBILITY_INDEX.json"
REQUIRED = {"train": 16, "validation": 3}
VISUAL_TIER_RESERVE = {"train": 4, "validation": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"immutable output exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _priority(row: dict[str, Any], matrix: dict[str, Any]) -> tuple[int, int, str]:
    current = matrix.get(row["session_id"], {})
    clean_ready = current.get("clean_state") == "PASSED_GRADE_B"
    return (0 if clean_ready else 1, -int(row.get("frame_count", 0)), row["session_id"])


def build(ledger: dict[str, Any], matrix_payload: dict[str, Any], eligibility: dict[str, Any]) -> dict[str, Any]:
    rows = ledger.get("rows", [])
    if len(rows) != 156 or len({row.get("session_id") for row in rows}) != 156:
        raise ValueError("conversion ledger must contain 156 unique sessions")
    matrix_rows = matrix_payload.get("rows", [])
    matrix = {row.get("session_id"): row for row in matrix_rows}
    if len(matrix) != 156:
        raise ValueError("current matrix must contain 156 unique sessions")

    existing_ready: dict[tuple[str, str], set[str]] = {}
    already_attempted: set[str] = set()
    for row in eligibility.get("rows", []):
        task = str(row.get("task", "")).upper()
        split = str(row.get("split", ""))
        session = str(row.get("session", ""))
        if session:
            already_attempted.add(session)
        if row.get("status") == "READY" and split in REQUIRED and task in {"CHIPS", "POKER"}:
            existing_ready.setdefault((task, split), set()).add(session)

    selections: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for task in ("CHIPS", "POKER"):
        summaries[task.lower()] = {}
        for split, minimum in REQUIRED.items():
            existing = existing_ready.get((task, split), set())
            needed = max(0, minimum - len(existing))
            source_rows = [
                row for row in rows
                if row.get("task") == task and row.get("split") == split
                and row.get("session_id") not in already_attempted
            ]
            metric = sorted(
                (row for row in source_rows if row.get("first_blocker") == "METRIC_GEOMETRY_READY"),
                key=lambda row: _priority(row, matrix),
            )
            visual = sorted(
                (row for row in source_rows if row.get("first_blocker") == "CALIBRATION_MISSING"),
                key=lambda row: _priority(row, matrix),
            )
            chosen_metric = metric[:needed]
            remaining = needed - len(chosen_metric)
            # Metric Poker has exactly the minimum denominator and historical
            # Robot projection conversion is imperfect.  Freeze a small
            # split-preserving reserve now so one zero-window/quality-C row
            # does not make the four-checkpoint route deterministically fail.
            reserve = VISUAL_TIER_RESERVE[split] if remaining > 0 else 0
            chosen_visual = visual[: remaining + reserve]
            for row, tier in [(item, "TIER_M_METRIC") for item in chosen_metric] + [
                (item, "TIER_V_VISUAL") for item in chosen_visual
            ]:
                current = matrix[row["session_id"]]
                if tier == "TIER_V_VISUAL":
                    route = "VISUAL_CLEAN_THEN_POSE_ONLY_ROBOT"
                elif current.get("clean_state") == "PASSED_GRADE_B":
                    route = "ROBOT_GEOMETRY_NOW"
                else:
                    route = "WAIT_CLEAN_TERMINAL_THEN_ROBOT_GEOMETRY"
                selections.append({
                    "task": task.lower(),
                    "session_id": row["session_id"],
                    "split": split,
                    "tier": tier,
                    "frame_count": int(row.get("frame_count", 0)),
                    "current_clean_state": current.get("clean_state"),
                    "current_robot_state": current.get("robot_current_state"),
                    "route": route,
                    "control_ground_truth": False,
                })
            selected_count = len(chosen_metric) + len(chosen_visual)
            summaries[task.lower()][split] = {
                "required_sessions": minimum,
                "existing_ready_sessions": len(existing),
                "selected_metric_sessions": len(chosen_metric),
                "selected_visual_tier_sessions": len(chosen_visual),
                "visual_tier_reserve_sessions": max(0, len(chosen_visual) - remaining),
                "remaining_session_deficit": max(0, needed - selected_count),
            }

    return {
        "schema_version": "visual-aux-robot-expansion-plan-v71-v1",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS_DETERMINISTIC_SELECTION" if all(
            item["remaining_session_deficit"] == 0
            for task in summaries.values() for item in task.values()
        ) else "BLOCKED_DATA_VOLUME",
        "required": {"train_sessions": 16, "validation_sessions": 3},
        "summaries": summaries,
        "selection": selections,
        "counts": {
            "sessions": len(selections),
            "metric": sum(row["tier"] == "TIER_M_METRIC" for row in selections),
            "visual_tier": sum(row["tier"] == "TIER_V_VISUAL" for row in selections),
            "ready_robot_now": sum(row["route"] == "ROBOT_GEOMETRY_NOW" for row in selections),
            "needs_clean": sum("CLEAN" in row["route"] for row in selections),
        },
        "constraints": {
            "split_movement_allowed": False,
            "future_information_allowed": False,
            "existing_zero_window_candidate_reused": False,
            "visual_trajectory_is_real_action": False,
            "selection_order": "clean-ready first, then longer frame count, then session_id",
            "visual_tier_reserve": VISUAL_TIER_RESERVE,
        },
        "claim_limit": (
            "Deterministic expansion planning only. Selection is not Robot authority, causal Robotized RGB, "
            "a checkpoint, contact truth, or real Robot action. Actual H50 windows must be recomputed after R1."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--eligibility", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(load(args.ledger), load(args.matrix), load(args.eligibility))
    payload["inputs"] = {
        "conversion_ledger": ref(args.ledger),
        "current_matrix": ref(args.matrix),
        "eligibility": ref(args.eligibility),
    }
    atomic_new(args.output, payload)
    print(json.dumps({"status": payload["status"], "counts": payload["counts"], "output": ref(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
