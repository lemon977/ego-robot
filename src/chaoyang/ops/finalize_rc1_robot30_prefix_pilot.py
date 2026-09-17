#!/usr/bin/env python3
"""Aggregate the two-session Robot30 scheduled-prefix pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha(value)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"object required: {path}")
    return value


def write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return float(ordered[index])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips-result", type=Path, required=True)
    parser.add_argument("--poker-result", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--attempt1-log", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    chips = load(args.chips_result); poker = load(args.poker_result); selection = load(args.selection)
    if chips.get("task") != "chips" or poker.get("task") != "poker":
        raise RuntimeError("two-task pilot identity mismatch")
    selected = selection.get("counts", {})
    if selected.get("chips", {}).get("selected") != 30 or selected.get("poker", {}).get("selected") != 30:
        raise RuntimeError("Robot30 selection is not frozen at 30 per task")
    starts = chips.get("rows", []) + poker.get("rows", [])
    walls = [float(row["wall_seconds"]) for row in starts]
    prefix_pass = sum(str(row.get("status", "")).startswith("PASSED") for row in starts)
    status = "PASSED" if prefix_pass == len(starts) and len(starts) == 6 else "FAILED_QUALITY_C"
    result = {
        "schema_version": "chaoyang-rc1-robot30-prefix-pilot-final-v1",
        "task_id": "rc1_robot30_prefix_schedule_pilot", "created_at": now(), "status": status,
        "selection": ref(args.selection),
        "selection_counts": selected,
        "pilot": {
            "sessions": 2, "scheduled_starts": len(starts), "prefix_state_pass": prefix_pass,
            "wall_seconds_p50": statistics.median(walls) if walls else None,
            "wall_seconds_p95": percentile(walls, 0.95),
            "chips": ref(args.chips_result), "poker": ref(args.poker_result),
        },
        "runtime_attempts": [
            {"attempt": 1, "status": "FAILED_RUNTIME", "retryable": True, "reason": "PREFIX_PATH_OMITTED_SESSION_ID", "log": ref(args.attempt1_log)},
            {"attempt": 2, "status": status},
        ],
        "collision_gate_complete": False,
        "render_gate_complete": False,
        "compositor_gate_complete": False,
        "robot30_expansion_status": "BLOCKED_PREREQ_DIGITAL_COLLISION_GATE",
        "selection_is_pass_count": False,
        "visual_aux_rc1_eligible_windows": 0,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "authority_promoted": False,
        "claim_limit": "Two-task causal scheduled-prefix solver timing and terminal-state closure only. Thirty selected rows per task are not thirty Robot passes; collision/render/compositor eligibility remains absent.",
    }
    result_path = args.output_root / "RESULT.json"; write(result_path, result)
    write(args.output_root / "METRICS.json", {"status": status, "pilot": result["pilot"], "visual_aux_rc1_eligible_windows": 0})
    write(args.output_root / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": status, "selection": "30_PER_TASK", "prefix_starts_passed": prefix_pass, "next_action": "DIGITAL_COLLISION_PREFIX_PILOT"})
    write(args.output_root / "RUN_RECEIPT.json", {"status": status, "result": ref(result_path), "created_at": now()})
    write(args.output_root / "ARTIFACT_MANIFEST.json", {"result": ref(result_path), "inputs": [ref(args.chips_result), ref(args.poker_result), ref(args.selection)]})
    write(args.output_root / "DECISION.md", "# Robot30 prefix pilot\n\n30+30 is a frozen target denominator, not a pass count. Prefix solver outputs remain ineligible until digital collision, render and compositor gates close.\n")
    write(args.output_root / "NEXT_ACTION.json", {"task_id": "rc1_robot30_digital_collision_pilot", "required": True, "then": "freeze pilot P95 and decide bounded expansion"})
    print(json.dumps({"status": status, "result": ref(result_path), "pilot": result["pilot"]}, ensure_ascii=False))
    return 0 if status == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
