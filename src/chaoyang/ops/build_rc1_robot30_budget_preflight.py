#!/usr/bin/env python3
"""Freeze Robot30 causal-window cost and offline-visual routing."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    if not isinstance(value, dict): raise RuntimeError(f"object required: {path}")
    return value


def write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink(): raise FileExistsError(path)
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--master-ledger", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(); args.output_root.mkdir(parents=True, exist_ok=False)
    selection = load(args.selection); master = load(args.master_ledger); pilot = load(args.pilot_result)
    by_session = {row["session_id"]: row for row in master["rows"]}
    p50 = float(pilot["pilot"]["wall_seconds_p50"]); p95 = float(pilot["pilot"]["wall_seconds_p95"])
    rows = []
    for selected in selection["rows"]:
        row = by_session[selected["session_id"]]
        starts = int(row["scheduled_h50_start_count"])
        rows.append({
            "task": selected["task"], "session_id": selected["session_id"],
            "target_rank": selected["target_rank"], "execution_route": selected["execution_route"],
            "requested_robot_tier": selected["requested_robot_tier"],
            "scheduled_starts": starts,
            "causal_estimate_p50_seconds": starts * p50,
            "causal_estimate_p95_seconds": starts * p95,
            "offline_visual_route": "RUN_PINNED_V77_FULLSESSION",
            "causal_training_route": "DEFERRED_CAPACITY_AND_CLEAN_BLOCKED",
        })
    counts = {}
    for task in ("chips", "poker"):
        task_rows = [row for row in rows if row["task"] == task]
        starts = sum(row["scheduled_starts"] for row in task_rows)
        counts[task] = {
            "selected_sessions": len(task_rows), "scheduled_starts": starts,
            "causal_estimate_p50_hours": starts * p50 / 3600.0,
            "causal_estimate_p95_hours": starts * p95 / 3600.0,
        }
    counts["total"] = {
        "selected_sessions": len(rows), "scheduled_starts": sum(row["scheduled_starts"] for row in rows),
        "causal_estimate_p50_hours": sum(row["scheduled_starts"] for row in rows) * p50 / 3600.0,
        "causal_estimate_p95_hours": sum(row["scheduled_starts"] for row in rows) * p95 / 3600.0,
    }
    result = {
        "schema_version": "chaoyang-rc1-robot30-budget-preflight-v1",
        "task_id": "rc1_robot30_budget_preflight", "created_at": now(), "status": "PASSED",
        "counts": counts, "rows": rows,
        "decisions": {
            "offline_visual_30_per_task": "CONTINUE_IN_IMMUTABLE_MICROBATCHES",
            "causal_scheduled_start_30_per_task": "DEFERRED",
            "causal_defer_reasons": [
                "T0 independent source-group capacity is 1 Chips and 3 Poker, below 16 train + 3 validation",
                "T2 causal Clean is BLOCKED_PREREQ for both pilots",
                f"Full 60-session causal prefix estimate is {counts['total']['causal_estimate_p95_hours']:.2f} serial CPU hours before collision/render/compositor",
            ],
        },
        "inputs": {"selection": ref(args.selection), "master_ledger": ref(args.master_ledger), "pilot": ref(args.pilot_result)},
        "authority_promoted": False,
        "claim_limit": "Pilot-derived compute forecast and routing decision only; estimates are not completed Robot counts or guaranteed wall-clock ETA.",
    }
    result_path = args.output_root / "RESULT.json"; write(result_path, result)
    csv_path = args.output_root / "ROBOT30_BUDGET_PREFLIGHT.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows); f.flush(); os.fsync(f.fileno())
    write(args.output_root / "METRICS.json", counts)
    write(args.output_root / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": "PASSED", "counts": counts, "next_action": "CONTINUE_OFFLINE_VISUAL_MICROBATCHES"})
    write(args.output_root / "RUN_RECEIPT.json", {"status": "PASSED", "result": ref(result_path), "created_at": now()})
    write(args.output_root / "ARTIFACT_MANIFEST.json", {"result": ref(result_path), "csv": ref(csv_path)})
    write(args.output_root / "DECISION.md", "# Robot30 budget decision\n\nContinue 30+30 offline visual terminals in immutable batches. Do not spend the projected causal-window budget until independent source-group and causal Clean gates can unlock checkpoint training.\n")
    write(args.output_root / "NEXT_ACTION.json", {"next": "CONTINUE_OFFLINE_VISUAL_MICROBATCHES", "causal_expansion": "DEFERRED"})
    print(json.dumps({"status": "PASSED", "result": ref(result_path), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
