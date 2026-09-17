#!/usr/bin/env python3
"""Freeze bounded 30-session Robot production targets per exact78 task."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, atomic_write, load_json, now_iso


CROSS = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v52/matrices/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
V77 = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/robot_v77_terminal_index_r22/attempts/attempt_0003/RESULT.json"


def _rank(row: dict[str, Any], terminal: str) -> tuple[int, int, str]:
    # Existing hard passes are immutable first choices.  Fresh metric candidates
    # precede visual-only candidates; existing quality-C rows remain last.
    if terminal == "PASSED": bucket = 0
    elif terminal == "NOT_EVALUATED" and row.get("metric_ready_wave0"): bucket = 1
    elif terminal == "FAILED_QUALITY_C" and row.get("metric_ready_wave0"): bucket = 2
    elif terminal == "NOT_EVALUATED" and row.get("three_upstream_ab"): bucket = 3
    else: bucket = 4
    return bucket, int(row.get("position", 10**9)), str(row["session_id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists(): raise RuntimeError(f"fresh output required: {args.output_root}")
    args.output_root.mkdir(parents=True)
    cross = load_json(CROSS); v77 = load_json(V77)
    terminal = {str(x["session"]): str(x["terminal_status"]) for x in v77["rows"]}
    selected: list[dict[str, Any]] = []
    counts: dict[str, Any] = {}
    for task in ("chips", "poker"):
        eligible = [x for x in cross["rows"] if x["task"] == task and x.get("three_upstream_ab")]
        eligible.sort(key=lambda x: _rank(x, terminal.get(str(x["session_id"]), "NOT_EVALUATED")))
        chosen = eligible[:30]
        if len(chosen) != 30: raise RuntimeError(f"{task} has only {len(chosen)} Robot candidates")
        for index, row in enumerate(chosen, 1):
            status = terminal.get(str(row["session_id"]), "NOT_EVALUATED")
            selected.append({
                "task": task, "target_rank": index, "session_id": row["session_id"],
                "requested_robot_tier": "METRIC_CONTACT_CANDIDATE" if row.get("metric_ready_wave0") else "POSE_ONLY_VISUAL_CANDIDATE",
                "baseline_terminal_status": status,
                "execution_route": "ADOPT_VERIFIED_HARD_PASS" if status == "PASSED" else ("BOUNDED_SUCCESSOR_REQUIRED" if status == "FAILED_QUALITY_C" else "FRESH_CAUSAL_PREFIX_RUN"),
                "three_upstream_ab": bool(row.get("three_upstream_ab")),
                "metric_ready_wave0": bool(row.get("metric_ready_wave0")),
                "control_ground_truth": False, "physical_deployment_authorized": False,
            })
        counts[task] = {
            "selected": 30,
            "existing_hard_pass": sum(x["baseline_terminal_status"] == "PASSED" for x in selected if x["task"] == task),
            "metric_candidates": sum(x["requested_robot_tier"] == "METRIC_CONTACT_CANDIDATE" for x in selected if x["task"] == task),
            "pose_only_candidates": sum(x["requested_robot_tier"] == "POSE_ONLY_VISUAL_CANDIDATE" for x in selected if x["task"] == task),
            "successor_required": sum(x["execution_route"] == "BOUNDED_SUCCESSOR_REQUIRED" for x in selected if x["task"] == task),
        }
    created = now_iso()
    selection_path = args.output_root / "ROBOT30_SELECTION.json"
    atomic_json(selection_path, {"schema_version": "chaoyang-rc1-robot30-selection-v1", "created_at": created, "rows": selected, "counts": counts, "inputs": [artifact_ref(CROSS), artifact_ref(V77)], "claim_limit": "Frozen production targets only; selection does not prove Robot quality, causality, contact or training eligibility."})
    csv_path = args.output_root / "ROBOT30_SELECTION.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
    result_path = args.output_root / "RESULT.json"
    atomic_json(result_path, {"schema_version": "chaoyang-rc1-robot30-selection-result-v1", "task_id": "rc1_robot30_selection", "status": "PASSED", "created_at": created, "counts": counts, "selection": artifact_ref(selection_path), "authority_promoted": False})
    atomic_json(args.output_root / "METRICS.json", counts)
    atomic_json(args.output_root / "NEXT_ACTION.json", {"task_id": "implement_v77_prefix_recompute_entry", "then": "run_robot30_in_4_12_30_batches"})
    atomic_write(args.output_root / "DECISION.md", b"# RC1 Robot30\n\nEach task now has 30 frozen targets. Existing hard passes remain evidence; all other rows require fresh causal-prefix execution or a bounded successor.\n")
    manifest_path = args.output_root / "ARTIFACT_MANIFEST.json"
    atomic_json(manifest_path, {"schema_version": "chaoyang-rc1-robot30-manifest-v1", "artifacts": [artifact_ref(x) for x in (selection_path, csv_path, result_path, args.output_root / "METRICS.json", args.output_root / "NEXT_ACTION.json", args.output_root / "DECISION.md")]})
    atomic_json(args.output_root / "RUN_RECEIPT.json", {"schema_version": "chaoyang-rc1-robot30-receipt-v1", "task_id": "rc1_robot30_selection", "status": "PASSED", "result": artifact_ref(result_path), "manifest": artifact_ref(manifest_path)})
    atomic_json(args.output_root / "RESULT_SUMMARY.json", {"task_id": "rc1_robot30_selection", "status": "PASSED", "counts": counts, "next_action": "implement_v77_prefix_recompute_entry"})
    print(json.dumps({"status": "PASSED", "counts": counts, "selection": str(selection_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
