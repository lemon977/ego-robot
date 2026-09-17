#!/usr/bin/env python3
"""Close unevaluated RC1 Robot30 causal rows without inventing results.

This finalizer is intentionally separate from the progress index builder.  It
may run only after every bounded offline successor has a real terminal.  Rows
selected for fresh causal production that were not executed are closed as
``NOT_EVALUATED_BUDGET``.  Existing offline geometry evidence is preserved,
but never promoted to causal training eligibility.
"""
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

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_prerequisites(t0: dict[str, Any], t2: dict[str, Any], t3: dict[str, Any]) -> None:
    if t0.get("status") != "PASSED":
        raise RuntimeError("T0 capacity result must be PASSED")
    for task in ("chips", "poker"):
        capacity = t0.get("capacity", {}).get(task, {})
        if capacity.get("pair_terminal") != "BLOCKED_DATA_VOLUME":
            raise RuntimeError(f"{task} must be explicitly BLOCKED_DATA_VOLUME")
    if t2.get("status") != "BLOCKED_PREREQ" or t2.get("fresh_clean_generated") is not False:
        raise RuntimeError("T2 must prove blocked causal Clean with no fresh output")
    if t3.get("status") != "PASSED":
        raise RuntimeError("T3 prefix-causal canary must be PASSED")
    if t3.get("production_entry", {}).get("full_session_training_eligibility") is not False:
        raise RuntimeError("T3 unexpectedly claims full-session eligibility")


def close_rows(
    progress: dict[str, Any],
    *,
    output_root: Path,
    prerequisite_refs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    if progress.get("schema_version") != "chaoyang-rc1-robot30-terminal-index-v1":
        raise RuntimeError("unexpected Robot30 progress schema")
    rows = progress.get("rows", [])
    if len(rows) != 60 or len({row.get("session_id") for row in rows}) != 60:
        raise RuntimeError("Robot30 progress must contain 60 unique rows")
    pending_successor = [row for row in rows if row.get("terminal_status") == "PENDING_BOUNDED_SUCCESSOR"]
    if pending_successor:
        raise RuntimeError(f"bounded successors remain pending: {len(pending_successor)}")

    counts = {
        task: {
            "selected": 0,
            "hard_geometry_pass_evidence": 0,
            "failed_quality_c": 0,
            "failed_runtime_final": 0,
            "not_evaluated_budget": 0,
            "training_eligible": 0,
        }
        for task in ("chips", "poker")
    }
    closed: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        task = str(row.get("task"))
        if task not in counts:
            raise RuntimeError(f"unsupported task: {task}")
        counts[task]["selected"] += 1
        terminal = str(row.get("terminal_status"))
        if terminal == "PENDING_CAUSAL_PRODUCTION":
            session = str(row["session_id"])
            receipt_path = output_root / "terminals" / session / "RESULT.json"
            receipt = {
                "schema_version": "chaoyang-rc1-robot30-causal-budget-terminal-v1",
                "created_at": now_iso(),
                "session_id": session,
                "task": task,
                "execution_status": "NOT_EVALUATED_BUDGET",
                "terminal_status": "NOT_EVALUATED_BUDGET",
                "algorithm_attempted": False,
                "quality_failure": False,
                "input_mode": "CAUSAL_TRAINING_INPUT_REQUIRED",
                "training_eligible": False,
                "reason_codes": [
                    "PAIR_BLOCKED_DATA_VOLUME",
                    "CAUSAL_CLEAN_BLOCKED_PREREQ",
                    "NO_RC1_DOWNSTREAM_CONSUMER_WITHIN_RELEASE",
                ],
                "prerequisites": prerequisite_refs,
                "control_ground_truth": False,
                "physical_deployment_authorized": False,
                "authority_promoted": False,
                "claim_limit": "Budget terminal only; the causal Robot algorithm was not run and no quality conclusion is made.",
            }
            receipt_path.parent.mkdir(parents=True, exist_ok=False)
            atomic_json(receipt_path, receipt)
            row.update(
                terminal_status="NOT_EVALUATED_BUDGET",
                input_mode="CAUSAL_TRAINING_INPUT_REQUIRED",
                training_eligible=False,
                result=artifact_ref(receipt_path),
            )
            counts[task]["not_evaluated_budget"] += 1
        elif terminal in {"PASSED_VERIFIED_PRIOR", "PASSED_OFFLINE_VISUAL_HARD_GEOMETRY"}:
            if row.get("hard_geometry_pass") is not True:
                raise RuntimeError(f"pass without hard geometry evidence: {row.get('session_id')}")
            counts[task]["hard_geometry_pass_evidence"] += 1
        elif terminal == "FAILED_QUALITY_C":
            counts[task]["failed_quality_c"] += 1
        elif terminal == "FAILED_RUNTIME_FINAL":
            counts[task]["failed_runtime_final"] += 1
        else:
            raise RuntimeError(f"nonterminal or unsupported status {terminal}: {row.get('session_id')}")
        if row.get("training_eligible") is True:
            counts[task]["training_eligible"] += 1
        closed.append(row)
    return closed, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-index", type=Path, required=True)
    parser.add_argument("--t0-result", type=Path, required=True)
    parser.add_argument("--t2-result", type=Path, required=True)
    parser.add_argument("--t3-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output required: {args.output_root}")

    progress = load(args.progress_index)
    t0, t2, t3 = load(args.t0_result), load(args.t2_result), load(args.t3_result)
    _validate_prerequisites(t0, t2, t3)
    refs = {
        "t0_capacity": artifact_ref(args.t0_result),
        "t2_causal_clean": artifact_ref(args.t2_result),
        "t3_causal_robot_canary": artifact_ref(args.t3_result),
    }
    args.output_root.mkdir(parents=True)
    rows, counts = close_rows(progress, output_root=args.output_root, prerequisite_refs=refs)
    index_path = args.output_root / "ROBOT30_FINAL_TERMINAL_INDEX.json"
    csv_path = args.output_root / "ROBOT30_FINAL_TERMINAL_INDEX.csv"
    payload = {
        "schema_version": "chaoyang-rc1-robot30-final-terminal-index-v1",
        "created_at": now_iso(),
        "status": "PASSED_TERMINAL_COVERAGE",
        "counts": counts,
        "rows": rows,
        "inputs": {"progress_index": artifact_ref(args.progress_index), **refs},
        "all_selected_rows_terminal": True,
        "rate_finalized": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "authority_promoted": False,
        "claim_limit": "Finite Robot30 release closure only; offline geometry evidence is not causal training data and budget terminals are not quality failures.",
    }
    atomic_json(index_path, payload)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["session_id", "task", "target_rank", "execution_route", "terminal_status", "hard_geometry_pass", "input_mode", "training_eligible"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    result = {
        "schema_version": "chaoyang-rc1-robot30-final-terminal-result-v1",
        "created_at": now_iso(),
        "status": "PASSED_TERMINAL_COVERAGE",
        "counts": counts,
        "index": artifact_ref(index_path),
        "csv": artifact_ref(csv_path),
        "rate_finalized": False,
        "authority_promoted": False,
        "claim_limit": payload["claim_limit"],
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
