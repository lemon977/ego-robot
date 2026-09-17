#!/usr/bin/env python3
"""Publish the standard immutable, non-governance receipt set for one Robot attempt."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.robot_target_reach_v75 import artifact_ref, atomic_new_json, load_json, now_iso  # noqa: E402


TERMINALS = {
    "PASSED",
    "FAILED_QUALITY_C",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "CANCELLED",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--terminal-status", choices=sorted(TERMINALS), required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--next-action", required=True)
    parser.add_argument("--primary-result", type=Path, action="append", required=True)
    parser.add_argument("--preserve-existing-result", action="store_true")
    args = parser.parse_args()
    attempt = args.attempt_dir.resolve(strict=True)
    results = [path.resolve(strict=True) for path in args.primary_result]
    result_refs = [artifact_ref(path) for path in results]
    generated_at = now_iso()
    result_path = attempt / "RESULT.json"
    if not args.preserve_existing_result:
        atomic_new_json(
            result_path,
            {
                "schema_version": "robot-v75-attempt-result-v1",
                "artifact_revision": "R7_ROBOT_5",
                "created_at": generated_at,
                "task_id": args.task_id,
                "terminal_status": args.terminal_status,
                "robot_tier": "NONE",
                "results": result_refs,
                "authority": False,
                "control_ground_truth": False,
                "physical_deployment_authorized": False,
                "claim_limit": "Attempt-level diagnostic/recovery result only; governance aggregator has not promoted it.",
            },
        )
    else:
        if result_path not in results:
            raise SystemExit("--preserve-existing-result requires RESULT.json among --primary-result")
        existing = load_json(result_path)
        if existing.get("terminal_status") != args.terminal_status:
            raise SystemExit("existing RESULT terminal_status differs from requested receipt status")
    task_result = artifact_ref(result_path)
    atomic_new_json(
        attempt / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "robot-v75-artifact-manifest-v1",
            "created_at": generated_at,
            "task_id": args.task_id,
            "artifacts": [task_result, *[ref for ref in result_refs if ref != task_result]],
        },
    )
    metrics = {
        "schema_version": "robot-v75-attempt-metrics-v1",
        "created_at": generated_at,
        "task_id": args.task_id,
        "terminal_status": args.terminal_status,
        "source_result_count": len(results),
    }
    for path in results:
        value = load_json(path)
        metrics[path.name] = {
            key: value.get(key)
            for key in ("terminal_status", "adopt_preflight_status", "governance_gate", "counts", "robot_tier", "candidate_robot_tier", "visual_train_eligible")
            if key in value
        }
    atomic_new_json(attempt / "METRICS.json", metrics)
    atomic_new_json(
        attempt / "RUN_RECEIPT.json",
        {
            "schema_version": "robot-v75-run-receipt-v1",
            "created_at": generated_at,
            "task_id": args.task_id,
            "terminal_status": args.terminal_status,
            "result": task_result,
            "producer": artifact_ref(Path(__file__)),
            "governance_current_modified": False,
            "authority": False,
        },
    )
    decision_path = attempt / "DECISION.md"
    if decision_path.exists() or decision_path.is_symlink():
        raise FileExistsError(decision_path)
    decision_path.write_text(
        f"# {args.task_id}\n\n状态：`{args.terminal_status}`\n\n{args.decision}\n\n"
        "本 attempt 不授予 Robot、接触、控制或物理部署 authority。\n",
        encoding="utf-8",
    )
    atomic_new_json(
        attempt / "NEXT_ACTION.json",
        {
            "schema_version": "robot-v75-next-action-v1",
            "created_at": generated_at,
            "task_id": args.task_id,
            "next_action": args.next_action,
            "requires_governance_aggregator": True,
        },
    )
    print(json.dumps({"task_id": args.task_id, "terminal_status": args.terminal_status, "result": task_result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
