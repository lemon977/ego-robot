#!/usr/bin/env python3
"""Recover the terminal governance publish after outputs were written before an API mismatch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_baseline_v1_convergence_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RESULT = ROOT / "RESULT.json"
RECEIPT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}_RESULT.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    result = load_json(RESULT); receipt = load_json(RECEIPT)
    if result.get("task_id") != TASK or result.get("task_terminal_status") != "REJECTED_QUALITY":
        raise RuntimeError("TERMINAL_RESULT_INVALID")
    if receipt.get("result", {}).get("sha256") != artifact_ref(RESULT)["sha256"]:
        raise RuntimeError("TERMINAL_RECEIPT_RESULT_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
        raise RuntimeError("TASK_NOT_RECOVERABLE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    terminal_at = result["terminal_at"]; result_ref = artifact_ref(RESULT)
    task.update(status="REJECTED_QUALITY", phase="CONVERGENCE_TERMINAL", attempt=1, pid=None,
                proc_start_ticks=None, gpu_id=None, heartbeat_at=None, updated_at=terminal_at,
                result=result_ref, last_attempt_terminal="REJECTED_QUALITY",
                last_attempt_reason="FINITE_WORK_EXECUTED_WITH_PRODUCT_QUALITY_GAPS")
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{"task_id": TASK, "attempt": 1,
        "status": "REJECTED_QUALITY", "created_at": now_iso(),
        "message": "Recovered terminal governance publish; no algorithm rerun or artifact rewrite.",
        "result": result_ref}])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_20260923",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX), "task_packets": [],
        "claim_limit": "Convergence successor is terminal rejected-quality; a new task requires new evidence or a genuinely different authorized contract."}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_BASELINE_V1_CONVERGENCE_TERMINAL_PUBLISH_RECOVERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REJECTED_QUALITY", "revision": published["governance_revision"],
                      "result": result_ref, "recovery": "PUBLISH_ONLY_NO_ALGORITHM_RERUN"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
