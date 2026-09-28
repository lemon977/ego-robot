#!/usr/bin/env python3
"""Atomically clear the stale current-task route after sealed R1 terminalization."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    publish_bundle,
)

R1 = "human_to_robot_completion_r1_20260922"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RESULT = REPO_ROOT / f"_run/current/{R1}/attempts/attempt_0001/RESULT.json"
ACTIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()

    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if state.get("next_task") is not None:
        raise RuntimeError("next_task is not empty")
    active = [row.get("task_id") for row in state.get("tasks", []) if row.get("status") in ACTIVE]
    if active:
        raise RuntimeError(f"active tasks remain: {active}")
    r1 = next((row for row in state.get("tasks", []) if row.get("task_id") == R1), None)
    if r1 is None or r1.get("status") != "REJECTED_QUALITY":
        raise RuntimeError("R1 is not terminal REJECTED_QUALITY")
    result = load_json(RESULT)
    if result.get("terminal") is not True or result.get("status") != "TERMINAL_WITH_GAPS":
        raise RuntimeError("R1 immutable result is not terminal")
    old_index = load_json(INDEX)
    rows = old_index.get("task_packets", [])
    if len(rows) != 1 or rows[0].get("task_id") != R1:
        raise RuntimeError("current index is not the stale R1 route")

    empty_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "NO_ACTIVE_TASKS_AFTER_R1_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_COMPLETION_R1_20260922_TERMINAL",
        "execution_revision": "HUMAN_TO_ROBOT_COMPLETION_R1_20260922_TERMINAL",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(INDEX),
        "terminal_predecessor": artifact_ref(RESULT),
        "task_packets": [],
        "claim_limit": "No executable current task; R1 remains immutable and rejected on quality.",
    }
    published = publish_bundle(
        copy.deepcopy(load_json(AUTHORITY_PATH)),
        state,
        event_type="HUMAN_TO_ROBOT_COMPLETION_R1_CURRENT_INDEX_CLEARED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=empty_index,
    )
    print(json.dumps({
        "status": "PASS_NO_ACTIVE_TASKS",
        "governance_revision": published["governance_revision"],
        "r1_result": artifact_ref(RESULT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
