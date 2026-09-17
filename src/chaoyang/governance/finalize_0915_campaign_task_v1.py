#!/usr/bin/env python3
"""CAS-finalize one registered 0915 campaign task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from chaoyang.governance.campaign_0915_task_specs_v1 import PLAN_REVISION, TASK_ORDER
from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    now_iso,
    publish_bundle,
)


CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TERMINAL = {"PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE", "BLOCKED_EXTERNAL", "CANCELLED"}
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=TASK_ORDER)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh finalization output required: {output}")
    result_path = args.result.resolve(strict=True)
    result = load_json(result_path)
    if result.get("task_id") != args.task_id:
        raise RuntimeError("result task identity mismatch")
    status = str(result.get("status"))
    if status not in TERMINAL:
        raise RuntimeError(f"unsupported terminal status: {status}")
    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before finalization")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state.get("tasks", []) if row.get("task_id") == args.task_id), None)
    if task is None or task.get("status") not in LIVE:
        raise RuntimeError("task is not live")
    if state.get("next_task", {}).get("task_id") != args.task_id:
        raise RuntimeError("task is not current next_task")
    current = load_json(CURRENT_INDEX)
    route = next((row for row in current.get("task_packets", []) if row.get("task_id") == args.task_id), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task is not current routable packet")

    output.mkdir(parents=True)
    predecessor = output / "PREDECESSOR_ACTIVE_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, predecessor)
    terminal_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{args.task_id.upper()}_TERMINAL",
        "plan_revision": PLAN_REVISION,
        "execution_revision": "0915_FULL_FUNNEL_V1",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(predecessor),
        "task_packets": [],
        "claim_limit": (
            "This 0915 task is terminal. No successor is executable until a separate "
            "CAS registration publishes exactly one new packet."
        ),
    }
    created = now_iso()
    task.update({
        "status": status,
        "updated_at": created,
        "heartbeat_at": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "result": artifact_ref(result_path),
    })
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": args.task_id,
        "session": None,
        "attempt": task.get("attempt", 1),
        "status": status,
        "created_at": created,
        "message": "Finite 0915 campaign task reached a recorded terminal.",
        "result": artifact_ref(result_path),
    }])[-100:]
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type=f"{args.task_id.upper()}_{status}",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=terminal_index,
    )
    final = {
        "schema_version": "finalize-0915-campaign-task-v1",
        "task_id": args.task_id,
        "status": status,
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "next_task": None,
    }
    (output / "FINALIZATION.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
