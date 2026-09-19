#!/usr/bin/env python3
"""CAS-finalize one current node from the 0915 Robot fifteen-hour DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

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
from chaoyang.governance.robot15h_task_specs_v1 import (
    EXECUTION_REVISION,
    PLAN_REVISION,
    TASK_ORDER,
    WINDOW_RUN_ID,
)


CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TERMINAL = {
    "PASSED",
    "REJECTED_QUALITY",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "CANCELLED",
    "BUDGET_EXHAUSTED",
}
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def _validate_terminal(result_path: Path, result: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("task_id") != packet.get("task_id"):
        errors.append("task packet/result identity mismatch")
    if result.get("window_run_id") != WINDOW_RUN_ID:
        errors.append("result window_run_id mismatch")
    write_set = packet.get("write_set", [])
    if not write_set:
        errors.append("task packet has no write_set")
    else:
        root = Path(str(write_set[0]))
        expected = root if root.is_absolute() else REPO_ROOT / root
        if result_path.parent.resolve() != expected.resolve():
            errors.append("RESULT is outside packet primary write root")
    if result.get("status") == "PASSED":
        if result.get("weights") != packet.get("weights"):
            errors.append("successful result weight identity differs from packet")
        for name in packet.get("required_outputs", []):
            if not (result_path.parent / str(name)).is_file():
                errors.append(f"required output is missing: {name}")
    return errors


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
        raise RuntimeError("CAS revision mismatch before Robot15h finalization")
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
    packet_path = REPO_ROOT / str(route.get("packet_path"))
    if route.get("packet_sha256") != artifact_ref(packet_path)["sha256"]:
        raise RuntimeError("task packet SHA differs from current route")
    packet = load_json(packet_path)
    errors = _validate_terminal(result_path, result, packet)
    if errors:
        raise RuntimeError("terminal bundle invalid: " + "; ".join(errors))

    output.mkdir(parents=True)
    predecessor = output / "PREDECESSOR_ACTIVE_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, predecessor)
    terminal_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{args.task_id.upper()}_TERMINAL",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(predecessor),
        "task_packets": [],
        "claim_limit": (
            "This Robot15h node is terminal.  No successor is executable until a new "
            "CAS registration publishes exactly one frozen DAG node."
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
        "message": "Robot15h frozen DAG node reached a recorded terminal.",
        "result": artifact_ref(result_path),
        "window_run_id": WINDOW_RUN_ID,
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
        "schema_version": "finalize-0915-robot15h-task-v1",
        "task_id": args.task_id,
        "window_run_id": WINDOW_RUN_ID,
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
