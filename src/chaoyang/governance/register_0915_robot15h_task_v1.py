#!/usr/bin/env python3
"""CAS-register one node from the frozen 0915 Robot fifteen-hour DAG."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance.robot15h_task_specs_v1 import (
    EXECUTION_REVISION,
    PLAN_REVISION,
    TASK_ORDER,
    WINDOW_CLOCK,
    WINDOW_RUN_ID,
    build_packet,
)


CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
TERMINAL = {
    "PASSED",
    "REJECTED_QUALITY",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "CANCELLED",
    "BUDGET_EXHAUSTED",
}


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    atomic_json(path, value)


def _validate_read_closure(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for raw in packet.get("read_set", []):
        path = Path(str(raw))
        resolved = path if path.is_absolute() else REPO_ROOT / path
        if not resolved.exists():
            errors.append(f"read_set path is not materialized: {raw}")
    weights = packet.get("weights")
    if isinstance(weights, list):
        for raw in weights:
            path = REPO_ROOT / str(raw)
            if not path.is_file():
                errors.append(f"weight is not materialized: {raw}")
    return errors


def _validate_window_clock(path: Path) -> dict[str, Any]:
    clock = load_json(path)
    if (
        clock.get("schema_version") != "0915-robot15h-window-clock-v1"
        or clock.get("run_id") != WINDOW_RUN_ID
        or clock.get("t0_policy")
        != "WRITE_ONCE_FROM_USER_AUTHORIZATION_NOT_RESET_BY_RETRY_OR_SUCCESSOR"
        or clock.get("started_at") != "2026-09-19T00:09:59+08:00"
        or clock.get("deadline_at") != "2026-09-19T15:09:59+08:00"
    ):
        raise RuntimeError("immutable window clock differs from the authorized T0")
    return clock


def _validate_dependencies(state: dict[str, Any], packet: dict[str, Any]) -> None:
    rows = {str(row.get("task_id")): row for row in state.get("tasks", [])}
    for task_id in packet.get("dag_dependencies", []):
        row = rows.get(str(task_id))
        if row is None:
            raise RuntimeError(f"DAG dependency is absent: {task_id}")
        if row.get("status") not in TERMINAL:
            raise RuntimeError(f"DAG dependency is not terminal: {task_id}={row.get('status')}")
        result = row.get("result")
        if not isinstance(result, dict) or validate_artifact_ref(result):
            raise RuntimeError(f"DAG dependency terminal result is not bound: {task_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=TASK_ORDER)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--window-clock", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh registration output required: {output}")
    expected_clock = (REPO_ROOT / WINDOW_CLOCK).resolve()
    if args.window_clock.resolve(strict=True) != expected_clock:
        raise RuntimeError("registration must use the one frozen campaign clock")
    clock = _validate_window_clock(expected_clock)

    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before Robot15h registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))

    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None:
        raise RuntimeError("registration requires next_task=null")
    if any(row.get("status") in LIVE for row in state.get("tasks", [])):
        raise RuntimeError("registration requires no live task")
    if any(row.get("task_id") == args.task_id for row in state.get("tasks", [])):
        raise RuntimeError(f"task already registered: {args.task_id}")

    packet = build_packet(args.task_id)
    packet_errors = _validate_packet(packet) + _validate_read_closure(packet)
    if packet_errors:
        raise RuntimeError("invalid Robot15h packet: " + "; ".join(packet_errors))
    _validate_dependencies(state, packet)

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    pointed = REPO_ROOT / str(pointer["index_path"])
    if pointed.resolve() != CURRENT_INDEX.resolve():
        raise RuntimeError("current packet pointer does not resolve to tasks/current/INDEX.json")
    current_ref = artifact_ref(CURRENT_INDEX)
    if (
        pointer.get("index_bytes") != current_ref["bytes"]
        or pointer.get("index_sha256") != current_ref["sha256"]
    ):
        raise RuntimeError("current task packet pointer SHA/bytes mismatch")
    current = load_json(CURRENT_INDEX)
    if current.get("task_packets") != [] or current.get("status") != "PASS_NO_ACTIVE_TASKS":
        raise RuntimeError("current packet index is not empty and terminal")

    packet_path = REPO_ROOT / "tasks/current" / args.task_id / "TASK_PACKET.json"
    if packet_path.parent.exists() or packet_path.parent.is_symlink():
        raise RuntimeError(f"fresh task packet directory required: {packet_path.parent}")
    output.mkdir(parents=True)
    frozen_index = output / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, frozen_index)
    _write_once(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{args.task_id.upper()}_ROUTABLE",
        "plan_revision": PLAN_REVISION,
        "execution_revision": EXECUTION_REVISION,
        "status": "PASS",
        "supersedes_index": artifact_ref(frozen_index),
        "task_packets": [{
            "task_id": args.task_id,
            "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
            "weights": copy.deepcopy(packet["weights"]),
            "window_run_id": WINDOW_RUN_ID,
        }],
        "claim_limit": (
            "Exactly one Robot15h DAG node is routable.  The frozen DAG does not grant "
            "parallel global execution authority."
        ),
    }
    created = now_iso()
    state["tasks"].append({
        "task_id": args.task_id,
        "phase": packet["phase"],
        "plan_execution_revision": EXECUTION_REVISION,
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": packet_ref,
        "window_run_id": WINDOW_RUN_ID,
        "window_clock": artifact_ref(expected_clock),
    })
    state["next_task"] = {
        "task_id": args.task_id,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": packet["expected_resource"],
        "stop_condition": "Reach one recorded terminal without changing the frozen T0 or quality gates.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": args.task_id,
        "session": None,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered one node from the frozen 0915 Robot fifteen-hour DAG.",
        "task_packet": packet_ref,
        "window_run_id": WINDOW_RUN_ID,
        "window_deadline_at": clock["deadline_at"],
    }])[-100:]

    result_path = output / "RESULT.json"
    _write_once(result_path, {
        "schema_version": "register-0915-robot15h-task-result-v1",
        "task_id": args.task_id,
        "status": "PASSED",
        "registered_at": created,
        "window_run_id": WINDOW_RUN_ID,
        "window_clock": artifact_ref(expected_clock),
        "task_packet": packet_ref,
        "weights": packet["weights"],
        "execution_started": False,
        "claim_limit": "Governance registration only; no algorithm or batch result.",
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type=f"{args.task_id.upper()}_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor_index,
    )
    _write_once(output / "RUN_RECEIPT.json", {
        "schema_version": "register-0915-robot15h-task-receipt-v1",
        "task_id": args.task_id,
        "status": "PASSED",
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "window_run_id": WINDOW_RUN_ID,
        "execution_started": False,
    })
    print(json.dumps({
        "status": "PASSED",
        "task_id": args.task_id,
        "window_run_id": WINDOW_RUN_ID,
        "governance_revision": published["governance_revision"],
        "weights": packet["weights"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
