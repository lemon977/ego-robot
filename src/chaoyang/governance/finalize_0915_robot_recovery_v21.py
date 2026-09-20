#!/usr/bin/env python3
"""CAS-finalize the single 0915 Robot recovery V2.1 parent task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)


TASK_ID = "0915_robot_quality_recovery_15h_v2"
PLAN_REVISION = "0915_ROBOT_RECOVERY_15H_V2_1"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
LIVE = {"PENDING", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
TERMINAL = {
    "PASSED",
    "REJECTED_QUALITY",
    "FAILED_QUALITY_C",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "BLOCKED_REFERENCE_PROOF",
    "UNKNOWN_VERIFICATION_REQUIRED",
    "CANCELLED",
}


def resolve_active_index(current: dict, expected_revision: int) -> tuple[dict, Path]:
    route = next((row for row in current.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is not None:
        return current, CURRENT_INDEX
    if (
        current.get("status") != "PASS_NO_ACTIVE_TASKS"
        or current.get("packet_revision") != f"{TASK_ID.upper()}_TERMINAL"
        or current.get("governance_revision") != expected_revision + 1
        or current.get("task_packets") != []
    ):
        raise RuntimeError("V2.1 parent is not current and no recoverable partial terminal exists")
    predecessor_ref = current.get("supersedes_index")
    errors = validate_artifact_ref(predecessor_ref) if isinstance(predecessor_ref, dict) else ["missing predecessor"]
    if errors:
        raise RuntimeError("invalid terminal predecessor: " + "; ".join(errors))
    path = Path(str(predecessor_ref["path"]))
    predecessor = load_json(path)
    if not any(row.get("task_id") == TASK_ID and row.get("execution_allowed") is True for row in predecessor.get("task_packets", [])):
        raise RuntimeError("terminal predecessor does not route V2.1 parent")
    return predecessor, path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh finalization output required: {output}")
    result_path = args.result.resolve(strict=True)
    result = load_json(result_path)
    if result.get("task_id") != TASK_ID or result.get("status") not in TERMINAL:
        raise RuntimeError("invalid V2.1 terminal result identity/status")
    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before V2.1 finalization")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID), None)
    if task is None or task.get("status") not in LIVE or state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("V2.1 parent is not the live current task")
    current, current_path = resolve_active_index(load_json(CURRENT_INDEX), args.expected_revision)
    route = next(row for row in current.get("task_packets", []) if row.get("task_id") == TASK_ID)
    if route.get("execution_allowed") is not True:
        raise RuntimeError("V2.1 parent route is not executable")
    packet_path = REPO_ROOT / str(route["packet_path"])
    packet = load_json(packet_path)
    if result_path.parent.resolve() != (REPO_ROOT / packet["write_set"][0]).resolve():
        raise RuntimeError("V2.1 terminal result is outside the primary write root")
    if result.get("status") == "PASSED":
        missing = [name for name in packet.get("required_outputs", []) if not (result_path.parent / name).is_file()]
        if missing:
            raise RuntimeError(f"V2.1 PASSED terminal is missing required outputs: {missing}")
        p0 = [
            result_path.parent / "R2_CONTRACT_SNAPSHOT_V1.json",
            result_path.parent / "LINEAGE_TOMBSTONE_V1.json",
        ]
        if not any(path.is_file() for path in p0):
            raise RuntimeError("V2.1 P0 has neither recovered snapshot nor lineage tombstone")

    output.mkdir(parents=True)
    predecessor = output / "PREDECESSOR_ACTIVE_INDEX.json"
    shutil.copyfile(current_path, predecessor)
    terminal_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{TASK_ID.upper()}_TERMINAL",
        "plan_revision": PLAN_REVISION,
        "execution_revision": PLAN_REVISION,
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(predecessor),
        "task_packets": [],
        "claim_limit": "V2.1 parent is terminal; a new CAS registration is required for any successor.",
    }
    created = now_iso()
    task.update({
        "status": result["status"],
        "updated_at": created,
        "heartbeat_at": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "result": artifact_ref(result_path),
    })
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID,
        "session": None,
        "attempt": task.get("attempt", 1),
        "status": result["status"],
        "created_at": created,
        "message": "Finite V2.1 campaign parent reached a recorded terminal.",
        "result": artifact_ref(result_path),
    }])[-100:]
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type=f"{TASK_ID.upper()}_{result['status']}",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=terminal_index,
    )
    final = {
        "schema_version": "finalize-0915-robot-recovery-v21-v1",
        "task_id": TASK_ID,
        "status": result["status"],
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "next_task": None,
    }
    atomic_json(output / "FINALIZATION.json", final)
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
