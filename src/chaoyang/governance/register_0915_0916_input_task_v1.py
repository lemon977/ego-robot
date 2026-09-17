#!/usr/bin/env python3
"""CAS-register the first bounded 0915/0916 campaign task."""

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
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "0915_0916_input_audit_clean_v1"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2,
                         sort_keys=True).encode() + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"immutable artifact conflict: {path}")
        return
    atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output required: {output}")

    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))

    packet = load_json(PACKET)
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid task packet: " + "; ".join(errors))
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("input/cleaning task must have weights=ABSENT")

    state = load_json(TASK_STATE_PATH)
    live = [row for row in state.get("tasks", []) if row.get("status") in {
        "PENDING", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE",
    }]
    if live or state.get("next_task") is not None:
        raise RuntimeError("registration requires no active or pending task")
    if any(row.get("task_id") == TASK_ID for row in state.get("tasks", [])):
        raise RuntimeError(f"task already registered: {TASK_ID}")

    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    predecessor_path = REPO_ROOT / pointer["index_path"]
    if predecessor_path.resolve() != CURRENT_INDEX.resolve():
        raise RuntimeError("clean baseline must point to tasks/current/INDEX.json")
    predecessor_ref = artifact_ref(predecessor_path)
    if (pointer.get("index_bytes") != predecessor_ref["bytes"]
            or pointer.get("index_sha256") != predecessor_ref["sha256"]):
        raise RuntimeError("task index pointer mismatch")

    output.mkdir(parents=True)
    frozen_predecessor = output / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(predecessor_path, frozen_predecessor)
    packet_ref = artifact_ref(PACKET)
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "0915_0916_INPUT_AUDIT_CLEAN_V1",
        "plan_revision": "0915_FULL_FUNNEL_0916_CLEAN_V1",
        "execution_revision": "0915_0916_V1",
        "status": "PASS",
        "supersedes_index": artifact_ref(frozen_predecessor),
        "task_packets": [{
            "task_id": TASK_ID,
            "packet_path": str(PACKET.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
            "weights": "ABSENT",
        }],
        "claim_limit": "One weightless processed-input audit and 0916 cleaning task; no model execution or authority promotion.",
    }

    created = now_iso()
    task = {
        "task_id": TASK_ID,
        "phase": packet["phase"],
        "plan_execution_revision": "0915_0916_V1",
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": packet_ref,
    }
    state["tasks"].append(task)
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "LOW_PRIORITY_CPU_IO; max two cleaning workers; weights ABSENT",
        "stop_condition": "220-session processed self-containment terminal plus 240-session 0916 cleaning terminal.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID,
        "session": None,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered processed-only 0915 audit and weightless 0916 cleaning task; execution not started.",
        "task_packet": packet_ref,
    }])[-100:]

    result_path = output / "RESULT.json"
    _write_once(result_path, {
        "schema_version": "register-0915-0916-input-task-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "registered_at": created,
        "task_packet": packet_ref,
        "execution_started": False,
        "weights": "ABSENT",
        "mask_model_policy": "SAM3.1_ONLY_FOR_LATER_TASK",
        "claim_limit": "Governance registration only; no dataset or algorithm result.",
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="0915_0916_INPUT_TASK_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor_index,
    )
    _write_once(output / "RUN_RECEIPT.json", {
        "schema_version": "register-0915-0916-input-task-receipt-v1",
        "status": "PASSED",
        "task_id": TASK_ID,
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_started": False,
    })
    print(json.dumps({
        "status": "PASSED",
        "task_id": TASK_ID,
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_started": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
