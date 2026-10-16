"""Finish the stopped T0 attempt after a partial governance publication."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, process_identity, publish_bundle,
)

TASK = "exact78_raw_to_robot_t0_input_freeze_v1_20260930"
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T0_V1.json"
SCRIPT = Path(__file__)


def main() -> int:
    writer = load_json(OUT / "WRITER.json")
    identity = process_identity(int(writer["pid"]))
    if identity["alive"] and identity["start_ticks"] == writer["proc_start_ticks"]:
        raise RuntimeError("ORIGINAL_WRITER_STILL_ALIVE")
    result_path = OUT / "RESULT.json"
    result = load_json(result_path)
    if result["status"] != "FAILED_RUNTIME" or "DECODE_FRAME_COUNT_MISMATCH:play_cards_0915_002:131:130" not in result["reason"]:
        raise RuntimeError("UNEXPECTED_FAILURE_EVIDENCE")
    original_receipt = load_json(RECEIPT)
    if original_receipt["result"] != artifact_ref(result_path):
        raise RuntimeError("ORIGINAL_RESULT_SHA_DRIFT")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(r for r in state["tasks"] if r["task_id"] == TASK)
    if row["status"] != "RUNNING" or row["pid"] != writer["pid"] or row["proc_start_ticks"] != writer["proc_start_ticks"]:
        raise RuntimeError("TASK_STATE_CONFLICT")
    active = [r for r in state["tasks"] if r["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK:
        raise RuntimeError("ANOTHER_ACTIVE_TASK")
    repair_path = OUT / "TERMINALIZATION_RECOVERY.json"
    if repair_path.exists():
        raise RuntimeError("RECOVERY_ALREADY_EXISTS")
    atomic_json(repair_path, {
        "schema_version": "exact78-raw-t0-terminalization-recovery-v1",
        "task_id": TASK, "status": "FAILED_RUNTIME_FINAL",
        "reason": "Decode frame count conflict in the preselected 0915 holdout; original immutable result and receipt retained byte-for-byte",
        "original_result": artifact_ref(result_path),
        "original_receipt": artifact_ref(RECEIPT),
        "original_writer": writer,
        "original_writer_alive_with_same_startticks": False,
        "repair_writer": {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"]},
        "next_action": "Register a new finite T0 successor; retain Cards002 mismatch as source/time evidence",
    })
    row.update(status="FAILED_RUNTIME_FINAL", pid=None, proc_start_ticks=None,
        updated_at=now_iso(), heartbeat_at=now_iso(),
        result=artifact_ref(result_path), terminal_receipt=artifact_ref(repair_path),
        quality="SOURCE_FRAME_COUNT_CONFLICT", adoption="NOT_ADOPTED", review="MACHINE_AUDITED")
    state["next_task"] = None
    index = {k: copy.deepcopy(v) for k, v in load_json(INDEX).items()
             if k not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}
    row_index = next(r for r in index["task_packets"] if r["task_id"] == TASK)
    row_index["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T0_V1_FAILED_RUNTIME_FINAL",
        plan_revision="EXACT78_RAW_TO_ROBOT_V1_20260930",
        execution_revision="EXACT78_RAW_T0_V1", status="PASS_NO_ACTIVE_TASKS",
        supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T0_V1_TERMINALIZATION_RECOVERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    print(json.dumps({"status": "FAILED_RUNTIME_FINAL", "governance_revision": published["governance_revision"],
        "original_result": artifact_ref(result_path), "recovery": artifact_ref(repair_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
