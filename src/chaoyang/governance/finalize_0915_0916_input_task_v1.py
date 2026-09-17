#!/usr/bin/env python3
"""CAS-finalize the weightless 0915/0916 input task."""

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
    load_json,
    now_iso,
    publish_bundle,
)


TASK_ID = "0915_0916_input_audit_clean_v1"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh finalization output required: {output}")
    result_path = args.result.resolve(strict=True)
    result = load_json(result_path)
    if result.get("task_id") != TASK_ID:
        raise RuntimeError("result task identity mismatch")
    status = result.get("status")
    if status not in {"PASSED", "FAILED_RUNTIME_FINAL"}:
        raise RuntimeError(f"unsupported terminal status: {status}")
    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before finalization")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state.get("tasks", [])
                 if row.get("task_id") == TASK_ID), None)
    if task is None or task.get("status") not in {
        "PENDING", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE",
    }:
        raise RuntimeError("task is not live")

    output.mkdir(parents=True)
    predecessor = output / "PREDECESSOR_ACTIVE_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, predecessor)
    terminal_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "0915_0916_INPUT_AUDIT_CLEAN_V1_TERMINAL",
        "plan_revision": "0915_FULL_FUNNEL_0916_CLEAN_V1",
        "execution_revision": "0915_0916_V1",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(predecessor),
        "task_packets": [],
        "claim_limit": (
            "The processed-input/0916-cleaning task is terminal; no algorithm "
            "successor is executable until separately registered."
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
        "task_id": TASK_ID,
        "session": None,
        "attempt": 1,
        "status": status,
        "created_at": created,
        "message": "0915 processed self-containment and 0916 cleaning reached a bounded terminal.",
        "result": artifact_ref(result_path),
    }])[-100:]
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type=f"0915_0916_INPUT_TASK_{status}",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=terminal_index,
    )
    final = {
        "schema_version": "finalize-0915-0916-input-task-v1",
        "status": status,
        "task_id": TASK_ID,
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
