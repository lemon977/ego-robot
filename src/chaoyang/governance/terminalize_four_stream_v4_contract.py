"""Fail-closed terminalization of the V4 heartbeat-only coordinator.

This is governance repair, not an algorithm run or a quality decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
)


TASK = "four_stream_full_pipeline_v4"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
TERMINAL = ATTEMPT / "V4_RUNTIME_TERMINAL_20260922.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if receipt["governance_revision"] != args.expected_revision:
        raise RuntimeError("governance CAS revision changed")
    state = load_json(TASK_STATE_PATH)
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    if row["status"] != "PENDING" or row.get("pid") is not None:
        raise RuntimeError("V4 is not recovered to an unowned PENDING state")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("V4 is not the sole current task")
    index = load_json(INDEX)
    entries = index.get("task_packets", [])
    if len(entries) != 1 or entries[0].get("task_id") != TASK:
        raise RuntimeError("unexpected current task packet index")
    old_claim = load_json(ATTEMPT / "CLAIM_WINDOW.json")
    identity = process_identity(int(old_claim["claim_pid"]))
    if identity["alive"] and identity["start_ticks"] == old_claim["claim_proc_start_ticks"]:
        raise RuntimeError("old V4 coordinator is still alive")
    for lane in ("exact78", "controller_manus", "hawor_retarget", "huro"):
        lane_state = load_json(ATTEMPT / "lanes" / lane / "STATE.json")
        writer = lane_state.get("writer", {})
        if writer.get("pid") is not None:
            raise RuntimeError(f"lane writer still registered: {lane}")
    terminal = load_json(TERMINAL)
    if terminal.get("status") != "FAILED_RUNTIME_FINAL" or terminal.get("quality_evaluated") is not False:
        raise RuntimeError("terminal receipt semantics mismatch")
    if (ATTEMPT / "RESULT.json").exists():
        raise RuntimeError("V4 acquired a RESULT after audit; stop and re-evaluate")
    time = now_iso()
    row.update(
        status="FAILED_RUNTIME_FINAL",
        phase="V4_CONTRACT_AND_COORDINATOR_TERMINAL",
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        heartbeat_at=None,
        updated_at=time,
        result=artifact_ref(TERMINAL),
        last_attempt_terminal="FAILED_RUNTIME_FINAL",
        last_attempt_reason="COORDINATOR_EXITED_WITHOUT_RESULT_AND_CONTRACT_CONFLICT",
    )
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK,
        "attempt": 1,
        "status": "FAILED_RUNTIME_FINAL",
        "created_at": time,
        "message": "Heartbeat-only coordinator closed without quality inference; successor must repair execution contract.",
        "result": artifact_ref(TERMINAL),
    }])[-100:]
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "FOUR_STREAM_FULL_PIPELINE_V4_TERMINAL",
        "plan_revision": "FOUR_STREAM_FULL_PIPELINE_V4",
        "execution_revision": "FOUR_STREAM_FULL_PIPELINE_V4",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "V4 is terminal without algorithm quality results; only a newly registered successor may execute.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="FOUR_STREAM_FULL_PIPELINE_V4_CONTRACT_TERMINAL",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor_index,
    )
    print(json.dumps({"status": "PASSED", "revision": published["governance_revision"], "result": str(TERMINAL)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
