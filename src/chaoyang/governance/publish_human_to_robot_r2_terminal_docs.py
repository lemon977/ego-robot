#!/usr/bin/env python3
"""Rebind terminal R2 shallow navigation through the governance publisher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    load_json,
    publish_bundle,
)

TASK = "human_to_robot_root_cause_gated_r2_20260923"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") != "REJECTED_QUALITY":
        raise RuntimeError("R2_NOT_TERMINAL_REJECTED_QUALITY")
    if state.get("next_task") is not None:
        raise RuntimeError("UNEXPECTED_NEXT_TASK")
    index = load_json(INDEX)
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("CURRENT_INDEX_NOT_EMPTY")
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_R2_TERMINAL_DOCS_REBOUND",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )
    print(json.dumps({
        "status": "PASSED",
        "task_id": TASK,
        "governance_revision": published["governance_revision"],
        "current_index_status": "PASS_NO_ACTIVE_TASKS",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
