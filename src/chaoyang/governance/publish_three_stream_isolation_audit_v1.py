#!/usr/bin/env python3
"""Publish the current three-stream isolation and provenance correction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    load_json,
    publish_bundle,
)


USER_AUTHORIZATION = REPO_ROOT / "tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json"
INVENTORY_RESULT = REPO_ROOT / "tasks/receipts/0915_ROBOT15H_WINDOW_START_INVENTORY_V1_RESULT.json"
CURRENT_TASK_INDEX = REPO_ROOT / "tasks/current/INDEX.json"


def main() -> int:
    receipt = load_json(RECEIPT_PATH)
    authority = load_json(AUTHORITY_PATH)
    task_state = load_json(TASK_STATE_PATH)
    current_index = load_json(CURRENT_TASK_INDEX)
    authorization = load_json(USER_AUTHORIZATION)
    inventory = load_json(INVENTORY_RESULT)

    active = [
        str(item.get("task_id"))
        for item in task_state.get("tasks", [])
        if isinstance(item, dict)
        and item.get("status") in {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    if active or current_index.get("task_packets"):
        raise RuntimeError(
            f"refusing three-stream authority repair with routable tasks: state={active}, "
            f"index_rows={len(current_index.get('task_packets', []))}"
        )
    fact = authorization.get("session_recording_fact", {})
    if fact.get("source") != "USER_CONFIRMED" or "recorded separately" not in str(fact.get("statement")):
        raise RuntimeError("0915 user recording attestation is missing or changed")
    if inventory.get("source_group_status") != "PASS_USER_CONFIRMED_AND_METADATA_CORROBORATED":
        raise RuntimeError("0915 source-group inventory no longer corroborates user attestation")
    if inventory.get("0916_consumed") is not False:
        raise RuntimeError("0915 inventory unexpectedly consumed 0916 data")

    published = publish_bundle(
        authority,
        task_state,
        event_type="THREE_STREAM_ISOLATION_AND_PROVENANCE_REPAIR_V1",
        expected_revision=int(receipt["governance_revision"]),
        generator_path=Path(__file__),
    )
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "governance_revision": published["governance_revision"],
                "generation_id": published["generation_id"],
                "algorithm_results_changed": False,
                "task_routing_changed": False,
                "provenance_corrected": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
