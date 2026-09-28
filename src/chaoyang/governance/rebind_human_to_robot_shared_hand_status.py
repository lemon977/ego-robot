"""Rebind the terminal shared-hand shallow STATUS projection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
)


TASK = "human_to_robot_shared_hand_delivery_20260924"
STATUS = REPO_ROOT / "docs/current/STATUS.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if (
        task is None
        or task.get("status") != "REJECTED_QUALITY"
        or state.get("next_task") is not None
    ):
        raise RuntimeError("SHARED_HAND_TERMINAL_PROJECTION_MISMATCH")
    receipt = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_SHARED_HAND_STATUS_REF_REBOUND",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(json.dumps({
        "status": "STATUS_REF_REBOUND",
        "governance_revision": receipt["governance_revision"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
