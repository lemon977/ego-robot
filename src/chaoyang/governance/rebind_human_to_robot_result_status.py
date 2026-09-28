"""Rebind the terminal shallow STATUS projection after its first publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "human_to_robot_result_breakthrough_20260924"
STATUS = REPO_ROOT / "docs/current/STATUS.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    status = load_json(STATUS)
    state = load_json(TASK_STATE_PATH)
    if status.get("latest_task") != TASK or status.get("latest_terminal_status") != "REJECTED_QUALITY":
        raise RuntimeError("WRONG_SHALLOW_STATUS")
    if state.get("next_task") is not None or next(
        (x for x in state["tasks"] if x.get("task_id") == TASK), {}
    ).get("status") != "REJECTED_QUALITY":
        raise RuntimeError("TASK_NOT_TERMINAL")
    status["governance_revision"] = args.expected_revision + 1
    status["generated_at"] = now_iso()
    atomic_json(STATUS, status)
    receipt = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_RESULT_STATUS_REF_REBOUND",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "STATUS_REF_REBOUND", "governance_revision": receipt["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
