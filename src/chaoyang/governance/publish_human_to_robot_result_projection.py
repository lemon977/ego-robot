"""Regenerate the current result projection from the terminal task ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, TASK_STATE_PATH,
    load_json, publish_bundle,
)

TASK = "human_to_robot_result_breakthrough_20260924"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((x for x in state["tasks"] if x.get("task_id") == TASK), None)
    if task is None or task.get("status") != "REJECTED_QUALITY" or state.get("next_task") is not None:
        raise RuntimeError("RESULT_TASK_NOT_TERMINAL")
    receipt = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_RESULT_CURRENT_PROJECTION_REPAIRED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "CURRENT_PROJECTION_REPAIRED",
                      "governance_revision": receipt["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
