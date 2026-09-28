"""Publish current navigation and terminal contract for the sealed 031 diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    load_json, publish_bundle,
)

TASK = "human_to_robot_031_surface_depth_diagnostic_20260923"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") != "PASSED" or state.get("next_task") is not None:
        raise RuntimeError("TASK_NOT_TERMINAL")
    result = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/RESULT.json"
    if load_json(result).get("status") != "TERMINAL_DEVELOPMENT_DIAGNOSTIC":
        raise RuntimeError("TERMINAL_RESULT_INVALID")
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_031_SURFACE_DEPTH_DOCS_SYNC",
                               expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "DOCS_SYNCED", "task_id": TASK,
                      "governance_revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
