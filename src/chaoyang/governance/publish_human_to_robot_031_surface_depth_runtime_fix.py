"""CAS-publish the one bounded non-seekable-pipe runtime correction."""
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
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    failure = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/RUNTIME_FAILURE.json"
    if load_json(failure).get("failure_stage") != "MANO_FACES_LOAD_BEFORE_OUTPUT_DIRECTORY":
        raise RuntimeError("RUNTIME_FAILURE_MISMATCH")
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_031_SURFACE_DEPTH_RUNTIME_FIX",
                               expected_revision=args.expected_revision, generator_path=Path(__file__))
    print(json.dumps({"status": "RUNTIME_FIX_PUBLISHED", "task_id": TASK,
                      "governance_revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
