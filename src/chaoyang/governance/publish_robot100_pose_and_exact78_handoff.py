"""CAS-publish the Robot100 pose diagnosis and 0902 Exact78 handoff document."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    load_json,
    publish_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    exact = [row for row in state.get("tasks", []) if row.get("task_id") == "exact78_inpainting_robot100_20260928"]
    old = [row for row in state.get("tasks", []) if row.get("task_id") == "robot100_batch_baseline_20260928"]
    if len(exact) != 1 or exact[0].get("status") != "PENDING":
        raise RuntimeError("EXACT78_HANDOFF_TASK_NOT_PENDING")
    if len(old) != 1 or old[0].get("status") != "PASSED":
        raise RuntimeError("OLD_ROBOT100_PREDECESSOR_NOT_TERMINAL")
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="ROBOT100_POSE_DIAGNOSIS_AND_EXACT78_HANDOFF_PUBLISHED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(json.dumps({"status": "PUBLISHED", "revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
