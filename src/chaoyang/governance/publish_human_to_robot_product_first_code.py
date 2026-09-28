"""CAS-publish the product-first code closure without promoting product quality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, TASK_STATE_PATH, load_json, publish_bundle,
)


TASK = "human_to_robot_product_first_cleanup_20260923"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="HUMAN_TO_ROBOT_PRODUCT_FIRST_CODE_CLOSURE_REGISTERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
