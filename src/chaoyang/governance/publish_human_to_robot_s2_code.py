#!/usr/bin/env python3
"""CAS-publish the bounded S2 code closure without promoting quality."""

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


TASK_ID = "human_to_robot_evidence_unlock_s2_20260923"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("S2_IS_NOT_CURRENT")
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_S2_CODE_CLOSURE_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(json.dumps({
        "status": "REGISTERED",
        "task_id": TASK_ID,
        "governance_revision": published["governance_revision"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
