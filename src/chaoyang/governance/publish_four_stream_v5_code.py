#!/usr/bin/env python3
"""CAS-publish the registered V5 algorithm code closure, without promoting quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, TASK_STATE_PATH, load_json, publish_bundle,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-revision", required=True, type=int)
    args = p.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    current = state.get("next_task", {}).get("task_id")
    if current != "four_stream_visual_delivery_v5":
        raise RuntimeError("V5 is not current")
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_V5_CODE_CLOSURE_REGISTERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "REGISTERED", "revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
