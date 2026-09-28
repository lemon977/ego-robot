#!/usr/bin/env python3
"""Complete the interrupted V5 terminal governance publish without rerunning data.

Usage: PYTHONPATH=src python -B -m \
  chaoyang.governance.recover_four_stream_v5_terminal_publish \
  --expected-revision <last committed receipt revision>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, load_json, publish_bundle,
)

TASK = "four_stream_visual_delivery_v5"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("LAST_COMMITTED_REVISION_CHANGED")
    state = load_json(TASK_STATE_PATH)
    authority = load_json(AUTHORITY_PATH)
    if (state.get("governance_revision") != args.expected_revision + 1
            or authority.get("governance_revision") != args.expected_revision + 1):
        raise RuntimeError("NOT_EXACTLY_ONE_PARTIAL_PUBLICATION")
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if (task is None or task.get("status") != "FAILED_QUALITY_C"
            or state.get("next_task") is not None):
        raise RuntimeError("V5_TERMINAL_STATE_NOT_WRITTEN")
    result_path = ROOT / "RESULT.json"
    result = load_json(result_path)
    if (result.get("status") != "FAILED_QUALITY_C" or result.get("release_status") != "INCOMPLETE"
            or result.get("review_videos_complete") != 11 or result.get("product_robot_videos") != 0
            or task.get("result") != artifact_ref(result_path)):
        raise RuntimeError("V5_TERMINAL_RESULT_MISMATCH")
    task_receipt = load_json(REPO_ROOT / "tasks/receipts/FOUR_STREAM_VISUAL_DELIVERY_V5_RESULT.json")
    if task_receipt.get("result") != artifact_ref(result_path):
        raise RuntimeError("TASK_RECEIPT_MISMATCH")
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if (index.get("packet_revision") != "FOUR_STREAM_VISUAL_DELIVERY_V5_TERMINAL"
            or index.get("task_packets") != []):
        raise RuntimeError("TERMINAL_TASK_INDEX_MISMATCH")
    published = publish_bundle(
        authority, state,
        event_type="FOUR_STREAM_VISUAL_DELIVERY_V5_TERMINAL_PUBLISH_RECOVERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "RECOVERED", "revision": published["governance_revision"],
                      "result": str(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
