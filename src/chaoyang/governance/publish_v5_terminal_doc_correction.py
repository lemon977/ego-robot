#!/usr/bin/env python3
"""CAS-publish a V5 terminal documentation correction without reopening execution.

Usage: PYTHONPATH=src python -B -m \
  chaoyang.governance.publish_v5_terminal_doc_correction \
  --expected-revision <validated current revision>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, TASK_STATE_PATH,
    load_json, publish_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    row = next((value for value in state["tasks"]
                if value.get("task_id") == "four_stream_visual_delivery_v5"), None)
    if row is None or row.get("status") != "FAILED_QUALITY_C" or state.get("next_task") is not None:
        raise RuntimeError("V5_NOT_TERMINAL")
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_V5_TERMINAL_DOC_RESUME_CAVEAT",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "PUBLISHED", "revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
