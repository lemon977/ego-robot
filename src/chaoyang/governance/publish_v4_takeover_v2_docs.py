"""Rebind governed current-doc SHA after the V4 V2 takeover handoff edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    load_json, publish_bundle,
)


TASK = "four_stream_full_pipeline_v4_takeover_v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("takeover is not the current task")
    row = next(x for x in state["tasks"] if x.get("task_id") == TASK)
    if row.get("status") != "PENDING" or row.get("pid") is not None:
        raise RuntimeError("active writer or unexpected task state")
    audit = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/ROUTE_AND_INPUT_AUDIT.json"
    if load_json(audit).get("task_id") != TASK:
        raise RuntimeError("missing current route audit")
    for name in ("PLAN.md", "FOUR_STREAM_FULL_PIPELINE_V4_HANDOFF_ZH.md", "AI1.md", "AI2.md", "AI4_HURO.md", "EXACT78.md", "README_ZH.md", "AI_WORK_ENTRY_ZH.md"):
        body = (REPO_ROOT / "docs/current" / name).read_text(encoding="utf-8")
        if TASK not in body:
            raise RuntimeError(f"current takeover marker missing from {name}")
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_V4_TAKEOVER_V2_DOC_REBIND",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": "DOCS_REBOUND", "revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
