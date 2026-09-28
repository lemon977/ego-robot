"""Regenerate governed references after the V4 terminal handoff edit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    load_json,
    publish_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if receipt["governance_revision"] != args.expected_revision:
        raise RuntimeError("governance revision changed")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None:
        raise RuntimeError("active route exists")
    row = next(item for item in state["tasks"] if item["task_id"] == "four_stream_full_pipeline_v4")
    if row["status"] != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("V4 terminal state changed")
    plan = (REPO_ROOT / "docs/current/PLAN.md").read_text(encoding="utf-8")
    handoff = (REPO_ROOT / "docs/current/FOUR_STREAM_FULL_PIPELINE_V4_HANDOFF_ZH.md").read_text(encoding="utf-8")
    if "V4 已于 2026-09-22 封为" not in plan or "原 V4 协调进程已自然退出" not in handoff:
        raise RuntimeError("terminal handoff edit missing")
    result = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="FOUR_STREAM_V4_TERMINAL_DOC_REBIND",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(json.dumps({"status": "PASSED", "revision": result["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
