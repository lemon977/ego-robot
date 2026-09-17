#!/usr/bin/env python3
"""Immutable rev9: recompute remaining work from delivered rows, not class labels."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import copy
import json
from pathlib import Path

from chaoyang.ops import build_rc1_robot30_research_delivery_revision_v7 as v7
from chaoyang.ops import build_rc1_robot30_research_delivery_revision_v8 as v8


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_independent_placement_successor_v1/delivery_index_rev_0008/ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0008.json"
OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_independent_placement_successor_v1/delivery_index_rev_0009/ROBOT30_VIDEO_DELIVERY_MATRIX_RESEARCH_REV_0009.json"
QUEUE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1/lane_e_robot30_successor/queue_prep/attempt_0001/POKER_ROBOT_VIDEO_QUEUE_V1.json"


def remaining(rows: list[dict], queue: dict) -> dict:
    missing = [row for row in rows if row["task"] == "poker" and row["delivery_full_video"] is None]
    queue_rows = {row["session_id"]: row for row in queue["rows"]}
    if len(missing) != 9 or any(row["session_id"] not in queue_rows for row in missing):
        raise RuntimeError("remaining Poker gaps are not closed by frozen input queue")
    blocked = [row["session_id"] for row in missing if queue_rows[row["session_id"]]["queue_class"] == "BLOCKED_PREREQ"]
    new_solve = [row["session_id"] for row in missing if queue_rows[row["session_id"]]["queue_class"] == "INPUT_CLOSURE_READY"]
    if blocked != ["play_cards_0902_049"] or len(new_solve) != 8:
        raise RuntimeError("Poker remaining-work partition changed")
    if any(row["delivery_class"] == "B_RENDER_ONLY" for row in missing):
        raise RuntimeError("unexpected render-only item")
    return {
        "poker_missing_fixed_30_rows": len(missing),
        "remaining_new_solve": len(new_solve),
        "remaining_new_solve_session_ids": new_solve,
        "render_only": 0,
        "blocked_upstream": len(blocked),
        "blocked_upstream_session_ids": blocked,
        "previously_labeled_C_NEW_SOLVE_REQUIRED_but_already_delivered": [
            row["session_id"] for row in rows if row["task"] == "poker" and
            row["delivery_class"] == "C_NEW_SOLVE_REQUIRED" and row["delivery_full_video"] is not None
        ],
        "queue_claim_limit": "INPUT_CLOSURE_READY does not mean executable: the generic independent placement and three fixed regressions are still blocked.",
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    result = copy.deepcopy(source)
    counts = v8.corrected_counts(result["rows"])
    work = remaining(result["rows"], queue)
    if work["previously_labeled_C_NEW_SOLVE_REQUIRED_but_already_delivered"] != ["play_cards_0901_001"]:
        raise RuntimeError("historical class-label exception changed")
    result.update({
        "schema_version": "rc1-robot30-research-delivery-matrix-rev9",
        "counts": counts,
        "normalized_watchable_counts": {task: counts[task]["all_watchable_full_video"] for task in counts},
        "remaining_work": work,
        "remaining_queue_source": v7.ref(QUEUE),
        "supersedes": v7.ref(SOURCE),
        "generator": v7.ref(Path(__file__)),
        "claim_limit": "Fixed Poker30: 21 watchable full videos, 8 undelivered input-closed but runner-blocked, 1 upstream/runtime blocked. Poker001 historical C_NEW_SOLVE_REQUIRED label is not current work. No formal Robot/training authority changed.",
    })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": v7.ref(OUTPUT)["sha256"],
                      "poker_watchable": counts["poker"]["all_watchable_full_video"],
                      "poker_remaining_new_solve": work["remaining_new_solve"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
