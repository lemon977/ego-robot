#!/usr/bin/env python3
"""Build the V3.2 four-stream shallow status from machine-owned state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from chaoyang.governance.common import (
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
)


TASK_ID = "four_stream_pretraining_baseline_v32"
RUN_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
OUTPUT = REPO_ROOT / "docs/current/STATUS.json"
LANES = ("exact78", "ai1", "ai2", "ai4_huro")


def _optional_object(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"state must be an object: {path}")
    return value


def build_status(
    *,
    task_state: Mapping[str, Any],
    generated_at: str,
    source_task_state_path: Path,
    run_root: Path = RUN_ROOT,
) -> dict[str, Any]:
    parents = [
        row for row in task_state.get("tasks", [])
        if isinstance(row, Mapping) and row.get("task_id") == TASK_ID
    ]
    if len(parents) > 1:
        raise RuntimeError("duplicate V3.2 parent task rows")
    parent = parents[0] if parents else None
    next_task = task_state.get("next_task")
    lanes: dict[str, Any] = {}
    for lane in LANES:
        state_path = run_root / "lanes" / lane / "STATE.json"
        state = _optional_object(state_path)
        if state is None:
            lanes[lane] = {
                "status": "REGISTERED_NOT_STARTED" if parent else "NOT_REGISTERED",
                "current_action": None,
                "latest_artifacts": [],
                "blocker": None,
                "next_step": "INITIALIZE_PARENT" if parent else "REGISTER_PARENT",
                "state_file": None,
            }
            continue
        if state.get("lane") != lane or state.get("parent_task_id") != TASK_ID:
            raise RuntimeError(f"lane state identity mismatch: {state_path}")
        lanes[lane] = {
            "status": state.get("status", "UNKNOWN"),
            "current_action": state.get("current_action"),
            "latest_artifacts": state.get("latest_artifacts", []),
            "blocker": state.get("blocker"),
            "next_step": state.get("next_step"),
            "claims": state.get("claims", {}),
            "state_file": artifact_ref(state_path),
        }
    return {
        "schema_version": "chaoyang-four-stream-shallow-status-v1",
        "generated_at": generated_at,
        "source": {
            "task_state": artifact_ref(source_task_state_path),
            "parent_task_id": TASK_ID,
            "run_root": str(run_root),
        },
        "parent": {
            "registered": parent is not None,
            "routable": bool(
                isinstance(next_task, Mapping) and next_task.get("task_id") == TASK_ID
            ),
            "status": parent.get("status") if parent else "NOT_REGISTERED",
            "attempt": parent.get("attempt") if parent else None,
            "task_packet": parent.get("task_packet") if parent else None,
        },
        "lanes": lanes,
        "global_authority": {
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
        },
        "claim_limit": (
            "Generated navigation only. Task ledger and immutable lane results remain authoritative; "
            "one lane cannot promote or block another lane's authority."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = build_status(
        task_state=load_json(TASK_STATE_PATH),
        generated_at=now_iso(),
        source_task_state_path=TASK_STATE_PATH,
    )
    atomic_json(args.output.resolve(), value)
    print(json.dumps({"status": "PASSED", "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

