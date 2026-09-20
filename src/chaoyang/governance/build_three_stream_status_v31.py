#!/usr/bin/env python3
"""Generate the shallow three-stream status from machine-owned facts.

This module deliberately does not accept free-form status text.  The current
task ledger is authoritative; optional lane STATE files may only add immutable
artifact references and a narrower lane blocker.
"""

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


DEFAULT_PARENT_TASK_ID = "three_stream_stable_baseline_v31"
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "_run/current/three_stream_stable_baseline_v31/attempts/attempt_0001"
)
DEFAULT_OUTPUT = REPO_ROOT / "docs/current/STATUS.json"
LANES = ("exact78", "ai1", "ai2")


def _load_optional(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    value = load_json(path)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"lane state must be an object: {path}")
    return value


def build_status(
    *,
    task_state: Mapping[str, Any],
    parent_task_id: str,
    run_root: Path,
    generated_at: str,
    source_task_state_path: Path | None = None,
) -> dict[str, Any]:
    rows = [
        row
        for row in task_state.get("tasks", [])
        if isinstance(row, Mapping) and row.get("task_id") == parent_task_id
    ]
    if len(rows) > 1:
        raise RuntimeError(f"duplicate parent task rows: {parent_task_id}")
    parent = rows[0] if rows else None
    next_task = task_state.get("next_task")
    parent_routable = bool(
        isinstance(next_task, Mapping)
        and next_task.get("task_id") == parent_task_id
    )

    lanes: dict[str, Any] = {}
    for lane in LANES:
        state_path = run_root / "lanes" / lane / "STATE.json"
        lane_state = _load_optional(state_path)
        if lane_state is None:
            lanes[lane] = {
                "status": "REGISTERED_NOT_STARTED" if parent else "NOT_REGISTERED",
                "current_action": None,
                "stable_baseline": None,
                "latest_artifacts": [],
                "blocker": None,
                "next_step": "WAIT_PARENT_REGISTRATION" if parent is None else "START_CPU_PREFLIGHT",
                "authority": "DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE",
                "state_file": None,
            }
            continue
        if lane_state.get("lane") != lane:
            raise RuntimeError(f"lane identity mismatch in {state_path}")
        latest = lane_state.get("latest_artifacts", [])
        if not isinstance(latest, list):
            raise RuntimeError(f"latest_artifacts must be a list: {state_path}")
        lanes[lane] = {
            "status": lane_state.get("status", "UNKNOWN"),
            "current_action": lane_state.get("current_action"),
            "stable_baseline": lane_state.get("stable_baseline"),
            "latest_artifacts": latest,
            "blocker": lane_state.get("blocker"),
            "next_step": lane_state.get("next_step"),
            "authority": lane_state.get(
                "authority", "DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE"
            ),
            "state_file": artifact_ref(state_path),
        }

    return {
        "schema_version": "chaoyang-three-stream-shallow-status-v1",
        "generated_at": generated_at,
        "source": {
            "task_state": artifact_ref(source_task_state_path or TASK_STATE_PATH),
            "parent_task_id": parent_task_id,
            "run_root": str(run_root),
        },
        "parent": {
            "registered": parent is not None,
            "routable": parent_routable,
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
            "Generated navigation projection only. Lane receipts and the current task ledger "
            "remain authoritative; this file must never be edited by a lane worker."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-task-id", default=DEFAULT_PARENT_TASK_ID)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    task_state = load_json(TASK_STATE_PATH)
    value = build_status(
        task_state=task_state,
        parent_task_id=args.parent_task_id,
        run_root=args.run_root.resolve(),
        generated_at=now_iso(),
        source_task_state_path=TASK_STATE_PATH,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, value)
    print(json.dumps({"status": "PASSED", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
