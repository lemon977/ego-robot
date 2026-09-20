#!/usr/bin/env python3
"""Record one CPU lane result without granting cross-lane or global authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


TASK_ID = "three_stream_stable_baseline_v31"
ATTEMPT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
LANES = ("exact78", "ai1", "ai2")


def _derived_status(result_status: str) -> tuple[str, str | None, str]:
    if result_status.startswith(("PASS", "PASSED")):
        return "CPU_PREFLIGHT_COMPLETE", None, "AWAIT_NEXT_FROZEN_STAGE"
    if result_status.startswith(("BLOCKED", "DIAGNOSTIC_")):
        return "BLOCKED_CPU_PREFLIGHT", result_status, "RESOLVE_RECORDED_DIRECT_BLOCKER"
    if result_status.startswith(("REJECT", "FAILED", "FAIL_")):
        return "REJECTED_CPU_PREFLIGHT", result_status, "REVIEW_FAILURE_NO_AUTOMATIC_RETRY"
    raise RuntimeError(f"unsupported machine result status: {result_status}")


def record(*, lane: str, result_path: Path) -> dict[str, Any]:
    if lane not in LANES:
        raise RuntimeError(f"unknown lane: {lane}")
    lane_root = (ATTEMPT_ROOT / "lanes" / lane).resolve(strict=True)
    result = result_path.resolve(strict=True)
    try:
        result.relative_to(lane_root)
    except ValueError as error:
        raise RuntimeError("lane result must be inside its assigned writer root") from error
    payload = load_json(result)
    machine_status = payload.get("status")
    if not isinstance(machine_status, str):
        raise RuntimeError("lane result has no string status")
    status, blocker, next_step = _derived_status(machine_status)
    state_path = lane_root / "STATE.json"
    previous = load_json(state_path)
    if previous.get("lane") != lane or previous.get("parent_task_id") != TASK_ID:
        raise RuntimeError("lane state identity mismatch")
    if previous.get("status") not in {
        "READY_CPU_PREFLIGHT",
        "CPU_PREFLIGHT_COMPLETE",
        "BLOCKED_CPU_PREFLIGHT",
        "REJECTED_CPU_PREFLIGHT",
    }:
        raise RuntimeError(f"lane state is not recordable: {previous.get('status')}")
    history_root = lane_root / "state_history"
    history_root.mkdir(parents=True, exist_ok=True)
    sequence = len(list(history_root.glob("STATE_*.json"))) + 1
    history_path = history_root / f"STATE_{sequence:04d}.json"
    if history_path.exists() or history_path.is_symlink():
        raise RuntimeError(f"lane state history collision: {history_path}")
    shutil.copyfile(state_path, history_path)
    created = now_iso()
    current = {
        **previous,
        "status": status,
        "current_action": "CPU_PREFLIGHT_RECORDED",
        "latest_artifacts": [artifact_ref(result)],
        "blocker": blocker,
        "next_step": next_step,
        "updated_at": created,
        "claims": {
            **previous.get("claims", {}),
            "PIPELINE_COMPLETE": False,
            "NUMERIC_QUALITY_PASS": False,
            "VISUAL_REVIEW_STATUS": "NOT_REVIEWED",
            "TRAINING_COMPLETE": False,
            "TRAINING_ELIGIBLE": False,
            "CONTROL_GROUND_TRUTH": False,
            "PHYSICAL_DEPLOYABLE": False,
        },
        "machine_result_status": machine_status,
        "previous_state": artifact_ref(history_path),
    }
    atomic_json(state_path, current)
    return {
        "status": "PASSED",
        "lane": lane,
        "lane_status": status,
        "machine_result_status": machine_status,
        "state": artifact_ref(state_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(record(lane=args.lane, result_path=args.result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
