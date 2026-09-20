#!/usr/bin/env python3
"""Record one immutable V3.2 lane result into its isolated state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


TASK_ID = "four_stream_pretraining_baseline_v32"
ATTEMPT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
LANES = ("exact78", "ai1", "ai2", "ai4_huro")


def _classify(status: str) -> tuple[str, str | None, str]:
    if status.startswith(("PASS", "PASSED", "COMPLETE")):
        return "MILESTONE_COMPLETE", None, "RUN_NEXT_FROZEN_STAGE"
    if status.startswith(("BLOCKED", "PAUSED", "WAIT_", "NO_ADMISSIBLE")):
        return "BLOCKED_OR_PAUSED", status, "RESOLVE_DIRECT_BLOCKER_OR_RESUME"
    if status.startswith(("REJECT", "FAIL", "FAILED")):
        return "REJECTED_QUALITY_OR_RUNTIME", status, "NO_AUTOMATIC_SAME_SIGNATURE_RETRY"
    raise RuntimeError(f"unsupported machine status: {status}")


def record(lane: str, result_path: Path) -> dict[str, Any]:
    if lane not in LANES:
        raise RuntimeError(f"unknown lane: {lane}")
    lane_root = (ATTEMPT_ROOT / "lanes" / lane).resolve(strict=True)
    result = result_path.resolve(strict=True)
    try:
        result.relative_to(lane_root)
    except ValueError as exc:
        raise RuntimeError("result must be under its assigned lane writer root") from exc
    payload = load_json(result)
    machine_status = payload.get("status")
    if not isinstance(machine_status, str):
        raise RuntimeError("result lacks string status")
    status, blocker, next_step = _classify(machine_status)
    state_path = lane_root / "STATE.json"
    previous = load_json(state_path)
    if previous.get("lane") != lane or previous.get("parent_task_id") != TASK_ID:
        raise RuntimeError("lane identity mismatch")
    history_root = lane_root / "state_history"
    history_root.mkdir(parents=True, exist_ok=True)
    sequence = len(list(history_root.glob("STATE_*.json"))) + 1
    history_path = history_root / f"STATE_{sequence:04d}.json"
    shutil.copyfile(state_path, history_path)
    current = {
        **previous,
        "status": status,
        "current_action": "MACHINE_RESULT_RECORDED",
        "latest_artifacts": [artifact_ref(result)],
        "blocker": blocker,
        "next_step": next_step,
        "machine_result_status": machine_status,
        "previous_state": artifact_ref(history_path),
        "updated_at": now_iso(),
    }
    atomic_json(state_path, current)
    return {"status": "PASSED", "lane": lane, "lane_status": status, "state": artifact_ref(state_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(record(args.lane, args.result), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

