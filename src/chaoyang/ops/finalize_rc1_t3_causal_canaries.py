#!/usr/bin/env python3
"""Seal Chips/Poker prefix-invariance canaries into one RC1 T3 receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha(value)}


def write(path: Path, payload: Any) -> None:
    if isinstance(payload, str):
        data = payload
    else:
        data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chips-result", type=Path, required=True)
    parser.add_argument("--poker-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    cases = {}
    for task, path in (("chips", args.chips_result), ("poker", args.poker_result)):
        data = json.loads(path.resolve(strict=True).read_text())
        passed = data.get("status") == "PASSED_CAUSAL_CANARY_NO_AUTHORITY" and all(data.get("gates", {}).values())
        cases[task] = {"pass": passed, "session": data.get("session"), "prefix_end": data.get("prefix_end"), "result": ref(path)}
    passed = all(row["pass"] for row in cases.values())
    status = "PASSED" if passed else "FAILED_QUALITY_C"
    result = {
        "schema_version": "chaoyang-rc1-t3-causal-robot-canary-index-v1",
        "task_id": "rc1_t3_v77_causal_robot", "created_at": now(), "status": status,
        "artifact_revision": "RC1_T3_PREFIX_CAUSAL_1", "cases": cases,
        "production_entry": {
            "mode": "PREFIX_RECOMPUTE_EMIT_TERMINAL_STATE_ONLY",
            "verified_tasks": [task for task, row in cases.items() if row["pass"]],
            "full_session_training_eligibility": False,
            "reason": "Scheduled-start producer and per-window hard gates remain to be executed.",
        },
        "forbidden_inputs": ["ARM_BIDIRECTIONAL_STATES.npz", "HAND_BIDIRECTIONAL_STATES.npz", "REVERSE_LOOKAHEAD", "full-trajectory placement"],
        "control_ground_truth": False, "physical_deployment_authorized": False,
        "authority_promoted": False,
        "claim_limit": "Two-task prefix-invariance development proof only; not full-session Robot eligibility, action truth, contact truth, physical accuracy or deployment authority.",
    }
    result_path = args.output_root / "RESULT.json"
    write(result_path, result)
    metrics = {"status": status, "passed_task_count": sum(row["pass"] for row in cases.values()), "expected_task_count": 2, "cases": cases}
    write(args.output_root / "METRICS.json", metrics)
    write(args.output_root / "DECISION.md", f"# RC1 T3 前缀因果决定\n\n状态：`{status}`。Chips/Poker 仅证明 prefix 重算的末帧不读取未来输入；批量 scheduled-start 生产与逐帧硬门尚未执行。\n")
    write(args.output_root / "NEXT_ACTION.json", {"status": status, "next_task_id": "rc1_t3_scheduled_start_producer" if passed else None})
    write(args.output_root / "RUN_RECEIPT.json", {"status": status, "created_at": now(), "result": ref(result_path), "authority_promoted": False})
    write(args.output_root / "ARTIFACT_MANIFEST.json", {"status": status, "result": ref(result_path), "metrics": ref(args.output_root / "METRICS.json")})
    write(args.output_root / "RESULT_SUMMARY.json", {"task_id": result["task_id"], "status": status, "passed_tasks": [k for k,v in cases.items() if v["pass"]], "next_action": "BUILD_SCHEDULED_START_PRODUCER" if passed else "STOP"})
    print(json.dumps({"status": status, "result": ref(result_path)}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
