#!/usr/bin/env python3
"""Fail-closed RC1 T4 paired-bundle and smoke preflight.

T4 must never turn an offline Robot review, an old Clean result, or a prefix
causality canary into a trainable pair.  This tool consumes only immutable T2,
T3 and T0 receipts and either authorizes the real bundle builder to run later
or publishes an explicit prerequisite terminal without creating RGB data.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.common import artifact_ref, atomic_json, atomic_write, now_iso


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def evaluate(t2: dict[str, Any], t3: dict[str, Any], capacity: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    if t2.get("status") != "PASSED":
        blockers.append({"code": "CAUSAL_CLEAN_TASK_NOT_PASSED", "observed": t2.get("status")})
    if t2.get("fresh_clean_generated") is not True:
        blockers.append({"code": "NO_FRESH_CAUSAL_CLEAN_RGB"})
    if t2.get("input_mode") != "CAUSAL_TRAINING_INPUT":
        blockers.append({"code": "CAUSAL_CLEAN_INPUT_MODE_INVALID", "observed": t2.get("input_mode")})

    if t3.get("status") != "PASSED":
        blockers.append({"code": "CAUSAL_ROBOT_CANARY_NOT_PASSED", "observed": t3.get("status")})
    production = t3.get("production_entry")
    if not isinstance(production, dict) or production.get("full_session_training_eligibility") is not True:
        blockers.append({"code": "CAUSAL_ROBOT_PRODUCTION_NOT_ELIGIBLE"})

    tasks = capacity.get("tasks")
    if not isinstance(tasks, dict):
        blockers.append({"code": "CAPACITY_REPORT_SCHEMA_INVALID"})
    else:
        for task in ("chips", "poker"):
            row = tasks.get(task)
            if not isinstance(row, dict) or row.get("capacity_ready_under_rc1") is not True:
                blockers.append({
                    "code": "PAIR_SOURCE_GROUP_CAPACITY_NOT_MET",
                    "task": task,
                    "observed_terminal": row.get("pair_terminal") if isinstance(row, dict) else None,
                })

    return ("BLOCKED_PREREQ" if blockers else "PASSED_PREFLIGHT"), blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2-result", required=True, type=Path)
    parser.add_argument("--t3-result", required=True, type=Path)
    parser.add_argument("--capacity-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)

    t2 = _load(args.t2_result)
    t3 = _load(args.t3_result)
    capacity = _load(args.capacity_report)
    status, blockers = evaluate(t2, t3, capacity)
    created = now_iso()
    result = {
        "schema_version": "chaoyang-rc1-t4-bundle-smoke-preflight-v1",
        "task_id": "rc1_t4_bundle_smoke",
        "created_at": created,
        "status": status,
        "blockers": blockers,
        "paired_bundle_generated": False,
        "smoke_training_started": False,
        "checkpoint_written": False,
        "input_mode_required": "CAUSAL_TRAINING_INPUT",
        "offline_bidirectional_artifacts_accepted": False,
        "review_mp4_accepted": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "authority_promoted": False,
        "lineage": {
            "t2_result": artifact_ref(args.t2_result),
            "t3_result": artifact_ref(args.t3_result),
            "capacity_report": artifact_ref(args.capacity_report),
        },
        "claim_limit": "RC1 T4 prerequisite closure only; no paired RGB, training smoke, checkpoint, control truth or deployment authority was produced.",
    }
    result_path = args.output_root / "RESULT.json"
    atomic_json(result_path, result)
    atomic_json(args.output_root / "METRICS.json", {
        "status": status,
        "blocker_count": len(blockers),
        "blocker_codes": [row["code"] for row in blockers],
        "paired_bundle_count": 0,
        "smoke_updates": 0,
    })
    atomic_json(args.output_root / "NEXT_ACTION.json", {
        "status": status,
        "next_task_id": "rc1_t5_batch_conversion",
        "reason": "Negative T4 terminal still closes the DAG dependency; T5 must report zero paired eligibility and the exact blockers.",
    })
    atomic_json(args.output_root / "RUN_RECEIPT.json", {
        "status": status,
        "created_at": created,
        "result": artifact_ref(result_path),
        "gpu_used": False,
        "authority_promoted": False,
    })
    atomic_json(args.output_root / "ARTIFACT_MANIFEST.json", {
        "status": status,
        "result": artifact_ref(result_path),
        "metrics": artifact_ref(args.output_root / "METRICS.json"),
    })
    atomic_json(args.output_root / "RESULT_SUMMARY.json", {
        "task_id": "rc1_t4_bundle_smoke",
        "status": status,
        "blocker_count": len(blockers),
        "paired_bundle_generated": False,
        "smoke_training_started": False,
        "next_action": "PUBLISH_NEGATIVE_TERMINAL_THEN_BUILD_T5_CONVERSION_REPORT",
    })
    atomic_write(
        args.output_root / "DECISION.md",
        (
            "# RC1 T4 paired bundle / smoke\n\n"
            "前置条件未闭合，任务按 fail-closed 规则终结。没有生成 paired RGB，"
            "没有启动 200 步训练，也没有写 checkpoint。离线双向 Robot 视频和旧 Clean "
            "不能替代因果训练输入。负面终态允许 T5 继续发布真实转化率与零配对窗口。\n"
        ).encode("utf-8"),
    )
    print(json.dumps({"status": status, "result": artifact_ref(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
