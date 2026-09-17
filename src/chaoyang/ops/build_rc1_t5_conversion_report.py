#!/usr/bin/env python3
"""Build the RC1 conversion report without upgrading development evidence.

The report combines the frozen exact156 first-blocker partition with RC1
capacity, bounded Mask/Clean/Robot prerequisites, T4 paired-bundle closure and
the current Robot30 progress index.  Pending rows or budget terminals keep
RATE_FINALIZED false; offline hard-geometry evidence never counts as paired
training data.
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


def summarize(
    exact: dict[str, Any],
    capacity: dict[str, Any],
    t1: dict[str, Any],
    t2: dict[str, Any],
    t3: dict[str, Any],
    t4: dict[str, Any],
    robot30: dict[str, Any],
) -> dict[str, Any]:
    exact_counts = exact.get("counts", {})
    if exact_counts.get("sessions") != 156:
        raise RuntimeError("exact conversion ledger must partition 156 sessions")
    first = exact_counts.get("first_blocker", {})
    if sum(int(value) for value in first.values()) != 156:
        raise RuntimeError("exact first-blocker counts do not sum to 156")

    cap_tasks = capacity.get("tasks")
    if not isinstance(cap_tasks, dict) or set(cap_tasks) != {"chips", "poker"}:
        raise RuntimeError("capacity report must contain exactly chips and poker")
    robot_counts = robot30.get("counts")
    if not isinstance(robot_counts, dict) or set(robot_counts) != {"chips", "poker"}:
        raise RuntimeError("Robot30 index must contain exactly chips and poker counts")

    tasks: dict[str, Any] = {}
    any_pending = False
    any_not_evaluated_budget = False
    for task in ("chips", "poker"):
        r = robot_counts[task]
        selected = int(r.get("selected", -1))
        if selected != 30:
            raise RuntimeError(f"Robot30 selected denominator must be 30 for {task}")
        pending_successor = int(r.get("pending_successor", 0))
        pending_causal = int(r.get("pending_causal_production", 0))
        not_evaluated_budget = int(r.get("not_evaluated_budget", 0))
        any_pending = any_pending or pending_successor > 0 or pending_causal > 0
        any_not_evaluated_budget = any_not_evaluated_budget or not_evaluated_budget > 0
        cap = cap_tasks[task]
        tasks[task] = {
            "independent_source_groups_total": int(cap.get("independent_source_groups_total", 0)),
            "train_source_groups": int(cap.get("train_source_groups", 0)),
            "validation_source_groups": int(cap.get("validation_source_groups", 0)),
            "scheduled_train_windows_upper_bound": int(cap.get("scheduled_train_windows_upper_bound", 0)),
            "scheduled_validation_windows_upper_bound": int(cap.get("scheduled_validation_windows_upper_bound", 0)),
            "data_minimum_met": bool(cap.get("capacity_ready_under_rc1")),
            "checkpoint_pair_status": str(cap.get("pair_terminal")),
            "robot30_selected": selected,
            "robot30_offline_hard_geometry_evidence": int(r.get("hard_geometry_pass_evidence", 0)),
            "robot30_failed_quality_c": int(r.get("failed_quality_c", 0)),
            "robot30_failed_runtime_final": int(r.get("failed_runtime_final", 0)),
            "robot30_pending_successor": pending_successor,
            "robot30_pending_causal_production": pending_causal,
            "robot30_not_evaluated_budget": not_evaluated_budget,
            "paired_eligible_windows": 0,
            "checkpoints_trained": 0,
        }

    t4_pair = bool(t4.get("paired_bundle_generated"))
    t4_smoke = bool(t4.get("smoke_training_started"))
    # A budget terminal closes the task operationally, but it does not make the
    # measured conversion denominator complete.  RC1 explicitly requires
    # RATE_FINALIZED=false whenever any selected row was not evaluated.
    rate_finalized = (
        not any_pending
        and not any_not_evaluated_budget
        and t4.get("status") in {"PASSED", "PASSED_PREFLIGHT"}
    )
    operationally_closed = (
        rate_finalized
        and t4_pair
        and t4_smoke
        and all(tasks[task]["data_minimum_met"] for task in tasks)
    )
    release_status = "COMPLETE" if operationally_closed else "INCOMPLETE"
    return {
        "schema_version": "chaoyang-rc1-conversion-summary-v1",
        "created_at": now_iso(),
        "status": "PASSED_TRUTHFUL_SNAPSHOT",
        "exact156_first_blocker": first,
        "conditional_conversion_rates": exact.get("conditional_conversion_rates", {}),
        "stage_terminals": {
            "T1_SAM31_MASK": t1.get("status"),
            "T2_CAUSAL_CLEAN": t2.get("status"),
            "T3_CAUSAL_ROBOT_CANARY": t3.get("status"),
            "T4_PAIRED_BUNDLE_SMOKE": t4.get("status"),
        },
        "tasks": tasks,
        "release_flags": {
            "PIPELINE_OPERATIONALLY_CLOSED": operationally_closed,
            "DATA_MINIMUM_MET_CHIPS": tasks["chips"]["data_minimum_met"],
            "DATA_MINIMUM_MET_POKER": tasks["poker"]["data_minimum_met"],
            "CHECKPOINT_PAIR_CHIPS": tasks["chips"]["checkpoint_pair_status"],
            "CHECKPOINT_PAIR_POKER": tasks["poker"]["checkpoint_pair_status"],
            "FOUR_CHECKPOINTS_TRAINED": False,
            "RATE_FINALIZED": rate_finalized,
            "RC1_RELEASE_STATUS": release_status,
        },
        "boundaries": {
            "offline_robot_hard_geometry_counts_as_training_input": False,
            "paired_bundle_generated": t4_pair,
            "smoke_training_started": t4_smoke,
            "control_ground_truth": False,
            "physical_deployment_authorized": False,
        },
        "claim_limit": "RC1 conversion and capacity accounting only; offline Robot hard geometry is not causal training data, and zero paired windows means no checkpoint claim.",
    }


def _markdown(summary: dict[str, Any]) -> str:
    first = summary["exact156_first_blocker"]
    flags = summary["release_flags"]
    lines = [
        "# RC1 转化率报告",
        "",
        "本报告由 immutable receipt 自动汇总。离线 Robot 硬几何证据不计为因果训练输入。",
        "",
        "## exact156 首个阻塞",
        "",
        "| 首个阻塞 | 数量 |",
        "|---|---:|",
    ]
    for key, value in first.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## RC1 任务状态", "", "| 任务 | T1 Mask | T2 Clean | T3 Robot canary | T4 bundle/smoke |", "|---|---|---|---|---|"])
    st = summary["stage_terminals"]
    lines.append(f"| exact78 | `{st['T1_SAM31_MASK']}` | `{st['T2_CAUSAL_CLEAN']}` | `{st['T3_CAUSAL_ROBOT_CANARY']}` | `{st['T4_PAIRED_BUNDLE_SMOKE']}` |")
    lines.extend(["", "## 两任务容量与 Robot30", "", "| 任务 | 独立来源组 | 验证组 | Robot30离线硬几何 | 待successor | 待因果生产 | 预算未评估 | paired窗口 | checkpoint pair |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for task in ("chips", "poker"):
        row = summary["tasks"][task]
        lines.append(
            f"| {task} | {row['independent_source_groups_total']} | {row['validation_source_groups']} | "
            f"{row['robot30_offline_hard_geometry_evidence']}/30 | {row['robot30_pending_successor']} | "
            f"{row['robot30_pending_causal_production']} | {row['robot30_not_evaluated_budget']} | "
            f"{row['paired_eligible_windows']} | `{row['checkpoint_pair_status']}` |"
        )
    lines.extend([
        "",
        "## 发布结论",
        "",
        f"- `RATE_FINALIZED={str(flags['RATE_FINALIZED']).lower()}`",
        f"- `PIPELINE_OPERATIONALLY_CLOSED={str(flags['PIPELINE_OPERATIONALLY_CLOSED']).lower()}`",
        f"- `RC1_RELEASE_STATUS={flags['RC1_RELEASE_STATUS']}`",
        "- `control_ground_truth=false`",
        "- `physical_deployment_authorized=false`",
        "",
        "未生成合法因果 paired bundle 或独立来源组不足时，不启动四支 checkpoint。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-ledger", required=True, type=Path)
    parser.add_argument("--capacity-report", required=True, type=Path)
    parser.add_argument("--t1-result", required=True, type=Path)
    parser.add_argument("--t2-result", required=True, type=Path)
    parser.add_argument("--t3-result", required=True, type=Path)
    parser.add_argument("--t4-result", required=True, type=Path)
    parser.add_argument("--robot30-index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    sources = {
        "exact_ledger": args.exact_ledger,
        "capacity_report": args.capacity_report,
        "t1_result": args.t1_result,
        "t2_result": args.t2_result,
        "t3_result": args.t3_result,
        "t4_result": args.t4_result,
        "robot30_index": args.robot30_index,
    }
    values = {key: _load(path) for key, path in sources.items()}
    summary = summarize(
        values["exact_ledger"], values["capacity_report"], values["t1_result"],
        values["t2_result"], values["t3_result"], values["t4_result"], values["robot30_index"],
    )
    summary["lineage"] = {key: artifact_ref(path) for key, path in sources.items()}
    summary_path = args.output_root / "QUALITY_SUMMARY.json"
    atomic_json(summary_path, summary)
    atomic_write(args.output_root / "CONVERSION_REPORT.md", _markdown(summary).encode("utf-8"))
    result_path = args.output_root / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "chaoyang-rc1-t5-conversion-result-v1",
        "task_id": "rc1_t5_batch_conversion",
        "created_at": now_iso(),
        "status": "PASSED",
        "rate_finalized": summary["release_flags"]["RATE_FINALIZED"],
        "release_status": summary["release_flags"]["RC1_RELEASE_STATUS"],
        "quality_summary": artifact_ref(summary_path),
        "conversion_report": artifact_ref(args.output_root / "CONVERSION_REPORT.md"),
        "authority_promoted": False,
        "claim_limit": summary["claim_limit"],
    })
    print(json.dumps({"status": "PASSED", "result": artifact_ref(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
