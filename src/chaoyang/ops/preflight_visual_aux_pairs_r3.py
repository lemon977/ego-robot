#!/usr/bin/env python3
"""Fail-closed CPU preflight for the two R3 Visual Aux task pairs.

This tool deliberately does not build RGB bundles or start training.  It binds
the evidence available to the current task packet and closes each task pair as
either READY_FOR_EPOCH0 or BLOCKED_PREREQ.  Projection-only eligibility is
never promoted to causal Robotized RGB, Occlusion Silver, or checkpoint proof.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REQUIREMENTS = {
    "train": {"sessions": 16, "windows": 256},
    "validation": {"sessions": 3, "windows": 48},
}
OUTPUT_NAMES = (
    "RESULT.json",
    "METRICS.json",
    "RUN_RECEIPT.json",
    "DECISION.md",
    "NEXT_ACTION.json",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_text(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stage_contract(algorithm: dict[str, Any]) -> dict[str, Any]:
    for item in algorithm.get("stages", []):
        if item.get("stage") == "HumanEgo Aux":
            return item
    raise RuntimeError("HumanEgo Aux stage missing from ALGORITHM_CONTRACT")


def observed(eligibility: dict[str, Any], task: str) -> dict[str, dict[str, int]]:
    summaries = eligibility.get("summaries", {}).get(task, {})
    result: dict[str, dict[str, int]] = {}
    for split in REQUIREMENTS:
        item = summaries.get(split, {})
        result[split] = {
            "sessions": int(item.get("ready_sessions", 0)),
            "windows": int(item.get("eligible_h50_windows", 0)),
        }
    return result


def build_pair(
    *,
    task: str,
    output: Path,
    status_receipt: Path,
    task_packet: Path,
    algorithm_contract: Path,
    eligibility_index: Path,
    prior_result: Path,
    robotized_ledger: Path,
    capacity_preflight: Path | None,
) -> None:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    receipt_data = load(status_receipt)
    packet_data = load(task_packet)
    algorithm_data = load(algorithm_contract)
    eligibility_data = load(eligibility_index)
    prior_data = load(prior_result)
    capacity_data = load(capacity_preflight) if capacity_preflight is not None else None
    humanego_contract = stage_contract(algorithm_data)

    if packet_data.get("task_id") != f"visual_aux_{task}_pair_v1":
        raise RuntimeError(f"wrong task packet for {task}")
    if eligibility_data.get("status") != "PASS_ELIGIBILITY_INDEX_BUILT":
        raise RuntimeError("eligibility index is not a completed projection preflight")

    seen = observed(eligibility_data, task)
    blockers: list[dict[str, Any]] = []
    for split, required in REQUIREMENTS.items():
        if seen[split]["sessions"] < required["sessions"]:
            blockers.append(
                {
                    "gate": f"{split}_session_minimum",
                    "observed": seen[split]["sessions"],
                    "required": required["sessions"],
                }
            )
        if seen[split]["windows"] < required["windows"]:
            blockers.append(
                {
                    "gate": f"{split}_H50_window_minimum",
                    "observed": seen[split]["windows"],
                    "required": required["windows"],
                }
            )

    ledger_exists = robotized_ledger.is_file()
    if not ledger_exists:
        blockers.append(
            {
                "gate": "formal_causal_robotized_rgb_ledger",
                "observed": "ABSENT",
                "required": str(robotized_ledger.resolve()),
            }
        )

    prior_pair = prior_data.get("pair_status", {}).get(task)
    if prior_pair not in {"PASS", "PASSED"}:
        blockers.append(
            {
                "gate": "frozen_paired_dataset_ledger",
                "observed": prior_pair or "ABSENT",
                "required": "PASS with exact Raw/Robotized frame, split, label and valid-mask closure",
            }
        )

    # The R3 contract requires formal Silver-bound causal inputs and a frozen
    # hardset before the paired Value Gate.  In this bounded preflight those
    # must be evidenced by the declared causal ledger; absence is fail-closed.
    if not ledger_exists:
        blockers.extend(
            [
                {
                    "gate": "formal_occlusion_silver_binding",
                    "observed": "NOT_EVIDENCED",
                    "required": "per-session formal Silver receipt bound by causal Robotized ledger",
                },
                {
                    "gate": "causal_compositor_binding",
                    "observed": "NOT_EVIDENCED",
                    "required": "CAUSAL_TRAINING_INPUT with donor_frame_id <= target_frame_id",
                },
                {
                    "gate": "frozen_occlusion_hardset_for_value_gate",
                    "observed": "NOT_EVIDENCED",
                    "required": "immutable task-specific hardset before checkpoint comparison",
                },
            ]
        )

    status = "READY_FOR_EPOCH0" if not blockers else "BLOCKED_PREREQ"
    inputs = {
        "current_status_receipt": ref(status_receipt),
        "task_packet": ref(task_packet),
        "algorithm_contract": ref(algorithm_contract),
        "projection_eligibility_index": ref(eligibility_index),
        "prior_pair_result": ref(prior_result),
    }
    if capacity_preflight is not None:
        inputs["capacity_preflight"] = ref(capacity_preflight)
    if ledger_exists:
        inputs["causal_robotized_rgb_ledger"] = ref(robotized_ledger)
    else:
        inputs["causal_robotized_rgb_ledger"] = {
            "path": str(robotized_ledger.resolve()),
            "status": "ABSENT",
        }

    capacity = None
    if capacity_data is not None:
        capacity_task = capacity_data.get("tasks", {}).get(task, {})
        capacity = {
            "status": capacity_task.get("capacity_status"),
            "train_potential_sessions": capacity_task.get("train", {}).get("unique_potential_sessions", 0),
            "validation_potential_sessions": capacity_task.get("validation", {}).get("unique_potential_sessions", 0),
            "actual_training_ready": False,
            "claim_limit": "Potential routing is not completed causal paired data.",
        }
    metrics = {
        "schema_version": "visual-aux-pair-preflight-r3-metrics-v1",
        "task": task,
        "pair_id": f"CHECKPOINT_PAIR_{task.upper()}",
        "status": status,
        "observed": seen,
        "required": REQUIREMENTS,
        "projection_only_ready_sessions": sum(item["sessions"] for item in seen.values()),
        "projection_only_ready_windows": sum(item["windows"] for item in seen.values()),
        "formal_causal_robotized_ledger_present": ledger_exists,
        "capacity_forecast": capacity,
        "published_checkpoints": 0,
        "published_loss_curves": 0,
        "control_ground_truth": False,
        "claim_limit": "Projection-only counts are not causal paired training eligibility or checkpoint authority.",
    }
    write_json(output / "METRICS.json", metrics)

    capacity_sentence = ""
    if capacity is not None:
        capacity_sentence = (
            f"旧路由容量预测显示 train={capacity['train_potential_sessions']}、"
            f"validation={capacity['validation_potential_sessions']} 个潜在 session，"
            "但这些 pending 路由不能计作已完成训练数据。\n\n"
        )
    decision = (
        f"# {task.capitalize()} Visual Aux R3 前置检查\n\n"
        f"终态：`{status}`。\n\n"
        "当前只存在投影资格证据，不存在可核验的正式因果 Robotized RGB ledger，"
        "也未闭合 Silver、causal compositor、冻结困难集、成对训练账本和 epoch-0。"
        "因此不得启动本任务的 Raw/Robotized checkpoint，也不得生成 loss 曲线占位文件。\n\n"
        f"训练/验证观测：train={seen['train']['sessions']} sessions/"
        f"{seen['train']['windows']} windows；validation={seen['validation']['sessions']} sessions/"
        f"{seen['validation']['windows']} windows。\n\n"
        f"{capacity_sentence}"
        "恢复条件：先发布逐会话 formal Silver + causal compositor 收据，构建严格配对 ledger，"
        "达到会话/窗口下限并冻结困难集；随后分别运行两支 epoch-0。\n"
    )
    write_text(output / "DECISION.md", decision)

    next_action = {
        "schema_version": "visual-aux-pair-preflight-r3-next-action-v1",
        "task": task,
        "status": status,
        "next_task": "BUILD_FORMAL_CAUSAL_ROBOTIZED_LEDGER",
        "prerequisites": [item["gate"] for item in blockers],
        "do_not_start_training": status != "READY_FOR_EPOCH0",
        "claim_limit": "Routing instruction only; it does not create a training artifact.",
    }
    write_json(output / "NEXT_ACTION.json", next_action)

    run_receipt = {
        "schema_version": "visual-aux-pair-preflight-r3-run-receipt-v1",
        "created_at": created_at,
        "execution_mode": "CPU_READ_ONLY_PREFLIGHT",
        "task": task,
        "status": "PASSED_PREFLIGHT_EXECUTION",
        "terminal_decision": status,
        "governance_revision_observed": receipt_data.get("governance_revision"),
        "inputs": inputs,
        "algorithm_id": humanego_contract.get("algorithm_id"),
        "weights": humanego_contract.get("weights"),
        "mutated_current_governance": False,
        "started_training": False,
        "claim_limit": "Successful preflight execution is not checkpoint eligibility.",
    }
    write_json(output / "RUN_RECEIPT.json", run_receipt)

    result = {
        "schema_version": "visual-aux-pair-preflight-r3-result-v1",
        "created_at": created_at,
        "task_id": packet_data["task_id"],
        "pair_id": f"CHECKPOINT_PAIR_{task.upper()}",
        "execution_status": "PASSED_PREFLIGHT_EXECUTION",
        "terminal_status": status,
        "blockers": blockers,
        "inputs": inputs,
        "metrics": ref(output / "METRICS.json"),
        "decision": ref(output / "DECISION.md"),
        "next_action": ref(output / "NEXT_ACTION.json"),
        "run_receipt": ref(output / "RUN_RECEIPT.json"),
        "checkpoint_authority": False,
        "checkpoint_count": 0,
        "loss_curve_count": 0,
        "control_ground_truth": False,
        "claim_limit": "R3 prerequisite audit only; no Silver, causal compositor, gold hardset, checkpoint, loss, policy or Robot action authority was created.",
    }
    write_json(output / "RESULT.json", result)

    manifest = {
        "schema_version": "visual-aux-pair-preflight-r3-artifact-manifest-v1",
        "created_at": created_at,
        "task": task,
        "status": status,
        "artifacts": {name: ref(output / name) for name in OUTPUT_NAMES},
        "claim_limit": "Immutable preflight evidence only.",
    }
    write_json(output / "ARTIFACT_MANIFEST.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-receipt", type=Path, required=True)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--algorithm-contract", type=Path, required=True)
    parser.add_argument("--eligibility-index", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--robotized-ledger", type=Path, required=True)
    parser.add_argument("--capacity-preflight", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"immutable output directory already exists: {args.output}")
    build_pair(
        task=args.task,
        output=args.output.resolve(),
        status_receipt=args.status_receipt.resolve(strict=True),
        task_packet=args.task_packet.resolve(strict=True),
        algorithm_contract=args.algorithm_contract.resolve(strict=True),
        eligibility_index=args.eligibility_index.resolve(strict=True),
        prior_result=args.prior_result.resolve(strict=True),
        robotized_ledger=args.robotized_ledger.resolve(),
        capacity_preflight=args.capacity_preflight.resolve(strict=True) if args.capacity_preflight else None,
    )
    print(json.dumps({"status": "PASS_PREFLIGHT_EXECUTION", "task": args.task, "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
