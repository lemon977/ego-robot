#!/usr/bin/env python3
"""Execute the ten R3 development task packets into immutable terminals.

This runner is deliberately narrow: it validates packet identity, read sets,
prerequisites and the packet's CPU dry-run command.  It never runs a model,
publishes authority, or edits current governance.  A prerequisite that needs a
real input manifest is fail-closed instead of being inferred from directory
names or historical chat.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any
import uuid


PROJECT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/TASK_PACKET_INDEX.json"
TERMINAL_STATUSES = {
    "PASSED_DEVELOPMENT",
    "BLOCKED_PREREQ",
    "FAILED_QUALITY_C",
    "FAILED_RUNTIME_FINAL",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def process_startticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[21])


def load_governance() -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        [sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        return {"status": "STATUS_CONFLICT", "errors": [combined.strip()]}, combined
    try:
        return json.loads(completed.stdout), combined
    except json.JSONDecodeError:
        return {"status": "STATUS_CONFLICT", "errors": ["governance validator returned non-JSON"]}, combined


def evaluate_prerequisites(
    prerequisites: list[str],
    governance: dict[str, Any],
    completed_statuses: dict[str, str],
    packet: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expression in prerequisites:
        passed = False
        reason = "UNRECOGNIZED_PREREQUISITE"
        if expression == "G0_CORE_GOVERNANCE=PASS":
            passed = governance.get("status") == "PASS"
            reason = "CURRENT_GOVERNANCE_PASS" if passed else "CURRENT_GOVERNANCE_NOT_PASS"
        elif expression == "governance_freshness=FRESH":
            passed = (governance.get("freshness") or {}).get("status") == "FRESH"
            reason = "CURRENT_GOVERNANCE_FRESH" if passed else "CURRENT_GOVERNANCE_NOT_FRESH"
        elif expression == "CONTROLLER_MANUS_INPUT_PRESENT":
            # Real inputs must be explicitly pinned by the packet.  This field
            # is intentionally absent in the initial development packets.
            manifest = packet.get("input_manifest")
            passed = isinstance(manifest, dict) and Path(str(manifest.get("path", ""))).is_file()
            reason = "PINNED_INPUT_MANIFEST_PRESENT" if passed else "PINNED_CONTROLLER_MANUS_INPUT_MANIFEST_MISSING"
        elif "=" in expression:
            task_id, expected = expression.rsplit("=", 1)
            observed = completed_statuses.get(task_id)
            if expected == "PASSED":
                passed = observed == "PASSED_DEVELOPMENT"
            elif expected == "TERMINAL":
                passed = observed in TERMINAL_STATUSES
            reason = f"UPSTREAM_{observed or 'NOT_TERMINAL'}"
        rows.append({"expression": expression, "passed": passed, "reason": reason})
    return rows


def execute_packet(
    packet_row: dict[str, Any],
    attempt_id: str,
    governance: dict[str, Any],
    governance_log: str,
    completed_statuses: dict[str, str],
    fencing_token: str,
) -> dict[str, Any]:
    packet_path = Path(packet_row["path"])
    if sha256(packet_path) != packet_row["sha256"]:
        raise RuntimeError(f"packet SHA mismatch: {packet_path}")
    packet = json.loads(packet_path.read_text())
    task_id = packet["task_id"]
    slug = packet_path.parent.name
    attempt = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_pipeline_contracts_r3/artifacts" / slug / "attempts" / attempt_id
    if attempt.exists():
        raise RuntimeError(f"refusing to overwrite immutable attempt: {attempt}")
    attempt.mkdir(parents=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()

    read_set = []
    missing_read_set = []
    for value in packet.get("read_set", []):
        path = Path(value)
        if path.is_file():
            read_set.append(artifact_ref(path))
        else:
            missing_read_set.append(str(path))

    prerequisites = evaluate_prerequisites(
        list(packet.get("prerequisites", [])), governance, completed_statuses, packet
    )
    write_json(attempt / "PREREQUISITE_CHECK.json", {
        "schema_version": "PIPELINE_CONTRACT_PREREQUISITE_CHECK_R3",
        "task_id": task_id,
        "governance_revision": governance.get("revision"),
        "read_set": read_set,
        "missing_read_set": missing_read_set,
        "prerequisites": prerequisites,
    })
    (attempt / "GOVERNANCE_VALIDATION.log").write_text(governance_log)

    dry_run_path = attempt / "DRY_RUN_RESULT.json"
    command = list(packet["command_argv"]) + ["--output", str(dry_run_path)]
    completed = subprocess.run(
        command,
        cwd=packet.get("workdir", str(PROJECT)),
        capture_output=True,
        text=True,
        check=False,
        timeout=int(packet.get("budgets", {}).get("wall_seconds", 600)),
    )
    (attempt / "COMMAND.log").write_text(completed.stdout + completed.stderr)
    dry_run_passed = completed.returncode == 0 and dry_run_path.is_file()

    blockers = [row["reason"] for row in prerequisites if not row["passed"]]
    blockers.extend(f"MISSING_READ_SET:{path}" for path in missing_read_set)
    if not dry_run_passed:
        terminal_status = "FAILED_RUNTIME_FINAL"
        blockers.append(f"DRY_RUN_EXIT_{completed.returncode}")
    elif blockers:
        terminal_status = "BLOCKED_PREREQ"
    else:
        terminal_status = "PASSED_DEVELOPMENT"

    metrics = {
        "schema_version": "PIPELINE_CONTRACT_TASK_METRICS_R3",
        "task_id": task_id,
        "contract_dry_run_passed": dry_run_passed,
        "read_set_files_verified": len(read_set),
        "missing_read_set_count": len(missing_read_set),
        "prerequisites_passed": sum(1 for row in prerequisites if row["passed"]),
        "prerequisites_total": len(prerequisites),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "real_model_inference_executed": False,
        "authority_promoted": False,
    }
    write_json(attempt / "METRICS.json", metrics)
    next_status = "READY_FOR_PINNED_REAL_INPUT_PACKET" if terminal_status == "PASSED_DEVELOPMENT" else "BLOCKED_PREREQ"
    next_action = {
        "schema_version": "PIPELINE_CONTRACT_TASK_NEXT_ACTION_R3",
        "task_id": task_id,
        "status": next_status,
        "next_action": (
            "Create a new immutable Task Packet with verified real input manifests before model/data execution."
            if terminal_status == "PASSED_DEVELOPMENT"
            else "Satisfy the listed prerequisite with an evidence-bound input manifest, then create a successor packet."
        ),
        "blockers": blockers,
    }
    write_json(attempt / "NEXT_ACTION.json", next_action)
    (attempt / "DECISION.md").write_text(
        f"# {task_id} 执行决定\n\n"
        f"终态：`{terminal_status}`。合同 dry-run：`{'PASSED' if dry_run_passed else 'FAILED'}`。\n\n"
        "本 attempt 只验证 CPU 合同、输入约束和失败闭包；未运行模型、未更新 current、未晋升 authority。\n"
        + (f"\n阻塞：{', '.join(blockers)}。\n" if blockers else "")
    )

    artifact_paths = [
        packet_path,
        attempt / "PREREQUISITE_CHECK.json",
        attempt / "GOVERNANCE_VALIDATION.log",
        attempt / "COMMAND.log",
        attempt / "METRICS.json",
        attempt / "NEXT_ACTION.json",
        attempt / "DECISION.md",
    ]
    if dry_run_path.is_file():
        artifact_paths.append(dry_run_path)
    manifest = {
        "schema_version": "PIPELINE_CONTRACT_TASK_ARTIFACT_MANIFEST_R3",
        "task_id": task_id,
        "artifacts": [artifact_ref(path) for path in artifact_paths],
    }
    write_json(attempt / "ARTIFACT_MANIFEST.json", manifest)
    result = {
        "schema_version": "PIPELINE_CONTRACT_TASK_RESULT_R3",
        "task_id": task_id,
        "attempt_id": attempt_id,
        "generated_at": generated_at,
        "terminal_status": terminal_status,
        "evaluation_scope": "DEVELOPMENT_CONTRACT_AND_SYNTHETIC_FIXTURE_ONLY",
        "contract_validation": "PASSED" if dry_run_passed else "FAILED",
        "blockers": blockers,
        "artifact_revision": "R7_0_DEVELOPMENT",
        "validity": "VALID_FOR_PINNED_REVISION",
        "authority_promoted": False,
        "real_model_inference_executed": False,
        "external_metric_accuracy": "UNKNOWN",
        "claim_limit": packet["claim_limit"],
        "artifact_manifest": artifact_ref(attempt / "ARTIFACT_MANIFEST.json"),
    }
    write_json(attempt / "RESULT.json", result)
    receipt = {
        "schema_version": "PIPELINE_CONTRACT_TASK_RUN_RECEIPT_R3",
        "task_id": task_id,
        "attempt_id": attempt_id,
        "generated_at": generated_at,
        "terminal_status": terminal_status,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "process_startticks": process_startticks(os.getpid()),
        "executor_epoch": packet.get("executor_epoch"),
        "fencing_token": fencing_token,
        "authority_promoted": False,
        "result": artifact_ref(attempt / "RESULT.json"),
        "artifact_manifest": artifact_ref(attempt / "ARTIFACT_MANIFEST.json"),
        "metrics": artifact_ref(attempt / "METRICS.json"),
    }
    write_json(attempt / "RUN_RECEIPT.json", receipt)
    return {
        "task_id": task_id,
        "slug": slug,
        "status": terminal_status,
        "attempt": str(attempt),
        "result": artifact_ref(attempt / "RESULT.json"),
        "receipt": artifact_ref(attempt / "RUN_RECEIPT.json"),
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--attempt-id", default="attempt_0002")
    parser.add_argument("--summary-attempt-root", type=Path, required=True)
    args = parser.parse_args()
    summary_root = args.summary_attempt_root.resolve()
    if summary_root.exists():
        raise SystemExit(f"refusing to overwrite immutable summary attempt: {summary_root}")

    index = json.loads(args.index.read_text())
    governance, governance_log = load_governance()
    fencing_token = f"pipeline-r3-{uuid.uuid4()}"
    completed_statuses: dict[str, str] = {}
    rows = []
    for packet_row in index["packets"]:
        row = execute_packet(
            packet_row, args.attempt_id, governance, governance_log, completed_statuses, fencing_token
        )
        completed_statuses[row["task_id"]] = row["status"]
        rows.append(row)

    summary_root.mkdir(parents=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    metrics = {
        "schema_version": "PIPELINE_CONTRACT_TASK_DISPATCH_METRICS_R3",
        "task_packet_count": len(rows),
        "terminal_count": len(rows),
        "status_counts": counts,
        "governance_status": governance.get("status"),
        "governance_revision": governance.get("revision"),
        "authority_promoted": False,
        "real_model_inference_count": 0,
    }
    write_json(summary_root / "METRICS.json", metrics)
    write_json(summary_root / "NEXT_ACTION.json", {
        "schema_version": "PIPELINE_CONTRACT_TASK_DISPATCH_NEXT_ACTION_R3",
        "status": "READY",
        "next_task_id": "DEPTH-20-REAL-INPUT-PREFLIGHT",
        "next_action": "Bind an immutable Controller+MANUS+Stereo input manifest, then execute DEPTH-20 on real data.",
        "current_governance_next_task_unchanged": True,
    })
    (summary_root / "DECISION.md").write_text(
        "# R3 十任务执行总结\n\n"
        f"共 {len(rows)} 个 packet 进入不可变终态：{json.dumps(counts, ensure_ascii=False)}。\n\n"
        "所有结论均为开发合同或明确前置阻塞；未运行模型、未修改 current、未晋升 authority。\n"
    )
    write_json(summary_root / "TASK_TERMINALS.json", {
        "schema_version": "PIPELINE_CONTRACT_TASK_TERMINAL_INDEX_R3",
        "generated_at": generated_at,
        "rows": rows,
    })
    manifest_paths = [summary_root / "METRICS.json", summary_root / "NEXT_ACTION.json", summary_root / "DECISION.md", summary_root / "TASK_TERMINALS.json", args.index]
    write_json(summary_root / "ARTIFACT_MANIFEST.json", {
        "schema_version": "PIPELINE_CONTRACT_TASK_DISPATCH_ARTIFACT_MANIFEST_R3",
        "artifacts": [artifact_ref(path) for path in manifest_paths],
    })
    write_json(summary_root / "RESULT.json", {
        "schema_version": "PIPELINE_CONTRACT_TASK_DISPATCH_RESULT_R3",
        "task_id": "pipeline_contracts_r3_dispatch",
        "attempt_id": summary_root.name,
        "generated_at": generated_at,
        "terminal_status": "PASSED_DEVELOPMENT_WITH_BLOCKED_PREREQUISITES" if counts.get("BLOCKED_PREREQ") else "PASSED_DEVELOPMENT",
        "status_counts": counts,
        "governance_status": governance.get("status"),
        "governance_revision": governance.get("revision"),
        "authority_promoted": False,
        "claim_limit": "Development contract execution only; no model quality, current authority, Gold accuracy, or physical truth.",
        "artifact_manifest": artifact_ref(summary_root / "ARTIFACT_MANIFEST.json"),
    })
    write_json(summary_root / "RUN_RECEIPT.json", {
        "schema_version": "PIPELINE_CONTRACT_TASK_DISPATCH_RUN_RECEIPT_R3",
        "task_id": "pipeline_contracts_r3_dispatch",
        "attempt_id": summary_root.name,
        "generated_at": generated_at,
        "terminal_status": "PASSED_DEVELOPMENT_WITH_BLOCKED_PREREQUISITES" if counts.get("BLOCKED_PREREQ") else "PASSED_DEVELOPMENT",
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "process_startticks": process_startticks(os.getpid()),
        "fencing_token": fencing_token,
        "authority_promoted": False,
        "result": artifact_ref(summary_root / "RESULT.json"),
        "artifact_manifest": artifact_ref(summary_root / "ARTIFACT_MANIFEST.json"),
    })
    print(json.dumps({"status": "TERMINAL", "summary": str(summary_root), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
