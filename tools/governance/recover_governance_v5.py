from __future__ import annotations

"""Fail-closed exact78 V5.2 governance recovery.

This command is the only allowed write in stage 0 before the current receipt is
fresh.  It verifies the old receipt, checks PID/startticks and the frozen Wave0
SHA, recounts Clean from immutable session finals, records dead workers as
failed attempts, and publishes a single new coherent governance generation.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from tools.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    freshness,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
    sha256_file,
    validate_artifact_ref,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "tasks/control/runs/20260913_exact78_v52_governance_recovery_v5"
WAVE0 = ROOT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
WAVE0_ROOT = WAVE0.parent
EXPECTED_WAVE0_SHA = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"
RECOVERABLE_TASKS = {"exact78_wave0_clean_v1", "exact78_upstream_c_successors_v1"}


def gpu_snapshot() -> dict[str, Any]:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    processes = []
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 2)]
            if len(parts) != 3:
                continue
            processes.append({
                "pid": int(parts[0]) if parts[0].isdigit() else None,
                "process_name": parts[1],
                "used_memory_mib": int(parts[2]) if parts[2].isdigit() else None,
                "classification": "DRIVER_RESIDUE_ONLY" if parts[1] == "[Not Found]" else "VISIBLE_COMPUTE_PROCESS",
            })
    devices = []
    if gpu.returncode == 0:
        for line in gpu.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 4:
                devices.append({
                    "index": int(parts[0]), "utilization_percent": int(parts[1]),
                    "memory_used_mib": int(parts[2]), "memory_total_mib": int(parts[3]),
                })
    return {
        "query_available": apps.returncode == 0 and gpu.returncode == 0,
        "processes": processes,
        "devices": devices,
        "driver_residue_is_not_executor_evidence": True,
    }


def verify_old_receipt(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for label, reference in receipt.get("files", {}).items():
        errors.extend(f"{label}: {error}" for error in validate_artifact_ref(reference))
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    for label, value in (("authority", authority), ("task_state", state)):
        if value.get("governance_revision") != receipt.get("governance_revision"):
            errors.append(f"{label} revision mismatch")
        if value.get("generation_id") != receipt.get("generation_id"):
            errors.append(f"{label} generation mismatch")
    return errors


def valid_clean_final(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return False, ["RESULT missing"]
    result = load_json(path)
    if not str(result.get("status", "")).startswith("PASS"):
        errors.append(f"non-pass status={result.get('status')}")
    if result.get("grade") not in {"A", "B"}:
        errors.append(f"invalid grade={result.get('grade')}")
    if result.get("downstream_authorized") is not True:
        errors.append("downstream_authorized is not true")
    for reference in result.get("artifacts", {}).values():
        if isinstance(reference, dict) and "path" in reference:
            errors.extend(validate_artifact_ref(reference))
    review = path.with_name("AGENT_REVIEW.json")
    if not review.is_file():
        errors.append("AGENT_REVIEW missing")
    return not errors, errors


def recount_clean() -> dict[str, Any]:
    selection = load_json(WAVE0)
    rows = selection.get("sessions", [])
    existing = 0
    new_passed: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for row in rows:
        if row.get("existing_clean"):
            errors = validate_artifact_ref(row["existing_clean"])
            if errors:
                partial.append({"session": row["session_id"], "kind": "existing_invalid", "errors": errors})
            else:
                existing += 1
        final = WAVE0_ROOT / "propainter_v1" / row["session_id"] / "RESULT.json"
        if final.is_file():
            valid, errors = valid_clean_final(final)
            if valid:
                new_passed.append({"session": row["session_id"], "result": artifact_ref(final)})
            else:
                partial.append({"session": row["session_id"], "kind": "new_invalid", "errors": errors})
    passed = existing + len(new_passed)
    return {
        "selected": len(rows), "existing_passed": existing,
        "new_passed": len(new_passed), "passed": passed,
        "pending": len(rows) - passed, "new_finals": new_passed,
        "invalid_or_partial": partial,
    }


def update_stage(authority: dict[str, Any], stage_name: str, **values: Any) -> None:
    stage = next(item for item in authority["stages"] if item["stage"] == stage_name)
    stage.update(values)


def ensure_task(state: dict[str, Any], task_id: str, phase: str, status: str) -> dict[str, Any]:
    task = next((item for item in state["tasks"] if item["task_id"] == task_id), None)
    if task is None:
        task = {
            "task_id": task_id, "phase": phase, "attempt": 0, "status": status,
            "updated_at": now_iso(), "heartbeat_at": None, "session": None,
            "pid": None, "proc_start_ticks": None, "gpu_id": None,
        }
        state["tasks"].append(task)
    return task


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    errors = verify_old_receipt(receipt)
    actual_wave_sha = sha256_file(WAVE0) if WAVE0.is_file() else None
    if actual_wave_sha != EXPECTED_WAVE0_SHA:
        errors.append(f"Wave0 SHA conflict expected={EXPECTED_WAVE0_SHA} actual={actual_wave_sha}")
    if receipt.get("governance_revision") != args.expected_revision:
        errors.append(
            f"CAS input mismatch expected={args.expected_revision} current={receipt.get('governance_revision')}"
        )
    authority = load_json(AUTHORITY_PATH)
    state = load_json(TASK_STATE_PATH)
    clean = recount_clean()
    gpu = gpu_snapshot()
    workers: list[dict[str, Any]] = []
    for task in state["tasks"]:
        if task.get("status") not in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
            continue
        pid = task.get("pid")
        identity = process_identity(int(pid)) if pid is not None else {"pid": None, "alive": False, "start_ticks": None}
        matches = bool(
            identity["alive"]
            and (task.get("proc_start_ticks") is None or identity["start_ticks"] == task.get("proc_start_ticks"))
        )
        workers.append({
            "task_id": task["task_id"], "recorded_status": task["status"],
            "recorded_pid": pid, "recorded_start_ticks": task.get("proc_start_ticks"),
            "observed": identity, "identity_matches": matches,
            "recovery": "KEEP_ACTIVE" if matches else "FAILED_RUNTIME_DEAD_WORKER",
        })
        if not matches and task["task_id"] not in RECOVERABLE_TASKS:
            errors.append(f"unexpected dead active task requires manual classification: {task['task_id']}")
    report = {
        "schema_version": "exact78-governance-recovery-v5",
        "created_at": now_iso(),
        "status": "PASS_READY_TO_APPLY" if not errors else "STATUS_CONFLICT",
        "source_revision": receipt.get("governance_revision"),
        "target_revision": int(receipt.get("governance_revision", 0)) + 1,
        "old_receipt": artifact_ref(RECEIPT_PATH),
        "wave0": {"path": str(WAVE0), "expected_sha256": EXPECTED_WAVE0_SHA, "actual_sha256": actual_wave_sha},
        "clean_recount": clean,
        "worker_audit": workers,
        "gpu_audit": gpu,
        "errors": errors,
        "claim_limit": "Governance recovery and factual recount only; no Clean, Robot, contact, training or deployment authority.",
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    output = RUN_ROOT / ("GOVERNANCE_CONFLICT.json" if errors else "GOVERNANCE_RECOVERY_V5.json")
    atomic_json(output, report)
    if errors or not args.apply:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2 if errors else 0

    for worker in workers:
        if worker["identity_matches"]:
            continue
        task = next(item for item in state["tasks"] if item["task_id"] == worker["task_id"])
        task["status"] = "PENDING"
        task["phase"] = "recovered_after_dead_guardian"
        task["updated_at"] = now_iso()
        task["heartbeat_at"] = None
        task["pid"] = None
        task["proc_start_ticks"] = None
        task["gpu_id"] = None
        task["last_attempt_terminal"] = "FAILED_RUNTIME"
        task["last_attempt_reason"] = "DEAD_WORKER_PID_IDENTITY_MISMATCH"
        task["recovery_receipt"] = artifact_ref(output)
        state["recent_events"].append({
            "task_id": task["task_id"], "session": task.get("session"),
            "attempt": task.get("attempt", 0), "status": "FAILED_RUNTIME",
            "created_at": now_iso(), "message": "Dead guardian recovered by PID/startticks audit.",
            "result": artifact_ref(output),
        })

    authority["waves"]["wave0_clean_passed"] = clean["passed"]
    authority["waves"]["wave0_clean_pending"] = clean["pending"]
    clean_evidence = [artifact_ref(WAVE0)] + [item["result"] for item in clean["new_finals"]]
    update_stage(
        authority, "Clean", total=58, passed=clean["passed"], grade_c=0,
        running=0, blocked=clean["pending"],
        authority_scope="EXACT78_WAVE0_FROZEN_PLUS_VERIFIED_FINALS",
        denominator_semantics="Wave0 frozen 58; only immutable Grade A/B finals with validated artifacts count as passed.",
        evidence=clean_evidence,
    )
    decision = "V5.2: one executor epoch, one canonical run signature and immutable finals are mandatory."
    if decision not in authority["decisions"]:
        authority["decisions"].append(decision)

    recovery_task = ensure_task(state, "governance_recovery_v5", "governance_recovery", "PASSED")
    recovery_task.update(
        status="PASSED", attempt=1, updated_at=now_iso(), heartbeat_at=None,
        pid=None, proc_start_ticks=None, gpu_id=None, result=artifact_ref(output),
    )
    stage0 = ensure_task(state, "exact78_v52_stage0", "contracts_and_fault_injection", "PENDING")
    stage0.update(status="PENDING", updated_at=now_iso(), heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    for task_id, phase in (
        ("exact78_v52_lane_a_clean", "clean"),
        ("exact78_v52_lane_b_upstream_c", "upstream_repair"),
        ("exact78_v52_lane_c_contact_robot", "contact_robot"),
        ("exact78_v52_lane_d_visual_aux", "training"),
        ("exact78_v52_lane_e_baseline_docs", "documentation"),
        ("exact78_v52_lane_f_cleanup", "maintenance"),
    ):
        lane = ensure_task(state, task_id, phase, "BLOCKED_PREREQ")
        lane.update(status="BLOCKED_PREREQ", updated_at=now_iso(), heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": "governance_recovery_v5", "session": None, "attempt": 1,
        "status": "PASSED", "created_at": now_iso(),
        "message": f"Recovered dead guardians and recounted Wave0 Clean={clean['passed']}/58.",
        "result": artifact_ref(output),
    }])[-100:]
    state["next_task"] = {
        "task_id": "exact78_v52_stage0", "session": None,
        "prerequisites": ["governance_recovery_v5"],
        "expected_resource": "CPU only; task packet, executor fencing, run-signature and fault-injection tests",
        "stop_condition": "Any receipt/revision/SHA conflict, ghost PID, duplicate executor or immutable-final overwrite acceptance.",
    }
    published = publish_bundle(
        authority, state, event_type="GOVERNANCE_RECOVERY_V5",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps(published, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
