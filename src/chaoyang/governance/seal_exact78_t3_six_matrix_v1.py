"""Bounded CPU-only T3 matrix Task Packet and immutable sealer.

``--check-only`` never writes. ``--seal`` is reserved for a later authorized
single-publisher window, after all six T0 sessions have terminal evidence.
The pre-registration task-state snapshot is part of MATRIX.json so the
registration CAS itself cannot make its evidence SHA appear to drift.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any

from chaoyang.governance.build_exact78_t3_six_matrix_v1 import build_matrix
from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK_ID = "exact78_raw_to_robot_t3_six_matrix_v1_20260930"
REVISION = "EXACT78_RAW_TO_ROBOT_T3_SIX_MATRIX_V1"
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T3_SIX_MATRIX_V1.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
FREEZE = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001/INPUT_FREEZE.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
COMMAND = "PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.governance.seal_exact78_t3_six_matrix_v1 --seal"
CODE_RELATIVE = (
    "src/chaoyang/governance/seal_exact78_t3_six_matrix_v1.py",
    "src/chaoyang/governance/build_exact78_t3_six_matrix_v1.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/v52_contracts.py",
    "src/chaoyang/governance/v71_contracts.py",
    "src/chaoyang/governance/current_r3_contracts.py",
)
WRITE_SET = (
    str(ATTEMPT.relative_to(REPO_ROOT)),
    str(PACKET.relative_to(REPO_ROOT)),
    str(RECEIPT.relative_to(REPO_ROOT)),
    "tasks/current/INDEX.json",
    "docs/governance",
)
TERMINAL = frozenset({"PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE", "FAILED_QUALITY_FINAL"})


class T3NotReady(RuntimeError):
    """A truthful six-session seal is not yet available."""


def _sha_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _frozen_cohort(freeze: dict[str, Any]) -> dict[str, Any]:
    selected = freeze.get("selected", {})
    counts = {"development": 0, "holdout": 0}
    for row in selected.values():
        role = row.get("role")
        if role not in counts:
            raise T3NotReady("T0_SELECTION_ROLE_DRIFT")
        counts[role] += 1
    limits = freeze.get("source_group_limits", {})
    if (len(selected) != 6 or counts != {"development": 4, "holdout": 2}
            or limits.get("0902") != "SPLIT_NOT_ESTABLISHED"
            or limits.get("0901") != "FROZEN_INDEPENDENT_GROUP"
            or limits.get("0915") != "FROZEN_INDEPENDENT_GROUP"):
        raise T3NotReady("T0_COHORT_OR_SPLIT_DRIFT")
    if any(row.get("source_path") is None for row in selected.values()):
        raise T3NotReady("T0_SOURCE_PATH_ABSENT")
    return {"development_sessions": 4, "pre_frozen_cross_group_holdout_sessions": 2,
            "expected_total": 6, "source_group_limits": limits,
            "0902_independent_holdout": "SPLIT_NOT_ESTABLISHED",
            "blind_test_claim": False}


def candidate_report() -> dict[str, Any]:
    """Read only T0, task-state eligibility, terminal receipts and sealed stages."""
    matrix = build_matrix()
    if artifact_ref(FREEZE) != matrix["inputs"]["t0_freeze"]:
        raise T3NotReady("T0_FREEZE_SHA_CHANGED_DURING_REPORT")
    freeze = load_json(FREEZE)
    cohort = _frozen_cohort(freeze)
    if set(matrix["sessions"]) != set(freeze["selected"]):
        raise T3NotReady("T0_MATRIX_SESSION_SET_DRIFT")
    report = copy.deepcopy(matrix)
    report["schema_version"] = "exact78-t3-six-session-sealed-report-v1"
    report["status"] = "PRE_REGISTER_SNAPSHOT_NOT_YET_SEALED"
    report["cohort"] = cohort
    report["external_evidence"] = {
        "c0_cleanup": "NOT_BOUND_TO_THIS_TASK",
        "corrected_metric_geometry": "NOT_EVALUATED_IN_THIS_MATRIX",
        "clean_visual_quality": "NOT_EVALUATED_IN_THIS_MATRIX",
        "depth_object6d_qualification": "NOT_EVALUATED_IN_THIS_MATRIX",
        "trusted_occlusion_geometry": "NOT_EVALUATED_IN_THIS_MATRIX",
    }
    for row in report["sessions"].values():
        row["review"] = {
            "automatic_execution_audit": row["execution"]["status"],
            "ai_visual_review": "NOT_EVALUATED",
            "user_visual_review": row["user_review"],
        }
    return report


def readiness(report: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers = []
    for session_id, row in report["sessions"].items():
        status = row["execution"]["status"]
        evidence = row.get("evidence", {})
        if status not in TERMINAL:
            blockers.append(f"{session_id}:NOT_TERMINAL:{status}")
            continue
        if not evidence.get("terminal_receipt") or not evidence.get("terminal_result"):
            blockers.append(f"{session_id}:TERMINAL_SHA_CHAIN_ABSENT")
        if status == "PASSED" and not all(evidence.get(key) for key in
                                            ("candidate_result", "hawor_result", "hawor_seal",
                                             "robot_result", "robot_seal", "robot_numeric_result")):
            blockers.append(f"{session_id}:PASSED_STAGE_SEALS_ABSENT")
    return not blockers, blockers


def _evidence_refs(report: dict[str, Any]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in report["sessions"].values():
        for ref in row.get("evidence", {}).values():
            if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                unique[(ref["path"], ref["sha256"])] = ref
    return [unique[key] for key in sorted(unique)]


def packet_value(report: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    ready, blockers = readiness(report)
    if not ready:
        raise T3NotReady("T3_SIX_TERMINAL_PROOF_NOT_READY:" + ";".join(blockers))
    stamp = now or datetime.now().astimezone()
    if stamp >= CAMPAIGN_END:
        raise T3NotReady("CAMPAIGN_DEADLINE_EXPIRED")
    state_ref = report["inputs"]["task_state_for_terminal_eligibility_only"]
    if artifact_ref(TASK_STATE_PATH) != state_ref:
        raise T3NotReady("PRE_REGISTER_TASK_STATE_SHA_DRIFT")
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": TASK_ID}
    runtime = [artifact_ref(REPO_ROOT / path) for path in CODE_RELATIVE]
    inputs = {"plan": artifact_ref(PLAN), "t0_freeze": artifact_ref(FREEZE),
              "t0_frame_time_map": report["inputs"]["t0_frame_time_map"],
              "pre_register_task_state": state_ref,
              "pre_register_matrix_sha256": _sha_value(report),
              "terminal_evidence": _evidence_refs(report)}
    config = {"mode": "CPU_ONLY_SIX_SESSION_REPORT", "gpu_count": 0,
              "cohort": report["cohort"], "matrix_schema": report["schema_version"],
              "review_policy": "AI_AND_USER_SEPARATE_NOT_PREFILLED",
              "external_quality_policy": "UNBOUND_EVIDENCE_REMAINS_NOT_EVALUATED"}
    signature = _sha_value({"inputs": inputs, "runtime_code": runtime,
                            "config": config, "command": COMMAND,
                            "schema": "exact78-t3-six-session-seal-run-v1"})
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Seal the six T0 Exact78 capability/quality matrix from terminal evidence only",
        "phase": "EXACT78_RAW_TO_ROBOT_T3_SIX_MATRIX",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": REVISION, "registered_at": stamp.isoformat(),
        "deadline_at": min(stamp + timedelta(seconds=3600), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     str(FREEZE.relative_to(REPO_ROOT)),
                     str(Path(report["inputs"]["t0_frame_time_map"]["path"]).relative_to(REPO_ROOT)),
                     str(TASK_STATE_PATH.relative_to(REPO_ROOT)),
                     "docs/governance/DOC_AUTHORITY_MAP.json",
                     "src/chaoyang/governance/build_exact78_t3_six_matrix_v1.py",
                     "src/chaoyang/governance/seal_exact78_t3_six_matrix_v1.py"],
        "expanded_read_set": [ref["path"] for ref in inputs["terminal_evidence"]],
        "write_set": list(WRITE_SET),
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "T0_SIX_FROZEN",
                          "SIX_TERMINAL_SHA_CHAINS", "CPU_ONLY", "NO_PARTIAL_READS",
                          "EXTERNAL_DATA_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["MATRIX.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 3600,
                    "session_attempts_max": 1},
        "stop_conditions": ["SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Report integrity only; 4 development + 2 cross-group holdouts, 0902 split absent; no automatic visual/product/training PASS",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature,
        "commands": [COMMAND],
    }
    if errors := _validate_packet(packet):
        raise T3NotReady("INVALID_TASK_PACKET:" + ";".join(errors))
    return packet


def publisher_blockers(state: dict[str, Any], index: dict[str, Any]) -> list[str]:
    blockers = []
    next_task = state.get("next_task")
    if next_task:
        blockers.append("NEXT_TASK_ACTIVE:" + str(next_task.get("task_id")))
    allowed = [row.get("task_id") for row in index.get("task_packets", [])
               if row.get("execution_allowed")]
    if allowed:
        blockers.append("EXECUTION_ALLOWED_PACKET:" + ",".join(map(str, allowed)))
    if any(row.get("status") in {"CLAIMED", "WAIT_GPU_RESOURCE", "RUNNING"}
           for row in state.get("tasks", [])):
        blockers.append("ACTIVE_TASK_ROW")
    return blockers


def dry_run() -> dict[str, Any]:
    report = candidate_report()
    ready, blockers = readiness(report)
    if not ready:
        return {"status": "DRY_RUN_NOT_READY", "blockers": blockers,
                "execution_counts": report["execution_counts"],
                "cohort": report["cohort"], "write_set_draft": list(WRITE_SET),
                "pre_register_task_state": report["inputs"]["task_state_for_terminal_eligibility_only"],
                "budgets_draft": {"gpu_count": 0, "workers_max": 1,
                                  "cpu_threads_total": 1, "wall_seconds_max": 3600,
                                  "session_attempts_max": 1},
                "registered": False}
    packet = packet_value(report)
    publisher = publisher_blockers(load_json(TASK_STATE_PATH), load_json(INDEX))
    return {"status": "DRY_RUN_EVIDENCE_READY_PUBLISHER_BUSY" if publisher else
            "DRY_RUN_READY", "publisher_blockers": publisher, "task_id": TASK_ID,
            "run_signature": packet["run_signature"],
            "pre_register_matrix_sha256": packet["inputs"]["pre_register_matrix_sha256"],
            "terminal_evidence_refs": len(packet["inputs"]["terminal_evidence"]),
            "read_set_count": len(packet["read_set"]), "write_set": packet["write_set"],
            "execution_counts": report["execution_counts"],
            "pre_register_task_state": packet["inputs"]["pre_register_task_state"],
            "cohort": report["cohort"], "registered": False}


def _index_copy() -> dict[str, Any]:
    value = load_json(INDEX)
    return {key: copy.deepcopy(item) for key, item in value.items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def _fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer):
        raise RuntimeError("T3_WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    rows = [row for row in state["tasks"] if row["task_id"] == TASK_ID]
    allowed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(rows) != 1 or rows[0]["status"] != "RUNNING"
            or rows[0]["run_signature"] != packet["run_signature"]
            or rows[0]["pid"] != writer["pid"]
            or rows[0]["proc_start_ticks"] != writer["proc_start_ticks"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(allowed) != 1 or allowed[0]["task_id"] != TASK_ID
            or allowed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("T3_ACTIVE_TASK_OR_PACKET_DRIFT")
    for ref in (*packet["runtime_code"], *packet["inputs"]["terminal_evidence"],
                packet["inputs"]["plan"], packet["inputs"]["t0_freeze"],
                packet["inputs"]["t0_frame_time_map"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("T3_FROZEN_SHA_DRIFT:" + ";".join(errors))


def register(report: dict[str, Any], packet: dict[str, Any]) -> None:
    """One governance CAS; never called by check-only."""
    if any(path.exists() for path in (ATTEMPT, PACKET, RECEIPT)):
        raise RuntimeError("T3_FRESH_ATTEMPT_REQUIRED")
    identity = process_identity(os.getpid())
    if (packet["writer"]["pid"] != os.getpid()
            or packet["writer"]["proc_start_ticks"] != identity["start_ticks"]):
        raise RuntimeError("T3_WRITER_IDENTITY_DRIFT")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    if (artifact_ref(TASK_STATE_PATH) != packet["inputs"]["pre_register_task_state"]
            or state.get("next_task")
            or any(row.get("status") in {"CLAIMED", "WAIT_GPU_RESOURCE", "RUNNING"}
                   for row in state["tasks"])
            or any(row.get("execution_allowed") for row in index["task_packets"])):
        raise RuntimeError("T3_GOVERNANCE_OR_STATE_CAS_CONFLICT")
    if _sha_value(report) != packet["inputs"]["pre_register_matrix_sha256"]:
        raise RuntimeError("T3_MATRIX_SHA_DRIFT")
    for ref in (*packet["runtime_code"], *packet["inputs"]["terminal_evidence"],
                packet["inputs"]["plan"], packet["inputs"]["t0_freeze"],
                packet["inputs"]["t0_frame_time_map"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("T3_PRE_REGISTER_SHA_DRIFT:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    atomic_json(ATTEMPT / "WRITER.json", packet["writer"])
    atomic_json(ATTEMPT / "MATRIX.json", report)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "T0_SIX_FROZEN_SESSIONS",
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(PACKET),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {"task_id": TASK_ID, "session": "T0_SIX_FROZEN_SESSIONS",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": "ONE_CPU_NO_GPU",
                          "stop_condition": "one six-session matrix seal or deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("T3_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK_ID,
        "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision=REVISION + "_RUNNING", plan_revision=packet["plan_revision"],
                 execution_revision=REVISION, status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=REVISION + "_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX,
        task_packet_index_value=index)


def finish(packet: dict[str, Any], *, failure: str | None = None) -> dict[str, Any]:
    _fence(packet)
    matrix_path = ATTEMPT / "MATRIX.json"
    report = load_json(matrix_path)
    if _sha_value(report) != packet["inputs"]["pre_register_matrix_sha256"]:
        failure = failure or "T3_MATRIX_SHA_DRIFT"
    ready, blockers = readiness(report)
    if not ready:
        failure = failure or "T3_TERMINAL_CHAIN_DRIFT:" + ";".join(blockers)
    if (ATTEMPT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("T3_TERMINAL_RESULT_ALREADY_EXISTS")
    status = "PASSED" if failure is None else "FAILED_RUNTIME_FINAL"
    result = {
        "schema_version": "exact78-t3-six-session-seal-result-v1",
        "task_id": TASK_ID, "status": status,
        "report_integrity": "SEALED_TERMINAL_SHA_CHAINS" if status == "PASSED" else "FAILED",
        "failure_code": failure, "matrix": artifact_ref(matrix_path),
        "execution_counts": report["execution_counts"],
        "cohort": report["cohort"],
        "ai_review": "NOT_EVALUATED", "user_review": "PER_SESSION_PENDING_OR_NOT_EVALUATED",
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "general_runnable_operation": None,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-t3-six-session-receipt-v1",
                          "task_id": TASK_ID, "status": status,
                          "result": artifact_ref(ATTEMPT / "RESULT.json"),
                          "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(ATTEMPT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="REPORT_INTEGRITY_ONLY" if status == "PASSED" else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("T3_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision=REVISION + "_TERMINAL",
                 plan_revision=packet["plan_revision"], execution_revision=REVISION,
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=REVISION + "_" + status,
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX,
        task_packet_index_value=index)
    return {"status": status, "result": artifact_ref(ATTEMPT / "RESULT.json"),
            "governance_revision": published["governance_revision"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(dry_run(), sort_keys=True, ensure_ascii=False))
        return 0
    report = candidate_report()
    packet = packet_value(report)
    register(report, packet)
    result = finish(packet)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
