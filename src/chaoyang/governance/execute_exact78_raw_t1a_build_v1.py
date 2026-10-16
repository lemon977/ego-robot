"""Freeze and test the CPU-only Exact78 Raw->Robot candidate implementation.

This task never imports or invokes HaWoR, SAM, Robot rendering, or Clean.  The
candidate files already exist when it registers; their bytes are pinned before
the negative tests run and are checked again before the immutable result.
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
import subprocess
import time
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref,
    validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_raw_to_robot_t1a_candidate_build_v1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T1A_BUILD_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
BUILD_PLAN = PLAN.parent / "02_T1A_BUILD_ZH.md"
T0 = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
CORRECTION = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
PARENT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"
T0_CORRECTION_SHA = "7cd873022e3a72b05fa88701cccb76dec7fda5321c4eed11b2aa14f69b95314c"
TEST = REPO_ROOT / "tests/unit/test_exact78_raw_to_robot_t1a.py"
CANDIDATE_PATHS = (
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/cli/exact78_submit.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/ops/run_exact78_raw_to_robot_v1.py",
    "contracts/exact78_motion_state_v1.schema.json",
    "tests/unit/test_exact78_raw_to_robot_t1a.py",
)
READ_ONLY_CANDIDATE_DEPENDENCIES = (
    "src/chaoyang/governance/exact78_operation_gate.py",
)
TEST_COMMAND = (
    "/usr/local/bin/python", "-B", "-m", "unittest", "discover", "-s", "tests/unit",
    "-p", "test_exact78_raw_to_robot_t1a.py", "-v",
)


def checked_inputs() -> tuple[dict[str, Any], datetime]:
    if artifact_ref(PARENT)["sha256"] != PARENT_SHA:
        raise RuntimeError("VISUAL_PARENT_SHA_DRIFT")
    if artifact_ref(CORRECTION / "RESULT.json")["sha256"] != T0_CORRECTION_SHA:
        raise RuntimeError("T0_CORRECTION_SHA_DRIFT")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_OPERATION_ALREADY_PROMOTED")
    correction = load_json(CORRECTION / "RESULT.json")
    if correction["status"] != "PASSED" or correction.get("models_run") is not False:
        raise RuntimeError("T0_CORRECTION_NOT_SEALED")
    if (correction["upstream_input_freeze"] != artifact_ref(T0 / "INPUT_FREEZE.json")
            or correction["upstream_frame_time_map"] != artifact_ref(T0 / "FRAME_TIME_MAP.json")):
        raise RuntimeError("T0_UPSTREAM_REF_DRIFT")
    upstream = load_json(T0 / "INPUT_FREEZE.json")
    selected = upstream["selected"]
    if len(selected) != 6 or sum(row["role"] == "development" for row in selected.values()) != 4:
        raise RuntimeError("T0_SAMPLE_SET_DRIFT")
    if upstream["source_group_limits"]["0902"] != "SPLIT_NOT_ESTABLISHED":
        raise RuntimeError("0902_HOLDOUT_CLAIM_DRIFT")
    deadline = datetime.fromisoformat(correction["campaign_deadline_at"])
    if deadline <= datetime.now().astimezone():
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    refs: dict[str, Any] = {
        "plan": artifact_ref(PLAN), "build_plan": artifact_ref(BUILD_PLAN),
        "t0_correction": artifact_ref(CORRECTION / "RESULT.json"),
        "t0_capability": artifact_ref(CORRECTION / "CORRECTED_CAPABILITY.json"),
        "t0_input_freeze": artifact_ref(T0 / "INPUT_FREEZE.json"),
        "t0_frame_time_map": artifact_ref(T0 / "FRAME_TIME_MAP.json"),
        "route": artifact_ref(ROUTE), "visual_parent": artifact_ref(PARENT),
        "candidate_code": [artifact_ref(REPO_ROOT / item)
                           for item in (*CANDIDATE_PATHS, *READ_ONLY_CANDIDATE_DEPENDENCIES)],
    }
    return refs, deadline


def packet_value(inputs: dict[str, Any], campaign_deadline: datetime) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {
        "pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK,
    }
    runtime = [artifact_ref(SCRIPT)]
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "test_command": TEST_COMMAND,
        "schema": "exact78-raw-t1a-build-v1", "weights": "NOT_RUN",
        "calibration": "NOT_CONSUMED_BY_BUILD",
    })).hexdigest()
    value = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Freeze and test one raw-only Exact78 candidate entry without model execution",
        "phase": "EXACT78_RAW_TO_ROBOT_T1A_BUILD", "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_RAW_T1A_BUILD_V1", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=3), campaign_deadline).isoformat(),
        "campaign_deadline_at": campaign_deadline.isoformat(),
        "read_set": [
            "AGENTS.md", str(PLAN.relative_to(REPO_ROOT)), str(BUILD_PLAN.relative_to(REPO_ROOT)),
            str((T0 / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
            str((T0 / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
            str((CORRECTION / "RESULT.json").relative_to(REPO_ROOT)),
            str(ROUTE.relative_to(REPO_ROOT)), str(SCRIPT.relative_to(REPO_ROOT)),
        ],
        "write_set": [
            str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
            str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance",
            *CANDIDATE_PATHS,
        ],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "T0_CORRECTION_SHA_PINNED",
                          "NO_MODEL_RUN", "NO_DELETION", "0916_READ_ONLY"],
        "required_outputs": ["CODE_FREEZE.json", "BUILD_AUDIT.json", "RESULT.json"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 10800, "session_attempts_max": 1},
        "stop_conditions": ["SHA_DRIFT", "DOUBLE_WRITER", "FORBIDDEN_READ_TEST_FAIL",
                            "TEST_FAIL", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Candidate source/build tests only; no model, video, visual quality, product, training or production entry",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "run_signature": signature,
        "commands": ["PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_raw_t1a_build_v1"],
    }
    errors = _validate_packet(value)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    return value


def index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def verify_refs(packet: dict[str, Any]) -> None:
    for name, value in packet["inputs"].items():
        values = value if isinstance(value, list) else [value]
        for ref in values:
            errors = validate_artifact_ref(ref)
            if errors:
                raise RuntimeError(f"INPUT_SHA_DRIFT:{name}:{';'.join(errors)}")
    for ref in packet["runtime_code"]:
        errors = validate_artifact_ref(ref)
        if errors:
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))


def fence(packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    index = load_json(INDEX)
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if len(allowed) != 1 or allowed[0]["task_id"] != TASK or allowed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]:
        raise RuntimeError("PACKET_ROUTING_CONFLICT")
    verify_refs(packet)


def heartbeat(packet: dict[str, Any], *, elapsed: float) -> None:
    fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T1A_BUILD_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    summary = OUT / "RESUME_SUMMARY.json"
    if not summary.exists():
        atomic_json(summary, {"schema_version": "exact78-raw-t1a-resume-v1",
            "status": "RUNNING_TESTS", "elapsed_seconds": round(elapsed, 3),
            "next_action": "Wait for bounded unit tests, then seal CODE_FREEZE/BUILD_AUDIT"})


def register(packet: dict[str, Any]) -> None:
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        raise RuntimeError("FRESH_T1A_TASK_REQUIRED")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    OUT.mkdir(parents=True)
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "Exact78_three_source", "updated_at": now, "heartbeat_at": now,
        "pid": packet["writer"]["pid"], "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(PACKET),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {"task_id": TASK, "session": "Exact78_three_source",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "one sealed build result or three-hour deadline"}
    errors = validate_task_state(state)
    if errors:
        raise RuntimeError("TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision="EXACT78_RAW_T1A_BUILD_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T1A_BUILD_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def run_tests(packet: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    fence(packet)
    environment = os.environ.copy()
    temp = OUT / "tmp"
    temp.mkdir()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp),
                       XDG_CACHE_HOME=str(OUT / "cache"), PYTHONNOUSERSITE="1")
    log = OUT / "unit_tests.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.Popen(TEST_COMMAND, cwd=REPO_ROOT, env=environment,
                                   stdout=handle, stderr=subprocess.STDOUT)
        last_heartbeat = started
        while process.poll() is None:
            if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
                process.terminate()
                process.wait(timeout=10)
                raise RuntimeError("T1A_DEADLINE")
            if time.monotonic() - last_heartbeat >= 50:
                heartbeat(packet, elapsed=time.monotonic() - started)
                last_heartbeat = time.monotonic()
            time.sleep(1)
        code = process.returncode
    audit = {"schema_version": "exact78-raw-t1a-build-audit-v1",
        "test_command": list(TEST_COMMAND), "exit_code": code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "test_log": artifact_ref(log), "models_run": False,
        "training_run": False, "test_suite": str(TEST.relative_to(REPO_ROOT))}
    return audit, log


def finish(packet: dict[str, Any], *, status: str, outputs: dict[str, Path],
           reason: str | None = None) -> dict[str, Any]:
    fence(packet)
    if status not in ("PASSED", "FAILED_RUNTIME_FINAL") or (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("INVALID_OR_DUPLICATE_TERMINAL_RESULT")
    result = {"schema_version": "exact78-raw-t1a-build-result-v1", "task_id": TASK,
        "status": status, "execution_complete": status == "PASSED",
        "campaign_deadline_at": packet["campaign_deadline_at"],
        "inputs": packet["inputs"], "outputs": {name: artifact_ref(path) for name, path in outputs.items()},
        "reason": reason, "models_run": False, "training_run": False,
        "source_mutated": False, "files_deleted": False, "robot100_modified": False,
        "general_operation": None, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "next_action": "Register T1-B fresh Cards042 full-session candidate run with this code freeze",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-raw-t1a-build-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT), quality="BUILD_TESTS_ONLY" if status == "PASSED" else "BUILD_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED")
    state["next_task"] = None
    errors = validate_task_state(state)
    if errors:
        raise RuntimeError("TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T1A_BUILD_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_RAW_T1A_BUILD_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    inputs, deadline = checked_inputs()
    packet = packet_value(inputs, deadline)
    if arguments.check_only:
        print(json.dumps({"status": "DRY_RUN_PASS", "task_id": TASK,
            "candidate_code": [ref["path"] for ref in inputs["candidate_code"]],
            "candidate_code_sha256": [ref["sha256"] for ref in inputs["candidate_code"]],
            "run_signature": packet["run_signature"],
            "campaign_deadline_at": packet["campaign_deadline_at"]}, ensure_ascii=False))
        return 0
    register(packet)
    try:
        audit, log = run_tests(packet)
        atomic_json(OUT / "BUILD_AUDIT.json", audit)
        if audit["exit_code"] != 0:
            return_code = 1
            reason = f"UNIT_TEST_EXIT_{audit['exit_code']}"
        else:
            return_code = 0
            reason = None
        freeze = {"schema_version": "exact78-raw-t1a-code-freeze-v1",
            "candidate_code": packet["inputs"]["candidate_code"],
            "schema": artifact_ref(REPO_ROOT / "contracts/exact78_motion_state_v1.schema.json"),
            "test_audit": artifact_ref(OUT / "BUILD_AUDIT.json"),
            "t0_correction": packet["inputs"]["t0_correction"],
            "claim_limit": packet["claim_limit"]}
        atomic_json(OUT / "CODE_FREEZE.json", freeze)
        print(json.dumps(finish(packet, status="PASSED" if return_code == 0 else "FAILED_RUNTIME_FINAL",
            outputs={"code_freeze": OUT / "CODE_FREEZE.json", "build_audit": OUT / "BUILD_AUDIT.json",
                     "unit_test_log": log}, reason=reason), ensure_ascii=False), flush=True)
        return return_code
    except Exception as error:
        if (OUT / "RESULT.json").exists() or RECEIPT.exists():
            raise RuntimeError("T1A_TERMINAL_PUBLICATION_NEEDS_AUDIT") from error
        outputs = {name: path for name, path in (
            ("code_freeze", OUT / "CODE_FREEZE.json"),
            ("build_audit", OUT / "BUILD_AUDIT.json"),
            ("unit_test_log", OUT / "unit_tests.log")) if path.exists()}
        print(json.dumps(finish(packet, status="FAILED_RUNTIME_FINAL", outputs=outputs,
            reason=f"{type(error).__name__}:{error}"), ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
