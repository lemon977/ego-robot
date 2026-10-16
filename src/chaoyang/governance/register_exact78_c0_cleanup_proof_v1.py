"""Register one bounded C0 proof, then exec its worker in the same fenced PID.

The default is no-op: choose --check-only or --register-and-run explicitly.
The C0 worker never deletes candidates. A future C1 needs its own authority.
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
import sys
from typing import Any

from chaoyang.governance import execute_exact78_c0_cleanup_proof_v1 as worker
from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


SCRIPT = Path(__file__)
TASK = worker.TASK
OUT = worker.OUT
PACKET = worker.PACKET
RECEIPT = worker.RECEIPT
INDEX = worker.INDEX
PLAN_DIR = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930"
PLAN = PLAN_DIR / "00_EXECUTION.md"
DRAFT = PLAN_DIR / "C0_TASK_PACKET_DRAFT.json"
MANIFEST = PLAN_DIR / "C0_CANDIDATES_V1.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
T0_CORRECTION = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001/RESULT.json"
VISUAL_PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
T0_CORRECTION_SHA = "7cd873022e3a72b05fa88701cccb76dec7fda5321c4eed11b2aa14f69b95314c"
VISUAL_PARENT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"
COMMAND = (
    "PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_c0_cleanup_proof_v1 "
    "--packet tasks/current/exact78_raw_to_robot_c0_cleanup_proof_v1_20260930/TASK_PACKET.json"
)
ACTIVE_STATUSES = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})


def validated_governance_revision() -> int:
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1", TMPDIR=str(REPO_ROOT / "_run"),
                       XDG_CACHE_HOME=str(REPO_ROOT / "_run/caches"))
    completed = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError("GOVERNANCE_VALIDATION_FAILED:" + completed.stdout[-2000:] + completed.stderr[-2000:])
    result = json.loads(completed.stdout)
    if result.get("status") != "PASS":
        raise RuntimeError("GOVERNANCE_NOT_PASS")
    return int(result["revision"])


def current_idle(expected_revision: int) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != expected_revision:
        raise RuntimeError("GOVERNANCE_REVISION_CHANGED")
    index_ref = receipt.get("files", {}).get("task_packet_index_current")
    if not isinstance(index_ref, dict) or validate_artifact_ref(index_ref):
        raise RuntimeError("INDEX_RECEIPT_REF_DRIFT")
    if Path(index_ref["path"]) != INDEX.resolve(strict=True):
        raise RuntimeError("INDEX_PATH_DRIFT")
    state = load_json(TASK_STATE_PATH)
    index = load_json(INDEX)
    if (state.get("next_task") is not None
            or any(row.get("status") in ACTIVE_STATUSES for row in state["tasks"])
            or any(row.get("execution_allowed") for row in index["task_packets"])):
        raise RuntimeError("ACTIVE_TASK_OR_PUBLISHER_CONFLICT")
    if any(row.get("task_id") == TASK for row in state["tasks"] + index["task_packets"]):
        raise RuntimeError("C0_TASK_ID_ALREADY_REGISTERED")
    if any(path.exists() for path in (OUT, OUT.parent.parent, PACKET, PACKET.parent, RECEIPT)):
        raise RuntimeError("C0_ATTEMPT_OR_RECEIPT_ALREADY_EXISTS")
    return state, index


def checked_inputs() -> tuple[dict[str, Any], dict[str, Any], datetime, dict[str, int]]:
    correction = load_json(T0_CORRECTION)
    if (artifact_ref(T0_CORRECTION)["sha256"] != T0_CORRECTION_SHA
            or correction.get("status") != "PASSED"):
        raise RuntimeError("T0_CORRECTION_REF_DRIFT")
    deadline = datetime.fromisoformat(correction["campaign_deadline_at"])
    if deadline <= datetime.now().astimezone():
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    if artifact_ref(VISUAL_PARENT)["sha256"] != VISUAL_PARENT_SHA:
        raise RuntimeError("VISUAL_PARENT_SHA_DRIFT")
    route = load_json(ROUTE)
    if route["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_OPERATION_ALREADY_PROMOTED")
    draft = load_json(DRAFT)
    if (draft.get("task_id") != TASK or draft.get("execution_allowed") is not False
            or draft.get("draft_status") != "NOT_REGISTERED_NOT_EXECUTABLE"
            or draft.get("commands") != [COMMAND]
            or draft.get("write_set") != [
                str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance",
            ]):
        raise RuntimeError("C0_DRAFT_SCOPE_OR_COMMAND_DRIFT")
    scope = draft["reference_scan_scope"]
    licenses = {
        "third_party_notices": artifact_ref(REPO_ROOT / "THIRD_PARTY_NOTICES.md")
        if (REPO_ROOT / "THIRD_PARTY_NOTICES.md").is_file() else "ABSENT",
        "project_license_files": [artifact_ref(path) for path in sorted(REPO_ROOT.glob("LICENSE*"))
                                  if path.is_file()],
    }
    inputs = {
        "candidate_manifest": artifact_ref(MANIFEST), "plan": artifact_ref(PLAN),
        "packet_draft": artifact_ref(DRAFT), "route": artifact_ref(ROUTE),
        "t0_correction": artifact_ref(T0_CORRECTION),
        "visual_parent": artifact_ref(VISUAL_PARENT),
        "registration_code": artifact_ref(SCRIPT), "license_sources": licenses,
    }
    candidates = worker.candidate_rows({"inputs": {"candidate_manifest": inputs["candidate_manifest"]}})
    inventory = {"target_count": len(candidates),
                 "target_bytes": sum(row["file"]["bytes"] for row in candidates),
                 "reference_scope_gap_count": len(worker.reference_scope_gaps(scope))}
    return inputs, scope, deadline, inventory


def packet_value(inputs: dict[str, Any], scope: dict[str, Any],
                 campaign_deadline: datetime) -> dict[str, Any]:
    now = datetime.now().astimezone()
    identity = process_identity(os.getpid())
    writer = {"pid": os.getpid(), "proc_start_ticks": identity["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    runtime = [artifact_ref(worker.SCRIPT)]
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "reference_scan_scope": scope,
        "schema": "exact78-c0-proof-v1", "weights": "NOT_CONSUMED",
        "calibration": "NOT_CONSUMED",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Read-only per-file proof for exact project-local diagnostic media and generated cache",
        "phase": "EXACT78_RAW_TO_ROBOT_C0_CLEANUP_PROOF",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_C0_PROOF_V1", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=2), campaign_deadline).isoformat(),
        "campaign_deadline_at": campaign_deadline.isoformat(),
        "read_set": [
            "AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
            str(MANIFEST.relative_to(REPO_ROOT)), str(worker.SCRIPT.relative_to(REPO_ROOT)),
            str(SCRIPT.relative_to(REPO_ROOT)), str(RECEIPT_PATH.relative_to(REPO_ROOT)),
            str(TASK_STATE_PATH.relative_to(REPO_ROOT)), str(INDEX.relative_to(REPO_ROOT)),
        ],
        "reference_scan_scope": copy.deepcopy(scope),
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), str(INDEX.relative_to(REPO_ROOT)),
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "EXACT_CANDIDATE_LIST",
                          "NO_DELETION", "PROJECT_LOCAL_ONLY", "0916_READ_ONLY"],
        "required_outputs": ["C0_PROOF.json", "RESULT.json", "terminal_receipt",
                             "post_validate_governance_PASS"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 7200, "session_attempts_max": 1},
        "stop_conditions": ["SHA_DRIFT", "DOUBLE_WRITER", "REFERENCE_SCAN_INCOMPLETE",
                            "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "C0 proof only; no deletion. C1 requires separate exact-target authority and complete proof.",
        "inputs": inputs, "runtime_code": runtime, "writer": writer,
        "run_signature": signature, "commands": [COMMAND],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("C0_PACKET_INVALID:" + ";".join(errors))
    if packet["commands"] != [COMMAND] or packet["runtime_code"] != [artifact_ref(worker.SCRIPT)]:
        raise RuntimeError("C0_WORKER_COMMAND_OR_CODE_DRIFT")
    return packet


def register(packet: dict[str, Any], *, expected_revision: int,
             state: dict[str, Any], index: dict[str, Any]) -> int:
    # A second read just before the first write makes the CAS/publisher boundary explicit.
    current_idle(expected_revision)
    OUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(PACKET, packet)
    packet_ref = artifact_ref(PACKET)
    now = now_iso()
    state = copy.deepcopy(state)
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "Exact78_project_local_cleanup", "updated_at": now, "heartbeat_at": now,
        "pid": packet["writer"]["pid"], "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": packet_ref, "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": TASK, "session": "Exact78_project_local_cleanup",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "one immutable C0 result or two-hour campaign-bounded deadline",
    }
    errors = validate_task_state(state)
    if errors:
        raise RuntimeError("C0_TASK_STATE_INVALID:" + ";".join(errors))
    index = {key: copy.deepcopy(value) for key, value in index.items()
             if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}
    index["task_packets"].append({
        "task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision="EXACT78_C0_PROOF_V1_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state, event_type="EXACT78_C0_PROOF_REGISTERED",
        expected_revision=expected_revision, generator_path=SCRIPT,
        task_packet_index_path=INDEX, task_packet_index_value=index,
    )
    return int(published["governance_revision"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    revision = validated_governance_revision()
    state, index = current_idle(revision)
    inputs, scope, campaign_deadline, inventory = checked_inputs()
    packet = packet_value(inputs, scope, campaign_deadline)
    if args.check_only:
        print(json.dumps({
            "status": "CHECK_ONLY_NO_WRITES", "governance_revision": revision,
            "task_id": TASK, "worker_command": packet["commands"][0],
            "worker_code": packet["runtime_code"][0],
            "registration_code": inputs["registration_code"],
            "run_signature_preview": packet["run_signature"],
            "campaign_deadline_at": packet["campaign_deadline_at"],
            "task_deadline_preview": packet["deadline_at"], **inventory,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    published_revision = register(packet, expected_revision=revision, state=state, index=index)
    if load_json(PACKET) != packet or load_json(OUT / "WRITER.json") != packet["writer"]:
        raise RuntimeError("C0_POST_REGISTER_PACKET_OR_WRITER_DRIFT")
    if process_identity(os.getpid())["start_ticks"] != packet["writer"]["proc_start_ticks"]:
        raise RuntimeError("C0_POST_REGISTER_PID_FENCE_DRIFT")
    print(json.dumps({"status": "REGISTERED_EXECING_SAME_PID", "task_id": TASK,
                      "governance_revision": published_revision,
                      "packet": artifact_ref(PACKET)}, ensure_ascii=False), flush=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       PYTHONNOUSERSITE="1", TMPDIR=str(OUT), TMP=str(OUT), TEMP=str(OUT),
                       XDG_CACHE_HOME=str(OUT / "cache"))
    os.chdir(REPO_ROOT)
    os.execvpe(sys.executable, [sys.executable, "-B", "-m",
        "chaoyang.governance.execute_exact78_c0_cleanup_proof_v1",
        "--packet", str(PACKET.relative_to(REPO_ROOT))], environment)
    raise AssertionError("execvpe returned")


if __name__ == "__main__":
    raise SystemExit(main())
