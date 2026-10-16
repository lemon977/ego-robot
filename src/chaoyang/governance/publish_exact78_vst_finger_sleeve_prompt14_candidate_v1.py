"""Guarded CAS publisher for one Prompt14 development attempt.

Check-only is read-only. Publication is unavailable until the separately
maintained algorithm builder emits the exact route. No route, CLI or shared
contract builder is changed by this module.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import secrets
import threading
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    load_json, now_iso, process_identity, publish_bundle, validate_artifact_ref,
    validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance import exact78_vst_finger_sleeve_prompt14_cpu_prep_v1 as prep
from chaoyang.governance import execute_exact78_vst_finger_sleeve_prompt14_candidate_v1 as executor


PACKET = executor.PACKET
INDEX = executor.INDEX
ATTEMPT = prep.ATTEMPT
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{prep.TASK_ID.upper()}.json"
ACTIVE = frozenset({"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"})
HEARTBEAT_SECONDS = 20
H20_NO_NEW_HEAVY_DISPATCH = datetime.fromisoformat("2026-10-01T14:47:25.116287+08:00")
H24_HARD_DEADLINE = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
WALL_SECONDS_MAX = 900


def _now() -> datetime:
    return datetime.now(H20_NO_NEW_HEAVY_DISPATCH.tzinfo)


def dispatch_gate(at: datetime | None = None) -> str:
    """Require the entire one-attempt wall budget to fit before H20."""
    current = at or _now()
    if current.tzinfo is None:
        raise RuntimeError("PROMPT14_NAIVE_DISPATCH_TIME")
    if current + timedelta(seconds=WALL_SECONDS_MAX) > H20_NO_NEW_HEAVY_DISPATCH:
        return "H20_HEAVY_DISPATCH_CLOSED"
    if current + timedelta(seconds=WALL_SECONDS_MAX) > H24_HARD_DEADLINE:
        return "H24_DEADLINE_EXCEEDED"
    return "DISPATCH_WINDOW_OPEN"


def packet_value() -> dict[str, Any]:
    """Build an exact one-attempt packet in memory; this function writes nothing."""
    inputs = prep.frozen_inputs()
    code = executor.runtime_code()
    now = _now()
    deadline = min(now + timedelta(seconds=WALL_SECONDS_MAX), H24_HARD_DEADLINE)
    identity = process_identity(os.getpid())
    if not identity["alive"] or type(identity["start_ticks"]) is not int:
        raise RuntimeError("PROMPT14_WRITER_IDENTITY_UNAVAILABLE")
    writer = {"pid": os.getpid(), "proc_start_ticks": identity["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": prep.TASK_ID}
    packet: dict[str, Any] = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": prep.TASK_ID,
        "objective": "Three frozen 0915 Cards001 sleeve-anchor text queries against sealed Mask13",
        "phase": "EXACT78_VST_FINGER_SLEEVE_PROMPT14_DEV_CANDIDATE",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": prep.TASK_ID.upper(),
        "execution_mode": "EXACT78_CANDIDATE",
        "candidate_operation": prep.OPERATION, "route_key": prep.ROUTE_KEY,
        "registered_at": now.isoformat(),
        "deadline_at": deadline.isoformat(),
        "source_session": prep.SOURCE_SESSION, "anchor_frames": list(prep.FRAMES),
        "single_change": {"role": prep.ROLE, "from": prep.OLD_PROMPT, "to": prep.NEW_PROMPT},
        "read_set": [str(Path(ref["path"]).relative_to(REPO_ROOT)) for ref in (
            inputs["sealed_mask13_result"], inputs["sealed_mask13_anchor"],
            inputs["sealed_mask13_packet"], inputs["sam_checkpoint"],
            inputs["sam_vendor_manifest"],
            *(row["refs"]["sam_input"] for row in inputs["anchors"]))],
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)),
                      str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)),
                      str(INDEX.relative_to(REPO_ROOT)), "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "MASK13_SEALED",
                          "EXACT_CANDIDATE_ROUTE_STAGED", "ONE_GPU",
                          "NO_0916", "NO_CLEAN", "NO_DELETION"],
        "required_outputs": ["candidate/FINGER_SLEEVE_ANCHORS.json",
                             "candidate/RESULT.json", "GPU_RECEIPT.json",
                             "GPU_CLEANUP_AUDIT.json", "RESULT.json",
                             "TERMINAL_RECEIPT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "model_calls_max": 3, "render_calls": 0, "clean_calls": 0,
                    "training_steps": 0, "wall_seconds_max": 900,
                    "session_attempts_max": 1},
        "stop_conditions": ["INPUT_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "GPU_LEASE_UNAVAILABLE", "THREE_MODEL_CALL_LIMIT",
                            "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": executor.route_contract(code)["claim_limit"],
        "writer": writer, "inputs": inputs, "runtime_code": code,
        "config": {"output_prob_thresh": 0.5, "instance_score_gate": 0.5},
        "commands": [executor.expected_command()],
    }
    packet["run_signature"] = executor.run_signature(packet)
    if errors := _validate_packet(packet):
        raise RuntimeError("PROMPT14_PACKET_INVALID:" + ";".join(errors))
    if len(packet["read_set"]) != 8 or len({row["frame"] for row in inputs["anchors"]}) != 3:
        raise RuntimeError("PROMPT14_READ_SET_OR_ANCHOR_GRID_DRIFT")
    return packet


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def route_readiness(packet: dict[str, Any]) -> str:
    """The common CAS can emit this route only when its builder already stages it."""
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    expected = executor.route_contract(packet["runtime_code"])
    published = load_json(ALGORITHM_CONTRACT_PATH).get("special_status_contracts", {}).get(prep.ROUTE_KEY)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH)).get("special_status_contracts", {}).get(prep.ROUTE_KEY)
    if staged != expected:
        return "NOT_ROUTABLE_UNTIL_ROUTE_STAGED"
    if published is not None and published != expected:
        raise RuntimeError("PROMPT14_PUBLISHED_ROUTE_DRIFT")
    if published is None and prep.OPERATION in _maintained_operations():
        raise RuntimeError("PROMPT14_UNPUBLISHED_OPERATION_ALREADY_MAINTAINED")
    if published is not None and prep.OPERATION not in _maintained_operations():
        raise RuntimeError("PROMPT14_PUBLISHED_OPERATION_NOT_MAINTAINED")
    return "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS"


def readiness(packet: dict[str, Any]) -> str:
    window = dispatch_gate()
    if window != "DISPATCH_WINDOW_OPEN":
        return window
    if any(path.exists() or path.is_symlink() for path in (ATTEMPT, PACKET, RECEIPT)):
        return "FRESH_ATTEMPT_REQUIRED"
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if (state.get("next_task")
            or any(row.get("status") in ACTIVE for row in state.get("tasks", []))
            or any(row.get("execution_allowed") is True for row in index.get("task_packets", []))):
        return "ACTIVE_CONFLICT"
    if any(row.get("task_id") == prep.TASK_ID for row in state.get("tasks", [])):
        return "PRIOR_TASK_REQUIRES_NEW_VERSION"
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("PROMPT14_RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    return route_readiness(packet)


def validate_for_publication(packet: dict[str, Any]) -> None:
    """Reject a modified in-memory packet before writing WRITER or Task Packet."""
    fresh = packet_value()
    frozen_keys = ("schema_version", "task_id", "objective", "phase", "plan_revision",
                   "execution_revision", "execution_mode", "candidate_operation", "route_key",
                   "source_session", "anchor_frames", "single_change", "read_set", "write_set",
                   "prerequisites", "required_outputs", "budgets", "stop_conditions",
                   "claim_limit", "inputs", "runtime_code", "config", "commands")
    if any(packet.get(key) != fresh.get(key) for key in frozen_keys):
        raise RuntimeError("PROMPT14_PACKET_VALUE_DRIFT")
    if packet.get("run_signature") != executor.run_signature(packet):
        raise RuntimeError("PROMPT14_RUN_SIGNATURE_DRIFT")
    try:
        registered = datetime.fromisoformat(packet["registered_at"])
        deadline = datetime.fromisoformat(packet["deadline_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("PROMPT14_DEADLINE_INVALID") from exc
    if (registered.tzinfo is None or deadline.tzinfo is None
            or (deadline - registered).total_seconds() != WALL_SECONDS_MAX
            or deadline > H24_HARD_DEADLINE
            or deadline > H20_NO_NEW_HEAVY_DISPATCH
            or registered > _now()
            or deadline <= _now()
            or dispatch_gate() != "DISPATCH_WINDOW_OPEN"):
        raise RuntimeError("PROMPT14_DEADLINE_DRIFT")
    writer = packet.get("writer", {})
    identity = process_identity(os.getpid())
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != identity.get("start_ticks")
            or writer.get("executor_epoch") != 1
            or writer.get("scope") != prep.TASK_ID
            or not isinstance(writer.get("fencing_token"), str)
            or len(writer["fencing_token"]) != 48):
        raise RuntimeError("PROMPT14_WRITER_FENCE_DRIFT")
    if errors := _validate_packet(packet):
        raise RuntimeError("PROMPT14_PACKET_SCHEMA_INVALID:" + ";".join(errors))


def register(packet: dict[str, Any]) -> None:
    validate_for_publication(packet)
    expected_revision = load_json(RECEIPT_PATH)["governance_revision"]
    status = readiness(packet)
    if status != "READY_TO_PUBLISH_ROUTE_AND_PACKET_SAME_CAS":
        raise RuntimeError("PROMPT14_REGISTER_BLOCKED:" + status)
    ATTEMPT.mkdir(parents=True)
    atomic_json(ATTEMPT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    stamp = now_iso()
    state["tasks"].append({
        "task_id": prep.TASK_ID, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": prep.SOURCE_SESSION,
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"], "gpu_id": 0,
        "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": prep.TASK_ID, "session": prep.SOURCE_SESSION,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "Three sleeve SAM calls or 900-second deadline",
    }
    if errors := validate_task_state(state):
        raise RuntimeError("PROMPT14_TASK_STATE_INVALID:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": prep.TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision=packet["execution_revision"] + "_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_REGISTERED",
                   expected_revision=expected_revision,
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def _active_writer(packet: dict[str, Any]) -> dict[str, Any]:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    state = load_json(TASK_STATE_PATH)
    rows = [row for row in state.get("tasks", []) if row.get("status") in ACTIVE]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != identity.get("start_ticks")
            or load_json(ATTEMPT / "WRITER.json") != writer
            or len(rows) != 1 or rows[0].get("task_id") != prep.TASK_ID
            or rows[0].get("run_signature") != packet["run_signature"]
            or rows[0].get("task_packet") != artifact_ref(PACKET)):
        raise RuntimeError("PROMPT14_DOUBLE_WRITER_OR_ACTIVE_STATE_DRIFT")
    return state


def heartbeat(packet: dict[str, Any]) -> None:
    expected_revision = load_json(RECEIPT_PATH)["governance_revision"]
    state = copy.deepcopy(_active_writer(packet))
    row = next(row for row in state["tasks"] if row["task_id"] == prep.TASK_ID)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    if errors := validate_task_state(state):
        raise RuntimeError("PROMPT14_HEARTBEAT_STATE_INVALID:" + ";".join(errors))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_HEARTBEAT",
                   expected_revision=expected_revision,
                   generator_path=Path(__file__))


def seal_terminal(packet: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    expected_revision = load_json(RECEIPT_PATH)["governance_revision"]
    state = copy.deepcopy(_active_writer(packet))
    if load_json(ATTEMPT / "RESULT.json") != result or RECEIPT.exists():
        raise RuntimeError("PROMPT14_TERMINAL_RESULT_DRIFT_OR_DUPLICATE")
    cleanup_proven = result.get("cleanup_proven") is True
    status = result.get("status")
    if status not in {"PASSED", "FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE"}:
        raise RuntimeError("PROMPT14_TERMINAL_STATUS_INVALID")
    if status == "PASSED" and not cleanup_proven:
        raise RuntimeError("PROMPT14_SUCCESS_WITHOUT_GPU_CLEANUP_PROOF")
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECEIPT, {
        "schema_version": "exact78-vst-finger-sleeve-prompt14-terminal-receipt-v1",
        "task_id": prep.TASK_ID, "status": status,
        "result": artifact_ref(ATTEMPT / "RESULT.json"),
        "deletion_authorized": False,
    })
    row = next(row for row in state["tasks"] if row["task_id"] == prep.TASK_ID)
    row.update(status=status if cleanup_proven else "FAILED_RUNTIME_RETRYABLE",
               pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(ATTEMPT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="YELLOW_CABLE_UNRESOLVED" if status == "PASSED" else "NOT_ADMITTED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    if cleanup_proven:
        state["next_task"] = None
    else:
        state["blockers"].append({
            "name": "EXACT78_PROMPT14_GPU_CLEANUP_UNPROVEN", "status": "OPEN",
            "scope": "ALL_NEW_TASK_PACKETS",
            "resolution": "Independently prove child process exit, physical GPU release, and lease release.",
        })
        state["next_task"] = {
            "task_id": prep.TASK_ID, "session": prep.SOURCE_SESSION,
            "prerequisites": ["PROVE_GPU_CHILD_EXIT", "PROVE_GPU_LEASE_RELEASE"],
            "expected_resource": "NONE_UNTIL_INDEPENDENT_CLEANUP_AUDIT",
            "stop_condition": "No new Task Packet until cleanup proof is published.",
        }
    if errors := validate_task_state(state):
        raise RuntimeError("PROMPT14_TERMINAL_STATE_INVALID:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item.get("task_id") == prep.TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision=packet["execution_revision"] +
                 ("_CLEANUP_HOLD" if not cleanup_proven else "_TERMINAL"),
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="BLOCKED_GPU_CLEANUP_PROOF" if not cleanup_proven else "PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type=packet["execution_revision"] + "_" + status,
                               expected_revision=expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(ATTEMPT / "RESULT.json")}


def publish_and_run(packet: dict[str, Any]) -> dict[str, Any]:
    register(packet)
    window = dispatch_gate()
    if window != "DISPATCH_WINDOW_OPEN":
        result = {
            "schema_version": "exact78-vst-finger-sleeve-prompt14-execution-v1",
            "task_id": prep.TASK_ID, "status": "FAILED_RUNTIME_FINAL",
            "execution_complete": False, "cleanup_proven": True,
            "run_signature": packet["run_signature"], "reason": window,
            "model_calls": 0, "gpu_called": False,
            "clean_called": False, "training_eligible": False,
            "next_action": "H20 closed before dispatch; terminalize without a GPU call",
        }
        atomic_json(ATTEMPT / "RESULT.json", result)
        return seal_terminal(packet, result)
    stop = threading.Event()
    heartbeat_errors: list[str] = []

    def beat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                heartbeat(packet)
            except Exception as exc:
                heartbeat_errors.append(f"{type(exc).__name__}:{exc}")
                return

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        try:
            result = executor.run_registered(packet)
        except Exception as exc:
            path = ATTEMPT / "RESULT.json"
            if path.exists():
                result = load_json(path)
            else:
                result = {
                    "schema_version": "exact78-vst-finger-sleeve-prompt14-execution-v1",
                    "task_id": prep.TASK_ID, "status": "FAILED_RUNTIME_FINAL",
                    "execution_complete": False, "cleanup_proven": False,
                    "run_signature": packet["run_signature"],
                    "reason": f"{type(exc).__name__}:{exc}",
                    "clean_called": False, "training_eligible": False,
                    "next_action": "Independent GPU child, physical GPU, and lease audit required",
                }
                atomic_json(path, result)
    finally:
        stop.set()
        thread.join(timeout=60)
    if thread.is_alive():
        raise RuntimeError("PROMPT14_HEARTBEAT_THREAD_NOT_STOPPED")
    if heartbeat_errors:
        atomic_json(ATTEMPT / "HEARTBEAT_HOLD.json", {
            "schema_version": "exact78-vst-finger-sleeve-prompt14-heartbeat-hold-v1",
            "task_id": prep.TASK_ID, "status": "BLOCKED_HEARTBEAT_AUDIT",
            "error": heartbeat_errors[0], "result": artifact_ref(ATTEMPT / "RESULT.json"),
            "next_action": "Independently audit active writer, GPU cleanup, and governance CAS",
        })
        raise RuntimeError("PROMPT14_HEARTBEAT_UNPROVEN")
    return seal_terminal(packet, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--publish-and-run", action="store_true")
    args = parser.parse_args(argv)
    packet = packet_value()
    if args.check_only:
        status = readiness(packet)
        print(json.dumps({"status": status, "task_id": prep.TASK_ID,
                          "anchor_frames": list(prep.FRAMES),
                          "runtime_code_refs": len(packet["runtime_code"]),
                          "run_signature": packet["run_signature"],
                          "registered": False, "gpu_called": False,
                          "heartbeat_seconds": HEARTBEAT_SECONDS}, sort_keys=True))
        return 0
    outcome = publish_and_run(packet)
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
