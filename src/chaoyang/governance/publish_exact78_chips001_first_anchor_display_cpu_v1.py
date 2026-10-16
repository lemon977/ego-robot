"""Fenced route-plus-packet CAS publisher, prepared but not invoked.

The main contract builder must stage ROUTE_KEY first. This module never edits
that builder or the CLI; check_only only reads and reports readiness.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    load_json, now_iso, publish_bundle, validate_artifact_ref, validate_task_state,
)


ROUTE_KEY = "exact78_0901_chips001_first_anchor_display_cpu_v1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SCRIPT = Path(__file__)


def route_contract(runtime_code: list[dict[str, Any]]) -> dict[str, Any]:
    from chaoyang.ops import run_exact78_chips001_first_anchor_display_cpu_v1 as operation

    return {
        "execution_status": "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION",
        "candidate_operation": operation.OPERATION,
        "task_packet_required": True,
        "code_closure": runtime_code,
        "output_schema": "exact78-chips001-display-cpu-execution-result-v1",
        "model_calls_authorized": 0, "render_calls_authorized": 0,
        "training_steps_authorized": 0,
        "current_runnable_operation": None,
        "claim_limit": "0901 Chips001 CPU display q/source diagnostic only; no video, visual, motion, control, product or training qualification.",
    }


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _index_copy() -> dict[str, Any]:
    current = load_json(INDEX)
    return {key: copy.deepcopy(value) for key, value in current.items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def route_readiness(packet: dict[str, Any]) -> str:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    published = load_json(ALGORITHM_CONTRACT_PATH)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    expected = route_contract(packet["runtime_code"])
    if published.get("special_status_contracts", {}).get(ROUTE_KEY) is not None:
        raise RuntimeError("CHIPS001_ROUTE_ALREADY_PUBLISHED_SEPARATELY")
    if (staged.get("special_status_contracts", {}).get(ROUTE_KEY) != expected
            or expected["candidate_operation"] in _maintained_operations()):
        raise RuntimeError("CHIPS001_ROUTE_NOT_STAGED_IN_CURRENT_R3_AND_CLI")
    return "STAGED_ROUTE_READY_FOR_SAME_CAS_AS_PACKET"


def readiness(packet: dict[str, Any]) -> str:
    from chaoyang.governance import execute_exact78_chips001_first_anchor_display_cpu_v1 as executor

    route_readiness(packet)
    if any(path.exists() for path in (executor.ATTEMPT, executor.PACKET_PATH,
                                      executor.RECEIPT_PATH)):
        raise RuntimeError("CHIPS001_FRESH_ATTEMPT_REQUIRED")
    state = load_json(TASK_STATE_PATH)
    if (state.get("next_task") or any(row.get("status") in
            {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"} for row in state.get("tasks", []))):
        raise RuntimeError("CHIPS001_ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(INDEX).get("task_packets", [])):
        raise RuntimeError("CHIPS001_ACTIVE_ROUTE_CONFLICT")
    return "READY_FOR_ROUTE_PACKET_SAME_CAS"


def check_only() -> dict[str, Any]:
    from chaoyang.governance import execute_exact78_chips001_first_anchor_display_cpu_v1 as executor

    draft = executor.packet_value()
    try:
        status = readiness(draft)
    except RuntimeError as error:
        status = str(error)
    return {"status": "PREP_ONLY_NOT_REGISTERED_NOT_EXECUTED",
            "route_readiness": status, "task_id": executor.TASK_ID,
            "run_signature": draft["run_signature"],
            "route_key": ROUTE_KEY,
            "active_packet_written": False, "candidate_executed": False}


def publish() -> dict[str, Any]:
    """Future explicit call only: packet/index/route in one governance CAS."""
    from chaoyang.governance import execute_exact78_chips001_first_anchor_display_cpu_v1 as executor
    from chaoyang.ops import run_exact78_chips001_first_anchor_display_cpu_v1 as operation

    packet = executor.finalize_packet(executor.packet_value())
    readiness(packet)
    if packet["inputs"] != operation._refs():
        raise RuntimeError("CHIPS001_PREPUBLISH_INPUT_REF_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("CHIPS001_PREPUBLISH_CODE_REF_DRIFT:" + ";".join(errors))
    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    receipt_revision = load_json(RECEIPT_PATH)["governance_revision"]
    stamp = now_iso()
    packet_ref = _virtual_ref(executor.PACKET_PATH, packet)
    state["tasks"].append({
        "task_id": executor.TASK_ID, "phase": packet["phase"],
        "attempt": 1, "status": "RUNNING", "session": operation.SESSION,
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": packet_ref,
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": executor.TASK_ID, "session": operation.SESSION,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "TWO_CPU_THREADS_NO_GPU",
        "stop_condition": "one 799-frame CPU attempt or 900-second deadline",
    }
    if errors := validate_task_state(state):
        raise RuntimeError("CHIPS001_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": executor.TASK_ID,
        "packet_path": str(executor.PACKET_PATH.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE",
        "execution_allowed": True,
    })
    index.update(packet_revision="EXACT78_0901_CHIPS001_FIRST_ANCHOR_CPU_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    executor.ATTEMPT.mkdir(parents=True, exist_ok=False)
    atomic_json(executor.ATTEMPT / "WRITER.json", packet["writer"])
    executor.PACKET_PATH.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(executor.PACKET_PATH, packet)
    if artifact_ref(executor.PACKET_PATH)["sha256"] != packet_ref["sha256"]:
        raise RuntimeError("CHIPS001_PACKET_VIRTUAL_SHA_DRIFT")
    published = publish_bundle(
        authority, state, event_type="EXACT78_0901_CHIPS001_FIRST_ANCHOR_CPU_REGISTERED",
        expected_revision=receipt_revision, generator_path=SCRIPT,
        task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"packet": packet, "governance_revision": published["governance_revision"]}


def main() -> int:
    # No --publish argument is exposed while this remains PREP_ONLY.
    print(json.dumps(check_only(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
