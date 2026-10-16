#!/usr/bin/env python3
"""Single-session, CPU-only Exact78 admission producer (not yet registered).

``--check-only`` is read-only. Direct ``--register-and-run`` is deliberately
disabled: a future formal task must register an algorithm-contract operation
and enter its worker through ``chaoyang run``. No model, GPU lease, production
route or 0916 data is used here.
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
import shlex
import signal
import subprocess
import time
from typing import Any

from chaoyang.cli.exact78_first_submit_bridge_candidate import plan_single
from chaoyang.governance.common import (
    AUTHORITY_PATH, BASELINE_REGISTRY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    DATA_ROOT, ColdStartInputError, validate_producer_reads,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
ALGORITHM_CONTRACT = REPO_ROOT / "docs/governance/ALGORITHM_CONTRACT.json"
OPERATION = "run_exact78_newraw_cpu_admission_candidate_v1"
ROUTE_KEY = "exact78_newraw_cpu_admission_candidate_v1"
WALL_SECONDS_MAX = 7200
ACTIVE = frozenset({"CLAIMED", "WAIT_GPU_RESOURCE", "RUNNING"})
FORBIDDEN = frozenset({
    "preprocess", "all_data", "trackingdata", "processed", "robot", "clean", "pico", "manus",
})


def _reject_derived(descriptor: dict[str, Any], *, allowed_root: Path) -> None:
    root = allowed_root.resolve(strict=True)
    session = Path(descriptor["session_root"]).resolve(strict=True)
    if not session.is_relative_to(root):
        raise ColdStartInputError("SOURCE_OUTSIDE_ALLOWED_ROOT")
    if any(part.lower() in FORBIDDEN for part in session.relative_to(root).parts):
        raise ColdStartInputError("OLD_DERIVED_SESSION_FORBIDDEN")
    for role, ref in descriptor["source_files"].items():
        raw_path = Path(ref["path"])
        if any(part.lower() in FORBIDDEN for part in raw_path.parts):
            raise ColdStartInputError(f"OLD_DERIVED_SOURCE_FORBIDDEN:{role}")
        path = raw_path.resolve(strict=True)
        if not path.is_relative_to(session):
            raise ColdStartInputError(f"SOURCE_ROLE_OUTSIDE_SESSION:{role}")
        if any(part.lower() in FORBIDDEN for part in path.relative_to(session).parts):
            raise ColdStartInputError(f"OLD_DERIVED_SOURCE_FORBIDDEN:{role}")


def _precheck_declared_paths(session: Path, *, allowed_root: Path) -> None:
    """Reject derived manifest roles before hashing or decoding those files."""
    if session.is_symlink():
        raise ColdStartInputError("SYMLINK_SESSION_FORBIDDEN")
    resolved = session.resolve(strict=True)
    root = allowed_root.resolve(strict=True)
    if not resolved.is_dir() or not resolved.is_relative_to(root):
        raise ColdStartInputError("SOURCE_OUTSIDE_ALLOWED_ROOT")
    if any(part.lower() in FORBIDDEN for part in resolved.relative_to(root).parts):
        raise ColdStartInputError("OLD_DERIVED_SESSION_FORBIDDEN")
    manifest_path = resolved / "clip_manifest.json"
    if manifest_path.is_file():
        if manifest_path.is_symlink():
            raise ColdStartInputError("SYMLINK_SOURCE_FORBIDDEN")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, dict):
            raise ColdStartInputError("TRACKER_MANIFEST_PATHS_INVALID")
        # The tracker manifest may retain historical sidecar fields such as
        # humanego_all_data. They are not producer inputs: inspect exactly the
        # roles that cold_start_guard selects, before it hashes/opens them.
        consumed = {
            "video": files.get("video"),
            "source_stereo_video": files.get("source_stereo_video"),
            "camera_params": files.get("camera_params", "camera_params.json"),
            "slam": files.get("slam", f"slam_trajectory_{resolved.name}.jsonl"),
            "clip_manifest": "clip_manifest.json",
        }
        for role, raw in consumed.items():
            if not isinstance(raw, str) or any(part.lower() in FORBIDDEN for part in Path(raw).parts):
                raise ColdStartInputError(f"OLD_DERIVED_SOURCE_FORBIDDEN:{role}")


def preflight(session: Path, *, allowed_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Reuse the CPU bridge, but reject derived sources before registration."""
    _precheck_declared_paths(session, allowed_root=allowed_root)
    prepared = plan_single(session, allowed_root=allowed_root)
    if prepared.get("status") != "BLOCKED_RUNNER_NOT_GENERALIZED":
        raise ColdStartInputError("FIRST_SUBMIT_BRIDGE_STATUS_DRIFT")
    descriptor = prepared["descriptor"]
    _reject_derived(descriptor, allowed_root=allowed_root)
    return prepared


def _paths(task_id: str) -> tuple[Path, Path, Path]:
    return (REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001",
            REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json",
            REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json")


def _index_copy() -> dict[str, Any]:
    value = load_json(INDEX)
    return {key: copy.deepcopy(item) for key, item in value.items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def _receipt_revision() -> int:
    receipt = load_json(RECEIPT_PATH)
    for key, path in (("task_state", TASK_STATE_PATH), ("task_packet_index_current", INDEX)):
        ref = receipt.get("files", {}).get(key)
        if not isinstance(ref, dict) or ref.get("path") != str(path):
            raise RuntimeError(f"GOVERNANCE_RECEIPT_BINDING_DRIFT:{key}")
        if errors := validate_artifact_ref(ref):
            raise RuntimeError(f"GOVERNANCE_RECEIPT_SHA_DRIFT:{key}:" + ";".join(errors))
    return int(receipt["governance_revision"])


def _registration_window() -> None:
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if (state.get("next_task")
            or any(row.get("status") in ACTIVE for row in state.get("tasks", []))
            or any(row.get("execution_allowed") is True for row in index.get("task_packets", []))):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("PRODUCTION_ROUTE_ALREADY_SELECTED")


def packet_value(prepared: dict[str, Any], *, script: Path = Path(__file__)) -> dict[str, Any]:
    """Build a non-runnable blueprint until the CLI operation is contracted."""
    descriptor = prepared["descriptor"]
    task_id = prepared["target_draft"]["task_id"]
    output, packet_path, receipt = _paths(task_id)
    runtime_code = [artifact_ref(path) for path in (
        script, REPO_ROOT / "src/chaoyang/cli/exact78_first_submit_bridge_candidate.py",
        REPO_ROOT / "src/chaoyang/cli/exact78_submit.py",
        REPO_ROOT / "src/chaoyang/cli/exact78_newraw_cpu_submit_candidate.py",
        REPO_ROOT / "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
        REPO_ROOT / "src/chaoyang/governance/common.py",
        REPO_ROOT / "src/chaoyang/governance/register_single_task_packet.py",
        REPO_ROOT / "src/chaoyang/cli/__init__.py",
        REPO_ROOT / "src/chaoyang/governance/exact78_operation_gate.py",
        REPO_ROOT / "src/chaoyang/paths.py",
    )]
    operation_path = REPO_ROOT / "src/chaoyang/ops" / f"{OPERATION}.py"
    if operation_path.is_file():
        runtime_code.append(artifact_ref(operation_path))
    source = descriptor["source_files"]
    inputs = {
        "source_session": descriptor["session_id"],
        "source_root": descriptor["session_root"],
        "source_files": source, "raw_video": descriptor["raw_video"],
        "submission_input_sha256": descriptor["submission_input_sha256"],
        "descriptor_sha256": prepared["descriptor_sha256"],
        "calibration": (source.get("camera_params") or source.get("raw/camera_params.json") or "ABSENT"),
    }
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": task_id}
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run "
               f"{OPERATION} --session {shlex.quote(descriptor['session_root'])} "
               f"--output-root {shlex.quote(str(output))}")
    config = {"cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
              "session_attempts_max": 1, "time_mode": descriptor["time_mode"],
              "physical_eye": descriptor["physical_eye"],
              "frame_count": descriptor["decoded_frames"],
              "world_pose_authority": False, "metric_authority": False}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime_code, "config": config,
        "command": command, "schema": "exact78-first-submit-cpu-producer-v1",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "Seal one new raw session's source, decoded frame map and diagnostic capabilities",
        "phase": "EXACT78_NEWRAW_CPU_ADMISSION", "plan_revision": "EXACT78_NEWRAW_CPU_ADMISSION_V1",
        "execution_revision": task_id.upper(), "execution_mode": "EXACT78_CANDIDATE",
        "deadline_at": (datetime.now().astimezone() + timedelta(seconds=WALL_SECONDS_MAX)).isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                     "src/chaoyang/cli/exact78_first_submit_bridge_candidate.py",
                     "src/chaoyang/governance/exact78_hawor_descriptor_adapter_candidate.py"],
        "write_set": [str(output.relative_to(REPO_ROOT)), str(packet_path.relative_to(REPO_ROOT)),
                      str(receipt.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "ONE_KNOWN_FORMAT_RAW_SESSION",
                          "SOURCE_SHA_PINNED", "EXTERNAL_DATA_READ_ONLY", "NO_MODEL_RUN",
                          "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["admission/DESCRIPTOR.json", "admission/FRAME_TIME_MAP.json",
                             "admission/SEAL.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": WALL_SECONDS_MAX,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "One-session CPU admission evidence only; no HaWoR, Robot, visual, product, training or production qualification.",
        "writer": writer, "inputs": inputs, "runtime_code": runtime_code,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    return packet


def _route_valid(contract: dict[str, Any], packet: dict[str, Any]) -> bool:
    """Require exact file refs for the candidate entry and its CPU code seam."""
    status_contracts = contract.get("special_status_contracts")
    route = status_contracts.get(ROUTE_KEY) if isinstance(status_contracts, dict) else None
    if not isinstance(route, dict):
        return False
    if (route.get("candidate_operation") != OPERATION
            or route.get("execution_status") != "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION"
            or route.get("task_packet_required") is not True
            or route.get("current_runnable_operation") is not None
            or route.get("output_schema") != "exact78-first-submit-cpu-admission-result-v1"
            or route.get("model_calls_authorized") != 0
            or route.get("render_calls_authorized") != 0
            or route.get("training_steps_authorized") != 0
            or not isinstance(route.get("code_closure"), list)):
        return False
    required = {str(REPO_ROOT / relative) for relative in (
        f"src/chaoyang/ops/{OPERATION}.py",
        "src/chaoyang/governance/execute_exact78_first_submit_cpu_candidate_v1.py",
        "src/chaoyang/cli/exact78_newraw_cpu_submit_candidate.py",
        "src/chaoyang/cli/exact78_first_submit_bridge_candidate.py",
        "src/chaoyang/cli/exact78_submit.py",
        "src/chaoyang/cli/__init__.py",
        "src/chaoyang/governance/exact78_operation_gate.py",
        "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    )}
    runtime = packet.get("runtime_code", [])
    closure = route["code_closure"]
    if not isinstance(runtime, list):
        return False
    for path in required:
        runtime_refs = [ref for ref in runtime if isinstance(ref, dict) and ref.get("path") == path]
        route_refs = [ref for ref in closure if isinstance(ref, dict) and ref.get("path") == path]
        if len(runtime_refs) != 1 or len(route_refs) != 1 or runtime_refs[0] != route_refs[0]:
            return False
    return True


def _assert_cli_admission_ready(packet: dict[str, Any]) -> str:
    """Permit published or same-CAS staged candidate route, never an absent one."""
    operation_path = REPO_ROOT / "src/chaoyang/ops" / f"{OPERATION}.py"
    if not operation_path.is_file():
        raise RuntimeError("ALGORITHM_CONTRACT_CLI_OPERATION_ABSENT")
    operation_ref = artifact_ref(operation_path)
    if operation_ref not in packet.get("runtime_code", []):
        raise RuntimeError("ALGORITHM_CONTRACT_CLI_CODE_NOT_PINNED")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("ALGORITHM_CONTRACT_CLI_CODE_SHA_DRIFT:" + ";".join(errors))
    published = load_json(ALGORITHM_CONTRACT)
    if _route_valid(published, packet):
        readiness = "PUBLISHED_CPU_CANDIDATE_ROUTE"
    else:
        from chaoyang.cli import _maintained_operations

        status_contracts = published.get("special_status_contracts", {})
        if (isinstance(status_contracts, dict) and ROUTE_KEY in status_contracts
                or OPERATION in _maintained_operations()):
            raise RuntimeError("ALGORITHM_CONTRACT_PUBLISHED_ROUTE_DRIFT")
        from chaoyang.governance.current_r3_contracts import build_algorithm_contract

        staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                          load_json(AUTHORITY_PATH))
        if not _route_valid(staged, packet):
            raise RuntimeError("ALGORITHM_CONTRACT_CLI_ROUTE_NOT_STAGED")
        readiness = "READY_TO_PUBLISH_CPU_ROUTE_WITH_PACKET"
    expected = ["PYTHONPATH=src", "/usr/local/bin/python", "-B", "-m", "chaoyang.cli",
                "run", OPERATION, "--session", packet["inputs"]["source_root"],
                "--output-root", str(_paths(packet["task_id"])[0])]
    if (len(packet.get("commands", [])) != 1
            or shlex.split(packet["commands"][0]) != expected):
        raise RuntimeError("ALGORITHM_CONTRACT_CLI_COMMAND_DRIFT")
    return readiness


def register(packet: dict[str, Any]) -> None:
    """CAS-publish only after a future contract/CLI route is separately added."""
    _assert_cli_admission_ready(packet)
    _registration_window()
    task_id = packet["task_id"]
    if (not isinstance(task_id, str) or not task_id.startswith("exact78_newraw_")
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        raise RuntimeError("NEWRAW_CANDIDATE_PACKET_REQUIRED")
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    output, packet_path, receipt = _paths(task_id)
    if any(path.exists() for path in (output, packet_path, receipt)):
        raise RuntimeError("FRESH_TARGET_TASK_REQUIRED")
    if packet["writer"]["pid"] != os.getpid() or packet["writer"]["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]:
        raise RuntimeError("WRITER_IDENTITY_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    for role, ref in packet["inputs"]["source_files"].items():
        if errors := validate_artifact_ref(ref):
            raise RuntimeError(f"SOURCE_SHA_DRIFT:{role}:" + ";".join(errors))
    _reject_derived({"session_root": packet["inputs"]["source_root"],
                     "source_files": packet["inputs"]["source_files"]}, allowed_root=DATA_ROOT)
    revision = _receipt_revision()
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_copy()
    output.mkdir(parents=True)
    atomic_json(output / "WRITER.json", packet["writer"])
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    stamp = now_iso()
    state["tasks"].append({
        "task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": packet["inputs"]["source_session"],
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(packet_path),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": task_id, "session": packet["inputs"]["source_session"],
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "one immutable admission result or deadline",
    }
    if errors := validate_task_state(state):
        raise RuntimeError("TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(packet_path)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision=packet["execution_revision"] + "_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_REGISTERED",
                   expected_revision=revision,
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def _fence(packet: dict[str, Any], *, sources: bool = False,
           allow_writer_descendant: bool = False) -> None:
    task_id = packet["task_id"]
    output, packet_path, _ = _paths(task_id)
    writer = packet["writer"]
    if allow_writer_descendant:
        from chaoyang.governance.exact78_operation_gate import _caller_is_writer_descendant
        caller_valid = _caller_is_writer_descendant(
            writer["pid"], writer["proc_start_ticks"], os.getpid())
    else:
        caller_valid = (writer["pid"] == os.getpid()
                        and writer["proc_start_ticks"] == process_identity(os.getpid())["start_ticks"])
    if not caller_valid or load_json(output / "WRITER.json") != writer:
        raise RuntimeError("WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    row = [item for item in state["tasks"] if item["task_id"] == task_id]
    if (len(row) != 1 or row[0]["status"] != "RUNNING"
            or row[0]["run_signature"] != packet["run_signature"]
            or (state.get("next_task") or {}).get("task_id") != task_id):
        raise RuntimeError("TASK_STATE_FENCE_DRIFT")
    allowed = [item for item in load_json(INDEX)["task_packets"]
               if item.get("execution_allowed") is True]
    if (len(allowed) != 1 or allowed[0]["task_id"] != task_id
            or allowed[0]["packet_sha256"] != artifact_ref(packet_path)["sha256"]):
        raise RuntimeError("PACKET_ROUTING_FENCE_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    if sources:
        for role, ref in packet["inputs"]["source_files"].items():
            if errors := validate_artifact_ref(ref):
                raise RuntimeError(f"SOURCE_SHA_DRIFT:{role}:" + ";".join(errors))


def heartbeat(packet: dict[str, Any]) -> None:
    _fence(packet)
    revision = _receipt_revision()
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_HEARTBEAT",
                   expected_revision=revision,
                   generator_path=Path(__file__))


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"IMMUTABLE_OUTPUT_ALREADY_EXISTS:{path}")
    atomic_json(path, value)


def produce_admission(prepared: dict[str, Any], packet: dict[str, Any],
                      *, cli_worker: bool = False) -> dict[str, Any]:
    """Write only the three immutable CPU admission artifacts in this attempt."""
    _fence(packet, sources=True, allow_writer_descendant=cli_worker)
    if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
        raise RuntimeError("CPU_ADMISSION_DEADLINE_EXPIRED")
    descriptor = prepared["descriptor"]
    if (packet["inputs"]["descriptor_sha256"] != prepared["descriptor_sha256"]
            or packet["inputs"]["source_files"] != descriptor["source_files"]
            or packet["inputs"]["submission_input_sha256"] != descriptor["submission_input_sha256"]):
        raise RuntimeError("PREPARED_DESCRIPTOR_PACKET_DRIFT")
    _reject_derived(descriptor, allowed_root=DATA_ROOT)
    source = {"kind": descriptor["source_kind"], "session_id": descriptor["session_id"],
              "session_root": descriptor["session_root"], "files": descriptor["source_files"]}
    validate_producer_reads(source, [Path(ref["path"]) for ref in descriptor["source_files"].values()])
    output, _, _ = _paths(packet["task_id"])
    admission = output / "admission"
    admission.mkdir(exist_ok=False)
    descriptor_path = admission / "DESCRIPTOR.json"
    map_path = admission / "FRAME_TIME_MAP.json"
    seal_path = admission / "SEAL.json"
    frame_map = {
        "schema_version": "exact78-first-submit-frame-map-v1",
        **{key: descriptor[key] for key in (
            "session_id", "source_kind", "physical_eye", "crop_xywh", "time_mode",
            "capture_to_decode_alignment", "decoded_frames", "raw_video", "frame_rows")},
    }
    _write_new_json(descriptor_path, descriptor)
    _write_new_json(map_path, frame_map)
    descriptor_ref, map_ref = artifact_ref(descriptor_path), artifact_ref(map_path)
    seal = {
        "schema_version": "exact78-first-submit-admission-seal-v1",
        "status": "SEALED_CPU_ADMISSION", "task_id": packet["task_id"],
        "descriptor": descriptor_ref, "frame_time_map": map_ref,
        "descriptor_sha256": prepared["descriptor_sha256"],
    }
    _write_new_json(seal_path, seal)
    _fence(packet, sources=True, allow_writer_descendant=cli_worker)
    return {"descriptor": descriptor_ref, "frame_time_map": map_ref,
            "seal": artifact_ref(seal_path)}


def finish(packet: dict[str, Any], admission: dict[str, Any] | None,
           *, error: str | None = None) -> dict[str, Any]:
    _fence(packet)
    revision = _receipt_revision()
    task_id = packet["task_id"]
    output, _, receipt = _paths(task_id)
    if (output / "RESULT.json").exists() or receipt.exists():
        raise RuntimeError("DUPLICATE_TERMINAL_RESULT")
    status = "PASSED" if admission is not None and error is None else "FAILED_RUNTIME_FINAL"
    source_sha_errors = [f"{role}:{';'.join(errors)}"
                         for role, ref in packet["inputs"]["source_files"].items()
                         if (errors := validate_artifact_ref(ref))]
    if admission is not None:
        for ref in admission.values():
            if errors := validate_artifact_ref(ref):
                status, error = "FAILED_RUNTIME_FINAL", "ADMISSION_REF_DRIFT:" + ";".join(errors)
    if source_sha_errors:
        status, error = "FAILED_RUNTIME_FINAL", "SOURCE_SHA_DRIFT_AFTER_PRODUCTION"
    if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
        status, error = "FAILED_RUNTIME_FINAL", "CPU_ADMISSION_DEADLINE_EXPIRED"
    result = {
        "schema_version": "exact78-first-submit-cpu-admission-result-v1",
        "task_id": task_id, "status": status, "failure_code": error,
        "source_session": packet["inputs"]["source_session"],
        "source_root": packet["inputs"]["source_root"],
        "source_files": packet["inputs"]["source_files"],
        "submission_input_sha256": packet["inputs"]["submission_input_sha256"],
        "descriptor_sha256": packet["inputs"]["descriptor_sha256"],
        "run_signature": packet["run_signature"], "admission": admission or {},
        "model_calls": 0, "gpu_count": 0, "source_write_attempted": False,
        "source_sha_stable_at_terminal": not source_sha_errors,
        "source_sha_errors": source_sha_errors,
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
        "general_operation": None, "next_action": "Register a separate HaWoR consumer only after terminal admission receipt is pinned",
    }
    _write_new_json(output / "RESULT.json", result)
    _write_new_json(receipt, {
        "schema_version": "exact78-first-submit-cpu-admission-receipt-v1",
        "task_id": task_id, "status": status,
        "result": artifact_ref(output / "RESULT.json"), "deletion_authorized": False,
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == task_id)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(output / "RESULT.json"),
               terminal_receipt=artifact_ref(receipt),
               quality="CPU_ADMISSION_ONLY" if status == "PASSED" else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDIT")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == task_id:
            item["execution_allowed"] = False
    index.update(packet_revision=packet["execution_revision"] + "_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_" + status,
                   expected_revision=revision,
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return result


def run_registered_child(packet: dict[str, Any]) -> dict[str, Any]:
    """Writer-owned orchestration; the child enters only through gated CLI."""
    _fence(packet, sources=True)
    attempt, _, _ = _paths(packet["task_id"])
    cache = attempt / "cache"
    cache.mkdir(exist_ok=False)
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1", "TMPDIR": str(cache), "TMP": str(cache),
        "TEMP": str(cache), "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "huggingface"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "",
    })
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run", OPERATION,
               "--session", packet["inputs"]["source_root"],
               "--output-root", str(attempt)]
    log_path = attempt / "worker.log"
    child: subprocess.Popen | None = None
    error: str | None = None
    admission: dict[str, Any] | None = None
    with log_path.open("x", encoding="utf-8") as log:
        try:
            child = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            last_beat = time.monotonic()
            while child.poll() is None:
                if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=15)
                    error = "CPU_ADMISSION_DEADLINE_EXPIRED"
                    break
                if time.monotonic() - last_beat >= 25:
                    heartbeat(packet)
                    last_beat = time.monotonic()
                time.sleep(1)
            if error is None and child.returncode != 0:
                error = f"CPU_ADMISSION_CLI_EXIT_{child.returncode}"
            if error is None:
                stage = load_json(attempt / "ADMISSION_STAGE.json")
                admission = stage.get("admission")
                if (stage.get("schema_version") != "exact78-first-submit-cpu-admission-stage-v1"
                        or stage.get("task_id") != packet["task_id"]
                        or stage.get("source_session") != packet["inputs"]["source_session"]
                        or stage.get("descriptor_sha256") != packet["inputs"]["descriptor_sha256"]
                        or stage.get("status") != "SEALED_CPU_ADMISSION_STAGE"
                        or not isinstance(admission, dict)
                        or set(admission) != {"descriptor", "frame_time_map", "seal"}):
                    raise RuntimeError("CPU_ADMISSION_STAGE_DRIFT")
                for ref in admission.values():
                    if errors := validate_artifact_ref(ref):
                        raise RuntimeError("CPU_ADMISSION_STAGE_REF_DRIFT:" + ";".join(errors))
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=15)
    return finish(packet, admission if error is None else None, error=error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    if args.register_and_run:
        raise RuntimeError("DIRECT_PRODUCER_EXECUTION_FORBIDDEN_USE_CHAOYANG_RUN")
    prepared = preflight(args.session)
    if args.check_only:
        descriptor = prepared["descriptor"]
        print(json.dumps({
            "status": "DRY_RUN_CPU_ADMISSION_READY_NOT_REGISTERED",
            "task_id": prepared["target_draft"]["task_id"],
            "source_session": descriptor["session_id"],
            "source_kind": descriptor["source_kind"],
            "decoded_frames": descriptor["decoded_frames"],
            "time_mode": descriptor["time_mode"],
            "physical_eye": descriptor["physical_eye"],
            "descriptor_sha256": prepared["descriptor_sha256"],
            "registration_started": False, "model_calls": 0,
            "visual_qualified": False, "training_eligible": False,
        }, ensure_ascii=False, sort_keys=True))
        return 0
    raise AssertionError("DIRECT_EXECUTION_GUARD_UNREACHABLE")


if __name__ == "__main__":
    raise SystemExit(main())
