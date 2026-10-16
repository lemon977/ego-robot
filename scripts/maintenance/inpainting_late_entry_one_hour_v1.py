"""Finite two-session automatic V1 seed model wrapper on existing governance.

This is an experimental runner, not an additional queue or production entry.
The sealed Source and RGB TEXT proposals are allowed inputs; historical manual
seeds, POINT masks, Clean, q and placement are forbidden producer inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.inpainting_candidate.contracts import verify_ref, verify_sealed_stage
from chaoyang.inpainting_candidate.resources import (
    CANDIDATE_CARD_LOCK_ROOT, DeviceMapping, GpuCardLease, ResourceError, read_gpu_inventory)

TASK_ID = "inpainting_late_entry_mask_a02_20261013"
OPERATION = "run_inpainting_late_entry_mask_v1"
SCOPE = "LATE_ENTRY_DEVICE_MASK_ONE_HOUR"
SPEC_SCHEMA = "inpainting-late-entry-model-spec-v1"
MODEL_RESULT_SCHEMA = "inpainting-late-entry-mask-model-result-v1"
CALL_UNIT = "SAM_ADAPTER_INFERENCE_API_INVOCATION"
INFRA_PATH = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"


def _infra():
    definition = importlib.util.spec_from_file_location("_auto_seed_existing_publisher", INFRA_PATH)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def _model_helpers():
    path = REPO_ROOT / "scripts/maintenance/inpainting_candidate_model_task_v1.py"
    definition = importlib.util.spec_from_file_location("_auto_seed_existing_model_helpers", path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def _require(condition, code):
    if not condition:
        raise ValueError(code)


def _deadline(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(stamp.tzinfo is not None, "AUTO_SEED_TIMEZONE_REQUIRED")
    return stamp


def _output_capacity(spec):
    """Finite actual-write proof when the shared mount reports zero free blocks.

    This is not a free-space claim or a capacity reservation. I/O errors stop
    the finite worker. It cannot authorize a full-data run or external writes.
    """
    required=spec["budgets"]["output_bytes_estimate"]
    available=os.statvfs(REPO_ROOT)
    if available.f_bavail*available.f_frsize >= required:
        return
    reference=spec.get("inputs",{}).get("capacity_proof")
    _require(isinstance(reference,dict),"LATE_ENTRY_OUTPUT_CAPACITY_UNPROVEN")
    verify_ref(reference)
    proof=load_json(Path(reference['path']))
    _require(Path(reference['path'])==REPO_ROOT/'_run/inpainting_late_entry_cpu_20261013/attempt_0001/CAPACITY_PROOF.json'
        and proof.get('status')=='BOUNDED_WRITE_FSYNC_PASS_STATVFS_UNVERIFIED'
        and proof.get('actual_written_fsynced_bytes',0)>=required
        and proof.get('own_zero_scratch_released') is True
        and proof.get('files_deleted')==0
        and 0<=(datetime.now(timezone.utc)-_deadline(proof['completed_at'])).total_seconds()<1800,
        'LATE_ENTRY_RECENT_BOUND_ACTUAL_WRITE_PROOF_REQUIRED')


def validate_spec(value):
    _require(value.get("spec_schema", value.get("schema")) == SPEC_SCHEMA
             and value.get("task_id") == TASK_ID
             and value.get("experiment_scope") == SCOPE, "AUTO_SEED_SPEC_SCOPE_REQUIRED")
    rows = value.get("sessions", [])
    _require([row.get("session_key") for row in rows] == ["001", "050"]
             and [row.get("native_frames") for row in rows] == [238, 128],
             "AUTO_SEED_EXACT_TWO_FULL_SESSIONS_REQUIRED")
    bounds = value.get("budgets", {})
    _require(type(bounds.get("wall_seconds_max")) is int
             and 0 < bounds["wall_seconds_max"] <= 3600
             and bounds.get("gpu_count") == 1
             and bounds.get("cpu_threads_per_worker") == 2
             and bounds.get("model_calls_unit") == CALL_UNIT
             and type(bounds.get("model_calls_max")) is int
             and 0 < bounds["model_calls_max"] <= 40
             and bounds.get("training_steps_max", 0) == 0,
             "AUTO_SEED_FINITE_GPU_CALL_CPU_BOUNDS_REQUIRED")
    _require(_deadline(value["global_deadline_at"]) > datetime.now(timezone.utc),
             "AUTO_SEED_GLOBAL_TIMEBOX_EXHAUSTED")
    _require(value.get("runtime_code") and value.get("model_weights"),
             "AUTO_SEED_CODE_AND_WEIGHTS_REQUIRED")
    for row in rows:
        for name in ("source_receipt", "text_receipt"):
            _require(isinstance(row.get(name), dict), "AUTO_SEED_SOURCE_AND_TEXT_REQUIRED")
    for reference in [*value.get("inputs", {}).values(), *value["runtime_code"],
                      *value["model_weights"],
                      *[row[name] for row in rows for name in ("source_receipt", "text_receipt")]]:
        verify_ref(reference)
    for row in rows:
        source_path = Path(row["source_receipt"]["path"])
        source = load_json(source_path)
        verify_sealed_stage(source, source["component_signature"], output_root=source_path.parent)
        _require(source.get("native_frames") == row["native_frames"],
                 "AUTO_SEED_SOURCE_NATIVE_COUNT_DRIFT")
        _require(Path(row["text_receipt"]["path"]).name == "SAM_TEXT.json",
                 "AUTO_SEED_ONLY_TEXT_PROPOSALS_ALLOWED")


def authorize_operation(args):
    """Reject before any algorithm import unless current candidate is fenced."""
    _require(len(args) == 2 and args[0] == "--packet", "AUTO_SEED_EXACT_PACKET_ARGUMENT_REQUIRED")
    packet_path = Path(args[1]).resolve(strict=True)
    _require(packet_path == REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json",
             "AUTO_SEED_PACKET_PATH_REQUIRED")
    infra = _infra()
    receipt, state, index = infra._snapshot()
    packet = load_json(packet_path)
    row, _ = _model_helpers().own_row(packet, state, index)
    _require(row.get("status") == "RUNNING" and packet.get("operation") == OPERATION
             and packet.get("experiment_scope") == SCOPE
             and packet.get("algorithm_execution_allowed") is True
             and packet.get("deletion_allowed") is False
             and packet.get("model_stages") == ["Mask"], "AUTO_SEED_RUNNING_MASK_PACKET_REQUIRED")
    expected = [sys.executable, "-B", "-m", "chaoyang.cli", "run", OPERATION,
                "--packet", str(packet_path)]
    _require(packet.get("command_argv") == expected
             and packet.get("commands") == [shlex.join(expected)], "AUTO_SEED_COMMAND_DRIFT")
    _require(datetime.now(timezone.utc) < _deadline(packet["deadline_at"])
             <= _deadline(packet["global_deadline_at"]), "AUTO_SEED_FINITE_DEADLINE_REQUIRED")
    from chaoyang.governance.exact78_operation_gate import _caller_is_writer_descendant
    writer = packet["writer"]
    _require(_caller_is_writer_descendant(writer["pid"], writer["proc_start_ticks"], os.getpid()),
             "AUTO_SEED_CALLER_NOT_WRITER_DESCENDANT")
    contract_ref = receipt["files"]["algorithm_contract"]
    verify_ref(contract_ref)
    contract = load_json(Path(contract_ref["path"]))
    from chaoyang.governance.inpainting_candidate_registration import (
        VISUAL_ROUTE_KEY, visual_route_contract)
    route = contract.get("special_status_contracts", {}).get(VISUAL_ROUTE_KEY)
    _require(route == visual_route_contract(REPO_ROOT).get(VISUAL_ROUTE_KEY)
             and route.get("late_entry_operation") == OPERATION
             and all(ref in packet["runtime_code"] for ref in route["code_closure"]),
             "AUTO_SEED_REGISTERED_ROUTE_OR_CLOSURE_REQUIRED")
    for reference in packet["runtime_code"]:
        _require(Path(reference["path"]).resolve(strict=True).is_relative_to(REPO_ROOT),
                 "AUTO_SEED_CODE_OUTSIDE_PROJECT")
        verify_ref(reference)
    for reference in packet["model_weights"]:
        verify_ref(reference)
    verify_ref(packet["inputs"]["spec"])
    spec = load_json(Path(packet["inputs"]["spec"]["path"]))
    validate_spec(spec)
    for name in ("sessions", "runtime_code", "model_weights", "config", "calibration",
                 "schema", "budgets", "global_deadline_at"):
        _require(packet[name] == spec[name], "AUTO_SEED_PACKET_SPEC_DRIFT:" + name)
    return packet


def validate_model_result(packet, helpers):
    root = Path(packet["output_dir"])
    reference = artifact_ref(root / "MODEL_RESULT.json")
    value = load_json(Path(reference["path"]))
    rows = value.get("sessions", [])
    _require(value.get("schema") == MODEL_RESULT_SCHEMA
             and value.get("task_id") == TASK_ID
             and value.get("run_signature") == packet["run_signature"]
             and value.get("experiment_scope") == SCOPE
             and value.get("planned_sessions") == 2
             and value.get("completed_mask_sessions") == 2
             and value.get("candidate_adopted") is False
             and value.get("visual_quality") == "NOT_REVIEWED"
             and [row.get("session_key") for row in rows] == ["001", "050"],
             "AUTO_SEED_MODEL_BATCH_BINDING_OR_COMPLETENESS_FAILED")
    accounting = value.get("model_call_accounting", {})
    calls = accounting.get("attempted_api_calls")
    _require(accounting.get("unit") == CALL_UNIT
             and accounting.get("maximum") == packet["budgets"]["model_calls_max"]
             and type(calls) is int and 0 < calls <= packet["budgets"]["model_calls_max"],
             "AUTO_SEED_MODEL_CALL_ACCOUNTING_FAILED")
    for planned, row in zip(packet["sessions"], rows):
        count = planned["native_frames"]
        _require(row.get("source") == planned["source_receipt"]
                 and row.get("native_frames") == count
                 and row.get("manual_seed_reads") == 0
                 and row.get("execution") == "COMPLETE_MASK_ONLY", "AUTO_SEED_SESSION_IDENTITY_FAILED")
        mask_ref = row["mask"]
        mask_path = root / "sessions" / planned["session_key"] / "Mask/RECEIPT.json"
        _require(Path(mask_ref["path"]) == mask_path, "AUTO_SEED_MASK_PATH_DRIFT")
        verify_ref(mask_ref)
        stage = load_json(mask_path)
        signature = planned.get("mask_signature", stage["component_signature"])
        verify_sealed_stage(stage, signature, output_root=mask_path.parent)
        _require(stage.get("stage") == "Mask" and stage.get("status") == "SEALED_COMPLETE"
                 and stage.get("native_input_frames") == count
                 and stage.get("native_output_frames") == count
                 and stage.get("producer_manual_seed_reads") == 0,
                 "AUTO_SEED_MASK_COUNTS_OR_FORBIDDEN_INPUT_FAILED")
        array_refs = [ref for ref in stage["outputs"] if Path(ref["path"]).name == "MASK_ARRAYS.npz"]
        _require(len(array_refs) == 1, "AUTO_SEED_ACTUAL_MASK_ARRAYS_REQUIRED")
        import numpy as np
        with np.load(array_refs[0]["path"], allow_pickle=False) as arrays:
            _require(bool(arrays.files), "AUTO_SEED_EMPTY_MASK_ARRAYS")
            for name in arrays.files:
                _require(arrays[name].shape[0] == count, "AUTO_SEED_MASK_ARRAY_FRAME_COUNT:" + name)
        video = row["review"]["video"]
        _require(Path(video["path"]).resolve(strict=True).is_relative_to(root)
                 and row["review"].get("decoded_frames") == count,
                 "AUTO_SEED_REVIEW_BINDING_FAILED")
        verify_ref(video)
        _require(helpers.review_decode_count(Path(video["path"])) == count,
                 "AUTO_SEED_REVIEW_DECODE_COUNT_FAILED")
    return reference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve(strict=True)
    _require(spec_path.is_relative_to(REPO_ROOT / "inpainting_optimization"),
             "AUTO_SEED_PROJECT_SPEC_REQUIRED")
    spec = load_json(spec_path)
    validate_spec(spec)
    infra, helpers = _infra(), _model_helpers()
    _require(artifact_ref(INFRA_PATH) == spec["infrastructure"], "AUTO_SEED_INFRA_SHA_DRIFT")
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    _require(check.returncode == 0, "AUTO_SEED_GOVERNANCE_REQUIRED:" + check.stdout[-800:])
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    root = Path(spec["output_dir"]).resolve()
    _require(not root.exists() and root.is_relative_to(REPO_ROOT / "_run"),
             "AUTO_SEED_FRESH_OUTPUT_REQUIRED")
    _output_capacity(spec)
    mapping = DeviceMapping.from_inventory(spec["physical_gpu_index"], read_gpu_inventory())
    packet_path = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
    _require(not packet_path.exists(), "AUTO_SEED_FRESH_PACKET_REQUIRED")
    writer = infra._writer_identity(TASK_ID, 1)
    started = datetime.now(timezone.utc)
    deadline = min(started + timedelta(seconds=spec["budgets"]["wall_seconds_max"]),
                   _deadline(spec["global_deadline_at"]))
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", OPERATION,
               "--packet", str(packet_path)]
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": spec["objective"], "phase": SCOPE,
        "plan_revision": "INPAINTING_GENERALIZATION_199_0918_20261012",
        "execution_revision": TASK_ID.upper(), "execution_mode": "CANDIDATE_VISUAL_MODEL",
        "spec_schema": SPEC_SCHEMA, "experiment_scope": SCOPE,
        "operation": OPERATION, "commands": [shlex.join(command)], "command_argv": command,
        "read_set": spec["read_set"], "read_set_expansions": spec.get("read_set_expansions", []),
        "write_set": [str(root), str(packet_path.parent), str(CANDIDATE_CARD_LOCK_ROOT),
                      "Existing CAS publisher generated projections only"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "SOURCE_TEXT_SHA_PINNED"],
        "required_outputs": ["MODEL_RESULT.json", "RESULT.json", "POST_VALIDATION.json"],
        "budgets": spec["budgets"], "model_stages": ["Mask"], "deletion_allowed": False,
        "algorithm_execution_allowed": True,
        "stop_conditions": ["global deadline", "writer conflict", "SHA drift", "card lease loss"],
        "claim_limit": "Two full-session automatic V1 seed Mask execution only; no visual adoption, Clean/Robot, training or deployment claim.",
        "inputs": {**spec["inputs"], "spec": artifact_ref(spec_path)},
        "runtime_code": spec["runtime_code"], "model_weights": spec["model_weights"],
        "weights": {"files": spec["model_weights"]}, "config": spec["config"],
        "calibration": spec["calibration"], "schema": spec["schema"], "sessions": spec["sessions"],
        "gpu_mapping": {"physical_index": mapping.physical_index, "uuid": mapping.uuid, "logical_index": 0},
        "writer": writer, "writer_file": str(root / "WRITER.json"), "packet_path": str(packet_path),
        "output_dir": str(root), "execution_started_at": started.isoformat(),
        "deadline_at": deadline.isoformat(), "global_deadline_at": spec["global_deadline_at"],
        "control_ground_truth": False, "physical_deployment_authorized": False}
    packet["run_signature"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    _require(not _validate_packet(packet), "AUTO_SEED_PACKET_INVALID")
    packet_path.parent.mkdir(parents=True, exist_ok=False)
    ref = infra._json_once(packet_path, packet)
    infra._json_once(root / "WRITER.json", writer)
    state["tasks"].append({"task_id": TASK_ID, "phase": SCOPE, "attempt": 0,
        "status": "WAIT_GPU_RESOURCE", "model_calls": 0, "updated_at": now_iso(),
        "heartbeat_at": now_iso(), "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "task_packet": ref, "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": TASK_ID, "expected_resource": "CANDIDATE_GPU_MASK_SINGLE_CARD"}
    index["task_packets"].append({"task_id": TASK_ID,
        "packet_path": str(packet_path.relative_to(REPO_ROOT)), "packet_sha256": ref["sha256"],
        "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, SCOPE + "_REGISTERED", expected_revision=receipt["governance_revision"])
    print(json.dumps({"status": "REGISTERED", "task_id": TASK_ID}), flush=True)
    errors, exit_code, child = [], None, None
    child_holder = [None]
    try:
        lease = GpuCardLease(CANDIDATE_CARD_LOCK_ROOT, mapping, task_id=TASK_ID,
            run_signature=packet["run_signature"], fencing_token=writer["fencing_token"])
        while not lease._held:
            try:
                lease.acquire()
            except ResourceError as error:
                if "WAIT_GPU_RESOURCE" not in str(error):
                    raise
                _require(datetime.now(timezone.utc) < deadline, "WAIT_GPU_RESOURCE_FINITE_DEADLINE")
                receipt, state, index = infra._snapshot()
                row, _ = helpers.own_row(packet, state, index)
                row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                infra._publish(state, index, SCOPE + "_WAIT_GPU_RESOURCE",
                    expected_revision=receipt["governance_revision"])
                time.sleep(min(10, max(0.1, (deadline - datetime.now(timezone.utc)).total_seconds())))
        with helpers.lease_until_child_stops(lease, child_holder):
            receipt, state, index = infra._snapshot()
            row, _ = helpers.own_row(packet, state, index)
            row.update(status="RUNNING", attempt=1, heartbeat_at=now_iso(), updated_at=now_iso())
            infra._publish(state, index, SCOPE + "_CARD_ACQUIRED", expected_revision=receipt["governance_revision"])
            infra._json_once(root / "CARD_LEASE.json", lease.owner)
            cache = root / "runtime_cache"
            cache.mkdir()
            env = mapping.worker_environment()
            env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO_ROOT / "src"),
                "PYTHONUNBUFFERED": "1", "TMPDIR": str(cache), "XDG_CACHE_HOME": str(cache),
                "TORCH_HOME": str(cache), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
                "TRITON_CACHE_DIR": str(cache / "triton"), "HF_HOME": str(cache / "huggingface")})
            with (root / "WORKER.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                child_holder[0] = child
                heartbeat = time.monotonic()
                while child.poll() is None:
                    lease.assert_owned()
                    _require(datetime.now(timezone.utc) < deadline, "AUTO_SEED_FINITE_DEADLINE")
                    if time.monotonic() - heartbeat >= 60:
                        receipt, state, index = infra._snapshot()
                        row, _ = helpers.own_row(packet, state, index)
                        row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                        infra._publish(state, index, SCOPE + "_HEARTBEAT", expected_revision=receipt["governance_revision"])
                        heartbeat = time.monotonic()
                    time.sleep(1)
                exit_code = child.returncode
                if exit_code:
                    errors.append("MODEL_WORKER_EXIT:" + str(exit_code))
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
        if child is not None and child.poll() is None:
            raise RuntimeError("CANNOT_SEAL_WITH_LIVE_GPU_CHILD") from error
    for reference in [*spec["inputs"].values(), *spec["runtime_code"], *spec["model_weights"]]:
        try:
            verify_ref(reference)
        except Exception as error:
            errors.append("FINAL_SHA_CHECK:" + str(error))
    inner = artifact_ref(root / "MODEL_RESULT.json") if (root / "MODEL_RESULT.json").is_file() else None
    if not errors and inner:
        try:
            inner = validate_model_result(packet, helpers)
        except Exception as error:
            errors.append("MODEL_RESULT_FINAL_VERIFICATION:" + str(error))
    result = {"schema": "inpainting-auto-v1-seed-mask-executor-result-v1",
        "task_id": TASK_ID, "run_signature": packet["run_signature"], "experiment_scope": SCOPE,
        "status": "PASSED" if not errors and inner else "FAILED_RUNTIME_FINAL",
        "errors": errors, "worker_exit_code": exit_code, "model_result": inner,
        "model_result_consumable": not errors and inner is not None,
        "visual_quality": "NOT_EVALUATED", "candidate_adopted": False,
        "training_steps": 0, "files_deleted": 0, "completed_at": now_iso()}
    receipt, state, index = infra._snapshot()
    row, pointer = helpers.own_row(packet, state, index)
    row.update(status=result["status"], result=infra._json_once(root / "RESULT.json", result),
        pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    infra._publish(state, index, SCOPE + "_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    infra._json_once(root / "POST_VALIDATION.json", {"exit_code": check.returncode,
        "stdout": check.stdout, "stderr": check.stderr})
    print(json.dumps({"status": result["status"], "governance_exit": check.returncode, "errors": errors}), flush=True)
    return 0 if result["status"] == "PASSED" and check.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
