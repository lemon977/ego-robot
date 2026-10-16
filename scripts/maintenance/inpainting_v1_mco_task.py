"""Finite original-V1 M/C/O executor, reusing the existing CAS/card lifecycle.

This file registers a new task and launches only its registered operation.
Algorithm code is never imported by admission or operation authorization.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, now_iso, validate_artifact_ref
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.inpainting_candidate.contracts import verify_ref
from chaoyang.inpainting_candidate.resources import (
    CANDIDATE_CARD_LOCK_ROOT, DeviceMapping, GpuCardLease, ResourceError, read_gpu_inventory,
)

OPERATION = "run_inpainting_v1_mco_stage_v1"
ROUTE_KEY = "inpainting_original_v1_mco_candidate_v1"
SPEC_SCHEMA = "inpainting-v1-mco-stage-spec-v1"
RESULT_SCHEMA = "inpainting-v1-mco-stage-result-v1"
INFRA_PATH = REPO_ROOT / "scripts/humanego_ict/production_resume_runner_v1.py"
HELPERS_PATH = REPO_ROOT / "scripts/maintenance/inpainting_candidate_model_task_v1.py"


def _require(value, code):
    if not value:
        raise ValueError(code)


def _deadline(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(result.tzinfo is not None, "MCO_TIMEZONE_REQUIRED")
    return result


def _project_path(value, *, existing=False):
    path = Path(value)
    _require(path.is_absolute() and path.resolve(strict=existing) == path
             and path.is_relative_to(REPO_ROOT) and path != REPO_ROOT,
             "MCO_REAL_PROJECT_PATH_REQUIRED")
    return path


def _load_module(path, name):
    definition = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def _infra():
    infra = _load_module(INFRA_PATH, "_mco_existing_cas_publisher")
    from chaoyang.humanego_ict.project_mount_paths import resolve_input_path

    def snapshot():
        # CPFS and MNT can spell the same immutable receipt. Never rewrite it.
        with (REPO_ROOT / "_run/current/locks/governance.lock").open("rb") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_SH)
            try:
                receipt = load_json(REPO_ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json")
                values = []
                for key in ("task_state", "task_packet_index_current"):
                    ref = receipt["files"][key]
                    local = resolve_input_path(ref["path"], REPO_ROOT)
                    _require(not validate_artifact_ref({**ref, "path": str(local)}), "MCO_ALIAS_SHA_DRIFT")
                    values.append(load_json(local))
                return receipt, *values
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    infra._snapshot = snapshot
    return infra


def _helpers():
    return _load_module(HELPERS_PATH, "_mco_existing_model_lifecycle")


def validate_spec(spec):
    _require(spec.get("spec_schema") == SPEC_SCHEMA, "MCO_SPEC_SCHEMA_REQUIRED")
    _require(re.fullmatch(r"inpainting_v1_mco_[A-Za-z0-9_]+", str(spec.get("task_id", ""))) is not None,
             "MCO_NEW_TASK_ID_REQUIRED")
    mode, bounds = spec.get("mode"), spec.get("budgets", {})
    _require(mode in {"Mask", "Clean", "Boundary", "Automation"}, "MCO_MODE_REQUIRED")
    for key in ("wall_seconds_max", "cpu_threads_per_worker", "gpu_count", "model_calls_max",
                "render_calls_max", "training_steps_max", "output_bytes_estimate"):
        _require(type(bounds.get(key)) is int, "MCO_INTEGER_BUDGET_REQUIRED:" + key)
    _require(0 < bounds["wall_seconds_max"] <= 3600 and bounds["cpu_threads_per_worker"] == 2
             and bounds["training_steps_max"] == 0
             and 0 < bounds["output_bytes_estimate"] <= 536870912
             and bounds["model_calls_max"] >= 0 and bounds["render_calls_max"] >= 0,
             "MCO_FINITE_BUDGET_REQUIRED")
    cpu_donor = spec.get("config", {}).get("cpu_raw_donor", False)
    cpu_spatial = spec.get("config", {}).get("cpu_spatial_clean", False)
    cpu_seam = spec.get('config',{}).get('cpu_context_seam',False)
    _require(all(type(v)is bool for v in [cpu_donor,cpu_spatial,cpu_seam])
             and sum([cpu_donor,cpu_spatial,cpu_seam])<=1, "MCO_CPU_CLEAN_BOOL_EXCLUSIVITY_REQUIRED")
    cpu_clean = cpu_donor or cpu_spatial or cpu_seam
    cpu_equipment = spec.get("config", {}).get("cpu_equipment_rgb", False)
    _require(type(cpu_equipment) is bool and not (cpu_equipment and cpu_clean), "MCO_CPU_EQUIPMENT_BOOL_EXCLUSIVITY")
    if cpu_equipment:
        _require(mode == "Mask" and bounds["model_calls_max"] == 0 and bounds["render_calls_max"] == 0
                 and not spec.get("model_weights") and len(spec.get("sessions", [])) == 1,
                 "MCO_CPU_EQUIPMENT_ZERO_MODEL_REQUIRED")
    if cpu_clean:
        _require(mode == "Clean" and bounds["model_calls_max"] == 0
                 and bounds["render_calls_max"] == 0 and not spec.get("model_weights")
                 and len(spec.get("sessions", [])) == 1
                 and not any(spec.get("config", {}).get(k) for k in
                             ("paired_clean", "mask_causal_pair", "context_clean_pair")),
                 "MCO_CPU_DONOR_ZERO_NEURAL_RENDER_FULL_NATIVE_REQUIRED")
    if cpu_seam:
        _require(spec['config'].get('algorithm')in ['POISSON_WORKING_GRADIENT_RAW_FIXED_BOUNDARY','POISSON_RGB_UNCERTAIN_BOUNDARY_VETO']
                 and not any(spec['config'].get(k)for k in ['context_exclusion_pair','generic_object_probe']),
                 'MCO_CONTEXT_SEAM_EXACT_CPU_ALGORITHM_REQUIRED')
    _require(bounds["gpu_count"] == (0 if mode == "Boundary" or cpu_clean or cpu_equipment else 1), "MCO_MODE_CARD_BUDGET_REQUIRED")
    _require((mode != "Boundary" or bounds["model_calls_max"] == 0)
             and (mode in {"Boundary", "Automation"} or bounds["render_calls_max"] == 0), "MCO_MODE_CALL_BUDGET_REQUIRED")
    generic_probe=spec.get("config",{}).get("generic_object_probe",False)
    _require(type(generic_probe) is bool,"MCO_GENERIC_PROBE_BOOL_REQUIRED")
    if generic_probe:
        _require(mode=="Mask" and bounds["model_calls_max"]==1 and bounds["render_calls_max"]==0
                 and spec["config"].get("query")=="object" and spec["config"].get("anchor_frame")==0
                 and len(spec.get("sessions",[]))==1,"MCO_GENERIC_PROBE_ONE_QUERY_REQUIRED")
    _require(_deadline(spec["global_deadline_at"]) > datetime.now(timezone.utc), "MCO_GLOBAL_DEADLINE")
    _project_path(spec["output_dir"])
    _project_path(spec["review_dir"])
    rows = spec.get("sessions")
    _require(isinstance(rows, list) and rows and len(rows) <= 4, "MCO_BOUNDED_SESSIONS_REQUIRED")
    keys = [row.get("session_key") for row in rows]
    _require(all(isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_]+", key) for key in keys)
             and len(set(keys)) == len(keys)
             and all(type(row.get("native_frames")) is int and row["native_frames"] > 0 for row in rows),
             "MCO_FULL_NATIVE_UNIQUE_SESSIONS_REQUIRED")
    for row in rows:
        if "output_frames" in row:
            frames = row["output_frames"]
            _require(mode == "Boundary" and isinstance(frames, list) and frames
                     and all(type(frame) is int and 0 <= frame < row["native_frames"] for frame in frames)
                     and frames == sorted(set(frames)), "MCO_FROZEN_BOUNDARY_FRAME_SELECTION_REQUIRED")
    _require(isinstance(spec.get("inputs"), dict) and spec["inputs"]
             and isinstance(spec.get("runtime_code"), list) and spec["runtime_code"]
             and isinstance(spec.get("model_weights"), list), "MCO_FROZEN_INPUTS_REQUIRED")
    for field in ("calibration", "schema", "config", "read_set"):
        _require(field in spec, "MCO_SIGNATURE_DOMAIN_REQUIRED:" + field)
    for ref in spec["runtime_code"]:
        _project_path(ref["path"], existing=True)
    for ref in [*spec["inputs"].values(), *spec["runtime_code"], *spec["model_weights"]]:
        verify_ref(ref)
    _require(spec.get("infrastructure") == artifact_ref(INFRA_PATH), "MCO_INFRASTRUCTURE_SHA_DRIFT")
    if mode == "Automation":
        from chaoyang.inpainting_candidate.automation import verify_v1_request
        _require("request" in spec["inputs"], "MCO_AUTOMATION_REQUEST_REQUIRED")
        verify_v1_request(load_json(Path(spec["inputs"]["request"]["path"])), project_root=REPO_ROOT)
        for key in ("component_code", "component_weights", "component_config", "component_signatures"):
            _require(isinstance(spec.get(key), dict) and spec[key], "MCO_AUTOMATION_COMPONENT_REQUIRED:" + key)


def _output_capacity(spec):
    required = spec["budgets"]["output_bytes_estimate"]
    info = os.statvfs(REPO_ROOT)
    if info.f_bavail * info.f_frsize >= required:
        return
    ref = spec["inputs"].get("capacity_proof")
    _require(isinstance(ref, dict), "MCO_CAPACITY_PROOF_REQUIRED")
    _project_path(ref["path"], existing=True)
    verify_ref(ref)
    proof = load_json(Path(ref["path"]))
    age = (datetime.now(timezone.utc) - _deadline(proof["completed_at"])).total_seconds()
    _require(proof.get("schema") == "inpainting-v1-mco-capacity-proof-v1"
             and proof.get("status") == "BOUNDED_WRITE_FSYNC_PASS_STATVFS_UNVERIFIED"
             and type(proof.get("actual_written_fsynced_bytes")) is int
             and proof["actual_written_fsynced_bytes"] >= required
             and proof.get("own_scratch_truncated_after_probe") is True
             and proof.get("files_deleted") == 0 and 0 <= age < 1800,
             "MCO_RECENT_BOUNDED_CAPACITY_PROOF_REQUIRED")


def authorize_operation(args):
    """Validate live packet/writer/code before the caller imports its worker."""
    _require(len(args) == 2 and args[0] == "--packet", "MCO_EXACT_PACKET_ARGUMENT_REQUIRED")
    packet_path = _project_path(args[1], existing=True)
    _require(packet_path.parent.parent == REPO_ROOT / "tasks/current"
             and packet_path.name == "TASK_PACKET.json", "MCO_REGISTERED_PACKET_PATH_REQUIRED")
    packet = load_json(packet_path)
    _require(packet_path.parent.name == packet.get("task_id"), "MCO_PACKET_ID_PATH_MISMATCH")
    infra, helpers = _infra(), _helpers()
    receipt, state, index = infra._snapshot()
    row, _ = helpers.own_row(packet, state, index)
    _require(row["status"] == "RUNNING" and packet.get("operation") == OPERATION
             and packet.get("algorithm_execution_allowed") is True
             and packet.get("deletion_allowed") is False, "MCO_RUNNING_CANDIDATE_PACKET_REQUIRED")
    expected = [sys.executable, "-B", "-m", "chaoyang.cli", "run", OPERATION, "--packet", str(packet_path)]
    _require(packet.get("command_argv") == expected and packet.get("commands") == [shlex.join(expected)],
             "MCO_COMMAND_DRIFT")
    _require(datetime.now(timezone.utc) < _deadline(packet["deadline_at"])
             <= _deadline(packet["global_deadline_at"]), "MCO_FINITE_DEADLINE_REQUIRED")
    from chaoyang.governance.exact78_operation_gate import _caller_is_writer_descendant
    writer = packet["writer"]
    _require(_caller_is_writer_descendant(writer["pid"], writer["proc_start_ticks"], os.getpid()),
             "MCO_CALLER_NOT_WRITER_DESCENDANT")
    from chaoyang.humanego_ict.project_mount_paths import resolve_input_path
    ref = receipt["files"]["algorithm_contract"]
    local = resolve_input_path(ref["path"], REPO_ROOT)
    _require(not validate_artifact_ref({**ref, "path": str(local)}), "MCO_CONTRACT_SHA_DRIFT")
    route = load_json(local).get("special_status_contracts", {}).get(ROUTE_KEY)
    from chaoyang.governance.inpainting_candidate_registration import mco_route_contract
    _require(isinstance(route, dict) and route.get("operation") == OPERATION
             and route == mco_route_contract(REPO_ROOT).get(ROUTE_KEY)
             and route.get("task_packet_required") is True
             and all(ref in packet["runtime_code"] for ref in route.get("code_closure", []))
             and route.get("code_closure"), "MCO_REGISTERED_ROUTE_REQUIRED")
    allowed = route.get("allowed_modes", ["Mask", "Clean", "Boundary"])
    _require(packet.get("mode") in allowed, "MCO_ROUTE_MODE_REQUIRED")
    verify_ref(packet["inputs"]["spec"])
    spec = load_json(Path(packet["inputs"]["spec"]["path"]))
    validate_spec(spec)
    for field in ("mode", "sessions", "runtime_code", "model_weights", "calibration", "schema", "config",
                  "budgets", "global_deadline_at", "output_dir", "review_dir"):
        _require(packet[field] == spec[field], "MCO_PACKET_SPEC_DRIFT:" + field)
    _require(packet["inputs"] == {**spec["inputs"], "spec": packet["inputs"]["spec"]}, "MCO_PACKET_INPUT_DRIFT")
    if spec["mode"] == "Automation":
        for key in ("component_code", "component_weights", "component_config", "component_signatures", "reuse_stages",
                    "base_clean_signatures","clean_work_reuse"):
            _require(packet.get(key) == spec.get(key, {}), "MCO_AUTOMATION_COMPONENT_DRIFT:" + key)
        _require(packet.get("request_path") == spec["inputs"]["request"]["path"], "MCO_AUTOMATION_REQUEST_PATH_DRIFT")
    return packet


def validate_model_result(packet):
    root = Path(packet["output_dir"])
    reference = artifact_ref(root / "MODEL_RESULT.json")
    value = load_json(Path(reference["path"]))
    _require(value.get("schema") == RESULT_SCHEMA and value.get("status") == "SEALED_COMPLETE"
             and value.get("task_id") == packet["task_id"]
             and value.get("run_signature") == packet["run_signature"] and value.get("mode") == packet["mode"]
             and value.get("candidate_adopted") is False and value.get("control_ground_truth") is False,
             "MCO_RESULT_BINDING_REQUIRED")
    for name in ("model_calls", "render_calls"):
        _require(type(value.get(name)) is int and 0 <= value[name] <= packet["budgets"][name + "_max"],
                 "MCO_RESULT_CALL_BUDGET_EXCEEDED:" + name)
    planned, rows = packet["sessions"], value.get("sessions")
    _require(isinstance(rows, list) and len(rows) == len(planned)
             and [row.get("session_key") for row in rows] == [row["session_key"] for row in planned],
             "MCO_RESULT_SESSION_COMPLETENESS_REQUIRED")
    for row, original in zip(rows, planned):
        expected_frames = original.get("output_frames", list(range(original["native_frames"])))
        _require(row.get("native_frames") == original["native_frames"]
                 and row.get("native_output_frames") == len(expected_frames)
                 and row.get("completed_frame_indices") == expected_frames
                 and row.get("execution_complete") is True, "MCO_RESULT_FROZEN_OUTPUT_COVERAGE_REQUIRED")
    artifacts = value.get("artifacts")
    _require(isinstance(artifacts, list) and artifacts, "MCO_RESULT_ARTIFACTS_REQUIRED")
    for ref in artifacts:
        path = _project_path(ref["path"], existing=True)
        _require(path.is_relative_to(root) or path.is_relative_to(Path(packet["review_dir"])),
                 "MCO_OUTPUT_ARTIFACT_WRITE_SET_REQUIRED")
        verify_ref(ref)
    return reference


def _environment(root, mapping):
    env = mapping.worker_environment() if mapping is not None else dict(os.environ, CUDA_VISIBLE_DEVICES="")
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=str(REPO_ROOT / "src"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")
    for name, subdir in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"), ("TORCH_HOME", "torch"),
                        ("TORCH_EXTENSIONS_DIR", "torch_extensions"), ("TRITON_CACHE_DIR", "triton"),
                        ("HF_HOME", "huggingface"), ("NUMBA_CACHE_DIR", "numba"),
                        ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("CUDA_CACHE_PATH", "cuda"),
                        ("MPLCONFIGDIR", "matplotlib")):
        path = root / "runtime_cache" / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(_project_path(path, existing=True))
    env["TMP"] = env["TEMP"] = env["TMPDIR"]
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec_path = _project_path(str(args.spec.absolute()), existing=True)
    spec = load_json(spec_path)
    validate_spec(spec)
    infra, helpers = _infra(), _helpers()
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
                           cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    _require(check.returncode == 0, "MCO_GOVERNANCE_REQUIRED:" + check.stdout[-1500:])
    receipt, state, index = infra._snapshot()
    infra._no_active_writer(state, index)
    root, task_id = Path(spec["output_dir"]), spec["task_id"]
    _require(root.is_relative_to(REPO_ROOT / "_run") and not root.exists(), "MCO_FRESH_OUTPUT_REQUIRED")
    _output_capacity(spec)
    mapping = (DeviceMapping.from_inventory(spec.get("physical_gpu_index", 0), read_gpu_inventory())
               if spec["budgets"]["gpu_count"] else None)
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    _require(not packet_path.exists(), "MCO_FRESH_PACKET_REQUIRED")
    writer, started = infra._writer_identity(task_id, 1), datetime.now(timezone.utc)
    deadline = min(started + timedelta(seconds=spec["budgets"]["wall_seconds_max"]), _deadline(spec["global_deadline_at"]))
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", OPERATION, "--packet", str(packet_path)]
    packet = {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": spec.get("objective", "Finite original-V1 " + spec["mode"] + " candidate"),
        "phase": "INPAINTING_ORIGINAL_V1_MCO_" + spec["mode"].upper(),
        "plan_revision": spec.get("plan_revision", "INPAINTING_V1_MCO_EIGHT_HOUR_20261013"), "execution_revision": task_id.upper(),
        "execution_mode": "CANDIDATE_CPU_RENDER" if mapping is None else "CANDIDATE_VISUAL_MODEL",
        "operation": OPERATION, "commands": [shlex.join(command)], "command_argv": command,
        "read_set": spec["read_set"], "read_set_expansions": spec.get("read_set_expansions", []),
        "write_set": [str(root), spec["review_dir"], str(packet_path.parent), str(CANDIDATE_CARD_LOCK_ROOT),
                      "Existing CAS publisher generated projections only"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_WRITER", "FROZEN_ORIGINAL_V1_INPUTS"],
        "required_outputs": ["MODEL_RESULT.json", "RESULT.json", "POST_VALIDATION.json"],
        "budgets": spec["budgets"], "model_stages": [spec["mode"]], "mode": spec["mode"],
        "deletion_allowed": False, "algorithm_execution_allowed": True,
        "stop_conditions": ["finite deadline", "writer conflict", "source/code/weight SHA drift", "card lease loss"],
        "claim_limit": "Original-V1 offline visual candidate only; execution is not quality or adoption.",
        "inputs": {**spec["inputs"], "spec": artifact_ref(spec_path)},
        "runtime_code": spec["runtime_code"], "model_weights": spec["model_weights"],
        "weights": {"files": spec["model_weights"]}, "calibration": spec["calibration"],
        "schema": spec["schema"], "config": spec["config"], "sessions": spec["sessions"],
        "writer": writer, "writer_file": str(root / "WRITER.json"), "packet_path": str(packet_path),
        "output_dir": str(root), "review_dir": spec["review_dir"],
        "execution_started_at": started.isoformat(), "deadline_at": deadline.isoformat(),
        "global_deadline_at": spec["global_deadline_at"], "control_ground_truth": False,
        "physical_deployment_authorized": False}
    if mapping:
        packet["gpu_mapping"] = {"physical_index": mapping.physical_index, "uuid": mapping.uuid, "logical_index": 0}
    if spec["mode"] == "Automation":
        packet["request_path"] = spec["inputs"]["request"]["path"]
        for key in ("component_code", "component_weights", "component_config", "component_signatures", "reuse_stages",
                    "base_clean_signatures","clean_work_reuse"):
            packet[key] = spec.get(key, {})
    packet["run_signature"] = hashlib.sha256(json.dumps(packet, sort_keys=True).encode()).hexdigest()
    _require(not _validate_packet(packet), "MCO_TASK_PACKET_INVALID")
    packet_path.parent.mkdir(parents=True, exist_ok=False)
    packet_ref = infra._json_once(packet_path, packet)
    infra._json_once(root / "WRITER.json", writer)
    state["tasks"].append({"task_id": task_id, "phase": packet["phase"], "attempt": 0 if mapping else 1,
        "status": "WAIT_GPU_RESOURCE" if mapping else "RUNNING", "model_calls": 0,
        "updated_at": now_iso(), "heartbeat_at": now_iso(), "pid": writer["pid"],
        "proc_start_ticks": writer["proc_start_ticks"], "task_packet": packet_ref, "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": task_id, "expected_resource": "MCO_SINGLE_CARD" if mapping else "MCO_CPU_RENDER"}
    index["task_packets"].append({"task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": packet_ref["sha256"], "execution_allowed": True, "execution_class": "CURRENT_LEDGER_ROUTABLE"})
    infra._publish(state, index, "MCO_REGISTERED", expected_revision=receipt["governance_revision"])
    print(json.dumps({"status": "REGISTERED", "task_id": task_id, "mode": spec["mode"]}), flush=True)
    errors, exit_code, holder, lease = [], None, [None], None
    try:
        if mapping:
            lease = GpuCardLease(CANDIDATE_CARD_LOCK_ROOT, mapping, task_id=task_id,
                run_signature=packet["run_signature"], fencing_token=writer["fencing_token"])
            while not lease._held:
                _require(datetime.now(timezone.utc) < deadline, "MCO_WAIT_GPU_FINITE_DEADLINE")
                try:
                    lease.acquire()
                except ResourceError as error:
                    if "WAIT_GPU_RESOURCE" not in str(error):
                        raise
                    receipt, state, index = infra._snapshot()
                    row, _ = helpers.own_row(packet, state, index)
                    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                    infra._publish(state, index, "MCO_WAIT_GPU_RESOURCE", expected_revision=receipt["governance_revision"])
                    time.sleep(min(10, max(.1, (deadline - datetime.now(timezone.utc)).total_seconds())))
        context = helpers.lease_until_child_stops(lease, holder) if lease else nullcontext()
        with context:
            if mapping:
                receipt, state, index = infra._snapshot()
                row, _ = helpers.own_row(packet, state, index)
                row.update(status="RUNNING", attempt=1, heartbeat_at=now_iso(), updated_at=now_iso())
                infra._publish(state, index, "MCO_CARD_ACQUIRED", expected_revision=receipt["governance_revision"])
                infra._json_once(root / "CARD_LEASE.json", lease.owner)
            env = _environment(root, mapping)
            with (root / "WORKER.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                holder[0] = child
                last = time.monotonic()
                while child.poll() is None:
                    if lease:
                        lease.assert_owned()
                    _require(datetime.now(timezone.utc) < deadline, "MCO_FINITE_DEADLINE")
                    if time.monotonic() - last >= 60:
                        receipt, state, index = infra._snapshot()
                        row, _ = helpers.own_row(packet, state, index)
                        row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                        infra._publish(state, index, "MCO_HEARTBEAT", expected_revision=receipt["governance_revision"])
                        last = time.monotonic()
                    time.sleep(1)
                exit_code = child.returncode
                _require(exit_code == 0, "MCO_WORKER_EXIT:" + str(exit_code))
    except BaseException as error:
        errors.append(type(error).__name__ + ":" + str(error))
    finally:
        # Includes CPU render; do not seal a result while any owned child lives.
        helpers.stop_owned_child(holder[0])
    inner = None
    for ref in [*spec["inputs"].values(), *spec["runtime_code"], *spec["model_weights"]]:
        try:
            verify_ref(ref)
        except Exception as error:
            errors.append("FINAL_SHA_CHECK:" + str(error))
    if not errors:
        try:
            inner = validate_model_result(packet)
        except Exception as error:
            errors.append("FINAL_RESULT_CHECK:" + str(error))
    result = {"schema": "inpainting-v1-mco-executor-result-v1", "task_id": task_id,
        "run_signature": packet["run_signature"], "mode": spec["mode"],
        "status": "PASSED" if not errors and inner else "FAILED_RUNTIME_FINAL", "errors": errors,
        "worker_exit_code": exit_code, "model_result": inner, "model_result_consumable": not errors and inner is not None,
        "visual_quality": "NOT_REVIEWED", "candidate_adopted": False, "default_changed": False,
        "training_steps": 0, "files_deleted": 0, "control_ground_truth": False, "completed_at": now_iso()}
    receipt, state, index = infra._snapshot()
    row, pointer = helpers.own_row(packet, state, index)
    row.update(status=result["status"], result=infra._json_once(root / "RESULT.json", result),
               pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso())
    pointer.update(execution_allowed=False, execution_class="TERMINAL_LEDGER_EVIDENCE")
    state["next_task"] = None
    infra._publish(state, index, "MCO_SEALED", expected_revision=receipt["governance_revision"])
    check = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
                           cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    infra._json_once(root / "POST_VALIDATION.json", {"exit_code": check.returncode,
        "stdout": check.stdout, "stderr": check.stderr})
    print(json.dumps({"status": result["status"], "governance_exit": check.returncode, "errors": errors}), flush=True)
    return 0 if result["status"] == "PASSED" and check.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
