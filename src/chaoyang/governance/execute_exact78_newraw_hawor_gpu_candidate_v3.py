"""Unregistered, bounded GPU consumer of one terminal new-raw CPU admission.

Check-only is read-only. A precisely staged candidate route may be published
in the same governance CAS as its Task Packet, never as a route-only change.
No model is imported or executed at import.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import signal
import subprocess
import time
from types import SimpleNamespace
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref, stage_signature
from chaoyang.governance.register_single_task_packet import _validate_packet


OPERATION = "run_exact78_newraw_hawor_gpu_candidate_v3"
ROUTE_KEY = "exact78_newraw_hawor_gpu_candidate_v3"
GPU_LEASE = REPO_ROOT / "_run/current/GPU_LEASE.json"
GPU_LOCK = REPO_ROOT / "_run/current/GPU_LEASE.lock"
MODEL_ENV = REPO_ROOT / "_run/current/environments/hawor-py310-v1"
MODEL_PYTHON = MODEL_ENV / "bin/python"
STRACE = Path("/usr/bin/strace")
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
WALL_SECONDS_MAX = 7200
GPU_WAIT_SECONDS_MAX = 900

RUNTIME_RELATIVE = (
    "src/chaoyang/governance/execute_exact78_newraw_hawor_gpu_candidate_v3.py",
    "src/chaoyang/ops/run_exact78_newraw_hawor_gpu_candidate_v3.py",
    "src/chaoyang/pipeline/exact78_newraw_hawor_stage_candidate_v1.py",
    "src/chaoyang/governance/exact78_hawor_descriptor_adapter_candidate.py",
    "src/chaoyang/ops/run_exact78_hawor_raw_v3.py",
    "src/chaoyang/cli/exact78_first_submit_bridge_candidate.py",
    "src/chaoyang/cli/exact78_submit.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/governance/gpu_lease_v71.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/paths.py",
)
ROUTE_CLOSURE_RELATIVE = (
    *RUNTIME_RELATIVE[:9],
    "src/chaoyang/governance/gpu_lease_v71.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/cli/__init__.py",
)


def _result_refs(result_path: Path) -> dict[str, Any]:
    path = result_path.resolve(strict=True)
    if not path.is_relative_to(REPO_ROOT / "_run/current") or path.name != "RESULT.json":
        raise RuntimeError("PRODUCER_RESULT_SCOPE_DRIFT")
    result = load_json(path)
    if (result.get("schema_version") != "exact78-first-submit-cpu-admission-result-v1"
            or result.get("status") != "PASSED" or result.get("model_calls") != 0
            or result.get("gpu_count") != 0 or result.get("source_sha_stable_at_terminal") is not True):
        raise RuntimeError("PRODUCER_CPU_TERMINAL_NOT_PASSED")
    task_id = result.get("task_id")
    expected = REPO_ROOT / "_run/current" / str(task_id) / "attempts/attempt_0001/RESULT.json"
    if path != expected:
        raise RuntimeError("PRODUCER_RESULT_ATTEMPT_DRIFT")
    admission = result.get("admission", {})
    if set(admission) != {"descriptor", "frame_time_map", "seal"}:
        raise RuntimeError("THREE_ADMISSION_REFS_REQUIRED")
    receipt = REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json"
    refs = {
        "upstream_admission_descriptor": admission["descriptor"],
        "upstream_admission_frame_time_map": admission["frame_time_map"],
        "upstream_admission_seal": admission["seal"],
        "upstream_admission_result": file_ref(path),
        "upstream_admission_receipt": file_ref(receipt),
    }
    for ref in refs.values():
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("UPSTREAM_REF_SHA_DRIFT:" + ";".join(errors))
    return {"result": result, "refs": refs}


def _identity(result: dict[str, Any]) -> str:
    session = result.get("source_session")
    descriptor_sha = result.get("descriptor_sha256")
    if (not isinstance(session, str) or not re.fullmatch(r"[a-z0-9_]+", session)
            or not isinstance(descriptor_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", descriptor_sha)):
        raise RuntimeError("PRODUCER_IDENTITY_DRIFT")
    return f"exact78_newraw_hawor_{session}_{descriptor_sha[:16]}_v3"


def _paths(task_id: str) -> tuple[Path, Path, Path]:
    return (REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001",
            REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json",
            REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json")


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _model_assets() -> dict[str, dict[str, Any]]:
    from chaoyang.ops.run_exact78_hawor_raw_v3 import model_asset_refs
    return model_asset_refs()


def _runtime_refs() -> list[dict[str, Any]]:
    refs = [artifact_ref(REPO_ROOT / relative) for relative in RUNTIME_RELATIVE]
    refs += [artifact_ref(path) for path in sorted((REPO_ROOT / "vendor/HaWoR").rglob("*.py"))
             if path.is_file() and not path.is_symlink()]
    return refs


def packet_value(result_path: Path) -> dict[str, Any]:
    """Return a finite packet candidate without writing, leasing or running."""
    source = _result_refs(result_path)
    result, refs = source["result"], source["refs"]
    task_id = _identity(result)
    output, packet_path, receipt = _paths(task_id)
    if not MODEL_PYTHON.is_file() or not STRACE.is_file():
        raise RuntimeError("PINNED_MODEL_RUNTIME_OR_STRACE_ABSENT")
    assets = _model_assets()
    runtime = _runtime_refs()
    raw = result["source_files"]
    inputs = {
        **refs,
        "upstream_admission_descriptor_sha256": result["descriptor_sha256"],
        "submission_input_sha256": result["submission_input_sha256"],
        "source_session": result["source_session"],
        "source_root": result["source_root"],
        "source_files": raw,
        "raw_video": (raw["video"] if "video" in raw else raw["raw/vst.h264"]),
        "model_assets": assets,
        "model_python": artifact_ref(MODEL_PYTHON),
        "native_tracer": artifact_ref(STRACE),
        "route": artifact_ref(ROUTE),
    }
    command = (
        f"PYTHONPATH=src {shlex.quote(str(MODEL_PYTHON))} -B -m chaoyang.cli run "
        f"{OPERATION} --producer-result {shlex.quote(str(result_path.resolve(strict=True)))} "
        f"--output-root {shlex.quote(str(output / 'candidate/hawor'))}"
    )
    config = {
        "frame_count": load_json(Path(refs["upstream_admission_descriptor"]["path"]))["decoded_frames"],
        "source_kind": ("tracker" if "video" in raw else "vst0915"),
        "model_input_size": [1280, 960], "source_role": "new_session_candidate",
        "world_pose_authority": False, "metric_authority": False,
        "infiller_invoked": False, "model_calls": "one full HaWoR session",
        "gpu_count": 1, "cpu_threads_total": 2, "training_steps": 0,
        "native_open_audit_required": True,
        "resume_policy": "SAME_SIGNATURE_SEALED_STAGE_ONLY",
    }
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": "exact78-newraw-hawor-gpu-consumer-v3",
    })).hexdigest()
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": task_id}
    now = datetime.now().astimezone()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "Run sealed new-raw RGB through frozen HaWoR with project-local CUDA cache and classified native-open audit",
        "phase": "EXACT78_NEWRAW_HAWOR_GPU_CANDIDATE_V3",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": task_id.upper(), "execution_mode": "EXACT78_CANDIDATE",
        "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(seconds=WALL_SECONDS_MAX)).isoformat(),
        "read_set": ["AGENTS.md", str(ROUTE.relative_to(REPO_ROOT)),
                     *(str(Path(ref["path"]).relative_to(REPO_ROOT))
                       for ref in refs.values()),
                     "src/chaoyang/governance/execute_exact78_newraw_hawor_gpu_candidate_v3.py"],
        "write_set": [str(output.relative_to(REPO_ROOT)),
                      str(packet_path.relative_to(REPO_ROOT)),
                      str(receipt.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK",
                          "TERMINAL_CPU_ADMISSION_SEALED", "ONE_GPU_LEASE",
                          "SOURCE_SHA_PINNED", "EXTERNAL_DATA_READ_ONLY",
                          "NO_OLD_ACTION_OR_MASK", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["candidate/hawor/RESULT.json",
                             "candidate/hawor/HAWOR_CAMERA_PREDICTIONS.npz",
                             "candidate/hawor/FRAME_PROVENANCE.json",
                             "candidate/hawor/INPUT_AUDIT.json",
                             "candidate/NATIVE_OPEN_AUDIT.json",
                             "candidate/HAWOR_STAGE_SEAL.json", "RESULT.json",
                             "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": WALL_SECONDS_MAX,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "GPU_LEASE_UNAVAILABLE", "UNSEALED_PARTIAL_NOT_RESUMABLE",
                            "NATIVE_OPEN_AUDIT_FAIL", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "One non-T0 same-session camera HaWoR candidate; no Robot, visual, product, world, control or training qualification.",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("PACKET_PREFLIGHT:" + ";".join(errors))
    from chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1 import checked_spec
    spec = checked_spec(packet)
    if spec["frame_count"] != config["frame_count"] or spec["source_video"] != Path(inputs["raw_video"]["path"]):
        raise RuntimeError("SEALED_SPEC_PACKET_DRIFT")
    return packet


def _route_valid(contract: dict[str, Any], packet: dict[str, Any]) -> bool:
    contracts = contract.get("special_status_contracts", {})
    route = contracts.get(ROUTE_KEY) if isinstance(contracts, dict) else None
    if not isinstance(route, dict) or any((
        route.get("execution_status") != "CANDIDATE_FENCED_ONE_GPU_ONE_SESSION",
        route.get("candidate_operation") != OPERATION,
        route.get("task_packet_required") is not True,
        route.get("current_runnable_operation") is not None,
        route.get("output_schema") != "exact78-newraw-hawor-gpu-execution-result-v3",
        route.get("model_calls_authorized") != "ONE_FULL_HAWOR_SESSION_ONLY_WHEN_EXACT_PACKET_ACTIVE",
        route.get("training_steps_authorized") != 0,
    )):
        return False
    closure = route.get("code_closure")
    if not isinstance(closure, list) or len(closure) != len(ROUTE_CLOSURE_RELATIVE):
        return False
    expected_paths = {str(REPO_ROOT / relative) for relative in ROUTE_CLOSURE_RELATIVE}
    route_refs = {ref.get("path"): ref for ref in closure if isinstance(ref, dict)}
    runtime_refs = {ref.get("path"): ref for ref in packet.get("runtime_code", [])
                    if isinstance(ref, dict)}
    return (set(route_refs) == expected_paths
            and all(route_refs[path] == runtime_refs.get(path) for path in expected_paths))


def _assert_route_ready(packet: dict[str, Any]) -> str:
    """Accept only an exact published route or a same-CAS staged successor."""
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("PRODUCTION_ROUTE_ALREADY_SELECTED")
    operation_ref = artifact_ref(REPO_ROOT / "src/chaoyang/ops" / f"{OPERATION}.py")
    if operation_ref not in packet.get("runtime_code", []):
        raise RuntimeError("CANDIDATE_OPERATION_CODE_NOT_PINNED")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    published = load_json(ALGORITHM_CONTRACT_PATH)
    published_routes = published.get("special_status_contracts", {})
    published_route = published_routes.get(ROUTE_KEY) if isinstance(published_routes, dict) else None
    if published_route is not None:
        if not _route_valid(published, packet):
            raise RuntimeError("PUBLISHED_CANDIDATE_ROUTE_DRIFT")
        from chaoyang.cli import _maintained_operations
        if OPERATION not in _maintained_operations():
            raise RuntimeError("PUBLISHED_CANDIDATE_CLI_ROUTE_ABSENT")
        from chaoyang.governance.current_r3_contracts import build_algorithm_contract
        staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                          load_json(AUTHORITY_PATH))
        staged_routes = staged.get("special_status_contracts", {})
        if (not _route_valid(staged, packet)
                or staged_routes.get(ROUTE_KEY) != published_route):
            raise RuntimeError("PUBLISHED_CANDIDATE_BUILDER_DRIFT")
        readiness = "PUBLISHED_CANDIDATE_ROUTE"
    else:
        from chaoyang.cli import _maintained_operations
        if OPERATION in _maintained_operations():
            raise RuntimeError("PUBLISHED_CANDIDATE_CLI_ROUTE_DRIFT")
        from chaoyang.governance.current_r3_contracts import build_algorithm_contract
        staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                          load_json(AUTHORITY_PATH))
        if not _route_valid(staged, packet):
            raise RuntimeError("CANDIDATE_ROUTE_NOT_STAGED")
        readiness = "READY_TO_PUBLISH_ROUTE_WITH_PACKET"
    expected = ["PYTHONPATH=src", str(MODEL_PYTHON), "-B", "-m", "chaoyang.cli",
                "run", OPERATION, "--producer-result",
                packet["inputs"]["upstream_admission_result"]["path"],
                "--output-root", str(_paths(packet["task_id"])[0] / "candidate/hawor")]
    if len(packet.get("commands", [])) != 1 or shlex.split(packet["commands"][0]) != expected:
        raise RuntimeError("CANDIDATE_COMMAND_DRIFT")
    return readiness


def _index_copy() -> dict[str, Any]:
    value = load_json(INDEX)
    return {key: copy.deepcopy(item) for key, item in value.items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def _fence(packet: dict[str, Any], *, assets: bool = False) -> None:
    task_id = packet["task_id"]
    output, packet_path, _ = _paths(task_id)
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(output / "WRITER.json") != writer):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [item for item in state["tasks"]
              if item.get("status") in {"WAIT_GPU_RESOURCE", "RUNNING", "CLAIMED"}]
    rows = [item for item in load_json(INDEX)["task_packets"]
            if item.get("execution_allowed") is True]
    if (len(active) != 1 or active[0]["task_id"] != task_id
            or active[0].get("run_signature") != packet["run_signature"]
            or len(rows) != 1 or rows[0]["task_id"] != task_id
            or rows[0]["packet_sha256"] != artifact_ref(packet_path)["sha256"]):
        raise RuntimeError("ACTIVE_PACKET_OR_SIGNATURE_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    if assets:
        for ref in (*packet["inputs"]["source_files"].values(),
                    *packet["inputs"]["model_assets"].values(),
                    packet["inputs"]["model_python"],
                    packet["inputs"]["native_tracer"],
                    *(packet["inputs"][name] for name in (
                        "upstream_admission_descriptor", "upstream_admission_frame_time_map",
                        "upstream_admission_seal", "upstream_admission_result",
                        "upstream_admission_receipt"))):
            if errors := validate_artifact_ref(ref):
                raise RuntimeError("INPUT_SHA_DRIFT:" + ";".join(errors))


def register(packet: dict[str, Any]) -> None:
    """Publish an exact staged route and this packet in one governance CAS."""
    _assert_route_ready(packet)
    task_id = packet["task_id"]
    output, packet_path, receipt = _paths(task_id)
    if any(path.exists() for path in (output, packet_path, receipt)):
        raise RuntimeError("FRESH_TASK_ATTEMPT_REQUIRED")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    if (state.get("next_task") or any(item.get("status") in
            {"WAIT_GPU_RESOURCE", "RUNNING", "CLAIMED"} for item in state["tasks"])
            or any(item.get("execution_allowed") for item in index["task_packets"])):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    output.mkdir(parents=True)
    atomic_json(output / "WRITER.json", packet["writer"])
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    stamp = now_iso()
    state["tasks"].append({
        "task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "WAIT_GPU_RESOURCE", "session": packet["inputs"]["source_session"],
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": 0, "task_packet": artifact_ref(packet_path),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": task_id, "session": packet["inputs"]["source_session"],
        "prerequisites": packet["prerequisites"],
        "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "one full HaWoR session or finite packet deadline",
    }
    if errors := validate_task_state(state):
        raise RuntimeError("TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": task_id, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(packet_path)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision=packet["execution_revision"] + "_WAIT_GPU",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def _publish_state(packet: dict[str, Any], status: str | None = None) -> None:
    _fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    if status is not None:
        if row["status"] != "WAIT_GPU_RESOURCE" or status != "RUNNING":
            raise RuntimeError("GPU_STATE_TRANSITION_DRIFT")
        row["status"] = status
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    if errors := validate_task_state(state):
        raise RuntimeError("HEARTBEAT_STATE_PREFLIGHT:" + ";".join(errors))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] +
                              ("_GPU_ACQUIRED" if status else "_HEARTBEAT"),
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=Path(__file__))


def _resume_summary(packet: dict[str, Any], elapsed: float) -> None:
    output, _, _ = _paths(packet["task_id"])
    number = int(elapsed // 600)
    path = output / f"RESUME_SUMMARY_{number:04d}.json"
    if number and not path.exists():
        atomic_json(path, {
            "schema_version": "exact78-newraw-hawor-resume-v3",
            "elapsed_seconds": round(elapsed, 1),
            "stage_sealed": (output / "candidate/HAWOR_STAGE_SEAL.json").is_file(),
            "algorithm_attempts": int((output / "WORKER_STARTED.json").is_file()),
            "next_action": "Only a SHA-verified same-signature stage seal may be reused; unsealed partial is not input.",
        })


def _lease_args(packet: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(task_id=packet["task_id"], attempt_id="attempt_0001",
        executor_epoch=1, priority="CANARY", gpu_id=0, min_free_mib=20000,
        lease=GPU_LEASE, lock=GPU_LOCK)


def _acquire_lease(packet: dict[str, Any]) -> bool:
    from chaoyang.ops.run_gpu_command_with_v71_lease import try_acquire
    output, _, _ = _paths(packet["task_id"])
    began = time.monotonic()
    last_beat = began
    deadline = min(began + GPU_WAIT_SECONDS_MAX,
                   began + max(0, (datetime.fromisoformat(packet["deadline_at"])
                                   - datetime.now().astimezone()).total_seconds() - 120))
    while time.monotonic() < deadline:
        _fence(packet)
        if try_acquire(_lease_args(packet), packet["writer"]["fencing_token"]) is not None:
            return True
        elapsed = time.monotonic() - began
        if time.monotonic() - last_beat >= 50:
            _publish_state(packet)
            _resume_summary(packet, elapsed)
            last_beat = time.monotonic()
        time.sleep(5)
    (output / "gpu_wait.log").write_text(
        "GPU_WAIT_BUDGET_EXHAUSTED; algorithm_attempts=0\n", encoding="utf-8")
    return False


def _lease_update(packet: dict[str, Any], *, worker_pid: int | None = None,
                  release_reason: str | None = None) -> None:
    from chaoyang.governance.gpu_lease_v71 import heartbeat as beat, release
    from chaoyang.ops.run_gpu_command_with_v71_lease import load
    token = packet["writer"]["fencing_token"]
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = load(GPU_LEASE)
        if current.get("pid") != os.getpid() or current.get("fencing_token") != token:
            raise RuntimeError("GPU_LEASE_FENCE_DRIFT")
        updated = (release(current, pid=os.getpid(), fencing_token=token,
                           reason=release_reason) if release_reason is not None
                   else beat(current, pid=os.getpid(), fencing_token=token))
        if worker_pid is not None and release_reason is None:
            updated["gpu_process_pid"] = worker_pid
        atomic_json(GPU_LEASE, updated)


def _worker_environment(output: Path) -> dict[str, str]:
    cache = output / "cache"
    cache.mkdir(exist_ok=False)
    cuda_cache = cache / "nvidia-compute"
    cuda_cache.mkdir(exist_ok=False)
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_EXE",
                 "CONDA_SHLVL", "CONDA_PROMPT_MODIFIER", "_CE_CONDA", "_CE_M",
                 "LD_PRELOAD"):
        environment.pop(name, None)
    environment.update({
        "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1", "CONDA_PREFIX": str(MODEL_ENV),
        "PATH": f"{MODEL_ENV / 'bin'}:/usr/bin:/bin",
        "LD_LIBRARY_PATH": str(MODEL_ENV / "lib"),
        "TMPDIR": str(cache), "TMP": str(cache), "TEMP": str(cache),
        "XDG_CACHE_HOME": str(cache), "HF_HOME": str(cache / "huggingface"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "NUMBA_CACHE_DIR": str(cache / "numba"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "YOLO_CONFIG_DIR": str(cache / "ultralytics"),
        "CUDA_CACHE_PATH": str(cuda_cache),
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": "0",
    })
    return environment


def _native_open_audit(packet: dict[str, Any], trace: Path, output: Path) -> dict[str, Any]:
    """Fail on persistent external writable opens; disclose OS IPC separately.

    This trace records opens, not write syscalls. A successful O_RDWR is a
    conservative writable-open failure for persistent files, not proof that
    bytes changed. Failed external attempts are recorded, and /path/to/private_data write
    attempts remain fatal even when the kernel denies them.
    """
    inputs = packet["inputs"]
    permitted_external = {str(Path(ref["path"]).resolve())
                          for ref in inputs["source_files"].values()}
    producer = Path(inputs["upstream_admission_result"]["path"]).parent.resolve()
    consumer, _, _ = _paths(packet["task_id"])
    opened_external: set[str] = set()
    forbidden: set[str] = set()
    failed_write_attempts: set[str] = set()
    persistent_writable_opens: set[str] = set()
    transient_ipc_opens: set[str] = set()
    runtime_device_opens: set[str] = set()
    unknown_write_returns: set[str] = set()
    calls = 0
    successful_calls = 0
    quoted = re.compile(r'"((?:[^"\\]|\\.)*)"')
    returned_fd = re.compile(r"\s=\s(-?\d+)(?:\s|$)")
    with trace.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not any(token in line for token in ("open(", "openat(", "openat2(")):
                continue
            calls += 1
            match = quoted.search(line)
            if match is None:
                continue
            raw = match.group(1)
            write_open = any(flag in line for flag in
                             ("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC"))
            returned = returned_fd.search(line)
            successful = returned is not None and int(returned.group(1)) >= 0
            successful_calls += int(successful)
            if write_open:
                if not raw.startswith("/"):
                    forbidden.add("UNRESOLVED_RELATIVE_WRITABLE_OPEN:" + raw)
                else:
                    resolved = str(Path(raw).resolve(strict=False))
                    outside = not Path(resolved).is_relative_to(REPO_ROOT)
                    if outside and returned is None:
                        unknown_write_returns.add(resolved)
                        forbidden.add("UNRESOLVED_EXTERNAL_WRITABLE_OPEN_RETURN:" + resolved)
                    elif outside and not successful:
                        failed_write_attempts.add(resolved)
                    elif outside and successful:
                        if (re.fullmatch(r"/dev/shm/sem\.[A-Za-z0-9]+", raw)
                                or re.fullmatch(r"/proc/self/task/[0-9]+/comm", raw)):
                            transient_ipc_opens.add(raw)
                        elif (raw in {"/dev/null", "/dev/urandom", "/dev/random"}
                              or re.fullmatch(r"/dev/nvidia(?:[0-9]+|ctl|-uvm(?:-tools)?)", raw)
                              or re.fullmatch(r"/dev/dri/renderD[0-9]+", raw)):
                            runtime_device_opens.add(raw)
                        else:
                            persistent_writable_opens.add(resolved)
                            forbidden.add("OUTSIDE_PROJECT_PERSISTENT_WRITABLE_OPEN:" + resolved)
            if raw.startswith("/path/to/private_data/"):
                path = str(Path(raw).resolve(strict=False))
                if write_open:
                    forbidden.add("EXTERNAL_DATA_WRITE_ATTEMPT:" + path)
                elif path not in permitted_external:
                    forbidden.add("UNDECLARED_EXTERNAL_READ:" + path)
                elif successful:
                    opened_external.add(path)
            if any(token in raw.lower() for token in
                   ("preprocess/all_data", "trackingdata", "pico.jsonl", "manus",
                    "selected_hawor", "accepted_hawor_template")):
                forbidden.add("DERIVED_OR_HARDWARE_PATH:" + raw)
            if raw.startswith(str(REPO_ROOT / "_run/current/exact78_")):
                path = Path(raw).resolve(strict=False)
                if not (path.is_relative_to(producer) or path.is_relative_to(consumer)):
                    forbidden.add("OTHER_EXACT78_ATTEMPT_READ:" + raw)
    video = str(Path(inputs["raw_video"]["path"]).resolve())
    status = ("PASS_NATIVE_OPEN_SCOPE" if video in opened_external and not forbidden
              else "FAILED_OR_INCOMPLETE_NATIVE_OPEN_SCOPE")
    audit = {
        "schema_version": "exact78-newraw-hawor-native-open-audit-v3",
        "status": status, "trace": artifact_ref(trace),
        "open_calls_seen": calls,
        "successful_open_calls_seen": successful_calls,
        "permitted_source_refs": inputs["source_files"],
        "opened_external_paths": sorted(opened_external),
        "source_video_open_observed": video in opened_external,
        "failed_external_write_open_attempts": sorted(failed_write_attempts),
        "unresolved_external_write_open_returns": sorted(unknown_write_returns),
        "successful_persistent_external_writable_opens": sorted(persistent_writable_opens),
        "successful_transient_kernel_ipc_writable_opens": sorted(transient_ipc_opens),
        "successful_runtime_device_writable_opens": sorted(runtime_device_opens),
        "write_open_counts": {
            "failed_external_attempts": len(failed_write_attempts),
            "unresolved_external_returns": len(unknown_write_returns),
            "successful_persistent_external": len(persistent_writable_opens),
            "successful_transient_kernel_ipc": len(transient_ipc_opens),
            "successful_runtime_device": len(runtime_device_opens),
        },
        "forbidden_paths": sorted(forbidden),
        "scope_limit": "Open/openat/openat2 only: exact kernel IPC opens are disclosed, not persistent user artifacts; actual write syscalls are not proven or disproven. No model quality or clock claim.",
    }
    _write_new_json(output, audit)
    return audit


def _stage_signature(packet: dict[str, Any]) -> str:
    inputs = packet["inputs"]
    return stage_signature(
        inputs=[*(inputs[name] for name in (
            "upstream_admission_descriptor", "upstream_admission_frame_time_map",
            "upstream_admission_seal", "upstream_admission_result",
            "upstream_admission_receipt", "model_python", "native_tracer")),
            *(inputs["source_files"][name] for name in sorted(inputs["source_files"]))],
        code=packet["runtime_code"], config=packet["config"],
        weights=[inputs["model_assets"][name] for name in sorted(inputs["model_assets"])],
        calibration=(inputs["source_files"].get("camera_params")
                     or inputs["source_files"].get("raw/camera_params.json")
                     or "ABSENT"), schema_version="exact78-newraw-hawor-stage-v3")


def _verify_stage(packet: dict[str, Any]) -> dict[str, Any]:
    output, _, _ = _paths(packet["task_id"])
    result_path = output / "candidate/hawor/RESULT.json"
    result = load_json(result_path)
    if (result.get("schema_version") != "exact78-hawor-pure-rgb-result-v3"
            or result.get("status") != "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS"
            or result.get("session_id") != packet["inputs"]["source_session"]
            or result.get("source_frames_decoded") != packet["config"]["frame_count"]
            or result.get("model_input_frames") != packet["config"]["frame_count"]
            or result.get("world_pose_valid") is not False
            or result.get("training_eligible") is not False):
        raise RuntimeError("HAWOR_STAGE_RESULT_CLAIM_DRIFT")
    for name in ("npz", "frame_provenance", "input_audit"):
        if (not isinstance(result.get(name), dict)
                or validate_artifact_ref(result[name])):
            raise RuntimeError("HAWOR_STAGE_OUTPUT_SHA_DRIFT:" + name)
    from chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1 import checked_spec
    spec = checked_spec(packet)
    provenance = load_json(Path(result["frame_provenance"]["path"]))
    audit = load_json(Path(result["input_audit"]["path"]))
    rows = provenance.get("frame_rows")
    if (provenance.get("schema_version") != "exact78-hawor-frame-provenance-v3"
            or provenance.get("session_id") != spec["session_id"]
            or provenance.get("frame_count") != spec["frame_count"]
            or provenance.get("physical_eye") != spec["physical_eye"]
            or provenance.get("time_mode") != spec["time_mode"]
            or not isinstance(rows, list) or len(rows) != spec["frame_count"]):
        raise RuntimeError("HAWOR_FRAME_PROVENANCE_DRIFT")
    for index, (row, original) in enumerate(zip(rows, spec["frame_rows_sealed"])):
        if (row.get("decoded_display_index") != index
                or row.get("model_input_index") != index
                or row.get("output_index") != index
                or row.get("sbs_physical_eye") != spec["physical_eye"]
                or row.get("crop_xywh") != original.get("crop_xywh")
                or row.get("time_mode") != spec["time_mode"]
                or any(row.get(key) != original.get(key) for key in (
                    "pts", "best_effort_timestamp", "qpc_raw_tick",
                    "capture_timestamp_raw", "capture_to_decode_alignment"))):
            raise RuntimeError(f"HAWOR_FRAME_MAP_DRIFT:{index}")
    if (audit.get("schema_version") != "exact78-hawor-input-audit-v3"
            or audit.get("raw_source_ref") != packet["inputs"]["raw_video"]
            or audit.get("source_role") != "new_session_candidate"
            or audit.get("source_split_claim") != "NOT_ASSIGNED"
            or audit.get("infiller_invoked") is not False
            or audit.get("world_pose_emitted") is not False):
        raise RuntimeError("HAWOR_INPUT_AUDIT_DRIFT")
    import numpy as np
    with np.load(Path(result["npz"]["path"]), allow_pickle=False) as data:
        count = spec["frame_count"]
        shapes = {
            "prediction_valid": (2, count),
            "joints_3d_camera": (2, count, 21, 3),
            "joints_2d": (2, count, 21, 2),
            "root_orient_camera": (2, count, 3, 3),
            "hand_pose_rotmat": (2, count, 15, 3, 3),
            "betas": (2, count, 10),
            "intrinsics": (count, 3, 3),
        }
        if (any(name not in data or data[name].shape != shape
                for name, shape in shapes.items())
                or not np.array_equal(data["original_frame_indices"], np.arange(count))):
            raise RuntimeError("HAWOR_NPZ_FRAME_OR_SIDE_DRIFT")
        counts = {side: int(data["prediction_valid"][index].sum())
                  for index, side in enumerate(("left", "right"))}
    if result.get("model_prediction_frames") != counts:
        raise RuntimeError("HAWOR_PREDICTION_COUNT_DRIFT")
    native = load_json(output / "candidate/NATIVE_OPEN_AUDIT.json")
    native_lists = (
        "failed_external_write_open_attempts",
        "unresolved_external_write_open_returns",
        "successful_persistent_external_writable_opens",
        "successful_transient_kernel_ipc_writable_opens",
        "successful_runtime_device_writable_opens",
    )
    counts = native.get("write_open_counts", {})
    if (native.get("schema_version") != "exact78-newraw-hawor-native-open-audit-v3"
            or native.get("status") != "PASS_NATIVE_OPEN_SCOPE"
            or native.get("source_video_open_observed") is not True
            or native.get("forbidden_paths") != []
            or native.get("successful_persistent_external_writable_opens") != []
            or native.get("trace") != artifact_ref(output / "candidate/native_open_trace.log")
            or any(not isinstance(native.get(name), list) for name in native_lists)
            or counts != {
                "failed_external_attempts": len(native.get(native_lists[0], [])),
                "unresolved_external_returns": len(native.get(native_lists[1], [])),
                "successful_persistent_external": len(native.get(native_lists[2], [])),
                "successful_transient_kernel_ipc": len(native.get(native_lists[3], [])),
                "successful_runtime_device": len(native.get(native_lists[4], [])),
            }):
        raise RuntimeError("HAWOR_NATIVE_OPEN_AUDIT_FAILED")
    return {"result": artifact_ref(result_path),
            "native_open_audit": artifact_ref(output / "candidate/NATIVE_OPEN_AUDIT.json"),
            "prediction_counts": counts}


def _sealed_stage(packet: dict[str, Any]) -> dict[str, Any] | None:
    output, _, _ = _paths(packet["task_id"])
    path = output / "candidate/HAWOR_STAGE_SEAL.json"
    if not path.is_file():
        if (output / "candidate").exists():
            raise RuntimeError("UNSEALED_PARTIAL_CANNOT_RESUME")
        return None
    seal = load_json(path)
    verified = _verify_stage(packet)
    if (seal.get("schema_version") != "exact78-newraw-hawor-stage-seal-v3"
            or seal.get("status") != "SEALED_CAMERA_PREDICTIONS"
            or seal.get("task_id") != packet["task_id"]
            or seal.get("signature") != _stage_signature(packet)
            or seal.get("result") != verified["result"]
            or seal.get("native_open_audit") != verified["native_open_audit"]):
        raise RuntimeError("SEALED_STAGE_SIGNATURE_OR_REF_DRIFT")
    return {**verified, "seal": artifact_ref(path)}


def _run_worker(packet: dict[str, Any]) -> tuple[int, Path, int, dict[str, Any] | None]:
    output, _, _ = _paths(packet["task_id"])
    _fence(packet, assets=True)
    if not _acquire_lease(packet):
        return 3, output / "gpu_wait.log", 0, None
    worker: subprocess.Popen | None = None
    token = packet["writer"]["fencing_token"]
    try:
        _fence(packet, assets=True)
        _publish_state(packet, "RUNNING")
        resumed = _sealed_stage(packet)
        if resumed is not None:
            log = output / "sealed_resume.log"
            log.write_text("SAME_SIGNATURE_SEALED_STAGE_REUSED; algorithm_attempts=0\n",
                           encoding="utf-8")
            return 0, log, 0, resumed
        candidate = output / "candidate"
        candidate.mkdir(exist_ok=False)
        trace = candidate / "native_open_trace.log"
        command = [str(STRACE), "-f", "-s", "4096", "-e",
                   "trace=open,openat,openat2", "-o", str(trace),
                   str(MODEL_PYTHON), "-B", "-m", "chaoyang.cli", "run", OPERATION,
                   "--producer-result",
                   packet["inputs"]["upstream_admission_result"]["path"],
                   "--output-root", str(candidate / "hawor")]
        log = output / "worker.log"
        with log.open("x", encoding="utf-8") as stream:
            worker = subprocess.Popen(command, cwd=REPO_ROOT,
                env=_worker_environment(output), stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(output / "WORKER_STARTED.json", {
                "schema_version": "exact78-newraw-hawor-worker-start-v3",
                "task_id": packet["task_id"], "pid": worker.pid,
                "algorithm_attempts": 1, "started_at": now_iso(),
            })
            _lease_update(packet, worker_pid=worker.pid)
            began = time.monotonic()
            last_beat = began
            last_lease = began
            while worker.poll() is None:
                if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL)
                        worker.wait(timeout=20)
                    return 124, log, 1, None
                if time.monotonic() - last_beat >= 50:
                    _publish_state(packet)
                    _resume_summary(packet, time.monotonic() - began)
                    last_beat = time.monotonic()
                if time.monotonic() - last_lease >= 30:
                    _lease_update(packet, worker_pid=worker.pid)
                    last_lease = time.monotonic()
                time.sleep(2)
        if worker.returncode != 0:
            return worker.returncode, log, 1, None
        native = _native_open_audit(packet, trace, candidate / "NATIVE_OPEN_AUDIT.json")
        if native["status"] != "PASS_NATIVE_OPEN_SCOPE":
            return 2, log, 1, None
        verified = _verify_stage(packet)
        seal = {
            "schema_version": "exact78-newraw-hawor-stage-seal-v3",
            "status": "SEALED_CAMERA_PREDICTIONS", "task_id": packet["task_id"],
            "session_id": packet["inputs"]["source_session"],
            "result": verified["result"],
            "native_open_audit": verified["native_open_audit"],
            "signature": _stage_signature(packet),
            "downstream_use": "SEALED_STAGE_ONLY",
        }
        _write_new_json(candidate / "HAWOR_STAGE_SEAL.json", seal)
        return 0, log, 1, {**verified, "seal": artifact_ref(candidate / "HAWOR_STAGE_SEAL.json")}
    finally:
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=20)
        _lease_update(packet, release_reason="TASK_WORKER_EXIT")


def _next_action(status: str) -> str:
    if status == "PASSED":
        return "Consume only sealed HaWoR with a separate Robot Task Packet; inspect per-frame identity and gaps."
    return "No sealed HaWoR is available; inspect failure and register a new finite successor."


def _finish(packet: dict[str, Any], code: int, log: Path,
            attempts: int, stage: dict[str, Any] | None,
            error: str | None = None) -> dict[str, Any]:
    _fence(packet)
    if code == 0 and stage is not None:
        verified_stage = _sealed_stage(packet)
        if verified_stage != stage:
            raise RuntimeError("FINAL_STAGE_SEAL_DRIFT")
    output, _, receipt = _paths(packet["task_id"])
    if (output / "RESULT.json").exists() or receipt.exists():
        raise RuntimeError("DUPLICATE_TERMINAL_RESULT")
    inputs = packet["inputs"]
    post_refs = {
        **{f"raw:{name}": ref for name, ref in inputs["source_files"].items()},
        **{f"model:{name}": ref for name, ref in inputs["model_assets"].items()},
        "model_python": inputs["model_python"],
        "native_tracer": inputs["native_tracer"],
        **{name: inputs[name] for name in (
            "upstream_admission_descriptor", "upstream_admission_frame_time_map",
            "upstream_admission_seal", "upstream_admission_result",
            "upstream_admission_receipt")},
    }
    input_errors = [f"{name}:{';'.join(errors)}" for name, ref in
                    post_refs.items() if (errors := validate_artifact_ref(ref))]
    raw_sha_stable = not any(item.startswith("raw:") for item in input_errors)
    native_path = output / "candidate/NATIVE_OPEN_AUDIT.json"
    external_write_observed: bool | str = "NOT_EVALUATED"
    external_write_attempt_observed: bool | str = "NOT_EVALUATED"
    transient_kernel_ipc_opens: int | str = "NOT_EVALUATED"
    if native_path.is_file():
        native = load_json(native_path)
        external_write_observed = bool(
            native.get("successful_persistent_external_writable_opens", []))
        external_write_attempt_observed = bool(
            native.get("failed_external_write_open_attempts", [])) or any(
                "EXTERNAL_DATA_WRITE_ATTEMPT" in item
                for item in native.get("forbidden_paths", []))
        transient_kernel_ipc_opens = len(
            native.get("successful_transient_kernel_ipc_writable_opens", []))
    if code == 3 and attempts == 0:
        status, failure = "BLOCKED_RESOURCE", "NOT_EVALUATED_BUDGET"
    elif code == 0 and stage and not input_errors and error is None:
        status, failure = "PASSED", None
    else:
        status = "FAILED_RUNTIME_FINAL"
        failure = error or ("INPUT_SHA_DRIFT" if input_errors else
                            "UNSEALED_STAGE_OR_WORKER_FAILED")
    result = {
        "schema_version": "exact78-newraw-hawor-gpu-execution-result-v3",
        "task_id": packet["task_id"], "status": status,
        "failure_code": failure, "worker_returncode": code,
        "algorithm_attempts": attempts,
        "source_session": packet["inputs"]["source_session"],
        "source_frames_expected": packet["config"]["frame_count"],
        "producer_result": packet["inputs"]["upstream_admission_result"],
        "post_input_sha_errors": input_errors,
        "stage": stage or {}, "worker_log": artifact_ref(log),
        "stage_status": "SEALED_CAMERA_PREDICTIONS" if status == "PASSED" else "NOT_SEALED",
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
        "user_review": "PENDING", "general_operation": None,
        "source_sha_stable_at_terminal": raw_sha_stable,
        "external_write_open_observed": external_write_observed,
        "external_write_attempt_observed": external_write_attempt_observed,
        "transient_kernel_ipc_writable_opens": transient_kernel_ipc_opens,
        "files_deleted": False,
        "claim_limit": packet["claim_limit"],
        "next_action": _next_action(status),
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {
        "schema_version": "exact78-newraw-hawor-gpu-receipt-v3",
        "task_id": packet["task_id"], "status": status,
        "result": artifact_ref(output / "RESULT.json"),
        "deletion_authorized": False,
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(output / "RESULT.json"),
               terminal_receipt=artifact_ref(receipt),
               quality="CAMERA_PREDICTIONS_ONLY" if status == "PASSED" else
                       "NOT_EVALUATED_BUDGET" if status == "BLOCKED_RESOURCE" else
                       "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == packet["task_id"]:
            item["execution_allowed"] = False
    index.update(packet_revision=packet["execution_revision"] + "_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=packet["execution_revision"] + "_" + status,
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX,
        task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(output / "RESULT.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-result", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    packet = packet_value(args.producer_result)
    if args.check_only:
        try:
            readiness = _assert_route_ready(packet)
        except RuntimeError as exc:
            readiness = str(exc)
        print(json.dumps({
            "status": readiness,
            "task_id": packet["task_id"], "run_signature": packet["run_signature"],
            "source_session": packet["inputs"]["source_session"],
            "source_frames_expected": packet["config"]["frame_count"],
            "runtime_code_refs": len(packet["runtime_code"]),
            "model_asset_refs": len(packet["inputs"]["model_assets"]),
            "gpu_count": 1, "model_calls": "one full HaWoR session",
            "registered": False, "model_executed": False,
        }, ensure_ascii=False))
        return 0
    register(packet)
    try:
        code, log, attempts, stage = _run_worker(packet)
        result = _finish(packet, code, log, attempts, stage)
    except Exception as exc:
        output, _, _ = _paths(packet["task_id"])
        log = output / "executor_error.log"
        if not log.exists():
            log.write_text(f"{type(exc).__name__}:{exc}\n", encoding="utf-8")
        result = _finish(packet, 2, log,
                         int((output / "WORKER_STARTED.json").exists()), None,
                         error=f"{type(exc).__name__}:{exc}")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
