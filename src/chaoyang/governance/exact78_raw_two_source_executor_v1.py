"""Shared fenced executor for two independent T0 Exact78 development captures.

Each small entry module supplies one immutable target and receives its own
Task Packet, writer, GPU lease, attempt and terminal receipt.  Importing this
module or running --check-only never registers or starts an algorithm task.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time
from types import SimpleNamespace
from typing import Any

from chaoyang.cli.exact78_submit import build_request
from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
T0 = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
T0_CORRECTION = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001/RESULT.json"
T1A = REPO_ROOT / "_run/current/exact78_raw_to_robot_t1a_candidate_build_v1_20260930/attempts/attempt_0001/RESULT.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
GPU_LEASE = REPO_ROOT / "_run/current/GPU_LEASE.json"
GPU_LOCK = REPO_ROOT / "_run/current/GPU_LEASE.lock"
PYTHON = Path("/usr/local/bin/python")
OPERATION = "run_exact78_raw_to_robot_v3"
FROZEN_UPSTREAM = {
    T0 / "INPUT_FREEZE.json": "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88",
    T0 / "FRAME_TIME_MAP.json": "7b68b5b1b64b35e096f4ef8ea16665e73428fede09a5d7b741f8cad8b94de70a",
    T0_CORRECTION: "7cd873022e3a72b05fa88701cccb76dec7fda5321c4eed11b2aa14f69b95314c",
    T1A: "264960825aa55f0e5e037419501e7973b5446911dba17004677267cee2f0ad98",
}
RUNTIME_RELATIVE = (
    "src/chaoyang/ops/run_exact78_raw_to_robot_v3.py",
    "src/chaoyang/ops/run_exact78_raw_to_robot_v2.py",
    "src/chaoyang/ops/run_exact78_hawor_raw_v3.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/cli/exact78_submit.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/governance/gpu_lease_v71.py",
)
VENDOR_RELATIVE = (
    "vendor/HaWoR/scripts/scripts_test_video/detect_track_video.py",
    "vendor/HaWoR/scripts/scripts_test_video/hawor_video.py",
    "vendor/HaWoR/lib/pipeline/tools.py",
    "vendor/HaWoR/hawor/utils/process.py",
    "vendor/HaWoR/weights/hawor/model_config.yaml",
)
ASSET_RELATIVE = (
    "vendor/HaWoR/weights/hawor/checkpoints/hawor.ckpt",
    "vendor/HaWoR/weights/external/detector.pt",
    "vendor/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl",
    "vendor/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl",
    "_run/current/environments/hawor-py310-v1/.chaoyang-hardlink-snapshot/manifest.json.gz",
    "assets/robot/ROBOT_ASSET_PIN.json",
    "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json",
    "assets/robot/V2相机固定法兰PRO版本.STL",
)


@dataclass(frozen=True)
class Target:
    task_id: str
    session: Path
    session_id: str
    frame_count: int
    fps_nominal: int
    time_mode: str
    physical_eye: str
    receipt_name: str
    execution_revision: str
    objective: str
    claim_limit: str
    wall_seconds_max: int

    @property
    def output(self) -> Path:
        return REPO_ROOT / "_run/current" / self.task_id / "attempts/attempt_0001"

    @property
    def packet(self) -> Path:
        return REPO_ROOT / "tasks/current" / self.task_id / "TASK_PACKET.json"

    @property
    def receipt(self) -> Path:
        return REPO_ROOT / "tasks/receipts" / self.receipt_name

    @property
    def command(self) -> str:
        return (f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} "
                f"--session {self.session} --output-root {self.output / 'candidate'}")


def _index_copy() -> dict[str, Any]:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    return {key: copy.deepcopy(value) for key, value in index.items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def _candidate_route_readiness() -> tuple[bool, str]:
    """Allow one CAS to publish the staged builder route with its first packet."""
    from chaoyang.cli import _maintained_operations
    if OPERATION in _maintained_operations():
        return True, "PUBLISHED_CANDIDATE_ROUTE"
    from chaoyang.governance.common import BASELINE_REGISTRY_PATH
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH))
    route = staged["special_status_contracts"].get(
        "exact78_raw_to_robot_candidate_cross_source_v3_20260930", {})
    # Dict references are not hashable; compare their canonical SHA/path pairs.
    closure = {(ref.get("path"), ref.get("sha256"))
               for ref in route.get("code_closure", []) if isinstance(ref, dict)}
    required = {tuple(artifact_ref(REPO_ROOT / path)[key] for key in ("path", "sha256"))
                for path in ("src/chaoyang/ops/run_exact78_raw_to_robot_v3.py",
                             "src/chaoyang/ops/run_exact78_raw_to_robot_v2.py",
                             "src/chaoyang/ops/run_exact78_hawor_raw_v3.py",
                             "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py",
                             "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py")}
    if (route.get("candidate_operation") != OPERATION
            or route.get("current_runnable_operation") is not None
            or route.get("output_schema") != "exact78-raw-to-robot-candidate-result-v3"
            or not required.issubset(closure)):
        return False, "V3_CANDIDATE_ROUTE_NOT_STAGED"
    return True, "READY_TO_PUBLISH_CANDIDATE_ROUTE_WITH_PACKET"


def checked_inputs(target: Target) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only current contracts, T0 and the target's allowed raw source."""
    if datetime.now().astimezone() >= CAMPAIGN_END:
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    for path, sha in FROZEN_UPSTREAM.items():
        if artifact_ref(path)["sha256"] != sha:
            raise RuntimeError(f"FROZEN_UPSTREAM_SHA_DRIFT:{path}")
    if load_json(T0_CORRECTION).get("status") != "PASSED" or load_json(T1A).get("status") != "PASSED":
        raise RuntimeError("T0_OR_T1A_NOT_SEALED")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_OPERATION_ALREADY_PROMOTED")
    request = build_request(session=target.session)
    if len(request["sources"]) != 1 or request["sources"][0]["session_id"] != target.session_id:
        raise RuntimeError("FROZEN_TARGET_SESSION_DRIFT")
    source = request["sources"][0]
    from chaoyang.ops.run_exact78_raw_to_robot_v3 import _validate_t0_target

    audit = _validate_t0_target(source, freeze=T0 / "INPUT_FREEZE.json",
                                frame_map=T0 / "FRAME_TIME_MAP.json")
    if (audit["frame_count"] != target.frame_count
            or audit["time_mode"] != target.time_mode
            or audit["physical_eye"] != target.physical_eye):
        raise RuntimeError("FROZEN_TARGET_CAPABILITY_DRIFT")
    inputs = {
        "plan": artifact_ref(PLAN), "route": artifact_ref(ROUTE),
        "t0_input_freeze": artifact_ref(T0 / "INPUT_FREEZE.json"),
        "t0_frame_time_map": artifact_ref(T0 / "FRAME_TIME_MAP.json"),
        "t0_correction": artifact_ref(T0_CORRECTION),
        "t1a_result": artifact_ref(T1A),
        "source_files": source["files"],
        "raw_video": audit["raw_video"],
        "submission_input_sha256": request["submission_input_sha256"],
        "target_audit": audit,
        "assets": [artifact_ref(REPO_ROOT / item) for item in ASSET_RELATIVE],
        "vendor_code": [artifact_ref(path) for path in
                        sorted((REPO_ROOT / "vendor/HaWoR").rglob("*.py"))
                        if path.is_file() and not path.is_symlink()],
        "vendor_config": [artifact_ref(REPO_ROOT / item) for item in VENDOR_RELATIVE],
    }
    return inputs, request


def packet_value(target: Target, script: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": target.task_id}
    runtime = [artifact_ref(script), artifact_ref(Path(__file__)),
               *(artifact_ref(REPO_ROOT / item) for item in RUNTIME_RELATIVE),
               *inputs["vendor_code"]]
    config = {"session_id": target.session_id, "frame_count": target.frame_count,
              "fps_nominal": target.fps_nominal, "time_mode": target.time_mode,
              "physical_eye": target.physical_eye, "gpu_lease_count": 1,
              "cpu_threads_total": 2, "model_calls": "one full HaWoR session",
              "clean": "NOT_IN_TASK", "geometry": "NOT_IN_TASK", "training_steps": 0,
              "ffmpeg_compat": "vsync_0_for_ffmpeg_4_4"}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": target.command, "schema": "exact78-raw-two-source-executor-v1",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": target.task_id,
        "objective": target.objective,
        "phase": "EXACT78_RAW_TO_ROBOT_T2_TWO_SOURCE",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": target.execution_revision,
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(seconds=target.wall_seconds_max), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     str((T0 / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
                     str((T0 / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
                     str(T0_CORRECTION.relative_to(REPO_ROOT)),
                     str(T1A.relative_to(REPO_ROOT)), str(ROUTE.relative_to(REPO_ROOT)),
                     str(script.relative_to(REPO_ROOT))],
        "write_set": [str(target.output.relative_to(REPO_ROOT)),
                      str(target.packet.relative_to(REPO_ROOT)),
                      str(target.receipt.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "T0_T1A_SEALED",
                          "ONE_GPU_LEASE", "SOURCE_SHA_PINNED", "EXTERNAL_DATA_READ_ONLY",
                          "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["candidate/HAWOR_STAGE_SEAL.json",
                             "candidate/NATIVE_OPEN_AUDIT.json",
                             "candidate/hawor/RESULT.json",
                             "candidate/robot/RESULT.json",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/ROBOT_FRAME_STATES.json",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/ego_robot_diagnostic.mp4",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/fixed_external_full_robot.mp4",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/raw_motion_source_review.mp4",
                             "candidate/ROBOT_STAGE_SEAL.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": target.wall_seconds_max,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "GPU_LEASE_UNAVAILABLE",
                            "STAGE_SEAL_FAIL", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": target.claim_limit,
        "writer": writer, "inputs": inputs, "runtime_code": runtime, "config": config,
        "run_signature": signature, "commands": [target.command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    return packet


def _fence(target: Target, packet: dict[str, Any], *, assets: bool = False) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(target.output / "WRITER.json") != writer):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row["status"] in
              ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if (len(active) != 1 or active[0]["task_id"] != target.task_id
            or active[0]["run_signature"] != packet["run_signature"]):
        raise RuntimeError("DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    allowed = [row for row in load_json(REPO_ROOT / "tasks/current/INDEX.json")["task_packets"]
               if row.get("execution_allowed") is True]
    if (len(allowed) != 1 or allowed[0]["task_id"] != target.task_id
            or allowed[0]["packet_sha256"] != artifact_ref(target.packet)["sha256"]):
        raise RuntimeError("PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    if assets:
        for ref in (*packet["inputs"]["assets"], *packet["inputs"]["source_files"].values(),
                    *packet["inputs"]["vendor_config"]):
            if errors := validate_artifact_ref(ref):
                raise RuntimeError("INPUT_SHA_DRIFT:" + ";".join(errors))


def register(target: Target, script: Path, packet: dict[str, Any]) -> None:
    ready, reason = _candidate_route_readiness()
    if not ready:
        raise RuntimeError(reason)
    if any(path.exists() for path in (target.output, target.packet, target.receipt)):
        raise RuntimeError("FRESH_TARGET_TASK_REQUIRED")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    target.output.mkdir(parents=True)
    atomic_json(target.output / "WRITER.json", packet["writer"])
    target.packet.parent.mkdir(parents=True)
    atomic_json(target.packet, packet)
    stamp = now_iso()
    state["tasks"].append({
        "task_id": target.task_id, "phase": packet["phase"], "attempt": 1,
        "status": "WAIT_GPU_RESOURCE", "session": target.session_id,
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": 0, "task_packet": artifact_ref(target.packet),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {"task_id": target.task_id, "session": target.session_id,
        "prerequisites": packet["prerequisites"],
        "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "one full session or fixed campaign deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": target.task_id,
        "packet_path": str(target.packet.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(target.packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision=target.execution_revision + "_WAIT_GPU",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(REPO_ROOT / "tasks/current/INDEX.json"))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=target.execution_revision + "_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=script, task_packet_index_path=REPO_ROOT / "tasks/current/INDEX.json",
        task_packet_index_value=index)


def heartbeat(target: Target, script: Path, packet: dict[str, Any], elapsed: float) -> None:
    _fence(target, packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == target.task_id)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=target.execution_revision + "_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=script)
    number = int(elapsed // 600)
    if number and not (target.output / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(target.output / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-raw-two-source-resume-v1",
            "elapsed_seconds": round(elapsed, 1),
            "state": row["status"],
            "hawor_sealed": (target.output / "candidate/HAWOR_STAGE_SEAL.json").is_file(),
            "robot_sealed": (target.output / "candidate/ROBOT_STAGE_SEAL.json").is_file(),
            "next_action": "Wait for fenced single-lease worker or inspect terminal stage receipt",
        })


def _set_running(target: Target, script: Path, packet: dict[str, Any]) -> None:
    _fence(target, packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == target.task_id)
    if row["status"] != "WAIT_GPU_RESOURCE":
        raise RuntimeError("GPU_WAIT_STATE_DRIFT")
    row.update(status="RUNNING", heartbeat_at=now_iso(), updated_at=now_iso())
    if errors := validate_task_state(state):
        raise RuntimeError("RUNNING_STATE_PREFLIGHT:" + ";".join(errors))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=target.execution_revision + "_GPU_ACQUIRED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=script)


def _lease_args(target: Target) -> SimpleNamespace:
    return SimpleNamespace(task_id=target.task_id, attempt_id="attempt_0001",
        executor_epoch=1, priority="CANARY", gpu_id=0, min_free_mib=20000,
        lease=GPU_LEASE, lock=GPU_LOCK)


def _acquire_lease(target: Target, script: Path, packet: dict[str, Any], started: float) -> dict | None:
    from chaoyang.ops.run_gpu_command_with_v71_lease import try_acquire

    args = _lease_args(target)
    token = packet["writer"]["fencing_token"]
    deadline = min(datetime.fromisoformat(packet["deadline_at"]), CAMPAIGN_END)
    wait_until = min(time.monotonic() + 900, time.monotonic() + max(
        0, (deadline - datetime.now().astimezone()).total_seconds() - 120))
    last_beat = started
    while time.monotonic() < wait_until:
        _fence(target, packet)
        lease = try_acquire(args, token)
        if lease is not None:
            return lease
        if time.monotonic() - last_beat >= 50:
            heartbeat(target, script, packet, time.monotonic() - started)
            last_beat = time.monotonic()
        time.sleep(5)
    return None


def _update_lease(token: str, *, worker_pid: int | None = None,
                  release_reason: str | None = None) -> None:
    from chaoyang.governance.gpu_lease_v71 import heartbeat as lease_heartbeat, release
    from chaoyang.ops.run_gpu_command_with_v71_lease import load
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lease = load(GPU_LEASE)
        if lease.get("pid") != os.getpid() or lease.get("fencing_token") != token:
            raise RuntimeError("GPU_LEASE_FENCE_DRIFT")
        if release_reason is not None:
            updated = release(lease, pid=os.getpid(), fencing_token=token,
                              reason=release_reason)
        else:
            updated = lease_heartbeat(lease, pid=os.getpid(), fencing_token=token)
            if worker_pid is not None:
                updated["gpu_process_pid"] = worker_pid
        atomic_json(GPU_LEASE, updated)


def _worker_environment(target: Target) -> dict[str, str]:
    cache = target.output / "cache"
    cache.mkdir()
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"),
        YOLO_CONFIG_DIR=str(cache / "ultralytics"), OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="0",
    )
    return environment


def run_worker(target: Target, script: Path, packet: dict[str, Any]) -> tuple[int, Path, int]:
    _fence(target, packet, assets=True)
    started = time.monotonic()
    lease = _acquire_lease(target, script, packet, started)
    if lease is None:
        log = target.output / "gpu_wait.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write("GPU_WAIT_BUDGET_EXHAUSTED; algorithm_attempts=0\n")
        return 3, log, 0
    token = packet["writer"]["fencing_token"]
    worker: subprocess.Popen | None = None
    try:
        _set_running(target, script, packet)
        command = [str(PYTHON), "-B", "-m", "chaoyang.cli", "run", OPERATION,
                   "--session", str(target.session),
                   "--output-root", str(target.output / "candidate")]
        log = target.output / "worker.log"
        with log.open("x", encoding="utf-8") as stream:
            worker = subprocess.Popen(command, cwd=REPO_ROOT,
                env=_worker_environment(target), stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(target.output / "WORKER_STARTED.json", {
                "schema_version": "exact78-raw-worker-start-v1",
                "task_id": target.task_id, "pid": worker.pid,
                "algorithm_attempts": 1, "started_at": now_iso(),
            })
            _update_lease(token, worker_pid=worker.pid)
            last_beat = started
            last_lease_beat = time.monotonic()
            while worker.poll() is None:
                if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
                    os.killpg(worker.pid, signal.SIGTERM)
                    try:
                        worker.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(worker.pid, signal.SIGKILL)
                        worker.wait(timeout=20)
                    return 124, log, 1
                if time.monotonic() - last_beat >= 50:
                    heartbeat(target, script, packet, time.monotonic() - started)
                    last_beat = time.monotonic()
                if time.monotonic() - last_lease_beat >= 30:
                    _update_lease(token, worker_pid=worker.pid)
                    last_lease_beat = time.monotonic()
                time.sleep(2)
        return worker.returncode, log, 1
    finally:
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=20)
        _update_lease(token, release_reason="TASK_WORKER_EXIT")


def _verify_child(target: Target, child: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    if (child.get("status") != "CAMERA_DIAGNOSTIC_STAGES_COMPLETE"
            or child.get("execution_complete") is not True
            or child.get("session_id") != target.session_id
            or child.get("stage_status", {}).get("hawor") != "SEALED_CAMERA_PREDICTIONS"
            or child.get("stage_status", {}).get("robot") != "SEALED_CAMERA_DIAGNOSTIC"):
        return False, {}
    candidate = target.output / "candidate"
    robot = candidate / "robot/RESULT.json"
    if not robot.is_file():
        return False, {}
    result = load_json(robot)
    videos = result.get("verified_videos") or {}
    if (result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("frame_count") != target.frame_count
            or set(videos) != {"ego_robot", "external_full_robot", "raw_motion_source"}
            or any(item.get("decoded_frames") != target.frame_count
                   or validate_artifact_ref(item.get("video", {}))
                   for item in videos.values())):
        return False, {}
    if (child.get("stages", {}).get("robot") != artifact_ref(robot)
            or not (candidate / "HAWOR_STAGE_SEAL.json").is_file()
            or not (candidate / "ROBOT_STAGE_SEAL.json").is_file()
            or load_json(candidate / "NATIVE_OPEN_AUDIT.json").get("status")
               != "PASS_NATIVE_OPEN_SCOPE"):
        return False, {}
    return True, videos


def finish(target: Target, script: Path, packet: dict[str, Any], code: int,
           log: Path, algorithm_attempts: int,
           reason: str | None = None) -> dict[str, Any]:
    _fence(target, packet)
    child_path = target.output / "candidate/RESULT.json"
    child_ref = artifact_ref(child_path) if child_path.is_file() else None
    child, verification_error = {}, None
    if child_ref:
        try:
            child = load_json(child_path)
            complete, videos = _verify_child(target, child)
        except Exception as error:
            complete, videos = False, {}
            verification_error = f"{type(error).__name__}:{error}"
    else:
        complete, videos = False, {}
    input_sha_errors = [f"{ref['path']}:{';'.join(errors)}"
                        for ref in (*packet["inputs"]["assets"],
                                    *packet["inputs"]["source_files"].values(),
                                    *packet["inputs"]["vendor_config"])
                        if (errors := validate_artifact_ref(ref))]
    if code == 3 and algorithm_attempts == 0:
        status, failure_code = "BLOCKED_RESOURCE", "NOT_EVALUATED_BUDGET"
    elif code == 0 and complete and reason is None and not input_sha_errors:
        status, failure_code = "PASSED", None
    else:
        status = "FAILED_RUNTIME_FINAL"
        failure_code = (reason or verification_error or
                        ("INPUT_SHA_DRIFT" if input_sha_errors else
                         "STAGE_OR_VIDEO_VERIFICATION_FAILED"))
    if (target.output / "RESULT.json").exists() or target.receipt.exists():
        raise RuntimeError("DUPLICATE_TERMINAL_RESULT")
    result = {
        "schema_version": "exact78-raw-two-source-execution-result-v1",
        "task_id": target.task_id, "status": status,
        "failure_code": failure_code, "execution_complete": status == "PASSED",
        "worker_returncode": code, "algorithm_attempts": algorithm_attempts,
        "post_input_sha_errors": input_sha_errors,
        "child_result": child_ref, "worker_log": artifact_ref(log),
        "source_session": target.session_id, "source_frames_expected": target.frame_count,
        "source_time_mode": target.time_mode, "source_physical_eye": target.physical_eye,
        "stage_status": child.get("stage_status", {}),
        "robot_video_complete": status == "PASSED", "videos": videos if status == "PASSED" else {},
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
        "user_review": "PENDING", "general_operation": None,
        "source_mutated": False, "files_deleted": False, "robot100_modified": False,
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "next_action": "Inspect per-frame Robot/HaWoR and raw clock evidence; Clean/occlusion remain separate",
        "claim_limit": target.claim_limit,
    }
    atomic_json(target.output / "RESULT.json", result)
    atomic_json(target.receipt, {"schema_version": "exact78-raw-two-source-receipt-v1",
        "task_id": target.task_id, "status": status,
        "result": artifact_ref(target.output / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": target.claim_limit})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == target.task_id)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(target.output / "RESULT.json"),
               terminal_receipt=artifact_ref(target.receipt),
               quality="CAMERA_DIAGNOSTIC_ONLY" if status == "PASSED" else
                       "NOT_EVALUATED_BUDGET" if status == "BLOCKED_RESOURCE" else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == target.task_id:
            item["execution_allowed"] = False
    index.update(packet_revision=target.execution_revision + "_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(REPO_ROOT / "tasks/current/INDEX.json"))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=target.execution_revision + "_" + status,
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=script, task_packet_index_path=REPO_ROOT / "tasks/current/INDEX.json",
        task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(target.output / "RESULT.json")}


def main_for(target: Target, script: Path) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    inputs, request = checked_inputs(target)
    packet = packet_value(target, script, inputs)
    routable, route_reason = _candidate_route_readiness()
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_READY" if routable else
                          "DRY_RUN_NOT_ROUTABLE", "reason": route_reason,
            "task_id": target.task_id, "session_id": target.session_id,
            "source_frames": target.frame_count, "fps_nominal": target.fps_nominal,
            "time_mode": target.time_mode, "physical_eye": target.physical_eye,
            "submission_input_sha256": request["submission_input_sha256"],
            "run_signature": packet["run_signature"],
            "campaign_deadline_at": CAMPAIGN_END.isoformat(),
            "deadline_at": packet["deadline_at"], "gpu_lease_count": 1,
            "registered": False}, ensure_ascii=False))
        return 0
    if not routable:
        raise RuntimeError("CANDIDATE_OPERATION_NOT_REGISTERED")
    register(target, script, packet)
    try:
        code, log, attempts = run_worker(target, script, packet)
        reason = None
    except Exception as error:
        code = 1
        attempts = int((target.output / "WORKER_STARTED.json").is_file())
        reason = f"{type(error).__name__}:{error}"
        log = target.output / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(reason + "\n")
    terminal = finish(target, script, packet, code, log, attempts, reason)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


__all__ = ["Target", "checked_inputs", "packet_value", "main_for"]
