"""Finite fenced Task Packet executor for the sealed 0915 Cards001 CPU replay.

The route and packet must be published through the governance CAS. Check-only
does not mutate state; execute refuses an active task or an unstaged route.
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
import signal
import subprocess
import time
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
    validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.ops import run_exact78_0915_cards001_base_temporal_replay_v1 as operation


SCRIPT = Path(__file__)
OPERATION = operation.OPERATION
TASK_ID = operation.TASK_ID
ROUTE_KEY = "exact78_0915_cards001_base_temporal_replay_v1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
OUTPUT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
STAGE = OUTPUT / operation.OUTPUT_NAME
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
WALL_SECONDS = 3600
H20_LATEST_START = datetime.fromisoformat("2026-10-01T14:47:25.116287+08:00")
H24_HARD_CUTOFF = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
CODE = (
    "src/chaoyang/governance/execute_exact78_0915_cards001_base_temporal_replay_v1.py",
    "src/chaoyang/ops/run_exact78_0915_cards001_base_temporal_replay_v1.py",
    "src/chaoyang/pipeline/exact78_0915_cards001_base_temporal_replay_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/ops/immutable_artifact_io.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/v52_contracts.py",
    "src/chaoyang/governance/v71_contracts.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/cli/__main__.py",
    "src/chaoyang/paths.py",
)


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def packet_value() -> dict[str, Any]:
    preflight = operation.check_only()["preflight"]
    inputs = {key: preflight[key] for key in
              ("input_refs", "raw_source_ref", "robot_asset_pin", "mount_proxy")}
    runtime = [artifact_ref(REPO_ROOT / name) for name in CODE]
    now = datetime.now().astimezone()
    deadline = min(now + timedelta(seconds=WALL_SECONDS), H24_HARD_CUTOFF)
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": TASK_ID}
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run "
               + OPERATION + " --output-root " + str(STAGE))
    config = {"session_id": "play_cards_0915_001", "frame_count": 150,
              "spatial_candidate_count": 32, "top_k": 8,
              "gpu_count": 0, "model_calls": 0, "render_calls": 0,
              "training_steps": 0, "old_robot_q_input": False,
              "numeric_gates_changed": False, "ik_changed": False,
              "calibration_or_absent": "ABSENT_FOR_CAMERA_DIAGNOSTIC"}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": "exact78-0915-cards001-base-temporal-task-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Replay top eight automatic bases over 150 dual-side sealed 0915 frames",
        "phase": operation.PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_0915_CARDS001_BASE_TEMPORAL_REPLAY_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": deadline.isoformat(), "campaign_deadline_at": deadline.isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                     *(str(Path(path).relative_to(REPO_ROOT))
                       for path in list(inputs["input_refs"])[:4]),
                     str(Path(inputs["robot_asset_pin"]["path"]).relative_to(REPO_ROOT)),
                     str(Path(inputs["mount_proxy"]["path"]).relative_to(REPO_ROOT))],
        "write_set": [str(OUTPUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_0915_HAWOR",
                          "SOURCE_SHA_PINNED", "ROBOT_ASSETS_PINNED", "CPU_ONLY",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": [str((STAGE / name).relative_to(OUTPUT)) for name in
                             ("RESULT.json", "FRAME_STATUSES.json")]
                            + ["RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0, "training_steps": 0,
                    "wall_seconds_max": WALL_SECONDS, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "INCOMPLETE_2400_FRAME_SIDE_GRID", "BASELINE_COUNT_DRIFT",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "0915 development camera-frame CPU numeric candidate only; no visual, control or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("BASE_TEMPORAL_PACKET_PREFLIGHT:" + ";".join(errors))
    return packet


def route_readiness(packet: dict[str, Any]) -> str:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    published = load_json(ALGORITHM_CONTRACT_PATH)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    expected = {ref["path"]: ref for ref in packet["runtime_code"]}
    def valid(contract: dict[str, Any]) -> bool:
        route = contract.get("special_status_contracts", {}).get(ROUTE_KEY, {})
        refs = {ref.get("path"): ref for ref in route.get("code_closure", [])
                if isinstance(ref, dict)}
        return (route.get("candidate_operation") == OPERATION
                and route.get("execution_status") == "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION"
                and route.get("task_packet_required") is True
                and route.get("current_runnable_operation") is None
                and route.get("output_schema") == "exact78-0915-cards001-base-temporal-execution-v1"
                and route.get("model_calls_authorized") == 0
                and route.get("render_calls_authorized") == 0
                and route.get("training_steps_authorized") == 0
                and refs == expected)
    published_route = published.get("special_status_contracts", {}).get(ROUTE_KEY)
    if published_route is not None:
        if (OPERATION not in _maintained_operations() or not valid(published)
                or not valid(staged)
                or published_route != staged.get("special_status_contracts", {}).get(ROUTE_KEY)):
            raise RuntimeError("BASE_TEMPORAL_PUBLISHED_ROUTE_DRIFT")
        return "PUBLISHED_CANDIDATE_ROUTE"
    if OPERATION in _maintained_operations() or not valid(staged):
        raise RuntimeError("BASE_TEMPORAL_ROUTE_NOT_STAGED")
    return "READY_TO_PUBLISH_ROUTE_WITH_PACKET"


def readiness(packet: dict[str, Any]) -> str:
    if datetime.now().astimezone() >= H20_LATEST_START:
        raise RuntimeError("BASE_TEMPORAL_H20_NEW_DISPATCH_CUTOFF")
    if datetime.fromisoformat(packet["deadline_at"]) > H24_HARD_CUTOFF:
        raise RuntimeError("BASE_TEMPORAL_H24_PACKET_DEADLINE_DRIFT")
    if any(path.exists() for path in (OUTPUT, PACKET, RECEIPT)):
        raise RuntimeError("BASE_TEMPORAL_FRESH_TASK_REQUIRED")
    state = load_json(TASK_STATE_PATH)
    if (state.get("next_task") or any(row["status"] in
            ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in state["tasks"])):
        raise RuntimeError("BASE_TEMPORAL_ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        raise RuntimeError("BASE_TEMPORAL_ACTIVE_ROUTE_CONFLICT")
    return route_readiness(packet)


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(item) for key, item in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def register(packet: dict[str, Any]) -> None:
    readiness(packet)
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("BASE_TEMPORAL_RUNTIME_SHA_DRIFT:" + ";".join(errors))
    revision = load_json(RECEIPT_PATH)["governance_revision"]
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": operation.PHASE, "attempt": 1,
        "status": "RUNNING", "session": "play_cards_0915_001",
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": _virtual_ref(PACKET, packet),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK_ID, "session": "play_cards_0915_001",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": "TWO_CPU_THREADS_NO_GPU",
                          "stop_condition": "one CPU replay or hard deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("BASE_TEMPORAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    index["task_packets"].append({
        "task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": _virtual_ref(PACKET, packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_0915_BASE_TEMPORAL_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET)["sha256"] != index["task_packets"][-1]["packet_sha256"]:
        raise RuntimeError("BASE_TEMPORAL_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_0915_BASE_TEMPORAL_REGISTERED",
                   expected_revision=revision, generator_path=SCRIPT,
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(OUTPUT / "WRITER.json") != writer):
        raise RuntimeError("BASE_TEMPORAL_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    routed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(active) != 1 or active[0]["task_id"] != TASK_ID
            or active[0]["run_signature"] != packet["run_signature"]
            or len(routed) != 1 or routed[0]["task_id"] != TASK_ID
            or routed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("BASE_TEMPORAL_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")


def heartbeat(packet: dict[str, Any]) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_0915_BASE_TEMPORAL_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT)


class WorkerNotStoppedError(RuntimeError):
    pass


def _stop_fenced(worker: subprocess.Popen[Any], ticks: int) -> None:
    if worker.poll() is not None:
        return
    try:
        if process_identity(worker.pid)["start_ticks"] != ticks:
            raise WorkerNotStoppedError("BASE_TEMPORAL_WORKER_IDENTITY_LOST")
        os.killpg(worker.pid, signal.SIGTERM)
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if process_identity(worker.pid)["start_ticks"] != ticks:
                raise WorkerNotStoppedError("BASE_TEMPORAL_WORKER_IDENTITY_LOST")
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
    except (ProcessLookupError, ChildProcessError, OSError, KeyError,
            ValueError, subprocess.TimeoutExpired):
        if worker.poll() is None:
            raise WorkerNotStoppedError("BASE_TEMPORAL_WORKER_STOP_UNPROVEN")


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    if datetime.now().astimezone() >= H20_LATEST_START:
        raise RuntimeError("BASE_TEMPORAL_H20_NEW_DISPATCH_CUTOFF")
    cache = OUTPUT / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
               NUMBA_CACHE_DIR=str(cache / "numba"), TORCH_EXTENSIONS_DIR=str(cache / "torch"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    deadline = min(datetime.fromisoformat(packet["deadline_at"]), H24_HARD_CUTOFF)
    if (deadline - datetime.now().astimezone()).total_seconds() < 120:
        raise RuntimeError("BASE_TEMPORAL_BUDGET_TOO_SHORT")
    log = OUTPUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(
            ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run", OPERATION,
             "--output-root", str(STAGE)], cwd=REPO_ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            ticks = process_identity(worker.pid)["start_ticks"]
        except (OSError, KeyError, ValueError) as exc:
            if worker.poll() is None:
                raise WorkerNotStoppedError("BASE_TEMPORAL_WORKER_START_IDENTITY_LOST") from exc
            ticks = -1
        last = 0.0
        try:
            while worker.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= WALL_SECONDS or datetime.now().astimezone() >= deadline:
                    raise RuntimeError("BASE_TEMPORAL_HARD_DEADLINE")
                if elapsed - last >= 50:
                    heartbeat(packet)
                    last = elapsed
                time.sleep(2)
        finally:
            _stop_fenced(worker, ticks)
    return worker.returncode, log


def verify_candidate(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    result_path, rows_path = STAGE / "RESULT.json", STAGE / "FRAME_STATUSES.json"
    if not result_path.is_file() or not rows_path.is_file():
        return ["BASE_TEMPORAL_OUTPUT_MISSING"]
    try:
        value, frames = load_json(result_path), load_json(rows_path)
        if (value.get("schema_version") != "exact78-0915-cards001-base-temporal-replay-v1"
                or value.get("status") != "COMPLETE_CPU_NUMERIC_CANDIDATE_NOT_VISUAL_QUALIFICATION"
                or value.get("session_id") != "play_cards_0915_001"
                or value.get("frame_count") != 150 or len(value.get("candidates", [])) != 8
                or len(frames.get("rows", [])) != 2400
                or value.get("input_refs") != packet["inputs"]["input_refs"]
                or value.get("raw_source_ref") != packet["inputs"]["raw_source_ref"]
                or value.get("robot_asset_pin") != packet["inputs"]["robot_asset_pin"]
                or value.get("mount_proxy") != packet["inputs"]["mount_proxy"]
                or value.get("control_ground_truth") is not False
                or value.get("visual_qualified") is not False):
            errors.append("BASE_TEMPORAL_RESULT_SCOPE_DRIFT")
        keys = [(row["candidate_index"], row["frame"], row["side"])
                for row in frames["rows"]]
        expected = {(index, frame, side) for index in value["selected_indices"]
                    for frame in range(150) for side in ("left", "right")}
        if len(keys) != 2400 or set(keys) != expected:
            errors.append("BASE_TEMPORAL_FRAME_GRID_DRIFT")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append("BASE_TEMPORAL_RESULT_PARSE:" + type(exc).__name__)
    return errors


def finish(packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    errors = verify_candidate(packet)
    try:
        preflight = operation.check_only()["preflight"]
        for key in ("input_refs", "raw_source_ref", "robot_asset_pin", "mount_proxy"):
            if preflight[key] != packet["inputs"][key]:
                errors.append("BASE_TEMPORAL_POST_INPUT_DRIFT:" + key)
    except Exception as exc:
        errors.append("BASE_TEMPORAL_POST_INPUT_VERIFY:" + type(exc).__name__)
    for ref in packet["runtime_code"]:
        if drift := validate_artifact_ref(ref):
            errors.extend(drift)
    status = "PASSED" if code == 0 and not errors and reason is None else "FAILED_RUNTIME_FINAL"
    if (OUTPUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("BASE_TEMPORAL_DUPLICATE_TERMINAL")
    result = {"schema_version": "exact78-0915-cards001-base-temporal-execution-v1",
              "task_id": TASK_ID, "status": status, "worker_returncode": code,
              "failure_code": reason, "validation_errors": errors,
              "stage_result": artifact_ref(STAGE / "RESULT.json") if not errors else None,
              "worker_log": artifact_ref(log), "model_calls": 0, "render_calls": 0,
              "visual_qualified": False, "training_eligible": False,
              "adoption": "NOT_ADOPTED", "user_review": "PENDING",
              "next_action": ("Review sealed numeric comparison under a separate task"
                              if status == "PASSED" else "Inspect failure and publish a new successor")}
    atomic_json(OUTPUT / "RESULT.json", result)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECEIPT, {"schema_version": "exact78-0915-cards001-base-temporal-receipt-v1",
                          "task_id": TASK_ID, "status": status,
                          "result": artifact_ref(OUTPUT / "RESULT.json"),
                          "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(OUTPUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if status == "PASSED" else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("BASE_TEMPORAL_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_0915_BASE_TEMPORAL_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=f"EXACT78_0915_BASE_TEMPORAL_{status}",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT, task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUTPUT / "RESULT.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", action="store_true")
    choice.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    packet = packet_value()
    try:
        status = readiness(packet)
    except RuntimeError as exc:
        status, ready = str(exc), False
    else:
        ready = True
    if args.check_only:
        try:
            route = route_readiness(packet)
        except RuntimeError as exc:
            route = str(exc)
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
                          "reason": status, "task_id": TASK_ID,
                          "candidate_route": route,
                          "packet_preflight": "PASS", "run_signature": packet["run_signature"],
                          "frame_count": 150, "top_k": 8, "gpu_count": 0,
                          "wall_seconds_max": WALL_SECONDS}, ensure_ascii=False, sort_keys=True))
        return 0
    if not ready:
        raise RuntimeError(status)
    register(packet)
    try:
        code, log = run_worker(packet)
        failure = None
    except Exception as exc:
        if isinstance(exc, WorkerNotStoppedError):
            raise
        code, failure = 1, f"{type(exc).__name__}:{exc}"
        log = OUTPUT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    result = finish(packet, code, log, failure)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
