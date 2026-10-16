"""Finite fenced Cards001 two-seed CPU candidate; --check-only never writes.

Registration and terminal publication use the existing governance bundle, Task
Packet, live writer and CLI operation gate. Nothing here reads old Robot output.
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
import subprocess
import time
from typing import Any

import numpy as np

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
    validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.ops import run_exact78_newraw_two_seed_cpu_candidate_v1 as operation


SCRIPT = Path(__file__)
OPERATION = operation.OPERATION
ROUTE_KEY = "exact78_newraw_two_seed_cpu_candidate_v1"
TASK_ID = "exact78_newraw_two_seed_play_cards_0902_001_9541579d8de29395_v1"
HAWOR_OUTER = (REPO_ROOT / "_run/current/exact78_newraw_hawor_play_cards_0902_001_9541579d8de29395_v4"
               "/attempts/attempt_0001/RESULT.json")
TASK_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
OUTPUT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
CODE = (
    "src/chaoyang/governance/execute_exact78_newraw_two_seed_cpu_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_newraw_two_seed_cpu_candidate_v1.py",
    "src/chaoyang/pipeline/exact78_newraw_two_seed_cpu_candidate_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/governance/exact78_newraw_robot_handoff_candidate_v1.py",
    "src/chaoyang/governance/execute_exact78_newraw_robot_cpu_candidate_v1.py",
    "src/chaoyang/governance/exact78_hawor_descriptor_adapter_candidate.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/paths.py",
)


def _inputs() -> dict[str, Any]:
    handoff = operation._handoff(HAWOR_OUTER)
    inputs = {name: handoff[name] for name in (
        "hawor_outer_result", "hawor_receipt", "hawor_packet", "hawor_stage_seal",
        "hawor_native_open_audit", "hawor_inner_result", "hawor_npz",
        "hawor_frame_provenance", "hawor_input_audit", "raw_video")}
    inputs["source_files"] = handoff["source_files"]
    inputs["robot_asset_pin"] = artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json")
    inputs["mount_proxy"] = artifact_ref(REPO_ROOT / "assets/robot/hardware_handoff"
                                         "/kaihand_flange_adapter_v1/excluded_reference_only"
                                         "/MOUNT_VISUAL_PROXY_CONTRACT.json")
    inputs["route"] = artifact_ref(REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json")
    return inputs


def packet_value() -> dict[str, Any]:
    inputs = _inputs()
    runtime = [artifact_ref(REPO_ROOT / name) for name in CODE]
    now = datetime.now().astimezone()
    deadline = min(now + timedelta(minutes=20), TASK_END)
    if deadline <= now:
        raise RuntimeError("TWO_SEED_CAMPAIGN_DEADLINE_EXPIRED")
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID}
    command = (f"PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run {OPERATION} "
               f"--hawor-result {shlex.quote(str(HAWOR_OUTER))} "
               f"--output-root {shlex.quote(str(OUTPUT / 'candidate/two_seed_cpu'))}")
    config = {"session_id": operation.EXPECTED_SESSION, "frame_count": 201,
              "fps_nominal": 30.0, "source_kind": "tracker", "gpu_count": 0,
              "model_calls": 0, "render_calls": 0, "training_steps": 0,
              "single_change": "ALTERNATE_EXISTING_SEED_ONLY_ON_TEMPORAL_REJECTION",
              "numeric_gates_changed": False, "base_selection_changed": False,
              "calibration_or_absent": "ABSENT_FOR_CAMERA_DIAGNOSTIC",
              "old_robot_result_input": False}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": "exact78-newraw-two-seed-cpu-task-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Full 201-frame two-side sealed HaWoR CPU two-seed temporal fallback diagnostic",
        "phase": operation.PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_NEWRAW_TWO_SEED_CPU_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": deadline.isoformat(), "campaign_deadline_at": TASK_END.isoformat(),
        "read_set": ["AGENTS.md", "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                     *(str(Path(inputs[key]["path"]).relative_to(REPO_ROOT)) for key in (
                         "hawor_outer_result", "hawor_stage_seal", "hawor_native_open_audit",
                         "hawor_npz", "hawor_frame_provenance", "hawor_input_audit"))],
        "write_set": [str(OUTPUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_V4_HAWOR",
                          "SOURCE_SHA_PINNED", "ROBOT_ASSETS_PINNED", "CPU_ONLY",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["candidate/two_seed_cpu/RESULT.json",
                             "candidate/two_seed_cpu/TWO_SEED_FRAMES.json",
                             "candidate/two_seed_cpu/SHADOW_BASELINE.json",
                             "candidate/two_seed_cpu/BASE_SEARCH.json",
                             "candidate/two_seed_cpu/NUMERIC_STATES.npz",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0, "training_steps": 0,
                    "wall_seconds_max": 1200, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "INCOMPLETE_402_FRAME_SIDE_GRID", "DEFAULT_SCORE_DRIFT",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Cards001 development CPU numeric experiment only; no Robot video, world, visual, product, control or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("TWO_SEED_PACKET_PREFLIGHT:" + ";".join(errors))
    return packet


def _route_valid(contract: dict[str, Any], packet: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(ROUTE_KEY, {})
    expected = {ref["path"]: ref for ref in packet["runtime_code"]}
    routed = {ref.get("path"): ref for ref in route.get("code_closure", [])
              if isinstance(ref, dict)}
    return (route.get("candidate_operation") == OPERATION
            and route.get("execution_status") == "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION"
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-newraw-two-seed-cpu-execution-result-v1"
            and route.get("model_calls_authorized") == 0
            and route.get("render_calls_authorized") == 0
            and routed == expected)


def route_readiness(packet: dict[str, Any]) -> str:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    published = load_json(ALGORITHM_CONTRACT_PATH)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    pub_route = published.get("special_status_contracts", {}).get(ROUTE_KEY)
    if pub_route is not None:
        if (OPERATION not in _maintained_operations() or not _route_valid(published, packet)
                or not _route_valid(staged, packet) or
                pub_route != staged.get("special_status_contracts", {}).get(ROUTE_KEY)):
            raise RuntimeError("TWO_SEED_PUBLISHED_ROUTE_DRIFT")
        return "PUBLISHED_CANDIDATE_ROUTE"
    if OPERATION in _maintained_operations() or not _route_valid(staged, packet):
        raise RuntimeError("TWO_SEED_ROUTE_NOT_STAGED")
    return "READY_TO_PUBLISH_ROUTE_WITH_PACKET"


def readiness(packet: dict[str, Any]) -> str:
    if any(path.exists() for path in (OUTPUT, PACKET, RECEIPT)):
        raise RuntimeError("TWO_SEED_FRESH_TASK_REQUIRED")
    state = load_json(TASK_STATE_PATH)
    if (state.get("next_task") or any(row["status"] in
            ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in state["tasks"])):
        raise RuntimeError("TWO_SEED_ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        raise RuntimeError("TWO_SEED_ACTIVE_ROUTE_CONFLICT")
    return route_readiness(packet)


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _index_copy() -> dict[str, Any]:
    value = load_json(INDEX)
    return {key: copy.deepcopy(item) for key, item in value.items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def register(packet: dict[str, Any]) -> None:
    readiness(packet)
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("TWO_SEED_RUNTIME_SHA_DRIFT:" + ";".join(errors))
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": operation.PHASE, "attempt": 1,
        "status": "RUNNING", "session": operation.EXPECTED_SESSION,
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": _virtual_ref(PACKET, packet),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK_ID, "session": operation.EXPECTED_SESSION,
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": "TWO_CPU_THREADS_NO_GPU",
                          "stop_condition": "one 201-frame CPU run or twenty-minute deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("TWO_SEED_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    index["task_packets"].append({
        "task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": _virtual_ref(PACKET, packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_NEWRAW_TWO_SEED_CPU_RUNNING",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET)["sha256"] != index["task_packets"][-1]["packet_sha256"]:
        raise RuntimeError("TWO_SEED_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_TWO_SEED_CPU_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT, task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(OUTPUT / "WRITER.json") != writer):
        raise RuntimeError("TWO_SEED_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    routed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(active) != 1 or active[0]["task_id"] != TASK_ID
            or active[0]["run_signature"] != packet["run_signature"]
            or len(routed) != 1 or routed[0]["task_id"] != TASK_ID
            or routed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("TWO_SEED_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_TWO_SEED_CPU_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT)
    segment = int(elapsed // 600)
    resume = OUTPUT / f"RESUME_SUMMARY_{segment:04d}.json"
    if segment and not resume.exists():
        atomic_json(resume, {"schema_version": "exact78-newraw-two-seed-cpu-resume-v1",
                             "elapsed_seconds": round(elapsed, 1),
                             "candidate_result_exists":
                             (OUTPUT / "candidate/two_seed_cpu/RESULT.json").is_file(),
                             "next_action": "Wait for fenced worker or terminal failure"})


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    cache = OUTPUT / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
               XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
               NUMBA_CACHE_DIR=str(cache / "numba"), TORCH_EXTENSIONS_DIR=str(cache / "torch"),
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if (deadline - datetime.now().astimezone()).total_seconds() < 120:
        raise RuntimeError("TWO_SEED_BUDGET_TOO_SHORT")
    log = OUTPUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run",
                                   OPERATION, "--hawor-result", str(HAWOR_OUTER),
                                   "--output-root", str(OUTPUT / "candidate/two_seed_cpu")],
                                  cwd=REPO_ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT)
        last = 0.0
        try:
            while worker.poll() is None:
                elapsed = time.monotonic() - started
                if datetime.now().astimezone() >= deadline:
                    raise RuntimeError("TWO_SEED_HARD_DEADLINE")
                if elapsed - last >= 50:
                    heartbeat(packet, elapsed)
                    last = elapsed
                time.sleep(2)
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait(timeout=10)
    return worker.returncode, log


def verify_candidate(packet: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    path = OUTPUT / "candidate/two_seed_cpu/RESULT.json"
    if not path.is_file():
        return None, ["TWO_SEED_CANDIDATE_MISSING"]
    try:
        value = load_json(path)
        if (value.get("schema_version") != "exact78-newraw-two-seed-cpu-result-v1"
                or value.get("status") != "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS"
                or value.get("session_id") != operation.EXPECTED_SESSION
                or value.get("frame_count") != 201
                or value.get("old_robot_result_input") is not False
                or value.get("numeric_gates_changed") is not False):
            raise RuntimeError("TWO_SEED_CANDIDATE_SCOPE_DRIFT")
        expected_inputs = {
            "hawor_npz": packet["inputs"]["hawor_npz"],
            "frame_provenance": packet["inputs"]["hawor_frame_provenance"],
            "input_audit": packet["inputs"]["hawor_input_audit"],
            "raw_video": packet["inputs"]["raw_video"],
        }
        if value.get("inputs") != expected_inputs:
            raise RuntimeError("TWO_SEED_CANDIDATE_INPUT_REF_DRIFT")
        expected_outputs = {"BASE_SEARCH.json", "TWO_SEED_FRAMES.json",
                            "SHADOW_BASELINE.json", "NUMERIC_STATES.npz"}
        outputs = value.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != expected_outputs:
            raise RuntimeError("TWO_SEED_OUTPUT_REF_SET_DRIFT")
        for name, ref in outputs.items():
            if not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}:
                raise RuntimeError("TWO_SEED_OUTPUT_REF_FORMAT_DRIFT:" + name)
            if (Path(ref["path"]) != path.parent / name or validate_artifact_ref(ref)):
                raise RuntimeError("TWO_SEED_OUTPUT_SHA_DRIFT:" + name)
        rows = load_json(path.parent / "TWO_SEED_FRAMES.json")["rows"]
        shadow = load_json(path.parent / "SHADOW_BASELINE.json")["rows"]
        expected = {(frame, side) for frame in range(201) for side in ("left", "right")}
        if (len(rows) != 402 or len(shadow) != 402
                or {(row["frame"], row["side"]) for row in rows} != expected
                or {(row["frame"], row["side"]) for row in shadow} != expected):
            raise RuntimeError("TWO_SEED_INCOMPLETE_402_GRID")
        if any(row.get("default_seed_index") not in (0, 1) or
               len(row.get("candidates", [])) != 2 or
               row.get("selected_seed_index") not in (0, 1) for row in rows):
            raise RuntimeError("TWO_SEED_CANDIDATE_ARRAY_DRIFT")
        for row in rows:
            choices = row["candidates"]
            default = min(range(2), key=lambda i: tuple(choices[i]["score"]))
            selected = row["selected_seed_index"]
            if (row["default_seed_index"] != default
                    or row["alternate_selected"] != (selected != default)
                    or (selected != default and not (
                        choices[default]["status"] in
                        ("HOLD_ARM_VELOCITY", "HOLD_ARM_ACCELERATION")
                        and choices[selected]["status"].startswith("PASS")))
                    or row["status"] != choices[selected]["status"]
                    or row["pass_numeric"] != row["status"].startswith("PASS")):
                raise RuntimeError("TWO_SEED_SELECTION_OR_SCORE_DRIFT")
        if value.get("alternate_selected_count") != sum(row["alternate_selected"] for row in rows):
            raise RuntimeError("TWO_SEED_ALTERNATE_COUNT_DRIFT")
        with np.load(path.parent / "NUMERIC_STATES.npz", allow_pickle=False) as archive:
            if (archive["q_arm"].shape != (201, 2, 7)
                    or archive["q_arm_shadow"].shape != (201, 2, 7)
                    or archive["q_hand"].shape[:2] != (201, 2)
                    or archive["T_camera_base"].shape != (4, 4)):
                raise RuntimeError("TWO_SEED_STATES_SHAPE_DRIFT")
        if load_json(path.parent / "BASE_SEARCH.json").get("rule") != "same_frozen_grid_for_all_sources_v1":
            raise RuntimeError("TWO_SEED_BASE_RULE_DRIFT")
        return value, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"TWO_SEED_VALIDATION:{error}"]


def finish(packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    candidate, errors = verify_candidate(packet)
    try:
        handoff = operation._handoff(HAWOR_OUTER)
        for key in ("hawor_outer_result", "hawor_receipt", "hawor_packet",
                    "hawor_stage_seal", "hawor_native_open_audit", "hawor_inner_result",
                    "hawor_npz", "hawor_frame_provenance", "hawor_input_audit",
                    "raw_video", "source_files"):
            if packet["inputs"].get(key) != handoff[key]:
                errors.append("TWO_SEED_POST_INPUT_REF_DRIFT:" + key)
    except Exception as error:
        errors.append("TWO_SEED_POST_INPUT_VERIFY:" + type(error).__name__ + ":" + str(error))
    passed = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if passed else "FAILED_RUNTIME_FINAL"
    if (OUTPUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("TWO_SEED_DUPLICATE_TERMINAL")
    result = {"schema_version": "exact78-newraw-two-seed-cpu-execution-result-v1",
              "task_id": TASK_ID, "status": status,
              "source_session": operation.EXPECTED_SESSION, "source_frames_expected": 201,
              "worker_returncode": code, "failure_code": reason,
              "validation_errors": errors, "stage": {"result": artifact_ref(
                  OUTPUT / "candidate/two_seed_cpu/RESULT.json") if candidate else None},
              "hawor_outer_result": packet["inputs"]["hawor_outer_result"],
              "worker_log": artifact_ref(log), "model_calls": 0, "render_calls": 0,
              "old_robot_result_input": False,
              "old_robot_ab_status": "EVALUATOR_ONLY_AFTER_FRESH_RESULT",
              "visual_qualified": False, "product_qualified": False,
              "training_eligible": False, "user_review": "PENDING",
              "next_action": ("Run sealed shadow-vs-old evaluator before interpreting improvement"
                              if passed else "Inspect failed candidate; use a new successor"),
              "claim_limit": packet["claim_limit"]}
    atomic_json(OUTPUT / "RESULT.json", result)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECEIPT, {"schema_version": "exact78-newraw-two-seed-cpu-receipt-v1",
                          "task_id": TASK_ID, "status": status,
                          "result": artifact_ref(OUTPUT / "RESULT.json"),
                          "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(OUTPUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if passed else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("TWO_SEED_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_NEWRAW_TWO_SEED_CPU_TERMINAL",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=f"EXACT78_NEWRAW_TWO_SEED_CPU_{status}",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT, task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUTPUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--check-only", action="store_true")
    choice.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    packet = packet_value()
    try:
        status = readiness(packet)
    except RuntimeError as error:
        status = str(error)
        ready = False
    else:
        ready = True
    if args.check_only:
        try:
            candidate_route = route_readiness(packet)
        except RuntimeError as error:
            candidate_route = str(error)
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
                          "reason": status, "task_id": TASK_ID,
                          "packet_preflight": "PASS", "run_signature": packet["run_signature"],
                          "candidate_route": candidate_route,
                          "source_session": operation.EXPECTED_SESSION, "frame_count": 201,
                          "gpu_count": 0, "model_calls": 0, "render_calls": 0,
                          "wall_seconds_max": 1200}, ensure_ascii=False, sort_keys=True))
        return 0
    if not ready:
        raise RuntimeError(status)
    register(packet)
    try:
        code, log = run_worker(packet)
        failure = None
    except Exception as error:
        code, failure = 1, f"{type(error).__name__}:{error}"
        log = OUTPUT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    result = finish(packet, code, log, failure)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
