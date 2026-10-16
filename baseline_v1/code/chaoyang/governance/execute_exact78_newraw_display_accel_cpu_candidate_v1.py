"""One fenced Cards001 displayed-acceleration CPU experiment.

Check-only is read-only. Execution requires a published exact candidate route,
one fresh Task Packet and no other active writer.
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
from chaoyang.governance import execute_exact78_newraw_velocity_tube_cpu_candidate_v1 as prior
from chaoyang.ops import run_exact78_newraw_display_accel_cpu_candidate_v1 as operation


SCRIPT = Path(__file__)
OPERATION = operation.OPERATION
ROUTE_KEY = "exact78_newraw_display_accel_cpu_candidate_v1"
TASK_ID = "exact78_newraw_display_accel_play_cards_0902_001_9541579d8de29395_v1"
HAWOR_OUTER = (REPO_ROOT / "_run/current/exact78_newraw_hawor_play_cards_0902_001_9541579d8de29395_v4"
               "/attempts/attempt_0001/RESULT.json")
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
OUTPUT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json"
STAGE_ROOT = OUTPUT / "candidate/display_accel_cpu"
STAGE_FILES = {"BASE_SEARCH.json", "FRAME_NUMERIC_DIAGNOSTIC.json",
               "SHADOW_BASELINE.json", "NUMERIC_STATES.npz"}
CODE = (
    "src/chaoyang/governance/execute_exact78_newraw_display_accel_cpu_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_newraw_display_accel_cpu_candidate_v1.py",
    "src/chaoyang/pipeline/exact78_newraw_display_accel_cpu_candidate_v1.py",
    "tests/unit/test_exact78_newraw_display_accel_cpu_candidate_v1.py",
    "tests/unit/test_execute_exact78_newraw_display_accel_cpu_candidate_v1.py",
    *prior.CODE,
)


def _inputs() -> dict[str, Any]:
    handoff = operation._handoff(HAWOR_OUTER)
    inputs = {name: handoff[name] for name in (
        "hawor_outer_result", "hawor_receipt", "hawor_packet", "hawor_stage_seal",
        "hawor_native_open_audit", "hawor_inner_result", "hawor_npz",
        "hawor_frame_provenance", "hawor_input_audit", "raw_video")}
    inputs["source_files"] = handoff["source_files"]
    inputs["robot_asset_pin"] = artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json")
    inputs["mount_proxy"] = artifact_ref(
        REPO_ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1"
                    "/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json")
    inputs["route"] = artifact_ref(REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json")
    return inputs


def packet_value() -> dict[str, Any]:
    inputs = _inputs()
    runtime = [artifact_ref(REPO_ROOT / name) for name in dict.fromkeys(CODE)]
    now = datetime.now().astimezone()
    deadline = now + timedelta(minutes=30)
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK_ID}
    command = ("PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run " + OPERATION
               + " --hawor-result " + str(HAWOR_OUTER)
               + " --output-root " + str(STAGE_ROOT))
    config = {"session_id": operation.EXPECTED_SESSION, "frame_count": 201,
              "fps_nominal": 30.0, "source_kind": "tracker", "gpu_count": 0,
              "model_calls": 0, "render_calls": 0, "training_steps": 0,
              "single_change": operation.SINGLE_CHANGE,
              "numeric_gates_changed": False, "base_selection_changed": False,
              "hand_mount_changed": False, "old_robot_result_input": False,
              "calibration_or_absent": "ABSENT_FOR_CAMERA_DIAGNOSTIC"}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": "exact78-newraw-display-accel-cpu-task-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "Full 201-frame two-side sealed HaWoR displayed-acceleration CPU shadow",
        "phase": operation.PHASE, "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_NEWRAW_DISPLAY_ACCEL_CPU_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": deadline.isoformat(), "campaign_deadline_at": deadline.isoformat(),
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
        "required_outputs": [*(str((STAGE_ROOT / name).relative_to(OUTPUT))
                                for name in sorted(STAGE_FILES | {"RESULT.json"})),
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "render_calls": 0, "training_steps": 0,
                    "wall_seconds_max": 1800, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "INCOMPLETE_402_FRAME_SIDE_GRID", "BASELINE_COUNT_DRIFT",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Cards001 development CPU numeric shadow only; no Robot video, visual, product, control or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("DISPLAY_ACCEL_PACKET_PREFLIGHT:" + ";".join(errors))
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
            and route.get("output_schema") == "exact78-newraw-display-accel-cpu-execution-result-v1"
            and route.get("model_calls_authorized") == 0
            and route.get("render_calls_authorized") == 0
            and route.get("training_steps_authorized") == 0
            and routed == expected)


def route_readiness(packet: dict[str, Any]) -> str:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    published = load_json(ALGORITHM_CONTRACT_PATH)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    pub_route = published.get("special_status_contracts", {}).get(ROUTE_KEY)
    if pub_route is not None:
        if (OPERATION not in _maintained_operations() or not _route_valid(published, packet)
                or not _route_valid(staged, packet)
                or pub_route != staged.get("special_status_contracts", {}).get(ROUTE_KEY)):
            raise RuntimeError("DISPLAY_ACCEL_PUBLISHED_ROUTE_DRIFT")
        return "PUBLISHED_CANDIDATE_ROUTE"
    if OPERATION in _maintained_operations() or not _route_valid(staged, packet):
        raise RuntimeError("DISPLAY_ACCEL_ROUTE_NOT_STAGED")
    return "READY_TO_PUBLISH_ROUTE_WITH_PACKET"


def readiness(packet: dict[str, Any]) -> str:
    if any(path.exists() for path in (OUTPUT, PACKET, RECEIPT)):
        raise RuntimeError("DISPLAY_ACCEL_FRESH_TASK_REQUIRED")
    state = load_json(TASK_STATE_PATH)
    if (state.get("next_task") or any(row["status"] in
            ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in state["tasks"])):
        raise RuntimeError("DISPLAY_ACCEL_ACTIVE_TASK_CONFLICT")
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        raise RuntimeError("DISPLAY_ACCEL_ACTIVE_ROUTE_CONFLICT")
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
            raise RuntimeError("DISPLAY_ACCEL_RUNTIME_SHA_DRIFT:" + ";".join(errors))
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
                          "stop_condition": "one 201-frame CPU run or thirty-minute deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("DISPLAY_ACCEL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    index["task_packets"].append({
        "task_id": TASK_ID, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": _virtual_ref(PACKET, packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_NEWRAW_DISPLAY_ACCEL_CPU_RUNNING",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS", supersedes_index=artifact_ref(INDEX))
    OUTPUT.mkdir(parents=True)
    atomic_json(OUTPUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET)["sha256"] != index["task_packets"][-1]["packet_sha256"]:
        raise RuntimeError("DISPLAY_ACCEL_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_DISPLAY_ACCEL_CPU_REGISTERED",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT, task_packet_index_path=INDEX,
                   task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != process_identity(os.getpid())["start_ticks"]
            or load_json(OUTPUT / "WRITER.json") != writer):
        raise RuntimeError("DISPLAY_ACCEL_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    routed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(active) != 1 or active[0]["task_id"] != TASK_ID
            or active[0]["run_signature"] != packet["run_signature"]
            or len(routed) != 1 or routed[0]["task_id"] != TASK_ID
            or routed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("DISPLAY_ACCEL_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="EXACT78_NEWRAW_DISPLAY_ACCEL_CPU_HEARTBEAT",
                   expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                   generator_path=SCRIPT)
    segment = int(elapsed // 600)
    resume = OUTPUT / f"RESUME_SUMMARY_{segment:04d}.json"
    if segment and not resume.exists():
        atomic_json(resume, {"schema_version": "exact78-newraw-display-accel-cpu-resume-v1",
                             "elapsed_seconds": round(elapsed, 1),
                             "candidate_result_exists": (STAGE_ROOT / "RESULT.json").is_file(),
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
        raise RuntimeError("DISPLAY_ACCEL_BUDGET_TOO_SHORT")
    log = OUTPUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run",
                                   OPERATION, "--hawor-result", str(HAWOR_OUTER),
                                   "--output-root", str(STAGE_ROOT)],
                                  cwd=REPO_ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            worker_start_ticks = process_identity(worker.pid)["start_ticks"]
        except (OSError, KeyError, ValueError) as error:
            if worker.poll() is None:
                raise prior.WorkerNotStoppedError("DISPLAY_ACCEL_WORKER_START_IDENTITY_LOST") from error
            worker_start_ticks = -1
        last = 0.0
        try:
            while worker.poll() is None:
                elapsed = time.monotonic() - started
                if datetime.now().astimezone() >= deadline:
                    raise RuntimeError("DISPLAY_ACCEL_HARD_DEADLINE")
                if elapsed - last >= 50:
                    heartbeat(packet, elapsed)
                    last = elapsed
                time.sleep(2)
        finally:
            prior._stop_fenced_group(worker, worker_start_ticks)
    return worker.returncode, log


def verify_candidate(packet: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    path = STAGE_ROOT / "RESULT.json"
    if not path.is_file():
        return None, ["DISPLAY_ACCEL_CANDIDATE_MISSING"]
    try:
        value = load_json(path)
        if (value.get("schema_version") != "exact78-newraw-display-accel-cpu-result-v1"
                or value.get("status") != "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS"
                or value.get("session_id") != operation.EXPECTED_SESSION
                or value.get("frame_count") != 201 or value.get("fps_nominal") != 30.0
                or value.get("single_change") != operation.SINGLE_CHANGE
                or value.get("model_calls") != 0 or value.get("render_calls") != 0
                or value.get("gpu_count") != 0 or value.get("visual_qualified") is not False
                or value.get("training_eligible") is not False):
            raise RuntimeError("DISPLAY_ACCEL_CANDIDATE_SCOPE_DRIFT")
        expected_inputs = {"hawor_npz": packet["inputs"]["hawor_npz"],
                           "frame_provenance": packet["inputs"]["hawor_frame_provenance"],
                           "input_audit": packet["inputs"]["hawor_input_audit"],
                           "raw_video": packet["inputs"]["raw_video"]}
        if value.get("inputs") != expected_inputs:
            raise RuntimeError("DISPLAY_ACCEL_CANDIDATE_INPUT_REF_DRIFT")
        outputs = value.get("outputs")
        if not isinstance(outputs, dict) or set(outputs) != STAGE_FILES:
            raise RuntimeError("DISPLAY_ACCEL_OUTPUT_REF_SET_DRIFT")
        for name, ref in outputs.items():
            if (not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}
                    or Path(ref["path"]) != path.parent / name or validate_artifact_ref(ref)):
                raise RuntimeError("DISPLAY_ACCEL_OUTPUT_SHA_DRIFT:" + name)
        from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
        from chaoyang.pipeline.exact78_newraw_velocity_tube_cpu_candidate_v1 import velocity_bounds
        gates = {"position_mm": stage.POSITION_GATE_MM,
                 "orientation_deg": stage.ROTATION_GATE_DEG,
                 "arm_velocity_rad_per_frame": stage.ARM_VELOCITY_GATE_RAD,
                 "arm_acceleration_rad_per_frame2": stage.ARM_ACCELERATION_GATE_RAD}
        if value.get("original_numeric_gates") != gates:
            raise RuntimeError("DISPLAY_ACCEL_NUMERIC_GATES_DRIFT")
        rows = load_json(path.parent / "FRAME_NUMERIC_DIAGNOSTIC.json")["rows"]
        shadow = load_json(path.parent / "SHADOW_BASELINE.json")["rows"]
        expected = [(frame, side) for frame in range(201) for side in ("left", "right")]
        if (len(rows) != 402 or len(shadow) != 402
                or [(row["frame"], row["side"]) for row in rows] != expected
                or {(row["frame"], row["side"]) for row in shadow} != set(expected)):
            raise RuntimeError("DISPLAY_ACCEL_INCOMPLETE_402_GRID")
        counts = {side: sum(bool(row["pass_numeric"]) for row in rows if row["side"] == side)
                  for side in ("left", "right")}
        baseline = {side: sum(bool(row["arm"]["pass_numeric"]) for row in shadow
                              if row["side"] == side) for side in ("left", "right")}
        if (value.get("baseline_arm_pass") != baseline or baseline != {"left": 44, "right": 2}
                or value.get("candidate_arm_pass") != counts):
            raise RuntimeError("DISPLAY_ACCEL_PASS_COUNTS_DRIFT")
        lower, upper = stage._arm_limits(stage.load_pinned_robot_assets(stage.ROOT))
        row_map = {(row["frame"], row["side"]): row for row in rows}
        for row in rows:
            frame, side_name = row["frame"], row["side"]
            side = 0 if side_name == "left" else 1
            previous = (None if frame == 0 else
                        np.asarray(row_map[frame - 1, side_name]["display_q_rad"], dtype=float))
            prior_display = (None if frame < 2 else
                             np.asarray(row_map[frame - 2, side_name]["display_q_rad"], dtype=float))
            if row.get("previous_display_q_rad") != (
                    None if previous is None else previous.tolist()):
                raise RuntimeError("DISPLAY_ACCEL_PREVIOUS_DISPLAY_Q_DRIFT")
            displayed = np.asarray(row["display_q_rad"], dtype=float)
            if displayed.shape != (7,) or not np.isfinite(displayed).all():
                raise RuntimeError("DISPLAY_ACCEL_DISPLAY_Q_INVALID")
            dq = None if previous is None else float(np.max(np.abs(displayed - previous)))
            if row.get("display_dq_max_rad") != dq:
                raise RuntimeError("DISPLAY_ACCEL_DISPLAY_DQ_DRIFT")
            if dq is not None and dq > stage.ARM_VELOCITY_GATE_RAD + 1e-9:
                raise RuntimeError("DISPLAY_ACCEL_DISPLAY_VELOCITY_GATE")
            attempted = row.get("candidate_q_rad")
            if attempted is not None:
                attempted = np.asarray(attempted, dtype=float)
                low, high = velocity_bounds(lower[side], upper[side], previous)
                ddq = (None if prior_display is None else
                       float(np.max(np.abs(attempted - 2 * previous + prior_display))))
                candidate_dq = (None if previous is None else
                                float(np.max(np.abs(attempted - previous))))
                if (attempted.shape != (7,) or not np.isfinite(attempted).all()
                        or not np.allclose(row.get("solve_bounds_lower_rad"), low)
                        or not np.allclose(row.get("solve_bounds_upper_rad"), high)
                        or np.any(attempted < low - 1e-9) or np.any(attempted > high + 1e-9)
                        or row.get("candidate_dq_max_rad") != candidate_dq
                        or row.get("candidate_ddq_max_rad") != ddq):
                    raise RuntimeError("DISPLAY_ACCEL_ATTEMPT_OR_ACCEL_DRIFT")
            if row["status"].startswith("PASS") != row["pass_numeric"]:
                raise RuntimeError("DISPLAY_ACCEL_ROW_STATUS_DRIFT")
            if row["pass_numeric"] and (
                    row.get("position_mm", float("inf")) > stage.POSITION_GATE_MM
                    or row.get("rotation_deg", float("inf")) > stage.ROTATION_GATE_DEG
                    or row.get("branch_margin_m_min", -float("inf")) < -1e-8
                    or (row.get("candidate_ddq_max_rad") is not None
                        and row["candidate_ddq_max_rad"] > stage.ARM_ACCELERATION_GATE_RAD + 1e-9)):
                raise RuntimeError("DISPLAY_ACCEL_ACCEPTED_GATE_DRIFT")
        continuity = all(row["display_dq_max_rad"] is None or
                         row["display_dq_max_rad"] <= stage.ARM_VELOCITY_GATE_RAD + 1e-9
                         for row in rows)
        acceleration = all(not row["pass_numeric"] or row.get("candidate_ddq_max_rad") is None or
                           row["candidate_ddq_max_rad"] <= stage.ARM_ACCELERATION_GATE_RAD + 1e-9
                           for row in rows)
        hypothesis = counts["right"] > 2 and counts["left"] >= 44 and continuity and acceleration
        if (value.get("display_velocity_within_existing_gate") is not continuity
                or value.get("accepted_acceleration_within_existing_gate") is not acceleration
                or value.get("numeric_hypothesis_pass") is not hypothesis):
            raise RuntimeError("DISPLAY_ACCEL_SUMMARY_GATE_DRIFT")
        with np.load(path.parent / "NUMERIC_STATES.npz", allow_pickle=False) as archive:
            if (set(archive.files) != {"q_arm_attempt", "q_arm_display", "q_arm_shadow", "q_hand",
                                       "T_camera_base", "T_tool_hand_root",
                                       "display_joints_3d_camera", "prediction_valid",
                                       "display_drawable"}
                    or archive["q_arm_attempt"].shape != (201, 2, 7)
                    or archive["q_arm_display"].shape != (201, 2, 7)
                    or archive["q_arm_shadow"].shape != (201, 2, 7)
                    or archive["q_hand"].shape[:2] != (201, 2)
                    or archive["T_camera_base"].shape != (4, 4)):
                raise RuntimeError("DISPLAY_ACCEL_STATES_SHAPE_DRIFT")
            for row in rows:
                side = 0 if row["side"] == "left" else 1
                frame = row["frame"]
                if not np.allclose(archive["q_arm_display"][frame, side],
                                   row["display_q_rad"], atol=1e-9, rtol=0):
                    raise RuntimeError("DISPLAY_ACCEL_DISPLAY_Q_DRIFT")
                attempted = archive["q_arm_attempt"][frame, side]
                if row["candidate_q_rad"] is None:
                    if not np.isnan(attempted).all():
                        raise RuntimeError("DISPLAY_ACCEL_ATTEMPT_Q_DRIFT")
                elif not np.allclose(attempted, row["candidate_q_rad"], atol=1e-9, rtol=0):
                    raise RuntimeError("DISPLAY_ACCEL_ATTEMPT_Q_DRIFT")
        if load_json(path.parent / "BASE_SEARCH.json").get("rule") != "same_frozen_grid_for_all_sources_v1":
            raise RuntimeError("DISPLAY_ACCEL_BASE_RULE_DRIFT")
        return value, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"DISPLAY_ACCEL_VALIDATION:{error}"]


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
                errors.append("DISPLAY_ACCEL_POST_INPUT_REF_DRIFT:" + key)
    except Exception as error:
        errors.append("DISPLAY_ACCEL_POST_INPUT_VERIFY:" + type(error).__name__ + ":" + str(error))
    for key in ("robot_asset_pin", "mount_proxy", "route"):
        if ref_errors := validate_artifact_ref(packet["inputs"][key]):
            errors.append("DISPLAY_ACCEL_POST_ASSET_REF_DRIFT:" + key + ":" + ";".join(ref_errors))
    for ref in packet["runtime_code"]:
        if ref_errors := validate_artifact_ref(ref):
            errors.append("DISPLAY_ACCEL_POST_CODE_REF_DRIFT:" + ref["path"] + ":" + ";".join(ref_errors))
    passed = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if passed else "FAILED_RUNTIME_FINAL"
    if (OUTPUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DISPLAY_ACCEL_DUPLICATE_TERMINAL")
    result = {"schema_version": "exact78-newraw-display-accel-cpu-execution-result-v1",
              "task_id": TASK_ID, "status": status,
              "source_session": operation.EXPECTED_SESSION, "source_frames_expected": 201,
              "worker_returncode": code, "failure_code": reason,
              "validation_errors": errors,
              "stage": {"result": artifact_ref(STAGE_ROOT / "RESULT.json") if candidate else None},
              "hawor_outer_result": packet["inputs"]["hawor_outer_result"],
              "worker_log": artifact_ref(log), "model_calls": 0, "render_calls": 0,
              "old_robot_result_input": False,
              "old_robot_ab_status": "EVALUATOR_ONLY_AFTER_FRESH_RESULT",
              "visual_qualified": False, "product_qualified": False,
              "training_eligible": False, "user_review": "PENDING",
              "next_action": ("Compare sealed CPU numeric counts; render and review only under a separate task"
                              if passed else "Inspect failure; use new successor"),
              "claim_limit": packet["claim_limit"]}
    atomic_json(OUTPUT / "RESULT.json", result)
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(RECEIPT, {"schema_version": "exact78-newraw-display-accel-cpu-receipt-v1",
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
        raise RuntimeError("DISPLAY_ACCEL_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_NEWRAW_DISPLAY_ACCEL_CPU_TERMINAL",
                 plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=f"EXACT78_NEWRAW_DISPLAY_ACCEL_CPU_{status}",
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
                          "wall_seconds_max": 1800}, ensure_ascii=False, sort_keys=True))
        return 0
    if not ready:
        raise RuntimeError(status)
    register(packet)
    try:
        code, log = run_worker(packet)
        failure = None
    except Exception as error:
        if isinstance(error, prior.WorkerNotStoppedError):
            raise
        code, failure = 1, f"{type(error).__name__}:{error}"
        log = OUTPUT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    result = finish(packet, code, log, failure)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
