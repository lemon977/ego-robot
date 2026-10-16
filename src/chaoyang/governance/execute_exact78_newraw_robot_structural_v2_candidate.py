"""Unpublished finite Task Packet executor for 0901 Chips001 Robot structural v2.

Check-only is read-only. Registration requires a future staged builder route and
an idle ledger; it publishes that route and this packet in the same governance
CAS. The current builder/CLI are intentionally not modified by this module.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shlex
import subprocess
import time
from typing import Any

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
    validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance import execute_exact78_newraw_robot_cpu_candidate_v1 as v1_executor
from chaoyang.pipeline import exact78_newraw_robot_structural_v2_candidate as structural


OPERATION = structural.OPERATION
TASK_ID = structural.TASK_ID
ROUTE_KEY = "exact78_newraw_robot_structural_v2_candidate"
OUTPUT_SCHEMA = "exact78-newraw-robot-structural-v2-execution-result-v1"
STAGE_SCHEMA = "exact78-newraw-robot-structural-v2-candidate-result-v1"
WALL_SECONDS_MAX = 5400
H20_DEADLINE = datetime.fromisoformat("2026-10-01T14:47:25+08:00")
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
SOURCE_ROLES = tuple(structural._SOURCE_SHA)
ADMISSION_ROLES = structural._REQUIRED_ADMISSION
VIDEOS = structural._VIDEO_NAMES
RUNTIME_RELATIVE = (
    "src/chaoyang/governance/execute_exact78_newraw_robot_structural_v2_candidate.py",
    "src/chaoyang/ops/run_exact78_newraw_robot_structural_v2_candidate.py",
    "src/chaoyang/pipeline/exact78_newraw_robot_structural_v2_candidate.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_flange_successor_v1.py",
    "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/governance/exact78_newraw_robot_handoff_candidate_v1.py",
    "src/chaoyang/governance/exact78_hawor_descriptor_adapter_candidate.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/execute_exact78_newraw_robot_cpu_candidate_v1.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/paths.py",
)


def _paths() -> tuple[Path, Path, Path]:
    return (REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001",
            REPO_ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json",
            REPO_ROOT / "tasks/receipts" / f"{TASK_ID.upper()}.json")


def route_contract(runtime_code: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact entry to add to a future staged current_r3_contracts builder."""
    return {
        "execution_status": "CANDIDATE_FENCED_CPU_ONE_SESSION",
        "candidate_operation": OPERATION, "task_packet_required": True,
        "code_closure": runtime_code, "output_schema": OUTPUT_SCHEMA,
        "model_calls_authorized": 0,
        "render_calls_authorized": "ONE_FULL_799_FRAME_SESSION_ONLY_WHEN_EXACT_PACKET_ACTIVE",
        "training_steps_authorized": 0, "current_runnable_operation": None,
        "claim_limit": "One sealed 0901 Chips001 structural Robot diagnostic; no visual, physical, product or training pass.",
    }


def _route_valid(contract: dict[str, Any], packet: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(ROUTE_KEY)
    return isinstance(route, dict) and route == route_contract(packet["runtime_code"])


def _route_readiness(packet: dict[str, Any]) -> str:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("STRUCTURAL_V2_PRODUCTION_ROUTE_SELECTED")
    published = load_json(ALGORITHM_CONTRACT_PATH)
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH))
    pub = published.get("special_status_contracts", {}).get(ROUTE_KEY)
    if pub is None:
        if OPERATION in _maintained_operations() or not _route_valid(staged, packet):
            raise RuntimeError("STRUCTURAL_V2_ROUTE_NOT_STAGED")
        return "READY_TO_PUBLISH_ROUTE_WITH_PACKET"
    if (OPERATION not in _maintained_operations()
            or not _route_valid(published, packet)
            or not _route_valid(staged, packet)):
        raise RuntimeError("STRUCTURAL_V2_PUBLISHED_ROUTE_DRIFT")
    return "PUBLISHED_CANDIDATE_ROUTE"


def _deadline() -> datetime:
    now = datetime.now().astimezone()
    if now >= H20_DEADLINE:
        raise RuntimeError("STRUCTURAL_V2_H20_EXPIRED")
    return min(now + timedelta(seconds=WALL_SECONDS_MAX), H20_DEADLINE)


def packet_value() -> dict[str, Any]:
    """Build a fresh v2 packet value only; no file or governance write."""
    from chaoyang.governance.exact78_newraw_robot_handoff_candidate_v1 import verify_handoff

    checked = structural.check_only()
    outer_path = Path(checked["source_refs"]["hawor_outer_result"]["path"])
    cpu, cpu_refs, admission = v1_executor._cpu_probe(outer_path)
    handoff = verify_handoff(outer_result=outer_path, admission=admission)
    if (handoff["session_id"] != structural.SESSION_ID
            or handoff["frame_count"] != structural.FRAMES
            or handoff["source_kind"] != structural.SOURCE_KIND
            or any(handoff[role] != checked["source_refs"][role]
                   for role in SOURCE_ROLES if role in handoff)):
        raise RuntimeError("STRUCTURAL_V2_HANDOFF_DRIFT")
    attempt, packet_path, receipt = _paths()
    assets = v1_executor._asset_refs()
    original = load_json(structural._V1_PACKET)["inputs"]
    if assets != original["robot_assets"]:
        raise RuntimeError("STRUCTURAL_V2_ASSET_MANIFEST_DRIFT")
    inputs = {
        **cpu_refs,
        "upstream_admission_descriptor_sha256": cpu["descriptor_sha256"],
        "submission_input_sha256": cpu["submission_input_sha256"],
        "source_session": structural.SESSION_ID,
        "source_root": cpu["source_root"],
        "source_files": handoff["source_files"],
        **checked["source_refs"],
        "robot_assets": assets, "route": artifact_ref(ROUTE),
    }
    if any(inputs[role] != original[role] for role in ADMISSION_ROLES):
        raise RuntimeError("STRUCTURAL_V2_ADMISSION_MANIFEST_DRIFT")
    runtime = [artifact_ref(REPO_ROOT / relative) for relative in RUNTIME_RELATIVE]
    command = (f"PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run {OPERATION} "
               f"--output-root {shlex.quote(str(attempt / 'candidate/robot_v2'))}")
    config = {
        "frame_count": structural.FRAMES, "fps_nominal": structural.FPS,
        "source_kind": structural.SOURCE_KIND,
        "time_mode": handoff["time_mode"], "physical_eye": handoff["physical_eye"],
        "solver": "EXACT78_FRESH_ROBOT_STRUCTURAL_V2_TWO_CORRECTIONS_ONLY",
        "camera_rule": "BASE_TO_CAMERA_NO_INVERSION", "hand_root_parent": "FLANGE_LR",
        "base_search_rule_changed": False, "numeric_gates_changed": False,
        "display_acceleration_changed": False,
        "arm_position_gate_mm": 10.0, "arm_rotation_gate_deg": 5.0,
        "arm_velocity_gate_rad": 0.12, "arm_acceleration_gate_rad": 0.06,
        "hand_bone_gate_deg": 60.0, "hand_tip_gate_deg": 15.0,
        "hand_velocity_gate_rad": 0.08, "hand_acceleration_gate_rad": 0.06,
        "cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
        "training_steps": 0, "session_attempts_max": 1,
        "world_pose_authority": False, "timebase_authority": False,
        "visual_qualified": False, "resume_policy": "SEALED_HAWOR_ONLY_NO_PARTIAL",
    }
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": command, "schema": OUTPUT_SCHEMA,
    })).hexdigest()
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": TASK_ID}
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK_ID,
        "objective": "One sealed fresh HaWoR to 0901 Chips001 Robot structural v2 diagnostic",
        "phase": "EXACT78_NEWRAW_ROBOT_STRUCTURAL_V2_CANDIDATE",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": TASK_ID.upper(), "execution_mode": "EXACT78_CANDIDATE",
        "registered_at": now.isoformat(), "deadline_at": _deadline().isoformat(),
        "read_set": ["AGENTS.md", str(ROUTE.relative_to(REPO_ROOT)),
                     str(outer_path.relative_to(REPO_ROOT)),
                     *(str(Path(cpu_refs[role]["path"]).relative_to(REPO_ROOT))
                       for role in ADMISSION_ROLES if role != "upstream_admission_receipt")],
        "write_set": [str(attempt.relative_to(REPO_ROOT)),
                      str(packet_path.relative_to(REPO_ROOT)),
                      str(receipt.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK",
                          "TERMINAL_SAME_SESSION_HAWOR_SEALED", "SOURCE_SHA_PINNED",
                          "ROBOT_ASSETS_PINNED", "CPU_ONLY", "EXTERNAL_DATA_READ_ONLY",
                          "NO_OLD_ACTION_OR_FITTED_TEMPLATE", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": [
            "candidate/robot_v2/RESULT.json",
            "candidate/robot_v2/MOUNT_CAMERA_CHAIN_AUDIT.json",
            "candidate/robot_v2/structural_stage_ffmpeg44/ROBOT_FRAME_STATES.json",
            *(f"candidate/robot_v2/structural_stage_ffmpeg44/render/{name}"
              for name in VIDEOS.values()),
            "RESULT.json", "terminal_receipt",
        ],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "model_calls": 0, "training_steps": 0,
                    "wall_seconds_max": WALL_SECONDS_MAX, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_OR_ASSET_SHA_DRIFT", "DOUBLE_WRITER",
                            "HAWOR_UNSEALED_OR_NOT_TERMINAL", "FRAME_MAP_DRIFT",
                            "H20_DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "0901 Chips001 two-structure Robot camera diagnostic only; no Clean, physical occlusion, world/timebase, visual/product, control or training qualification.",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("STRUCTURAL_V2_PACKET_PREFLIGHT:" + ";".join(errors))
    return packet


def check_only() -> dict[str, Any]:
    """Report source/code/packet readiness even while v1 still owns current."""
    packet = packet_value()
    attempt, packet_path, receipt = _paths()
    if any(path.exists() for path in (attempt, packet_path, receipt)):
        raise RuntimeError("STRUCTURAL_V2_FRESH_TASK_REQUIRED")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("STRUCTURAL_V2_RUNTIME_DRIFT:" + ";".join(errors))
    try:
        route = _route_readiness(packet)
    except RuntimeError as exc:
        if str(exc) != "STRUCTURAL_V2_ROUTE_NOT_STAGED":
            raise
        route = "ROUTE_TEMPLATE_READY_BUILDER_NOT_STAGED"
    state = load_json(TASK_STATE_PATH)
    active = [row["task_id"] for row in state.get("tasks", [])
              if row.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}]
    return {"status": "READY", "scope": "CODE_INPUT_PACKET_PREP_ONLY",
            "task_id": TASK_ID, "run_signature": packet["run_signature"],
            "frame_count": structural.FRAMES, "fps_nominal": structural.FPS,
            "route_status": route, "registration_blocked_by_active": active,
            "registration_started": False, "render_calls": 0, "model_calls": 0,
            "deadline_at_if_registered_now": packet["deadline_at"]}


def _receipt_revision() -> int:
    return v1_executor._receipt_revision()


def _index_copy() -> dict[str, Any]:
    return v1_executor._index_copy()


def _fence(packet: dict[str, Any], *, inputs: bool = False) -> None:
    attempt, packet_path, _ = _paths()
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != process_identity(os.getpid())["start_ticks"]
            or load_json(attempt / "WRITER.json") != writer):
        raise RuntimeError("STRUCTURAL_V2_WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID]
    active = [row for row in load_json(INDEX).get("task_packets", [])
              if row.get("execution_allowed") is True]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or rows[0].get("run_signature") != packet["run_signature"]
            or rows[0].get("pid") != writer["pid"]
            or rows[0].get("proc_start_ticks") != writer["proc_start_ticks"]
            or (state.get("next_task") or {}).get("task_id") != TASK_ID
            or len(active) != 1 or active[0].get("task_id") != TASK_ID
            or active[0].get("packet_sha256") != artifact_ref(packet_path)["sha256"]):
        raise RuntimeError("STRUCTURAL_V2_ACTIVE_PACKET_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("STRUCTURAL_V2_RUNTIME_SHA_DRIFT:" + ";".join(errors))
    if inputs:
        refs = [*packet["inputs"]["source_files"].values(),
                *packet["inputs"]["robot_assets"].values(),
                *(packet["inputs"][role] for role in (*ADMISSION_ROLES, *SOURCE_ROLES))]
        for ref in refs:
            if errors := validate_artifact_ref(ref):
                raise RuntimeError("STRUCTURAL_V2_INPUT_SHA_DRIFT:" + ";".join(errors))


def register(packet: dict[str, Any]) -> None:
    """Publish the already staged route and this packet in one CAS, after v1."""
    _route_readiness(packet)
    attempt, packet_path, receipt = _paths()
    if any(path.exists() for path in (attempt, packet_path, receipt)):
        raise RuntimeError("STRUCTURAL_V2_FRESH_TASK_REQUIRED")
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != process_identity(os.getpid())["start_ticks"]
            or writer.get("scope") != TASK_ID):
        raise RuntimeError("STRUCTURAL_V2_WRITER_IDENTITY_DRIFT")
    fresh = packet_value()
    for key in ("inputs", "runtime_code", "config", "run_signature", "commands"):
        if fresh[key] != packet[key]:
            raise RuntimeError("STRUCTURAL_V2_PACKET_FRESH_PREFLIGHT_DRIFT:" + key)
    if errors := _validate_packet(packet):
        raise RuntimeError("STRUCTURAL_V2_PACKET_PREFLIGHT:" + ";".join(errors))
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_copy()
    if (state.get("next_task")
            or any(row.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
                   for row in state.get("tasks", []))
            or any(row.get("execution_allowed") for row in index["task_packets"])):
        raise RuntimeError("STRUCTURAL_V2_ACTIVE_TASK_CONFLICT")
    revision = _receipt_revision()
    attempt.mkdir(parents=True)
    atomic_json(attempt / "WRITER.json", writer)
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK_ID, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": structural.SESSION_ID,
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(packet_path),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {"task_id": TASK_ID, "session": structural.SESSION_ID,
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": "CPU_TWO_THREADS_GPU_ZERO",
                          "stop_condition": "one complete Robot diagnostic or finite H20 deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("STRUCTURAL_V2_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": TASK_ID,
        "packet_path": str(packet_path.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(packet_path)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(packet_revision=packet["execution_revision"] + "_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_REGISTERED",
                   expected_revision=revision, generator_path=Path(__file__),
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def heartbeat(packet: dict[str, Any]) -> None:
    _fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
    if errors := validate_task_state(state):
        raise RuntimeError("STRUCTURAL_V2_HEARTBEAT_PREFLIGHT:" + ";".join(errors))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=structural.TASK_ID.upper() + "_HEARTBEAT",
                   expected_revision=_receipt_revision(), generator_path=Path(__file__))


def _verify_stage(packet: dict[str, Any]) -> dict[str, Any]:
    attempt, _, _ = _paths()
    root = attempt / "candidate/robot_v2"
    result = load_json(root / "RESULT.json")
    stage_root = root / "structural_stage_ffmpeg44"
    stage_path = stage_root / "RESULT.json"
    if (result.get("schema_version") != STAGE_SCHEMA
            or result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("session_id") != structural.SESSION_ID
            or result.get("frame_count") != structural.FRAMES
            or result.get("fps_nominal") != structural.FPS
            or result.get("source_hawor_result") != packet["inputs"]["hawor_outer_result"]
            or result.get("stage_result") != artifact_ref(stage_path)
            or result.get("base_search_rule_changed") is not False
            or result.get("numeric_gates_changed") is not False
            or result.get("display_acceleration_changed") is not False
            or result.get("ego_camera_rule") != "BASE_TO_CAMERA_NO_INVERSION"
            or result.get("hand_root_parent") != "FLANGE_LR"
            or any(result.get(name) is not False for name in (
                "visual_qualified", "product_qualified", "training_eligible",
                "control_ground_truth", "old_robot_state_consumed"))):
        raise RuntimeError("STRUCTURAL_V2_RESULT_CLAIM_DRIFT")
    stage = load_json(stage_path)
    if (stage.get("schema_version") != "exact78-fresh-robot-stage-result-v1"
            or stage.get("session_id") != structural.SESSION_ID
            or stage.get("frame_count") != structural.FRAMES
            or any(stage.get("inputs", {}).get(key) != packet["inputs"][role]
                   for key, role in (("hawor_npz", "hawor_npz"),
                                     ("frame_provenance", "hawor_frame_provenance"),
                                     ("input_audit", "hawor_input_audit"),
                                     ("raw_video", "raw_video")))):
        raise RuntimeError("STRUCTURAL_V2_INNER_STAGE_DRIFT")
    numeric_states_path = stage_root / "ROBOT_CAMERA_DIAGNOSTIC_STATES.npz"
    render_audit_path = stage_root / "RENDER_AUDIT.json"
    render_audit = load_json(render_audit_path)
    if (stage.get("outputs", {}).get("states") != artifact_ref(numeric_states_path)
            or stage.get("outputs", {}).get("render_audit") != artifact_ref(render_audit_path)
            or render_audit.get("frame_count") != structural.FRAMES
            or not isinstance(render_audit.get("frame_rows"), list)
            or len(render_audit["frame_rows"]) != structural.FRAMES
            or any(row.get("frame") != i for i, row in enumerate(render_audit["frame_rows"]))):
        raise RuntimeError("STRUCTURAL_V2_NUMERIC_OR_RENDER_GRID_DRIFT")
    state_path = stage_root / "ROBOT_FRAME_STATES.json"
    states_ref = artifact_ref(state_path)
    states = load_json(state_path)
    rows = states.get("rows")
    if (stage.get("outputs", {}).get("frame_states") != states_ref
            or states.get("frame_count") != structural.FRAMES
            or not isinstance(rows, list) or len(rows) != 2 * structural.FRAMES):
        raise RuntimeError("STRUCTURAL_V2_FRAME_GRID_DRIFT")
    for i, row in enumerate(rows):
        if (row.get("frame") != i // 2 or row.get("side") != ("left", "right")[i % 2]
                or not isinstance(row.get("arm"), dict)
                or not isinstance(row.get("hand"), dict)):
            raise RuntimeError(f"STRUCTURAL_V2_FRAME_ROW_DRIFT:{i}")
    audit_path = root / "MOUNT_CAMERA_CHAIN_AUDIT.json"
    audit = load_json(audit_path)
    if (result.get("mount_camera_chain_audit") != artifact_ref(audit_path)
            or audit.get("frame_count") != structural.FRAMES
            or audit.get("camera_rule") != "BASE_TO_CAMERA_NO_INVERSION"
            or audit.get("hand_root_rule") != "CAMERA_AT_FLANGE_LR_AT_PROXY_MOUNT"
            or not isinstance(audit.get("max_relative_matrix_error"), (int, float))
            or not math.isfinite(audit["max_relative_matrix_error"])
            or audit["max_relative_matrix_error"] > 1e-8
            or audit.get("physical_mount_verified") is not False):
        raise RuntimeError("STRUCTURAL_V2_CHAIN_AUDIT_DRIFT")
    videos = result.get("video_refs")
    if not isinstance(videos, dict) or set(videos) != set(VIDEOS):
        raise RuntimeError("STRUCTURAL_V2_VIDEO_SET_DRIFT")
    for role, filename in VIDEOS.items():
        video_path = stage_root / "render" / filename
        item = stage.get("outputs", {}).get("videos", {}).get(role)
        if (videos[role] != artifact_ref(video_path)
                or not isinstance(item, dict)
                or item.get("video") != videos[role]
                or item.get("decoded_frames") != structural.FRAMES
                or not isinstance(item.get("encoded_duration_seconds"), (int, float))
                or not math.isfinite(item["encoded_duration_seconds"])
                or abs(item["encoded_duration_seconds"] - structural.FRAMES / structural.FPS)
                > 1.0 / structural.FPS
                or item.get("width") != 640 or item.get("height") != 480):
            raise RuntimeError("STRUCTURAL_V2_VIDEO_OR_FRAME_DRIFT:" + role)
    return {"result": artifact_ref(root / "RESULT.json"),
            "stage_result": artifact_ref(stage_path), "frame_states": states_ref,
            "chain_audit": artifact_ref(audit_path), "videos": videos,
            "decoded_frames": structural.FRAMES}


def _run_worker(packet: dict[str, Any]) -> tuple[int, Path, dict[str, Any] | None, str | None]:
    attempt, _, _ = _paths()
    log_path = attempt / "worker.log"
    worker: subprocess.Popen | None = None
    error: str | None = None
    stage: dict[str, Any] | None = None
    with log_path.open("x", encoding="utf-8") as log:
        try:
            _fence(packet, inputs=True)
            if (attempt / "candidate").exists():
                raise RuntimeError("STRUCTURAL_V2_PARTIAL_CANNOT_RESUME")
            environment = v1_executor._worker_environment(attempt)
            command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run",
                       OPERATION, "--output-root", str(attempt / "candidate/robot_v2")]
            worker = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            last_beat = time.monotonic()
            while worker.poll() is None:
                if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
                    v1_executor._stop_worker(worker)
                    error = "STRUCTURAL_V2_H20_OR_WALL_DEADLINE"
                    break
                if time.monotonic() - last_beat >= 25:
                    heartbeat(packet)
                    last_beat = time.monotonic()
                time.sleep(1)
            if error is None and worker.returncode != 0:
                error = f"STRUCTURAL_V2_CLI_EXIT_{worker.returncode}"
            if error is None:
                stage = _verify_stage(packet)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
        finally:
            if worker is not None and worker.poll() is None:
                v1_executor._stop_worker(worker)
    return (worker.returncode if worker is not None else 1), log_path, stage, error


def finish(packet: dict[str, Any], *, worker_returncode: int,
           worker_log: Path, stage: dict[str, Any] | None,
           error: str | None = None) -> dict[str, Any]:
    """Seal terminal v2 outcome; never promote partial outputs."""
    _fence(packet)
    attempt, _, receipt = _paths()
    if (attempt / "RESULT.json").exists() or receipt.exists():
        raise RuntimeError("STRUCTURAL_V2_DUPLICATE_TERMINAL")
    if worker_log != attempt / "worker.log" or not worker_log.is_file():
        raise RuntimeError("STRUCTURAL_V2_WORKER_LOG_ABSENT")
    refs = [*packet["inputs"]["source_files"].values(),
            *packet["inputs"]["robot_assets"].values(),
            *(packet["inputs"][role] for role in (*ADMISSION_ROLES, *SOURCE_ROLES))]
    input_errors = [";".join(errors) for ref in refs
                    if (errors := validate_artifact_ref(ref))]
    if worker_returncode == 0 and stage is not None and error is None and not input_errors:
        if stage != _verify_stage(packet):
            raise RuntimeError("STRUCTURAL_V2_FINAL_STAGE_DRIFT")
        status, failure = "PASSED", None
    else:
        status = "FAILED_RUNTIME_FINAL"
        failure = error or ("STRUCTURAL_V2_INPUT_SHA_DRIFT" if input_errors
                            else "STRUCTURAL_V2_UNSEALED_OR_WORKER_FAILED")
    result = {
        "schema_version": OUTPUT_SCHEMA, "task_id": TASK_ID, "status": status,
        "failure_code": failure, "worker_returncode": worker_returncode,
        "worker_log": artifact_ref(worker_log),
        "source_session": structural.SESSION_ID,
        "source_frames_expected": structural.FRAMES,
        "fps_nominal": structural.FPS,
        "hawor_outer_result": packet["inputs"]["hawor_outer_result"],
        "cpu_admission_result": packet["inputs"]["upstream_admission_result"],
        "post_input_sha_errors": input_errors, "stage": stage or {},
        "render_complete": status == "PASSED", "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "control_ground_truth": False, "user_review": "PENDING",
        "general_operation": None, "files_deleted": False,
        "claim_limit": packet["claim_limit"],
        "next_action": "Inspect F0/F399/F798, maximal hand-root jump and full PASS/HOLD grid; human review pending.",
    }
    atomic_json(attempt / "RESULT.json", result)
    atomic_json(receipt, {"schema_version": "exact78-newraw-robot-structural-v2-receipt-v1",
                          "task_id": TASK_ID, "status": status,
                          "result": artifact_ref(attempt / "RESULT.json"),
                          "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK_ID)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(attempt / "RESULT.json"),
               terminal_receipt=artifact_ref(receipt),
               quality="CAMERA_DIAGNOSTIC_ONLY" if status == "PASSED" else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDIT_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("STRUCTURAL_V2_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK_ID:
            item["execution_allowed"] = False
    index.update(packet_revision=packet["execution_revision"] + "_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=packet["execution_revision"] + "_" + status,
        expected_revision=_receipt_revision(), generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(attempt / "RESULT.json")}


def register_and_run(packet: dict[str, Any]) -> dict[str, Any]:
    register(packet)
    code, log, stage, error = _run_worker(packet)
    return finish(packet, worker_returncode=code, worker_log=log,
                  stage=stage, error=error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        print(json.dumps(check_only(), ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(register_and_run(packet_value()), ensure_ascii=False,
                         sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
