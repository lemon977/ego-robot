"""Read-only packet blueprint for one non-T0 sealed HaWoR -> CPU Robot task.

This candidate deliberately has no register-and-run shortcut. A separate
successor must publish its algorithm route and packet in one governance CAS,
then enter only through ``chaoyang run`` with a live fenced writer.
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

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH,
    RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref, atomic_json,
    canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
    validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.governance.register_single_task_packet import _validate_packet


OPERATION = "run_exact78_newraw_robot_cpu_candidate_v1"
ROUTE_KEY = "exact78_newraw_robot_cpu_candidate_v1"
WALL_SECONDS_MAX = 7200
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ASSET_PIN = REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json"
MOUNT = (REPO_ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1"
         "/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json")
FLANGE = REPO_ROOT / "assets/robot/V2相机固定法兰PRO版本.STL"
RUNTIME_RELATIVE = (
    "src/chaoyang/governance/execute_exact78_newraw_robot_cpu_candidate_v1.py",
    "src/chaoyang/governance/exact78_newraw_robot_handoff_candidate_v1.py",
    "src/chaoyang/ops/run_exact78_newraw_robot_cpu_candidate_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/governance/exact78_hawor_descriptor_adapter_candidate.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/register_single_task_packet.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/paths.py",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _cpu_probe(outer_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from chaoyang.governance.exact78_hawor_descriptor_adapter_candidate import (
        build_spec_from_sealed,
    )

    outer = _load(outer_path)
    cpu_ref = outer.get("producer_result")
    if not isinstance(cpu_ref, dict):
        raise RuntimeError("HAWOR_OUTER_CPU_RESULT_REF_REQUIRED")
    cpu_path = Path(cpu_ref["path"])
    if file_ref(cpu_path) != cpu_ref:
        raise RuntimeError("CPU_PRODUCER_RESULT_SHA_DRIFT")
    cpu = _load(cpu_path)
    if (cpu.get("schema_version") != "exact78-first-submit-cpu-admission-result-v1"
            or cpu.get("status") != "PASSED"
            or cpu.get("source_sha_stable_at_terminal") is not True):
        raise RuntimeError("CPU_PRODUCER_NOT_TERMINAL_PASSED")
    admission = cpu.get("admission")
    if not isinstance(admission, dict) or set(admission) != {
            "descriptor", "frame_time_map", "seal"}:
        raise RuntimeError("CPU_ADMISSION_THREE_REFS_REQUIRED")
    receipt = REPO_ROOT / "tasks/receipts" / f"{cpu['task_id'].upper()}.json"
    refs = {
        "upstream_admission_descriptor": admission["descriptor"],
        "upstream_admission_frame_time_map": admission["frame_time_map"],
        "upstream_admission_seal": admission["seal"],
        "upstream_admission_result": cpu_ref,
        "upstream_admission_receipt": file_ref(receipt),
    }
    task_id = f"exact78_newraw_robot_{cpu['source_session']}_{cpu['descriptor_sha256'][:16]}_v1"
    probe = {
        "task_id": task_id, "execution_mode": "EXACT78_CANDIDATE",
        "run_signature": "newraw-robot-handoff-probe-" + file_ref(outer_path)["sha256"],
        "inputs": {**refs, "upstream_admission_descriptor_sha256": cpu["descriptor_sha256"],
                   "submission_input_sha256": cpu["submission_input_sha256"],
                   "source_session": cpu["source_session"],
                   "source_root": cpu["source_root"]},
    }
    spec = build_spec_from_sealed(
        packet=probe, descriptor_ref=refs["upstream_admission_descriptor"],
        frame_map_ref=refs["upstream_admission_frame_time_map"],
        seal_ref=refs["upstream_admission_seal"],
        producer_result_ref=refs["upstream_admission_result"],
        producer_receipt_ref=refs["upstream_admission_receipt"],
    )
    return cpu, refs, spec


def _asset_refs() -> dict[str, dict[str, Any]]:
    pin = _load(ASSET_PIN)
    rows = pin.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("ROBOT_ASSET_PIN_EMPTY")
    refs = {"asset_pin": artifact_ref(ASSET_PIN),
            "mount_proxy": artifact_ref(MOUNT), "flange_mesh": artifact_ref(FLANGE)}
    for row in rows:
        relative = row.get("path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            raise RuntimeError("ROBOT_ASSET_PIN_ROW_INVALID")
        path = (REPO_ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(REPO_ROOT / "assets/robot"):
            raise RuntimeError("ROBOT_ASSET_OUTSIDE_PINNED_ROOT")
        ref = artifact_ref(path)
        if row.get("sha256") != ref["sha256"] or row.get("bytes") != ref["bytes"]:
            raise RuntimeError("ROBOT_ASSET_SHA_DRIFT:" + relative)
        refs[relative] = ref
    return refs


def packet_value(outer_result: Path) -> dict[str, Any]:
    """Return one non-runnable candidate Task Packet; no registration or render."""
    from chaoyang.governance.exact78_newraw_robot_handoff_candidate_v1 import (
        verify_handoff,
    )

    outer_path = outer_result.resolve(strict=True)
    cpu, cpu_refs, admission = _cpu_probe(outer_path)
    handoff = verify_handoff(outer_result=outer_path, admission=admission)
    task_id = f"exact78_newraw_robot_{handoff['session_id']}_{cpu['descriptor_sha256'][:16]}_v1"
    attempt = REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001"
    packet_path = REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    receipt = REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json"
    assets = _asset_refs()
    inputs = {
        **cpu_refs,
        "upstream_admission_descriptor_sha256": cpu["descriptor_sha256"],
        "submission_input_sha256": cpu["submission_input_sha256"],
        "source_session": handoff["session_id"], "source_root": cpu["source_root"],
        "source_files": handoff["source_files"],
        **{name: handoff[name] for name in (
            "hawor_outer_result", "hawor_receipt", "hawor_packet",
            "hawor_stage_seal", "hawor_native_open_audit", "hawor_inner_result",
            "hawor_npz", "hawor_frame_provenance", "hawor_input_audit", "raw_video")},
        "robot_assets": assets, "route": artifact_ref(ROUTE),
    }
    runtime_code = [artifact_ref(REPO_ROOT / name) for name in RUNTIME_RELATIVE]
    command = (
        f"PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run {OPERATION} "
        f"--hawor-result {shlex.quote(str(outer_path))} "
        f"--output-root {shlex.quote(str(attempt / 'candidate/robot'))}"
    )
    config = {
        "frame_count": handoff["frame_count"], "source_kind": handoff["source_kind"],
        "time_mode": handoff["time_mode"], "physical_eye": handoff["physical_eye"],
        "solver": "EXACT78_FRESH_ROBOT_V1_UNCHANGED_FFMPEG44_ENCODER",
        "arm_velocity_gate_rad": 0.12, "arm_acceleration_gate_rad": 0.06,
        "hand_velocity_gate_rad": 0.08, "hand_acceleration_gate_rad": 0.06,
        "cpu_threads_total": 2, "gpu_count": 0, "training_steps": 0,
        "model_calls": 0, "session_attempts_max": 1,
        "world_pose_authority": False, "timebase_authority": False,
        "visual_qualified": False, "resume_policy": "SEALED_HAWOR_ONLY_NO_PARTIAL",
    }
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime_code, "config": config,
        "command": command, "schema": "exact78-newraw-robot-cpu-consumer-v1",
    })).hexdigest()
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": task_id}
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": task_id,
        "objective": "One sealed new-raw HaWoR session through unchanged CPU Robot camera diagnostic",
        "phase": "EXACT78_NEWRAW_ROBOT_CPU_CANDIDATE",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": task_id.upper(), "execution_mode": "EXACT78_CANDIDATE",
        "registered_at": now.isoformat(),
        "deadline_at": (now + timedelta(seconds=WALL_SECONDS_MAX)).isoformat(),
        "read_set": ["AGENTS.md", str(ROUTE.relative_to(REPO_ROOT)),
                     str(outer_path.relative_to(REPO_ROOT)),
                     *(str(Path(cpu_refs[name]["path"]).relative_to(REPO_ROOT))
                       for name in ("upstream_admission_result",
                                    "upstream_admission_descriptor",
                                    "upstream_admission_frame_time_map",
                                    "upstream_admission_seal"))],
        "write_set": [str(attempt.relative_to(REPO_ROOT)),
                      str(packet_path.relative_to(REPO_ROOT)),
                      str(receipt.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK",
                          "TERMINAL_SAME_SESSION_HAWOR_SEALED", "SOURCE_SHA_PINNED",
                          "ROBOT_ASSETS_PINNED", "CPU_ONLY", "EXTERNAL_DATA_READ_ONLY",
                          "NO_OLD_ACTION_OR_FITTED_TEMPLATE", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["candidate/robot/RESULT.json",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/ROBOT_FRAME_STATES.json",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/ego_robot_diagnostic.mp4",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/fixed_external_full_robot.mp4",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/raw_motion_source_review.mp4",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": WALL_SECONDS_MAX,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_OR_ASSET_SHA_DRIFT", "DOUBLE_WRITER",
                            "HAWOR_UNSEALED_OR_NOT_TERMINAL", "FRAME_MAP_DRIFT",
                            "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "One-session camera Robot diagnostic only; no Clean, physical occlusion, world/timebase, visual/product, control or training qualification.",
        "writer": writer, "inputs": inputs, "runtime_code": runtime_code,
        "config": config, "run_signature": signature, "commands": [command],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("ROBOT_PACKET_PREFLIGHT:" + ";".join(errors))
    return packet


def _paths(task_id: str) -> tuple[Path, Path, Path]:
    return (REPO_ROOT / "_run/current" / task_id / "attempts/attempt_0001",
            REPO_ROOT / "tasks/current" / task_id / "TASK_PACKET.json",
            REPO_ROOT / "tasks/receipts" / f"{task_id.upper()}.json")


def _route_valid(contract: dict[str, Any], packet: dict[str, Any]) -> bool:
    routes = contract.get("special_status_contracts", {})
    route = routes.get(ROUTE_KEY) if isinstance(routes, dict) else None
    if (not isinstance(route, dict)
            or route.get("execution_status") != "CANDIDATE_FENCED_CPU_ONE_SESSION"
            or route.get("candidate_operation") != OPERATION
            or route.get("task_packet_required") is not True
            or route.get("current_runnable_operation") is not None
            or route.get("output_schema") != "exact78-newraw-robot-cpu-execution-result-v1"
            or route.get("model_calls_authorized") != 0
            or route.get("render_calls_authorized") != "ONE_FULL_SESSION_ONLY_WHEN_EXACT_PACKET_ACTIVE"
            or route.get("training_steps_authorized") != 0):
        return False
    closure = route.get("code_closure")
    if not isinstance(closure, list) or len(closure) != len(RUNTIME_RELATIVE):
        return False
    expected = {str(REPO_ROOT / relative) for relative in RUNTIME_RELATIVE}
    routed = {ref.get("path"): ref for ref in closure if isinstance(ref, dict)}
    runtime = {ref.get("path"): ref for ref in packet.get("runtime_code", [])
               if isinstance(ref, dict)}
    return set(routed) == expected and all(routed[path] == runtime.get(path)
                                            for path in expected)


def _assert_route_ready(packet: dict[str, Any]) -> str:
    """A route may be published already or staged for the same registration CAS."""
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("PRODUCTION_ROUTE_ALREADY_SELECTED")
    operation_ref = artifact_ref(REPO_ROOT / "src/chaoyang/ops" / f"{OPERATION}.py")
    if operation_ref not in packet.get("runtime_code", []):
        raise RuntimeError("ROBOT_OPERATION_CODE_NOT_PINNED")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("ROBOT_RUNTIME_SHA_DRIFT:" + ";".join(errors))
    published = load_json(ALGORITHM_CONTRACT_PATH)
    routes = published.get("special_status_contracts", {})
    published_route = routes.get(ROUTE_KEY) if isinstance(routes, dict) else None
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract
    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH))
    staged_routes = staged.get("special_status_contracts", {})
    staged_route = staged_routes.get(ROUTE_KEY) if isinstance(staged_routes, dict) else None
    if published_route is not None:
        if (not _route_valid(published, packet) or not _route_valid(staged, packet)
                or staged_route != published_route):
            raise RuntimeError("PUBLISHED_ROBOT_ROUTE_OR_BUILDER_DRIFT")
        if OPERATION not in _maintained_operations():
            raise RuntimeError("PUBLISHED_ROBOT_CLI_ROUTE_ABSENT")
        readiness = "PUBLISHED_CANDIDATE_ROUTE"
    else:
        if OPERATION in _maintained_operations():
            raise RuntimeError("UNPUBLISHED_ROBOT_CLI_ROUTE_DRIFT")
        if not _route_valid(staged, packet):
            raise RuntimeError("ROBOT_ROUTE_NOT_STAGED")
        readiness = "READY_TO_PUBLISH_ROUTE_WITH_PACKET"
    output, _, _ = _paths(packet["task_id"])
    expected = ["PYTHONPATH=src", "/usr/local/bin/python", "-B", "-m",
                "chaoyang.cli", "run", OPERATION, "--hawor-result",
                packet["inputs"]["hawor_outer_result"]["path"], "--output-root",
                str(output / "candidate/robot")]
    if (len(packet.get("commands", [])) != 1
            or shlex.split(packet["commands"][0]) != expected):
        raise RuntimeError("ROBOT_COMMAND_DRIFT")
    return readiness


def _receipt_revision() -> int:
    receipt = load_json(RECEIPT_PATH)
    for key, path in (("task_state", TASK_STATE_PATH),
                      ("task_packet_index_current", INDEX)):
        ref = receipt.get("files", {}).get(key)
        if not isinstance(ref, dict) or ref.get("path") != str(path):
            raise RuntimeError(f"GOVERNANCE_RECEIPT_BINDING_DRIFT:{key}")
        if errors := validate_artifact_ref(ref):
            raise RuntimeError(f"GOVERNANCE_RECEIPT_SHA_DRIFT:{key}:" + ";".join(errors))
    return int(receipt["governance_revision"])


def _index_copy() -> dict[str, Any]:
    index = load_json(INDEX)
    return {key: copy.deepcopy(value) for key, value in index.items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def register(packet: dict[str, Any]) -> None:
    """CAS-register one route plus one immutable packet; never infer prior output."""
    _assert_route_ready(packet)
    task_id = packet["task_id"]
    output, packet_path, receipt = _paths(task_id)
    if any(path.exists() for path in (output, packet_path, receipt)):
        raise RuntimeError("FRESH_ROBOT_TASK_REQUIRED")
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != process_identity(os.getpid())["start_ticks"]
            or writer.get("scope") != task_id):
        raise RuntimeError("ROBOT_WRITER_IDENTITY_DRIFT")
    fresh = packet_value(Path(packet["inputs"]["hawor_outer_result"]["path"]))
    if any(fresh[key] != packet[key] for key in
           ("inputs", "runtime_code", "config", "run_signature", "commands")):
        raise RuntimeError("ROBOT_PACKET_FRESH_PREFLIGHT_DRIFT")
    if errors := _validate_packet(packet):
        raise RuntimeError("ROBOT_PACKET_PREFLIGHT:" + ";".join(errors))
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_copy()
    if (state.get("next_task")
            or any(row.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
                   for row in state.get("tasks", []))
            or any(row.get("execution_allowed") is True
                   for row in index.get("task_packets", []))):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    revision = _receipt_revision()
    output.mkdir(parents=True)
    atomic_json(output / "WRITER.json", writer)
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    stamp = now_iso()
    state["tasks"].append({
        "task_id": task_id, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": packet["inputs"]["source_session"],
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(packet_path),
        "run_signature": packet["run_signature"], "quality": "PENDING",
        "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT",
    })
    state["next_task"] = {
        "task_id": task_id, "session": packet["inputs"]["source_session"],
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU_TWO_THREADS_GPU_ZERO",
        "stop_condition": "one full Robot camera diagnostic or finite deadline",
    }
    if errors := validate_task_state(state):
        raise RuntimeError("ROBOT_TASK_STATE_PREFLIGHT:" + ";".join(errors))
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
                   expected_revision=revision, generator_path=Path(__file__),
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def _fence(packet: dict[str, Any], *, inputs: bool = False) -> None:
    task_id = packet["task_id"]
    output, packet_path, _ = _paths(task_id)
    writer = packet["writer"]
    if (writer.get("pid") != os.getpid()
            or writer.get("proc_start_ticks") != process_identity(os.getpid())["start_ticks"]
            or load_json(output / "WRITER.json") != writer):
        raise RuntimeError("ROBOT_WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    rows = [row for row in state.get("tasks", []) if row.get("task_id") == task_id]
    active = [row for row in load_json(INDEX).get("task_packets", [])
              if row.get("execution_allowed") is True]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or rows[0].get("run_signature") != packet["run_signature"]
            or rows[0].get("pid") != writer["pid"]
            or rows[0].get("proc_start_ticks") != writer["proc_start_ticks"]
            or (state.get("next_task") or {}).get("task_id") != task_id
            or len(active) != 1 or active[0].get("task_id") != task_id
            or active[0].get("packet_sha256") != artifact_ref(packet_path)["sha256"]):
        raise RuntimeError("ROBOT_ACTIVE_PACKET_OR_SIGNATURE_DRIFT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("ROBOT_RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    if inputs:
        refs = [*packet["inputs"]["source_files"].values(),
                *packet["inputs"]["robot_assets"].values(),
                *(packet["inputs"][name] for name in (
                    "upstream_admission_descriptor", "upstream_admission_frame_time_map",
                    "upstream_admission_seal", "upstream_admission_result",
                    "upstream_admission_receipt", "hawor_outer_result", "hawor_receipt",
                    "hawor_packet", "hawor_stage_seal", "hawor_native_open_audit",
                    "hawor_inner_result", "hawor_npz", "hawor_frame_provenance",
                    "hawor_input_audit", "raw_video"))]
        for ref in refs:
            if errors := validate_artifact_ref(ref):
                raise RuntimeError("ROBOT_INPUT_SHA_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any]) -> None:
    _fence(packet)
    revision = _receipt_revision()
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(updated_at=now_iso(), heartbeat_at=now_iso())
    if errors := validate_task_state(state):
        raise RuntimeError("ROBOT_HEARTBEAT_STATE_PREFLIGHT:" + ";".join(errors))
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type=packet["execution_revision"] + "_HEARTBEAT",
                   expected_revision=revision, generator_path=Path(__file__))


def _verify_stage(packet: dict[str, Any]) -> dict[str, Any]:
    """Independently check the immutable Robot camera result and frame table."""
    attempt, _, _ = _paths(packet["task_id"])
    root = attempt / "candidate/robot"
    result_path = root / "RESULT.json"
    result = load_json(result_path)
    count = packet["config"]["frame_count"]
    session = packet["inputs"]["source_session"]
    if (result.get("schema_version") != "exact78-fresh-robot-ffmpeg44-successor-result-v1"
            or result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or result.get("session_id") != session
            or result.get("frame_count") != count
            or result.get("visual_qualified") is not False
            or result.get("product_qualified") is not False
            or result.get("training_eligible") is not False
            or result.get("control_ground_truth") is not False):
        raise RuntimeError("ROBOT_STAGE_RESULT_CLAIM_DRIFT")
    stage_path = root / "v1_stage_with_ffmpeg44_encoder/RESULT.json"
    if result.get("stage_result") != artifact_ref(stage_path):
        raise RuntimeError("ROBOT_V1_STAGE_RESULT_SHA_DRIFT")
    stage = load_json(stage_path)
    if (stage.get("schema_version") != "exact78-fresh-robot-stage-result-v1"
            or stage.get("session_id") != session
            or stage.get("frame_count") != count
            or stage.get("inputs", {}).get("hawor_npz") != packet["inputs"]["hawor_npz"]
            or stage.get("inputs", {}).get("frame_provenance")
               != packet["inputs"]["hawor_frame_provenance"]
            or stage.get("inputs", {}).get("input_audit")
               != packet["inputs"]["hawor_input_audit"]
            or stage.get("inputs", {}).get("raw_video") != packet["inputs"]["raw_video"]):
        raise RuntimeError("ROBOT_V1_STAGE_INPUT_DRIFT")
    states_path = stage_path.parent / "ROBOT_FRAME_STATES.json"
    states_ref = artifact_ref(states_path)
    if stage.get("outputs", {}).get("frame_states") != states_ref:
        raise RuntimeError("ROBOT_FRAME_STATES_SHA_DRIFT")
    states = load_json(states_path)
    rows = states.get("rows")
    if (states.get("schema_version") != "exact78-fresh-robot-frame-states-v1"
            or states.get("session_id") != session
            or states.get("frame_count") != count
            or states.get("world_pose_verified") is not False
            or states.get("timebase_verified") is not False
            or states.get("visual_qualified") is not False
            or states.get("training_eligible") is not False
            or not isinstance(rows, list) or len(rows) != 2 * count):
        raise RuntimeError("ROBOT_FRAME_ROWS_CLAIM_DRIFT")
    for index, row in enumerate(rows):
        if (row.get("frame") != index // 2
                or row.get("side") != ("left", "right")[index % 2]
                or row.get("training_eligible") is not False
                or row.get("world_pose_verified") is not False
                or not isinstance(row.get("motion_source"), str)
                or not isinstance(row.get("arm"), dict)
                or not isinstance(row.get("hand"), dict)):
            raise RuntimeError(f"ROBOT_FRAME_ROW_DRIFT:{index}")
    names = {
        "ego_robot": "ego_robot_diagnostic.mp4",
        "external_full_robot": "fixed_external_full_robot.mp4",
        "raw_motion_source": "raw_motion_source_review.mp4",
    }
    videos = result.get("verified_videos")
    if not isinstance(videos, dict) or set(videos) != set(names):
        raise RuntimeError("ROBOT_THREE_VIDEO_SET_DRIFT")
    for name, filename in names.items():
        item = videos[name]
        path = stage_path.parent / "render" / filename
        if (not isinstance(item, dict) or item.get("decoded_frames") != count
                or item.get("width") != 640 or item.get("height") != 480
                or item.get("video") != artifact_ref(path)
                or stage.get("outputs", {}).get("videos", {}).get(name) != item):
            raise RuntimeError(f"ROBOT_VIDEO_SHA_OR_FRAME_DRIFT:{name}")
    return {"result": artifact_ref(result_path), "stage_result": artifact_ref(stage_path),
            "frame_states": states_ref,
            "videos": {name: videos[name]["video"] for name in names},
            "decoded_frames": count}


def _worker_environment(attempt: Path) -> dict[str, str]:
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
        "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
        "CUDA_VISIBLE_DEVICES": "",
    })
    return environment


def _stop_worker(worker: subprocess.Popen) -> None:
    """Reap the whole renderer group even if it ignores the first signal."""
    if worker.poll() is not None:
        return
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        worker.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        worker.wait(timeout=15)


def _run_worker(packet: dict[str, Any]) -> tuple[int, Path, dict[str, Any] | None, str | None]:
    attempt, _, _ = _paths(packet["task_id"])
    log_path = attempt / "worker.log"
    worker: subprocess.Popen | None = None
    error: str | None = None
    stage: dict[str, Any] | None = None
    with log_path.open("x", encoding="utf-8") as log:
        try:
            _fence(packet, inputs=True)
            if (attempt / "candidate").exists():
                raise RuntimeError("ROBOT_PARTIAL_CANNOT_RESUME")
            environment = _worker_environment(attempt)
            command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run",
                       OPERATION, "--hawor-result",
                       packet["inputs"]["hawor_outer_result"]["path"],
                       "--output-root", str(attempt / "candidate/robot")]
            worker = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                      stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            last_beat = time.monotonic()
            while worker.poll() is None:
                if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
                    _stop_worker(worker)
                    error = "ROBOT_CPU_DEADLINE_EXPIRED"
                    break
                if time.monotonic() - last_beat >= 25:
                    heartbeat(packet)
                    last_beat = time.monotonic()
                time.sleep(1)
            if error is None and worker.returncode != 0:
                error = f"ROBOT_CLI_EXIT_{worker.returncode}"
            if error is None:
                stage = _verify_stage(packet)
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
        finally:
            if worker is not None and worker.poll() is None:
                _stop_worker(worker)
    return (worker.returncode if worker is not None else 1), log_path, stage, error


def finish(packet: dict[str, Any], *, worker_returncode: int,
           worker_log: Path, stage: dict[str, Any] | None,
           error: str | None = None) -> dict[str, Any]:
    """Seal a finite terminal result; a partial stage never becomes PASSED."""
    _fence(packet)
    attempt, _, receipt = _paths(packet["task_id"])
    if (attempt / "RESULT.json").exists() or receipt.exists():
        raise RuntimeError("ROBOT_DUPLICATE_TERMINAL_RESULT")
    if worker_log != attempt / "worker.log" or not worker_log.is_file():
        raise RuntimeError("ROBOT_WORKER_LOG_ABSENT")
    refs = [*packet["inputs"]["source_files"].values(),
            *packet["inputs"]["robot_assets"].values(),
            *(packet["inputs"][name] for name in (
                "upstream_admission_descriptor", "upstream_admission_frame_time_map",
                "upstream_admission_seal", "upstream_admission_result",
                "upstream_admission_receipt", "hawor_outer_result", "hawor_receipt",
                "hawor_packet", "hawor_stage_seal", "hawor_native_open_audit",
                "hawor_inner_result", "hawor_npz", "hawor_frame_provenance",
                "hawor_input_audit", "raw_video"))]
    input_errors = [";".join(errors) for ref in refs
                    if (errors := validate_artifact_ref(ref))]
    if worker_returncode == 0 and stage is not None and error is None and not input_errors:
        if stage != _verify_stage(packet):
            raise RuntimeError("ROBOT_FINAL_STAGE_DRIFT")
        status, failure = "PASSED", None
    else:
        status = "FAILED_RUNTIME_FINAL"
        failure = error or ("ROBOT_INPUT_SHA_DRIFT" if input_errors else
                            "ROBOT_UNSEALED_OR_WORKER_FAILED")
    result = {
        "schema_version": "exact78-newraw-robot-cpu-execution-result-v1",
        "task_id": packet["task_id"], "status": status, "failure_code": failure,
        "worker_returncode": worker_returncode, "worker_log": artifact_ref(worker_log),
        "source_session": packet["inputs"]["source_session"],
        "source_frames_expected": packet["config"]["frame_count"],
        "hawor_outer_result": packet["inputs"]["hawor_outer_result"],
        "cpu_admission_result": packet["inputs"]["upstream_admission_result"],
        "post_input_sha_errors": input_errors, "stage": stage or {},
        "render_complete": status == "PASSED", "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "control_ground_truth": False, "user_review": "PENDING",
        "general_operation": None, "files_deleted": False,
        "claim_limit": packet["claim_limit"],
        "next_action": "Inspect per-frame Robot states and three full camera videos; Clean/occlusion remain separate.",
    }
    atomic_json(attempt / "RESULT.json", result)
    atomic_json(receipt, {
        "schema_version": "exact78-newraw-robot-cpu-receipt-v1",
        "task_id": packet["task_id"], "status": status,
        "result": artifact_ref(attempt / "RESULT.json"),
        "deletion_authorized": False,
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == packet["task_id"])
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(attempt / "RESULT.json"),
               terminal_receipt=artifact_ref(receipt),
               quality="CAMERA_DIAGNOSTIC_ONLY" if status == "PASSED" else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDIT_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("ROBOT_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
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
        expected_revision=_receipt_revision(), generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(attempt / "RESULT.json")}


def register_and_run(packet: dict[str, Any]) -> dict[str, Any]:
    """Only a future explicitly contracted successor may call this path."""
    register(packet)
    code, log, stage, error = _run_worker(packet)
    return finish(packet, worker_returncode=code, worker_log=log,
                  stage=stage, error=error)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hawor-result", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--register-and-run", action="store_true")
    args = parser.parse_args(argv)
    packet = packet_value(args.hawor_result)
    if args.register_and_run:
        print(json.dumps(register_and_run(packet), ensure_ascii=False, sort_keys=True))
        return 0
    try:
        readiness = _assert_route_ready(packet)
    except RuntimeError as exc:
        readiness = str(exc)
    print(json.dumps({
        "status": readiness,
        "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "frame_count": packet["config"]["frame_count"],
        "source_kind": packet["config"]["source_kind"],
        "registration_started": False, "render_calls": 0,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
