"""Unregistered one-attempt, CPU-only Cards042 Clean+Robot diagnostic packet.

Only --check-only is appropriate until a maintained candidate route and a new
finite Task Packet are published together by the sole governance publisher.
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

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.ops import run_exact78_cards042_clean_robot_diagnostic_v1 as operation
from chaoyang.pipeline import exact78_clean_robot_diagnostic_cards042_v1 as composite
from chaoyang.pipeline import exact78_robot_render_direction_v1 as direction


TASK = operation.TASK
SCRIPT = Path(__file__)
OUT = operation.ATTEMPT
SPEC = operation.SPEC
PACKET = operation.PACKET
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVIEW_INDEX = REPO_ROOT / "review/EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1.md"
DESIGN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/23_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_CANDIDATE_ZH.md"
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
PYTHON = Path("/usr/local/bin/python")
OPERATION = operation.OPERATION
CONTRACT_KEY = "exact78_cards042_clean_robot_diagnostic_candidate_v1_20261001"
COMMAND = f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} --spec {SPEC}"
SEALED_OUTERS = {
    "scene_result": REPO_ROOT / "_run/current/exact78_raw_to_robot_t2_cards042_frontend_v1_20260930/attempts/attempt_0001/RESULT.json",
    "action_result": REPO_ROOT / "_run/current/exact78_raw_to_robot_t2_0902_cards042_v1_20260930/attempts/attempt_0001/RESULT.json",
    "render_result": REPO_ROOT / "_run/current/exact78_robot_render_direction_cards042_v1_20260930/attempts/attempt_0001/RESULT.json",
}


def frozen_spec() -> dict[str, Any]:
    inputs = {key: artifact_ref(path) for key, path in SEALED_OUTERS.items()}
    binding = composite.prepare_binding(inputs)
    if (binding["frame_count"] != 171 or binding["fps_nominal"] != 30
            or binding["geometry_decision"] != "UNKNOWN"):
        raise RuntimeError("COMPOSITE_BINDING_DOMAIN_DRIFT")
    return {"task_id": TASK, "session_id": composite.SESSION,
            "inputs": inputs, "output_root": str(OUT / "candidate")}


def virtual_json_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def runtime_refs() -> list[dict[str, Any]]:
    paths = [SCRIPT,
             REPO_ROOT / "src/chaoyang/ops/run_exact78_cards042_clean_robot_diagnostic_v1.py",
             REPO_ROOT / "src/chaoyang/pipeline/exact78_clean_robot_diagnostic_cards042_v1.py",
             REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
             *(REPO_ROOT / relative for relative in direction.PINNED_CODE_SHA),
             REPO_ROOT / "src/chaoyang/governance/common.py",
             REPO_ROOT / "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
             REPO_ROOT / "src/chaoyang/governance/exact78_operation_gate.py",
             REPO_ROOT / "src/chaoyang/governance/current_r3_contracts.py",
             REPO_ROOT / "src/chaoyang/cli/__init__.py",
             REPO_ROOT / "src/chaoyang/ops/immutable_artifact_io.py"]
    direction.verify_code_pins()
    return [artifact_ref(path) for path in paths]


def packet_value(spec_ref: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"composite_spec": spec_ref, "sealed_fresh_outers": spec["inputs"],
              "robot_asset_pin": artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json"),
              "candidate_design": artifact_ref(DESIGN)}
    runtime = runtime_refs()
    config = {"session_id": composite.SESSION, "frames": 171, "fps": 30,
              "candidate_role": "CLEAN_ROBOT_DIAGNOSTIC_ONLY",
              "scene_version": "SEALED_FRESH_V1_NOT_V4", "gpu_count": 0,
              "model_calls": 0, "cpu_threads_max": 2,
              "object_mask_occlusion_authority": False,
              "clean_geometry_authority": False,
              "numeric_quality_gates_changed": False,
              "human_review": "PENDING"}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": COMMAND, "schema": "exact78-cards042-clean-robot-diagnostic-task-v1",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One 171-frame Cards042 Clean+Robot visual diagnostic with 4-panel review",
        "phase": "EXACT78_CLEAN_ROBOT_DIAGNOSTIC",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=2), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(DESIGN.relative_to(REPO_ROOT)),
                     *(str(path.relative_to(REPO_ROOT)) for path in SEALED_OUTERS.values()),
                     str(SPEC.relative_to(REPO_ROOT)),
                     "src/chaoyang/ops/run_exact78_cards042_clean_robot_diagnostic_v1.py",
                     "src/chaoyang/pipeline/exact78_clean_robot_diagnostic_cards042_v1.py"],
        "write_set": [str(OUT.relative_to(REPO_ROOT)),
                      str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)),
                      str(REVIEW_INDEX.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK",
                          "SEALED_FRESH_CARDS042_V1_SCENE_AND_ACTION",
                          "SOURCE_SHA_PINNED", "ROBOT_ASSETS_PINNED",
                          "NO_GPU_OR_MODEL", "NO_CLEAN_GEOMETRY",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["COMPOSITE_SPEC.json", "candidate/FRAME_STATUS.json",
                             "candidate/RESULT.json",
                             *(f"candidate/videos/{name}" for name in operation.VIDEO_NAMES.values()),
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 7200,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "DEADLINE", "FRAME_OR_VIDEO_VALIDATION_FAIL",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Offline Clean+Robot diagnostic only; no scene occlusion, visual PASS, geometry, product or training authority",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_COMPOSITE_PACKET:" + ";".join(errors))
    return packet


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def _route_valid(contract: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(CONTRACT_KEY, {})
    expected = [artifact_ref(REPO_ROOT / relative) for relative in (
        "src/chaoyang/ops/run_exact78_cards042_clean_robot_diagnostic_v1.py",
        "src/chaoyang/pipeline/exact78_clean_robot_diagnostic_cards042_v1.py",
        "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
        *direction.PINNED_CODE_SHA)]
    return (route.get("candidate_operation") == OPERATION
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-cards042-clean-robot-diagnostic-v1"
            and route.get("model_calls_authorized") == 0
            and route.get("training_steps_authorized") == 0
            and isinstance(route.get("code_closure"), list)
            and all(ref in route["code_closure"] for ref in expected))


def _candidate_route_readiness() -> tuple[bool, str]:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.common import ALGORITHM_CONTRACT_PATH, BASELINE_REGISTRY_PATH

    if OPERATION in _maintained_operations():
        return ((True, "PUBLISHED_CANDIDATE_ROUTE")
                if _route_valid(load_json(ALGORITHM_CONTRACT_PATH))
                else (False, "COMPOSITE_PUBLISHED_ROUTE_DRIFT"))
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH))
    if not _route_valid(staged):
        return False, "COMPOSITE_CANDIDATE_ROUTE_NOT_STAGED"
    return True, "READY_TO_PUBLISH_COMPOSITE_ROUTE_WITH_PACKET"


def readiness() -> tuple[bool, str]:
    if datetime.now().astimezone() >= CAMPAIGN_END:
        return False, "CAMPAIGN_DEADLINE_EXPIRED"
    if any(path.exists() for path in (OUT, PACKET, RECEIPT, REVIEW_INDEX)):
        return False, "FRESH_COMPOSITE_TASK_REQUIRED"
    state = load_json(TASK_STATE_PATH)
    if any(row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")
           for row in state["tasks"]):
        return False, "ACTIVE_TASK_CONFLICT"
    if state.get("next_task") or any(row.get("execution_allowed")
                                     for row in load_json(INDEX)["task_packets"]):
        return False, "ACTIVE_ROUTE_CONFLICT"
    return _candidate_route_readiness()


def register(packet: dict[str, Any], spec: dict[str, Any]) -> None:
    ready, reason = readiness()
    if not ready:
        raise RuntimeError(reason)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    if (state.get("next_task")
            or any(row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")
                   for row in state["tasks"])
            or any(row.get("execution_allowed") for row in index["task_packets"])):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if not _candidate_route_readiness()[0]:
        raise RuntimeError("COMPOSITE_ROUTE_CHANGED_BEFORE_CAS")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("COMPOSITE_CODE_CHANGED_BEFORE_CAS:" + ";".join(errors))
    for ref in (*spec["inputs"].values(), packet["inputs"]["robot_asset_pin"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("COMPOSITE_INPUT_CHANGED_BEFORE_CAS:" + ";".join(errors))
    OUT.mkdir(parents=True)
    atomic_json(SPEC, spec)
    if artifact_ref(SPEC) != packet["inputs"]["composite_spec"]:
        raise RuntimeError("COMPOSITE_SPEC_PRECOMPUTED_SHA_DRIFT")
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": spec["session_id"], "updated_at": now,
        "heartbeat_at": now, "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": spec["session_id"],
        "prerequisites": packet["prerequisites"], "expected_resource": "TWO_CPU_THREADS_NO_GPU",
        "stop_condition": "one full 171-frame diagnostic result or deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("COMPOSITE_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK,
        "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("COMPOSITE_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if (len(active) != 1 or active[0]["task_id"] != TASK
            or active[0]["run_signature"] != packet["run_signature"]):
        raise RuntimeError("COMPOSITE_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    routes = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if (len(routes) != 1 or routes[0]["task_id"] != TASK
            or routes[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("COMPOSITE_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("COMPOSITE_RUNTIME_CODE_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (OUT / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-clean-robot-diagnostic-resume-v1",
            "elapsed_seconds": round(elapsed, 1),
            "candidate_result_exists": (OUT / "candidate/RESULT.json").is_file(),
            "next_action": "Wait for fenced CPU worker or inspect terminal receipt"})


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    cache = OUT / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache), OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if (deadline - datetime.now().astimezone()).total_seconds() < 120:
        raise RuntimeError("COMPOSITE_BUDGET_TOO_SHORT")
    log = OUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen([str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
            OPERATION, "--spec", str(SPEC)], cwd=REPO_ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT)
        last_beat = 0.0
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if datetime.now().astimezone() >= deadline:
                    raise RuntimeError("COMPOSITE_DEADLINE")
                if elapsed - last_beat >= 50:
                    heartbeat(packet, elapsed)
                    last_beat = elapsed
                time.sleep(2)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    return process.returncode, log


def candidate_scope_valid(value: dict[str, Any], spec: dict[str, Any]) -> bool:
    return (value.get("status") == "COMPLETE_DIAGNOSTIC_VIDEO_NOT_VISUAL_PASS"
            and value.get("session_id") == composite.SESSION
            and value.get("frame_count") == 171
            and value.get("fps_nominal") == 30
            and value.get("diagnostic_video_complete") is True
            and value.get("inputs") == spec["inputs"]
            and value.get("time_mode") == "CLIP_INDEX_ONLY_TIMEBASE_UNVERIFIED"
            and value.get("geometry_decision") == "UNKNOWN"
            and value.get("display_policy") == composite.DISPLAY_POLICY
            and all(value.get(key) is False for key in (
                "scene_depth_consumed", "object_mask_consumed_for_occlusion",
                "visual_qualified", "product_qualified", "training_eligible",
                "control_ground_truth")))


def verify_candidate(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    path = OUT / "candidate/RESULT.json"
    if not path.is_file():
        return None, ["CANDIDATE_RESULT_MISSING"]
    try:
        value = load_json(path)
        if not candidate_scope_valid(value, spec):
            raise RuntimeError("CANDIDATE_IDENTITY_OR_CLAIM_DRIFT")
        if value["frame_status"] != artifact_ref(OUT / "candidate/FRAME_STATUS.json"):
            raise RuntimeError("CANDIDATE_FRAME_STATUS_SHA_DRIFT")
        rows = load_json(OUT / "candidate/FRAME_STATUS.json")["rows"]
        if len(rows) != 171:
            raise RuntimeError("CANDIDATE_FRAME_STATUS_COUNT")
        binding = composite.prepare_binding(spec["inputs"])
        for index, row in enumerate(rows):
            if (row.get("output_index") != index
                    or row.get("scene_decode_display_index") != index
                    or row.get("hawor_output_index") != index
                    or row.get("geometry_decision") != "UNKNOWN"
                    or row.get("display_policy") != composite.DISPLAY_POLICY
                    or row.get("raw_model_input") != binding["raw_rows"][index]["model_input"]
                    or row.get("clean") != binding["clean_rows"][index]["clean"]
                    or row.get("q_state_ref") != binding["robot_states_ref"]
                    or row.get("display_masks") != {
                        key: binding["mask_rows"][index]["masks"][key]
                        for key in ("removal", "protected")}
                    or row.get("mask_geometry_authority") is not False
                    or any(row.get(key) is not False for key in (
                        "scene_depth_consumed", "object_mask_consumed_for_occlusion",
                        "visual_qualified", "product_qualified", "training_eligible",
                        "control_ground_truth"))):
                raise RuntimeError(f"CANDIDATE_FRAME_SCOPE_DRIFT:{index}")
            if not all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in (
                    "composite_array_sha256", "review_panel_array_sha256",
                    "robot_rgb_array_sha256", "robot_labels_array_sha256",
                    "robot_depth_array_sha256", "pixel_source_array_sha256")):
                raise RuntimeError(f"CANDIDATE_FRAME_SHA_MISSING:{index}")
            staged = OUT / "candidate/composite_frames" / f"{index:06d}.png"
            if row.get("composite_frame_png") != artifact_ref(staged):
                raise RuntimeError(f"CANDIDATE_STAGED_FRAME_SHA_DRIFT:{index}")
        for key, name in operation.VIDEO_NAMES.items():
            size = (640, 480) if key == "ego_clean_robot" else (1280, 960)
            verified = operation.verify_video(OUT / "candidate/videos" / name, 171, size, 30)
            if value["videos"].get(key) != verified:
                raise RuntimeError(f"CANDIDATE_VIDEO_REF_DRIFT:{key}")
        return value, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"CANDIDATE_VALIDATION:{error}"]


def _write_review_index(candidate: dict[str, Any]) -> None:
    videos = candidate["videos"]
    fresh = videos["ego_clean_robot"]["video"]
    grid = videos["raw_mask_clean_composite"]["video"]
    raw = candidate["review_classes"]["raw_plus_motion_source"]["video"]
    external = candidate["review_classes"]["ego_robot_plus_fixed_external"]["sealed_external"]["video"]
    body = ("# Cards042 Clean＋Robot 诊断审阅（非视觉 PASS）\n\n"
            f"- [Raw＋动作来源]({raw['path']})\n"
            f"- [Clean＋Robot 第一人称诊断]({fresh['path']})；"
            f"[封存固定第三人称全臂]({external['path']})\n"
            f"- [Raw／Mask／Clean／合成四宫格]({grid['path']})\n\n"
            f"合成 RESULT：[{(OUT / 'candidate/RESULT.json').name}]({OUT / 'candidate/RESULT.json'})。"
            "Clean 和 Mask 仅供离线显示；遮挡几何 UNKNOWN，人审 PENDING。\n")
    with REVIEW_INDEX.open("x", encoding="utf-8") as stream:
        stream.write(body)


def finish(packet: dict[str, Any], spec: dict[str, Any], code: int,
           log: Path, reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    candidate, errors = verify_candidate(spec)
    complete = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if complete else "FAILED_RUNTIME_FINAL"
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_COMPOSITE_TERMINAL_RESULT")
    if complete:
        _write_review_index(candidate)
    result = {"schema_version": "exact78-cards042-clean-robot-diagnostic-outer-v1",
        "task_id": TASK, "status": status, "execution_complete": complete,
        "diagnostic_video_complete": complete,
        "worker_returncode": code, "reason": reason, "validation_errors": errors,
        "source_session": composite.SESSION, "source_frames_expected": 171,
        "composite_spec": artifact_ref(SPEC),
        "candidate_result": artifact_ref(OUT / "candidate/RESULT.json") if candidate else None,
        "review_index": artifact_ref(REVIEW_INDEX) if complete else None,
        "worker_log": artifact_ref(log), "model_calls": 0, "gpu_count": 0,
        "geometry_decision": "UNKNOWN", "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "control_ground_truth": False, "user_review": "PENDING",
        "source_mutated": False, "files_deleted": False,
        "next_action": "Review the three video classes and 171-frame status; fix Clean/Robot quality in separate finite lanes",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-cards042-clean-robot-diagnostic-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None,
               updated_at=now_iso(), heartbeat_at=now_iso(),
               result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("COMPOSITE_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_CARDS042_CLEAN_ROBOT_DIAGNOSTIC_V1_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = frozen_spec()
    ready, reason = readiness()
    spec_ref = virtual_json_ref(SPEC, spec)
    packet = packet_value(spec_ref, spec)
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "task_id": TASK, "command": COMMAND,
            "spec_sha256": spec_ref["sha256"], "run_signature": packet["run_signature"],
            "source_frames": 171, "gpu_count": 0, "model_calls": 0,
            "new_attempt_write_set": packet["write_set"][0]}, ensure_ascii=False))
        return 0
    if not ready:
        raise RuntimeError(reason)
    register(packet, spec)
    try:
        code, log = run_worker(packet)
        failure = None
    except Exception as error:
        code, failure = 1, f"{type(error).__name__}:{error}"
        log = OUT / "executor_error.log"
        with log.open("x", encoding="utf-8") as stream:
            stream.write(failure + "\n")
    terminal = finish(packet, spec, code, log, failure)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
