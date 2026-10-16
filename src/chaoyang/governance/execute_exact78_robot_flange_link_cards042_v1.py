"""Prepare/register one fenced Cards042 flange-link Robot-only successor.

--check-only is read-only and remains unroutable until the independent builder
route is staged. --execute is intentionally not invoked by this code-prep task.
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
import sys
import time
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.pipeline import exact78_fresh_robot_flange_successor_v1 as flange


TASK = "exact78_robot_flange_link_cards042_v1_20261001"
OPERATION = "run_exact78_robot_flange_link_cards042_v1"
CONTRACT_KEY = "exact78_robot_flange_link_cards042_candidate_v1_20261001"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
SPEC = OUT / "FLANGE_SPEC.json"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_ROBOT_FLANGE_LINK_CARDS042_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PYTHON = Path("/usr/local/bin/python")
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
COMMAND = f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} --spec {SPEC}"
VIDEO_FILES = {
    "ego_robot": "ego_robot_diagnostic.mp4",
    "external_full_robot": "fixed_external_full_robot.mp4",
    "raw_motion_source": "raw_motion_source_review.mp4",
}
CODE_PATHS = (
    "src/chaoyang/governance/execute_exact78_robot_flange_link_cards042_v1.py",
    "src/chaoyang/ops/run_exact78_robot_flange_link_cards042_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_flange_successor_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/common.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/paths.py",
)


def _virtual_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def frozen_spec() -> dict[str, Any]:
    spec = flange._SEALED_DEVELOPMENT["play_cards_0902_042"]
    hawor = REPO_ROOT / spec["root"]
    audit = load_json(hawor / "INPUT_AUDIT.json")
    raw = Path(audit["raw_source_ref"]["path"])
    flange.validate_sealed_development(
        hawor_npz=hawor / "HAWOR_CAMERA_PREDICTIONS.npz",
        frame_provenance=hawor / "FRAME_PROVENANCE.json",
        input_audit=hawor / "INPUT_AUDIT.json", raw_video=raw,
        source_kind="tracker", session_id="play_cards_0902_042")
    refs = {
        "hawor_result": artifact_ref(hawor / "RESULT.json"),
        "hawor_npz": artifact_ref(hawor / "HAWOR_CAMERA_PREDICTIONS.npz"),
        "frame_provenance": artifact_ref(hawor / "FRAME_PROVENANCE.json"),
        "input_audit": artifact_ref(hawor / "INPUT_AUDIT.json"),
        "hawor_seal": artifact_ref(hawor.parent / "HAWOR_STAGE_SEAL.json"),
        "native_open_audit": artifact_ref(hawor.parent / "NATIVE_OPEN_AUDIT.json"),
        "raw_video": artifact_ref(raw),
        "robot_asset_pin": artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json"),
        "mount_proxy": artifact_ref(flange.stage_v1.MOUNT),
        "flange_mesh": artifact_ref(flange.stage_v1.FLANGE),
    }
    if refs["raw_video"] != audit["raw_source_ref"]:
        raise RuntimeError("FLANGE_RAW_SOURCE_REF_DRIFT")
    return {
        "schema_version": "exact78-flange-link-cards042-spec-v1",
        "task_id": TASK, "session_id": "play_cards_0902_042",
        "source_kind": "tracker", "source_role": "DEVELOPMENT_ONLY",
        "frame_count": 171, "fps_nominal": 30.0,
        "inputs": refs, "output_root": str(OUT / "candidate"),
        "single_change": "IK_AND_RASTER_HAND_ROOT_PARENT_FLANGE_INSTEAD_OF_SIDE_TOOL",
        "numeric_gates_changed": False, "base_search_rule_changed": False,
        "renderer_camera_base_rule_changed": False,
        "old_robot_result_input": False,
    }


def runtime_refs() -> list[dict[str, Any]]:
    flange._check_code_pins()
    return [artifact_ref(REPO_ROOT / path) for path in CODE_PATHS]


def packet_value(spec: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    deadline = min(now + timedelta(hours=2), CAMPAIGN_END)
    writer = {"pid": os.getpid(),
              "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24),
              "scope": TASK}
    inputs = {"flange_spec": _virtual_ref(SPEC, spec),
              "source_refs": spec["inputs"]}
    runtime = runtime_refs()
    config = {"frames": 171, "fps_nominal": 30.0,
              "source_role": "DEVELOPMENT_ONLY", "model_calls": 0,
              "single_change": spec["single_change"],
              "numeric_gates_changed": False,
              "base_search_rule_changed": False,
              "renderer_camera_base_rule_changed": False,
              "old_robot_result_input": False}
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime, "config": config,
        "command": COMMAND, "schema": "exact78-flange-link-cards042-task-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One full 171-frame sealed-HaWoR Cards042 Robot-only flange-link candidate",
        "phase": "EXACT78_ROBOT_FLANGE_LINK_ONLY",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_ROBOT_FLANGE_LINK_CARDS042_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": deadline.isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(SPEC.relative_to(REPO_ROOT)),
                     *(str(Path(spec["inputs"][key]["path"]).relative_to(REPO_ROOT))
                       for key in ("hawor_result", "hawor_npz", "frame_provenance",
                                   "input_audit", "hawor_seal", "native_open_audit"))],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_CARDS042_HAWOR",
                          "SOURCE_SHA_PINNED", "ROBOT_ASSETS_PINNED", "CPU_ONLY",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["FLANGE_SPEC.json", "candidate/RESULT.json",
                             "candidate/MOUNT_CHAIN_AUDIT.json",
                             "candidate/flange_link_stage_ffmpeg44/RESULT.json",
                             *("candidate/flange_link_stage_ffmpeg44/render/" + name
                               for name in VIDEO_FILES.values()),
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 7200,
                    "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "CAMPAIGN_OR_TASK_DEADLINE", "VIDEO_DECODE_OR_TIME_FAIL",
                            "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Cards042 development-only flange-parent camera diagnostic; original numeric/base/camera-direction rules unchanged; no world, occlusion, visual, product or training pass",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_FLANGE_PACKET:" + ";".join(errors))
    return packet


def _route_valid(contract: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(CONTRACT_KEY, {})
    expected = [artifact_ref(REPO_ROOT / path) for path in CODE_PATHS
                if path.startswith("src/chaoyang/ops/") or
                path.startswith("src/chaoyang/pipeline/")]
    return (route.get("candidate_operation") == OPERATION
            and route.get("execution_status") == "CANDIDATE_FENCED_ROBOT_ONLY_SINGLE_SESSION"
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-flange-link-robot-successor-result-v1"
            and route.get("model_calls_authorized") == 0
            and route.get("training_steps_authorized") == 0
            and isinstance(route.get("code_closure"), list)
            and all(ref in route["code_closure"] for ref in expected))


def _candidate_route_readiness() -> tuple[bool, str]:
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.common import ALGORITHM_CONTRACT_PATH, BASELINE_REGISTRY_PATH

    if OPERATION in _maintained_operations():
        return ((True, "PUBLISHED_CANDIDATE_ROUTE") if _route_valid(load_json(ALGORITHM_CONTRACT_PATH))
                else (False, "FLANGE_PUBLISHED_ROUTE_DRIFT"))
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH), load_json(AUTHORITY_PATH))
    if not _route_valid(staged):
        return False, "FLANGE_CANDIDATE_ROUTE_NOT_STAGED"
    return True, "READY_TO_PUBLISH_CANDIDATE_ROUTE_WITH_PACKET"


def readiness() -> tuple[bool, str]:
    if datetime.now().astimezone() >= CAMPAIGN_END:
        return False, "CAMPAIGN_DEADLINE_EXPIRED"
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        return False, "FRESH_FLANGE_TASK_REQUIRED"
    state = load_json(TASK_STATE_PATH)
    if (state.get("next_task") or any(row["status"] in
                                  ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")
                                  for row in state["tasks"])):
        return False, "ACTIVE_TASK_CONFLICT"
    if any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        return False, "ACTIVE_ROUTE_CONFLICT"
    return _candidate_route_readiness()


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at",
                           "supersedes_index"}}


def _registration_values(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = _index_copy()
    stamp = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "play_cards_0902_042",
        "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": _virtual_ref(PACKET, packet),
        "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": "play_cards_0902_042",
        "prerequisites": packet["prerequisites"],
        "expected_resource": "TWO_CPU_THREADS_NO_GPU",
        "stop_condition": "one full Robot-only result or two-hour/campaign deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("FLANGE_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({
        "task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": _virtual_ref(PACKET, packet)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_ROBOT_FLANGE_LINK_CARDS042_V1_RUNNING",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS",
                 supersedes_index=artifact_ref(INDEX))
    return state, index


def register(packet: dict[str, Any], spec: dict[str, Any]) -> None:
    ready, reason = readiness()
    if not ready:
        raise RuntimeError(reason)
    if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
        raise RuntimeError("FLANGE_TASK_DEADLINE_EXPIRED")
    for ref in [*packet["runtime_code"], *spec["inputs"].values()]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("FLANGE_PRE_REGISTER_SHA_DRIFT:" + ";".join(errors))
    if not _candidate_route_readiness()[0]:
        raise RuntimeError("FLANGE_ROUTE_CHANGED_BEFORE_CAS")
    state, index = _registration_values(packet)  # preflight before any write
    OUT.mkdir(parents=True)
    atomic_json(SPEC, spec)
    if artifact_ref(SPEC) != packet["inputs"]["flange_spec"]:
        raise RuntimeError("FLANGE_SPEC_VIRTUAL_SHA_DRIFT")
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    if artifact_ref(PACKET)["sha256"] != index["task_packets"][-1]["packet_sha256"]:
        raise RuntimeError("FLANGE_PACKET_VIRTUAL_SHA_DRIFT")
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_ROBOT_FLANGE_LINK_CARDS042_V1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX,
        task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer["pid"] != os.getpid()
            or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("FLANGE_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if (len(active) != 1 or active[0]["task_id"] != TASK
            or active[0]["run_signature"] != packet["run_signature"]):
        raise RuntimeError("FLANGE_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    routes = [row for row in load_json(INDEX)["task_packets"]
              if row.get("execution_allowed")]
    if (len(routes) != 1 or routes[0]["task_id"] != TASK
            or routes[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("FLANGE_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("FLANGE_RUNTIME_CODE_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_ROBOT_FLANGE_LINK_CARDS042_V1_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (OUT / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-flange-link-resume-v1",
            "elapsed_seconds": round(elapsed, 1),
            "candidate_result_exists": (OUT / "candidate/RESULT.json").is_file(),
            "next_action": "Wait for fenced CPU worker or inspect terminal receipt"})


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    cache = OUT / "cache"
    cache.mkdir()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"), OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    if (deadline - datetime.now().astimezone()).total_seconds() < 120:
        raise RuntimeError("FLANGE_BUDGET_TOO_SHORT")
    log = OUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen([str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
            OPERATION, "--spec", str(SPEC)], cwd=REPO_ROOT, env=environment,
            stdout=stream, stderr=subprocess.STDOUT)
        last_beat = 0.0
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if datetime.now().astimezone() >= deadline:
                    raise RuntimeError("FLANGE_TASK_DEADLINE")
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


def verify_candidate(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    path = OUT / "candidate/RESULT.json"
    if not path.is_file():
        return None, ["FLANGE_CANDIDATE_RESULT_MISSING"]
    try:
        result = load_json(path)
        if (result.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
                or result.get("session_id") != "play_cards_0902_042"
                or result.get("frame_count") != 171
                or result.get("source_role") != "DEVELOPMENT_ONLY"
                or result.get("single_change") != spec["single_change"]
                or result.get("numeric_gates_changed") is not False
                or result.get("base_search_rule_changed") is not False
                or result.get("renderer_camera_base_rule_changed") is not False
                or result.get("visual_qualified") is not False
                or result.get("product_qualified") is not False
                or result.get("training_eligible") is not False
                or result.get("control_ground_truth") is not False):
            raise RuntimeError("FLANGE_CANDIDATE_IDENTITY_OR_SCOPE_DRIFT")
        for role, key in (("hawor_result", "sealed_hawor_result"),
                          ("hawor_seal", "hawor_stage_seal"),
                          ("native_open_audit", "hawor_native_open_audit")):
            if result.get(key) != spec["inputs"][role]:
                raise RuntimeError(f"FLANGE_CANDIDATE_SOURCE_REF_DRIFT:{role}")
        chain = result.get("mount_chain_audit")
        if (not isinstance(chain, dict)
                or Path(chain.get("path", "")).resolve() !=
                OUT / "candidate/MOUNT_CHAIN_AUDIT.json"
                or validate_artifact_ref(chain)):
            raise RuntimeError("FLANGE_MOUNT_CHAIN_AUDIT_DRIFT")
        audit = load_json(Path(chain["path"]))
        if (audit.get("frame_count") != 171
                or audit.get("max_relative_matrix_error", float("inf")) > 1e-8
                or audit.get("continuity_quality_pass") is not False
                or "NOT_CHECKED" not in audit.get("camera_projection_chain_claim", "")):
            raise RuntimeError("FLANGE_MOUNT_CHAIN_CONTENT_DRIFT")
        stage = result.get("stage_result")
        if (not isinstance(stage, dict)
                or Path(stage.get("path", "")).resolve() !=
                OUT / "candidate/flange_link_stage_ffmpeg44/RESULT.json"
                or validate_artifact_ref(stage)):
            raise RuntimeError("FLANGE_STAGE_RESULT_SHA_DRIFT")
        videos = result.get("verified_videos", {})
        if set(videos) != set(VIDEO_FILES):
            raise RuntimeError("FLANGE_VIDEO_SET_DRIFT")
        for role, filename in VIDEO_FILES.items():
            video = OUT / "candidate/flange_link_stage_ffmpeg44/render" / filename
            record = videos[role]
            if (record.get("video") != artifact_ref(video)
                    or record.get("decoded_frames") != 171
                    or record.get("width") != flange.stage_v1.WIDTH
                    or record.get("height") != flange.stage_v1.HEIGHT
                    or record.get("encoded_duration_seconds") != 171 / 30):
                raise RuntimeError(f"FLANGE_VIDEO_REF_OR_TIME_DRIFT:{role}")
            if record != flange.stage_v1._verify_video(
                    video, 171, (flange.stage_v1.WIDTH, flange.stage_v1.HEIGHT), 30.0):
                raise RuntimeError(f"FLANGE_VIDEO_DECODE_DRIFT:{role}")
        for role, ref in spec["inputs"].items():
            if errors := validate_artifact_ref(ref):
                raise RuntimeError(f"FLANGE_INPUT_SHA_DRIFT:{role}:{';'.join(errors)}")
        return result, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"FLANGE_CANDIDATE_VALIDATION:{error}"]


def finish(packet: dict[str, Any], spec: dict[str, Any], code: int,
           log: Path, reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    candidate, errors = verify_candidate(spec)
    complete = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if complete else "FAILED_RUNTIME_FINAL"
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_FLANGE_TERMINAL_RESULT")
    result = {
        "schema_version": "exact78-robot-flange-link-cards042-result-v1",
        "task_id": TASK, "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason, "validation_errors": errors,
        "source_session": spec["session_id"], "source_frames_expected": 171,
        "flange_spec": artifact_ref(SPEC),
        "candidate_result": artifact_ref(OUT / "candidate/RESULT.json") if candidate else None,
        "worker_log": artifact_ref(log),
        "single_change": spec["single_change"],
        "numeric_gates_changed": False, "base_search_rule_changed": False,
        "renderer_camera_base_rule_changed": False,
        "old_robot_result_input": False, "old_video_ab_status": "EVALUATOR_ONLY_AFTER_FRESH_RESULT",
        "ai_review": "PENDING", "user_review": "PENDING",
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "source_mutated": False, "files_deleted": False,
        "next_action": "Review mount-chain audit and F0/middle/final full Robot videos; evaluate old direction separately",
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {
        "schema_version": "exact78-robot-flange-link-cards042-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("FLANGE_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for row in index["task_packets"]:
        if row["task_id"] == TASK:
            row["execution_allowed"] = False
    index.update(packet_revision="EXACT78_ROBOT_FLANGE_LINK_CARDS042_V1_TERMINAL",
                 plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"],
                 status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_ROBOT_FLANGE_LINK_CARDS042_V1_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX,
        task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    spec = frozen_spec()
    packet = packet_value(spec)
    ready, reason = readiness()
    if args.check_only:
        route_ready, route_reason = _candidate_route_readiness()
        print(json.dumps({
            "status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "route_ready": route_ready,
            "route_reason": route_reason,
            "task_id": TASK, "command": COMMAND,
            "spec_sha256": packet["inputs"]["flange_spec"]["sha256"],
            "run_signature": packet["run_signature"],
            "source_frames": 171, "gpu_count": 0, "model_calls": 0,
            "new_attempt_write_set": packet["write_set"][0],
        }, ensure_ascii=False))
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
