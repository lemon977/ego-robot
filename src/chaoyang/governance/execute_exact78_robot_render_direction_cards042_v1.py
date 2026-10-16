"""Unregistered, bounded CPU-only Cards042 render-direction candidate runner.

``--check-only`` reads and validates frozen inputs but never publishes, renders,
or writes.  ``--execute`` remains unroutable until a new maintained-operation
contract is published and every active task has terminated.
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
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy
from chaoyang.pipeline import exact78_robot_render_direction_v1 as render_direction


TASK = "exact78_robot_render_direction_cards042_v1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
SPEC = OUT / "RENDER_SPEC.json"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_ROBOT_RENDER_DIRECTION_CARDS042_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
AUDIT = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/13_CAMERA_BASE_RENDER_DIRECTION_AUDIT_ZH.md"
DESIGN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/14_RENDER_DIRECTION_ONLY_CANDIDATE_ZH.md"
STAGE_RESULT = (REPO_ROOT / "_run/current/exact78_raw_to_robot_t2_0902_cards042_v1_20260930"
                "/attempts/attempt_0001/candidate/robot/v1_stage_with_ffmpeg44_encoder/RESULT.json")
STAGE_SHA = "27260cd52fa84a60c0c30bbf30425c62da4262f1d0b23f424a6f890eb173261d"
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
PYTHON = Path("/usr/local/bin/python")
OPERATION = "run_exact78_robot_render_direction_v1"
CONTRACT_KEY = "exact78_robot_render_direction_cards042_candidate_v1_20260930"
COMMAND = (f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} --spec {SPEC}")
VIDEO_FILES = {
    "ego_robot": "ego_robot_diagnostic.mp4",
    "external_full_robot": "fixed_external_full_robot.mp4",
    "raw_motion_source": "raw_motion_source_review.mp4",
}


def frozen_spec() -> dict[str, Any]:
    if artifact_ref(STAGE_RESULT)["sha256"] != STAGE_SHA:
        raise RuntimeError("RENDER_DIRECTION_STAGE_RESULT_SHA_DRIFT")
    stage = load_json(STAGE_RESULT)
    spec = {
        "task_id": TASK, "session_id": "play_cards_0902_042",
        "source_kind": "TRACKER_CLIP_RGB",
        "inputs": {"stage_result": artifact_ref(STAGE_RESULT),
                   "states": stage["outputs"]["states"],
                   **{key: stage["inputs"][key] for key in (
                       "hawor_npz", "frame_provenance", "input_audit", "raw_video")}},
        "output_root": str(OUT / "candidate"),
    }
    preview = render_direction.dry_run(spec)
    if (preview["frame_count"] != 171 or preview["nominal_fps"] != 30.0
            or preview["attempt_write_set_first"] != str(OUT.relative_to(REPO_ROOT))):
        raise RuntimeError("RENDER_DIRECTION_PREVIEW_DOMAIN_DRIFT")
    return spec


def virtual_json_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def runtime_refs() -> list[dict[str, Any]]:
    files = [SCRIPT, REPO_ROOT / "src/chaoyang/ops/run_exact78_robot_render_direction_v1.py",
             REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
             *(REPO_ROOT / relative for relative in render_direction.PINNED_CODE_SHA),
             REPO_ROOT / "src/chaoyang/governance/common.py",
             REPO_ROOT / "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
             REPO_ROOT / "src/chaoyang/governance/exact78_operation_gate.py",
             REPO_ROOT / "src/chaoyang/governance/current_r3_contracts.py",
             REPO_ROOT / "src/chaoyang/cli/__init__.py",
             REPO_ROOT / "src/chaoyang/ops/immutable_artifact_io.py"]
    refs = [artifact_ref(path) for path in files]
    for relative, sha in render_direction.PINNED_CODE_SHA.items():
        if artifact_ref(REPO_ROOT / relative)["sha256"] != sha:
            raise RuntimeError(f"RENDER_DIRECTION_RUNTIME_CODE_DRIFT:{relative}")
    return refs


def packet_value(spec_ref: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {"render_spec": spec_ref, "fixed_stage_inputs": spec["inputs"],
              "robot_asset_pin": artifact_ref(REPO_ROOT / "assets/robot/ROBOT_ASSET_PIN.json"),
              "direction_audit": artifact_ref(AUDIT), "candidate_design": artifact_ref(DESIGN)}
    runtime = runtime_refs()
    config = {"session_id": spec["session_id"], "frames": 171, "fps": 30,
              "render_only": True, "changed_expression": "T_CAMERA_BASE_INSTEAD_OF_INVERSE",
              "ik_calls": 0, "model_calls": 0, "gpu_count": 0, "numeric_gates_changed": False,
              "observation_plan": {
                  "all_frames": "IN_FRAME_EGO_AND_EXTERNAL_ROBOT_PIXELS_AND_VISIBILITY",
                  "cropped_pixels": "PENDING_SEPARATE_UNCLIPPED_GEOMETRY_DIAGNOSTIC",
                  "projection_frames": [0, 85, 170],
                  "projection_comparison": "FIXED_Q_BASE_MOUNT_TO_HAWOR_PREDICTED_WRIST_WITH_VALIDITY",
                  "old_video_ab": "EVALUATOR_ONLY_AFTER_FRESH_RESULT",
                  "ai_review": "PENDING", "human_review": "PENDING",
              }}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": COMMAND,
        "schema": "exact78-render-direction-cards042-task-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One full 171-frame Cards042 Robot render-only camera-direction candidate",
        "phase": "EXACT78_ROBOT_RENDER_DIRECTION_ONLY",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_ROBOT_RENDER_DIRECTION_CARDS042_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=1), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(AUDIT.relative_to(REPO_ROOT)),
                     str(DESIGN.relative_to(REPO_ROOT)), str(STAGE_RESULT.relative_to(REPO_ROOT)),
                     str(SPEC.relative_to(REPO_ROOT)),
                     "src/chaoyang/ops/run_exact78_robot_render_direction_v1.py",
                     "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
                     "assets/robot/ROBOT_ASSET_PIN.json"],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_CARDS042_STATES",
                          "SOURCE_SHA_PINNED", "ROBOT_ASSETS_PINNED", "NO_GPU_OR_MODEL",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["RENDER_SPEC.json", "candidate/RESULT.json",
                             *(f"candidate/render/{name}" for name in VIDEO_FILES.values()),
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 3600, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE",
                            "VIDEO_DECODE_OR_TIME_FAIL", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Render-only Cards042 diagnostic from fixed sealed q/base/mount; no IK, flange, visual, geometry, product or training qualification",
        "writer": writer, "inputs": inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    if errors := _validate_packet(packet):
        raise RuntimeError("INVALID_RENDER_DIRECTION_PACKET:" + ";".join(errors))
    return packet


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def _route_valid(contract: dict[str, Any]) -> bool:
    route = contract.get("special_status_contracts", {}).get(CONTRACT_KEY, {})
    expected = [artifact_ref(REPO_ROOT / relative) for relative in (
        "src/chaoyang/ops/run_exact78_robot_render_direction_v1.py",
        "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
        *render_direction.PINNED_CODE_SHA)]
    return (route.get("candidate_operation") == OPERATION
            and route.get("task_packet_required") is True
            and route.get("current_runnable_operation") is None
            and route.get("output_schema") == "exact78-robot-render-direction-only-v1"
            and route.get("model_calls_authorized") == 0
            and route.get("training_steps_authorized") == 0
            and isinstance(route.get("code_closure"), list)
            and all(ref in route["code_closure"] for ref in expected))


def _candidate_route_readiness() -> tuple[bool, str]:
    """Stage builder route and packet in one CAS, or use an exact published route."""
    from chaoyang.cli import _maintained_operations
    from chaoyang.governance.common import ALGORITHM_CONTRACT_PATH, BASELINE_REGISTRY_PATH

    if OPERATION in _maintained_operations():
        if _route_valid(load_json(ALGORITHM_CONTRACT_PATH)):
            return True, "PUBLISHED_CANDIDATE_ROUTE"
        return False, "RENDER_DIRECTION_PUBLISHED_ROUTE_DRIFT"
    from chaoyang.governance.current_r3_contracts import build_algorithm_contract

    staged = build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                      load_json(AUTHORITY_PATH))
    if not _route_valid(staged):
        return False, "RENDER_DIRECTION_CANDIDATE_ROUTE_NOT_STAGED"
    return True, "READY_TO_PUBLISH_CANDIDATE_ROUTE_WITH_PACKET"


def readiness() -> tuple[bool, str]:
    if datetime.now().astimezone() >= CAMPAIGN_END:
        return False, "CAMPAIGN_DEADLINE_EXPIRED"
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        return False, "FRESH_RENDER_DIRECTION_TASK_REQUIRED"
    state = load_json(TASK_STATE_PATH)
    if any(row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in state["tasks"]):
        return False, "ACTIVE_TASK_CONFLICT"
    if state.get("next_task") or any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
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
        raise RuntimeError("RENDER_DIRECTION_ROUTE_CHANGED_BEFORE_CAS")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RENDER_DIRECTION_CODE_CHANGED_BEFORE_CAS:" + ";".join(errors))
    for ref in (*spec["inputs"].values(), packet["inputs"]["robot_asset_pin"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RENDER_DIRECTION_INPUT_CHANGED_BEFORE_CAS:" + ";".join(errors))
    OUT.mkdir(parents=True)
    atomic_json(SPEC, spec)
    if artifact_ref(SPEC) != packet["inputs"]["render_spec"]:
        raise RuntimeError("RENDER_DIRECTION_SPEC_PRECOMPUTED_SHA_DRIFT")
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
        "stop_condition": "one full render-only result or one-hour deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("RENDER_DIRECTION_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_ROBOT_RENDER_DIRECTION_CARDS042_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_ROBOT_RENDER_DIRECTION_CARDS042_V1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("RENDER_DIRECTION_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("RENDER_DIRECTION_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    routes = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if len(routes) != 1 or routes[0]["task_id"] != TASK or routes[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]:
        raise RuntimeError("RENDER_DIRECTION_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RENDER_DIRECTION_RUNTIME_CODE_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_ROBOT_RENDER_DIRECTION_CARDS042_V1_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (OUT / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-render-direction-resume-v1",
            "elapsed_seconds": round(elapsed, 1), "candidate_result_exists": (OUT / "candidate/RESULT.json").is_file(),
            "next_action": "Wait for fenced CPU worker or inspect its terminal receipt"})


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
        raise RuntimeError("RENDER_DIRECTION_BUDGET_TOO_SHORT")
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
                    raise RuntimeError("RENDER_DIRECTION_DEADLINE")
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
        return None, ["CANDIDATE_RESULT_MISSING"]
    try:
        result = load_json(path)
        if (result.get("status") != "COMPLETE_RENDER_DIAGNOSTIC_NOT_VISUAL_PASS"
                or result.get("session_id") != spec["session_id"]
                or result.get("frame_count") != 171
                or result.get("inputs") != spec["inputs"]
                or len(result.get("frame_rows", [])) != 171
                or result.get("q_base_mount_recomputed") is not False
                or result.get("numeric_gates_changed") is not False
                or result.get("ego_camera_rule") != "T_CAMERA_BASE_BASE_TO_CAMERA_NO_INVERSION"
                or result.get("external_camera_rule") != "PINNED_FIXED_EXTERNAL_FROM_BASE_TO_CAMERA"
                or result.get("clean_or_depth_consumed") is not False
                or result.get("visual_qualified") is not False
                or result.get("training_eligible") is not False):
            raise RuntimeError("CANDIDATE_IDENTITY_OR_SCOPE_DRIFT")
        for index, row in enumerate(result["frame_rows"]):
            if (row.get("frame") != index or row.get("raw_decoded_display_index") != index
                    or not isinstance(row.get("ego_robot_pixels"), int)
                    or not isinstance(row.get("external_robot_pixels"), int)
                    or row["ego_robot_pixels"] < 0 or row["external_robot_pixels"] < 0
                    or not isinstance(row.get("final_visible"), bool)):
                raise RuntimeError(f"CANDIDATE_FRAME_OBSERVATION_DRIFT:{index}")
        videos = result["videos"]
        if set(videos) != set(VIDEO_FILES):
            raise RuntimeError("CANDIDATE_VIDEO_SET_DRIFT")
        for key, name in VIDEO_FILES.items():
            video_path = OUT / "candidate/render" / name
            if videos[key] != legacy._verify_video(video_path, 171, (legacy.WIDTH, legacy.HEIGHT), 30.0):
                raise RuntimeError(f"CANDIDATE_VIDEO_REF_OR_TIME_DRIFT:{key}")
        for key, ref in spec["inputs"].items():
            if errors := validate_artifact_ref(ref):
                raise RuntimeError(f"CANDIDATE_INPUT_SHA_DRIFT:{key}:{';'.join(errors)}")
        return result, []
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        return None, [f"CANDIDATE_VALIDATION:{error}"]


def finish(packet: dict[str, Any], spec: dict[str, Any], code: int,
           log: Path, reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    candidate, errors = verify_candidate(spec)
    complete = code == 0 and candidate is not None and not errors and reason is None
    status = "PASSED" if complete else "FAILED_RUNTIME_FINAL"
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_RENDER_DIRECTION_TERMINAL_RESULT")
    zero_visible = ([row["frame"] for row in candidate["frame_rows"]
                     if not row["final_visible"]] if candidate else [])
    result = {"schema_version": "exact78-robot-render-direction-cards042-result-v1",
        "task_id": TASK, "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason, "validation_errors": errors,
        "source_session": spec["session_id"], "source_frames_expected": 171,
        "render_spec": artifact_ref(SPEC),
        "candidate_result": artifact_ref(OUT / "candidate/RESULT.json") if candidate else None,
        "worker_log": artifact_ref(log), "q_base_mount_recomputed": False,
        "flange_repaired": False, "numeric_gates_changed": False,
        "zero_visible_frame_indices": zero_visible,
        "per_frame_robot_pixel_counts": "SEE_CANDIDATE_RESULT_FRAME_ROWS",
        "cropped_pixel_count_status": "NOT_MEASURED_REQUIRES_SEPARATE_UNCLIPPED_GEOMETRY_DIAGNOSTIC",
        "projection_review_frames": [0, 85, 170],
        "projection_review_status": "PENDING_FIXED_Q_VS_HAWOR_WRIST_EVALUATOR",
        "old_video_ab_status": "NOT_READ_EVALUATOR_ONLY_AFTER_FRESH_RESULT",
        "ai_review": "PENDING",
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "user_review": "PENDING",
        "source_mutated": False, "files_deleted": False,
        "next_action": "Review F0/middle/final and full-length ego/external videos against sealed q and HaWoR, then compare flange lane separately",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-robot-render-direction-cards042-receipt-v1",
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
        raise RuntimeError("RENDER_DIRECTION_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_ROBOT_RENDER_DIRECTION_CARDS042_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_ROBOT_RENDER_DIRECTION_CARDS042_V1_{status}",
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
