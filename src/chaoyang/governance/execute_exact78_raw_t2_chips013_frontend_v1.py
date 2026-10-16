"""Finite 0901 Chips013 fresh visual frontend candidate; --check-only is read-only.

The production RGB source is the same-capture SBS file.  The exported mono
clip is pinned solely for source/physical-eye identity, never model input.
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


TASK = "exact78_raw_to_robot_t2_chips013_frontend_v1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
SPEC = OUT / "FRONTEND_SPEC.json"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T2_CHIPS013_FRONTEND_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SESSION = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0901/potato_chips/get_potato_chips_0901_013")
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
BINDING = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/04_T2_FRONTEND_BINDING_ZH.md"
OPTICAL_AUDIT = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/05_T0_TIMEBASE_OPTICAL_AUDIT_ZH.md"
T0 = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
T0_CORRECTION = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001/RESULT.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
PYTHON = Path("/usr/local/bin/python")
OPERATION = "run_exact78_fresh_frontend_v2"
COMMAND = (f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} "
           f"--spec {SPEC} --phase all")
STAGES = ("MAP", "RAW", "MASK", "PREPARE", "CLEAN")
EXPECTED_SHA = {
    "stereo": "9c88384f9c86377c71651f6a81e614a4ea15b6f0d2704de72d4b8f8ac256829b",
    "video": "1d886dcc845d945c8169fd41c70312ec4070bf95c25e50948652e82db54e55e2",
    "camera": "baff3edf9856fa0566c2d6c216745b9a334fafc4403f83504839e805e5412a50",
    "clip_manifest": "63bbe0406ba57a86c2dd7937fbd8a1bb3e7b564b78d808a44ebb8f7b52bd8710",
}
T0_SHA = {
    T0 / "INPUT_FREEZE.json": "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88",
    T0 / "FRAME_TIME_MAP.json": "7b68b5b1b64b35e096f4ef8ea16665e73428fede09a5d7b741f8cad8b94de70a",
    T0_CORRECTION: "7cd873022e3a72b05fa88701cccb76dec7fda5321c4eed11b2aa14f69b95314c",
}
CODE_RELATIVE = (
    "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
    "src/chaoyang/ops/run_exact78_fresh_frontend_v1.py",
    "src/chaoyang/pipeline/vst_encoded_video_domain.py",
    "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/governance/gpu_lease_v71.py",
    "vendor/FoundationStereo/scripts/pico_stereo_depth.py",
    "vendor/ProPainter/inference_propainter.py",
)


def _exact_ref(ref: dict[str, Any], label: str) -> dict[str, Any]:
    if artifact_ref(ref["path"]) != ref:
        raise RuntimeError(f"T2_INPUT_SHA_DRIFT:{label}")
    return ref


def frozen_inputs() -> dict[str, Any]:
    if datetime.now().astimezone() >= CAMPAIGN_END:
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    for path, sha in T0_SHA.items():
        if artifact_ref(path)["sha256"] != sha:
            raise RuntimeError(f"T0_SHA_DRIFT:{path}")
    frozen = load_json(T0 / "INPUT_FREEZE.json")
    row = frozen["selected"].get(SESSION.name)
    if not row or row.get("source_path") != str(SESSION) or row.get("role") != "development":
        raise RuntimeError("CHIPS013_T0_IDENTITY_DRIFT")
    files = {key: _exact_ref(row["files"][key], key)
             for key in ("stereo", "video", "camera", "clip_manifest")}
    for key, sha in EXPECTED_SHA.items():
        if files[key]["sha256"] != sha:
            raise RuntimeError(f"CHIPS013_SOURCE_IDENTITY_DRIFT:{key}")
    manifest = load_json(Path(files["clip_manifest"]["path"]))
    stereo = manifest.get("source_stereo_video", {})
    mono = manifest.get("video", {})
    if (manifest.get("video_rectification", {}).get("eye") != "left"
            or stereo.get("frame_count") != 870 or stereo.get("fps") != 25.0
            or mono.get("frame_count") != 870 or mono.get("fps") != 25.0):
        raise RuntimeError("CHIPS013_CLIP_CONTRACT_DRIFT")
    camera = load_json(Path(files["camera"]["path"]))
    if ("extrinsics" in camera or camera.get("reference_resolution") != {"w": 2048, "h": 1536}
            or camera.get("left", {}).get("distortion", {}).get("model") != "equiDis62"):
        raise RuntimeError("CHIPS013_CAMERA_DOMAIN_DRIFT")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_OPERATION_ALREADY_PROMOTED")
    models = frozen["models"]
    model_refs = {
        "sam_checkpoint": _exact_ref(models["sam31_checkpoint"], "sam31_checkpoint"),
        "propainter_weights": {
            "raft-things.pth": _exact_ref(models["raft_weights"], "raft_weights"),
            "recurrent_flow_completion.pth": _exact_ref(models["flow_weights"], "flow_weights"),
            "ProPainter.pth": _exact_ref(models["propainter_weights"], "propainter_weights"),
        },
    }
    return {"source_files": files, "models": model_refs,
            "t0_input_freeze": artifact_ref(T0 / "INPUT_FREEZE.json"),
            "t0_frame_time_map": artifact_ref(T0 / "FRAME_TIME_MAP.json"),
            "t0_correction": artifact_ref(T0_CORRECTION),
            "plan": artifact_ref(PLAN), "binding": artifact_ref(BINDING),
            "optical_audit": artifact_ref(OPTICAL_AUDIT), "route": artifact_ref(ROUTE),
            "map_generator_code": artifact_ref(REPO_ROOT / CODE_RELATIVE[-2]),
            "propainter_entry": artifact_ref(REPO_ROOT / CODE_RELATIVE[-1])}


def spec_value(inputs: dict[str, Any]) -> dict[str, Any]:
    spec = {
        "task_id": TASK, "session_id": SESSION.name, "task": "chips",
        "source_family": "TRACKER_0901", "source_root": str(SESSION),
        "source_video": inputs["source_files"]["stereo"],
        "camera_params": inputs["source_files"]["camera"],
        "source_index": 0, "source_eye_size": [2048, 1536],
        "frame_count": 870, "source_fps": 25,
        "image_domain": "TRACKER_RECTIFIED_PRIMARY",
        "transform": "SAME_SESSION_EQUIDIS62_TO_PINHOLE",
        "map_rule": "GENERATE_FROM_CAPTURE_CAMERA_PARAMS",
        "map_generator_code": inputs["map_generator_code"],
        "frame_timing": inputs["t0_frame_time_map"],
        "sam_checkpoint": inputs["models"]["sam_checkpoint"],
        "propainter_entry": inputs["propainter_entry"],
        "propainter_weights": inputs["models"]["propainter_weights"],
        "output_root": str(OUT / "candidate"),
    }
    from chaoyang.ops.run_exact78_fresh_frontend_v2 import validate_spec
    validate_spec(spec)
    return spec


def virtual_json_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def packet_value(inputs: dict[str, Any], spec_ref: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    runtime = [artifact_ref(SCRIPT), *(artifact_ref(REPO_ROOT / path) for path in CODE_RELATIVE)]
    known = {REPO_ROOT / path for path in CODE_RELATIVE}
    for vendor in ("vendor/SAM3", "vendor/ProPainter"):
        runtime.extend(artifact_ref(path) for path in sorted((REPO_ROOT / vendor).rglob("*.py"))
                       if path.is_file() and not path.is_symlink() and path not in known)
    all_inputs = {**inputs, "frontend_spec": spec_ref}
    config = {"session_id": SESSION.name, "frame_count": 870, "fps_nominal": 25,
              "stages": list(STAGES), "gpu_lease_count": 1, "training_steps": 0,
              "visual_quality": "PENDING_REVIEW", "geometry": "CALIBRATION_EXTRINSICS_ABSENT",
              "mono_clip_role": "IDENTITY_ONLY_NOT_MODEL_INPUT",
              "eye_identity": "SOURCE_INDEX_0_FIRST_FRAME_COMPARISON_ONLY",
              "frame_timing_binding": "MONO_INDEX_PROJECTED_TO_SBS_UNVERIFIED",
              "time_mode": "TIMEBASE_UNVERIFIED"}
    signature = hashlib.sha256(canonical_bytes({"inputs": all_inputs,
        "runtime_code": runtime, "config": config, "command": COMMAND,
        "schema": "exact78-raw-t2-chips013-frontend-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One full native-25FPS Chips013 raw SBS→fresh MAP/Mask/Clean visual candidate",
        "phase": "EXACT78_RAW_TO_ROBOT_T2_CHIPS013_FRONTEND",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_RAW_T2_CHIPS013_FRONTEND_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=4), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     str((T0 / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
                     str((T0 / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
                     str(T0_CORRECTION.relative_to(REPO_ROOT)),
                     str(BINDING.relative_to(REPO_ROOT)),
                     str(OPTICAL_AUDIT.relative_to(REPO_ROOT)),
                     str(SPEC.relative_to(REPO_ROOT))],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json",
                      "docs/governance", "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "ONE_GPU_LEASE",
                          "SOURCE_SHA_PINNED", "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["FRONTEND_SPEC.json", "candidate/CAMERA_MAPS.npz",
                             *(f"candidate/{stage}.json" for stage in STAGES),
                             *(f"candidate/SEAL_{stage}.json" for stage in STAGES),
                             "candidate/propainter_model/frames/masked_in.mp4",
                             "candidate/propainter_model/frames/inpaint_out.mp4",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": 14400, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "GPU_LEASE_UNAVAILABLE",
                            "STAGE_SEAL_FAIL", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "Chips013 fresh visual scene candidate only; no metric geometry, Mask/Clean quality, Robot, product, or training authority",
        "writer": writer, "inputs": all_inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_T2_PACKET:" + ";".join(errors))
    return packet


def readiness() -> tuple[bool, str]:
    state = load_json(TASK_STATE_PATH)
    if any(row["status"] in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"} for row in state["tasks"]):
        return False, "ACTIVE_TASK_CONFLICT"
    if state.get("next_task") or any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        return False, "ACTIVE_ROUTE_CONFLICT"
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        return False, "FRESH_T2_TASK_REQUIRED"
    from chaoyang.cli import _maintained_operations
    if OPERATION not in _maintained_operations():
        return False, "T2_CANDIDATE_OPERATION_NOT_MAINTAINED"
    return True, "READY_FOR_REGISTRATION"


def register(packet: dict[str, Any], spec: dict[str, Any]) -> None:
    ready, reason = readiness()
    if not ready:
        raise RuntimeError(reason)
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), _index_copy()
    OUT.mkdir(parents=True)
    atomic_json(SPEC, spec)
    if artifact_ref(SPEC) != packet["inputs"]["frontend_spec"]:
        raise RuntimeError("FRONTEND_SPEC_PRECOMPUTED_SHA_DRIFT")
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": SESSION.name, "updated_at": now, "heartbeat_at": now,
        "pid": packet["writer"]["pid"], "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": 0, "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": SESSION.name,
        "prerequisites": packet["prerequisites"], "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "one full 870-frame Chips013 scene terminal result or campaign deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("T2_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_RAW_T2_CHIPS013_FRONTEND_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T2_CHIPS013_FRONTEND_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("T2_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("T2_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    routes = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if len(routes) != 1 or routes[0]["task_id"] != TASK or routes[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]:
        raise RuntimeError("T2_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("T2_CODE_SHA_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(row for row in state["tasks"] if row["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T2_CHIPS013_FRONTEND_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (OUT / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-raw-t2-resume-v1", "elapsed_seconds": round(elapsed, 1),
            "sealed_stages": [stage for stage in STAGES if (OUT / "candidate" / f"SEAL_{stage}.json").is_file()],
            "next_action": "Wait for fenced worker or inspect terminal result"})


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
    remaining = int((datetime.fromisoformat(packet["deadline_at"]) - datetime.now().astimezone()).total_seconds())
    if remaining < 600:
        raise RuntimeError("T2_REMAINING_BUDGET_TOO_SHORT")
    wait_seconds = min(600, max(0, remaining // 10))
    wall_seconds = min(12600, remaining - wait_seconds - 60)
    command = [str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
        "run_gpu_command_with_v71_lease", "--task-id", TASK,
        "--attempt-id", "attempt_0001_chips013_frontend", "--priority", "CANARY",
        "--gpu-id", "0", "--min-free-mib", "60000", "--wait-seconds", str(wait_seconds),
        "--wall-seconds", str(wall_seconds), "--receipt", str(OUT / "GPU_RECEIPT.json"),
        "--claim-limit", "One Chips013 visual scene candidate; no metric or training authority",
        "--", str(PYTHON), "-B", "-m", "chaoyang.cli", "run", OPERATION,
        "--spec", str(SPEC), "--phase", "all"]
    log = OUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        last_beat = started
        while process.poll() is None:
            if time.monotonic() - last_beat >= 50:
                heartbeat(packet, time.monotonic() - started)
                last_beat = time.monotonic()
            time.sleep(2)
    return process.returncode, log


def finish(packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    stage_refs: dict[str, Any] = {}
    stage_errors = []
    previous: dict[str, Any] | str = "ABSENT"
    for stage in STAGES:
        path = OUT / "candidate" / f"SEAL_{stage}.json"
        if not path.is_file():
            stage_errors.append(f"STAGE_SEAL_MISSING:{stage}")
            continue
        seal = load_json(path)
        if (seal.get("session_id") != SESSION.name or seal.get("stage") != stage
                or seal.get("frontend_spec") != packet["inputs"]["frontend_spec"]
                or seal.get("upstream_seal") != previous
                or seal.get("frame_count") != (0 if stage == "MAP" else 870)
                or not isinstance(seal.get("result"), dict)
                or validate_artifact_ref(seal["result"])):
            stage_errors.append(f"STAGE_SEAL_DRIFT:{stage}")
        previous = artifact_ref(path)
        stage_refs[stage] = previous
    if "CLEAN" in stage_refs:
        try:
            from chaoyang.ops.run_exact78_fresh_frontend_v2 import verify_internal_mp4
            videos = load_json(OUT / "candidate/CLEAN.json")["internal_videos"]
            for name in ("masked_in.mp4", "inpaint_out.mp4"):
                path = OUT / "candidate/propainter_model/frames" / name
                if videos[name] != verify_internal_mp4(path, 870, 25):
                    raise RuntimeError(f"INTERNAL_VIDEO_REF_DRIFT:{name}")
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            stage_errors.append(f"INTERNAL_VIDEO_VALIDATION:{error}")
    input_errors = []
    for key, ref in packet["inputs"]["source_files"].items():
        if errors := validate_artifact_ref(ref):
            input_errors.append(f"{key}:{';'.join(errors)}")
    models = packet["inputs"]["models"]
    for key, ref in {"sam_checkpoint": models["sam_checkpoint"],
                     **models["propainter_weights"]}.items():
        if errors := validate_artifact_ref(ref):
            input_errors.append(f"{key}:{';'.join(errors)}")
    gpu_path = OUT / "GPU_RECEIPT.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    complete = (code == 0 and gpu.get("status") == "PASSED" and not stage_errors
                and not input_errors and reason is None)
    status = "PASSED" if complete else (
        "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL")
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_T2_TERMINAL_RESULT")
    result = {"schema_version": "exact78-raw-t2-chips013-frontend-result-v1",
        "task_id": TASK, "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason,
        "stage_seal_errors": stage_errors, "post_input_sha_errors": input_errors,
        "source_session": SESSION.name, "source_frames_expected": 870, "source_fps": 25,
        "production_rgb_source": packet["inputs"]["source_files"]["stereo"],
        "mono_identity_only": packet["inputs"]["source_files"]["video"],
        "eye_identity": "SOURCE_INDEX_0_FIRST_FRAME_COMPARISON_ONLY",
        "frame_timing_binding": "MONO_INDEX_PROJECTED_TO_SBS_UNVERIFIED",
        "frontend_spec": artifact_ref(SPEC), "stage_seals": stage_refs,
        "gpu_receipt": artifact_ref(gpu_path) if gpu_path.is_file() else None,
        "worker_log": artifact_ref(log), "mask_quality": "NOT_YET_ADOPTED",
        "clean_quality": "NOT_YET_ADOPTED", "visual_qualified": False,
        "metric_geometry": "BLOCKED_CALIBRATION_ABSENT", "time_mode": "TIMEBASE_UNVERIFIED",
        "product_qualified": False, "training_eligible": False,
        "user_review": "PENDING", "source_mutated": False, "files_deleted": False,
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "next_action": "Review all 870 fresh device/object masks and synthetic Clean; metric geometry needs same-session extrinsics",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-raw-t2-chips013-frontend-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(row for row in state["tasks"] if row["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_OR_RESOURCE_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("T2_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = _index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T2_CHIPS013_FRONTEND_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_RAW_T2_CHIPS013_FRONTEND_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    inputs = frozen_inputs()
    spec = spec_value(inputs)
    packet = packet_value(inputs, virtual_json_ref(SPEC, spec))
    ready, reason = readiness()
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "task_id": TASK, "command": COMMAND,
            "spec_sha256": packet["inputs"]["frontend_spec"]["sha256"],
            "run_signature": packet["run_signature"], "source_frames": 870,
            "source_fps": 25, "production_rgb": "SAME_SESSION_ORIGINAL_SBS",
            "mono_clip_role": "IDENTITY_ONLY", "metric_geometry": "BLOCKED_CALIBRATION_ABSENT",
            "gpu_lease_count": 1}, ensure_ascii=False))
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
    terminal = finish(packet, code, log, failure)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
