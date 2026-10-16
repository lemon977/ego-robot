"""One bounded 0915 Cards001 raw-VST frontend candidate; no registration on import.

Only ``--check-only`` is read-only.  The normal entry is intentionally fenced by
the current governance ledger, a single writer, and one shared GPU lease.
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


TASK = "exact78_raw_to_robot_t2_0915_cards001_frontend_v1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
SPEC = OUT / "FRONTEND_SPEC.json"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T2_0915_CARDS001_FRONTEND_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SESSION = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/001")
SESSION_ID = "play_cards_0915_001"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
BINDING = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/04_T2_FRONTEND_BINDING_ZH.md"
T0 = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
T0_CORRECTION = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001/RESULT.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
CAMPAIGN_END = datetime.fromisoformat("2026-10-01T18:47:25.116287+08:00")
PYTHON = Path("/usr/local/bin/python")
OPERATION = "run_exact78_fresh_frontend_v2"
COMMAND = (f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run {OPERATION} "
           f"--spec {SPEC} --phase all")
T0_SHA = {
    T0 / "INPUT_FREEZE.json": "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88",
    T0 / "FRAME_TIME_MAP.json": "7b68b5b1b64b35e096f4ef8ea16665e73428fede09a5d7b741f8cad8b94de70a",
    T0_CORRECTION: "7cd873022e3a72b05fa88701cccb76dec7fda5321c4eed11b2aa14f69b95314c",
}
SOURCE_NAMES = ("vst.h264", "camera_params.json", "camera_params.meta.json",
                "vst.qpc.ts.jsonl", "vst.ts.jsonl")
CODE_RELATIVE = (
    "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
    "src/chaoyang/ops/run_exact78_fresh_frontend_v1.py",
    "src/chaoyang/pipeline/vst_encoded_video_domain.py",
    "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    "src/chaoyang/governance/gpu_lease_v71.py",
    "vendor/FoundationStereo/scripts/pico_stereo_depth.py",
    "vendor/ProPainter/inference_propainter.py",
)
STAGES = ("MAP", "RAW", "MASK", "PREPARE", "CLEAN")


def exact_ref(ref: dict[str, Any], label: str) -> dict[str, Any]:
    actual = artifact_ref(ref["path"])
    if actual != ref:
        raise RuntimeError(f"T2_0915_INPUT_SHA_DRIFT:{label}")
    return actual


def frozen_inputs() -> dict[str, Any]:
    if datetime.now().astimezone() >= CAMPAIGN_END:
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    for path, sha in T0_SHA.items():
        if artifact_ref(path)["sha256"] != sha:
            raise RuntimeError(f"T0_SHA_DRIFT:{path}")
    frozen = load_json(T0 / "INPUT_FREEZE.json")
    row = frozen["selected"].get(SESSION_ID)
    if not row or row.get("source_path") != str(SESSION) or row.get("role") != "development":
        raise RuntimeError("CARDS001_T0_IDENTITY_DRIFT")
    files = {name: exact_ref(row["files"][name], name) for name in SOURCE_NAMES}
    if (Path(files["vst.h264"]["path"]) != SESSION / "raw/vst.h264"
            or Path(files["camera_params.json"]["path"]) != SESSION / "raw/camera_params.json"
            or files["vst.h264"]["sha256"] != "7e12cde47c01f4a2e97125a3d19a0d10434825f84499363764febd6c58ef4b68"
            or files["camera_params.json"]["sha256"] != "19c8e5dbe471c91e84214a047882c1abf3087b865c863d8a398c497236be3352"):
        raise RuntimeError("CARDS001_RAW_CAPTURE_IDENTITY_DRIFT")
    camera = load_json(Path(files["camera_params.json"]["path"]))
    if (camera.get("left", {}).get("sourceIndex"), camera.get("right", {}).get("sourceIndex")) != (1, 0):
        raise RuntimeError("CARDS001_PHYSICAL_EYE_DRIFT")
    timing = load_json(T0 / "FRAME_TIME_MAP.json")["sessions"][SESSION_ID]
    if (timing.get("decoded_frames") != 150 or timing.get("nominal_fps_from_capture_metadata") != 30
            or timing.get("time_mode") != "TIMEBASE_UNVERIFIED"
            or len(timing.get("frame_rows", [])) != 150):
        raise RuntimeError("CARDS001_TIMEBASE_FREEZE_DRIFT")
    models = frozen["models"]
    model_refs = {"sam_checkpoint": exact_ref(models["sam31_checkpoint"], "sam31_checkpoint"),
                  "propainter_weights": {
                      "raft-things.pth": exact_ref(models["raft_weights"], "raft_weights"),
                      "recurrent_flow_completion.pth": exact_ref(models["flow_weights"], "flow_weights"),
                      "ProPainter.pth": exact_ref(models["propainter_weights"], "propainter_weights")}}
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_OPERATION_ALREADY_PROMOTED")
    return {"source_files": files, "models": model_refs,
            "t0_input_freeze": artifact_ref(T0 / "INPUT_FREEZE.json"),
            "t0_frame_time_map": artifact_ref(T0 / "FRAME_TIME_MAP.json"),
            "t0_correction": artifact_ref(T0_CORRECTION),
            "plan": artifact_ref(PLAN), "binding": artifact_ref(BINDING),
            "route": artifact_ref(ROUTE),
            "propainter_entry": artifact_ref(REPO_ROOT / CODE_RELATIVE[-1])}


def spec_value(inputs: dict[str, Any]) -> dict[str, Any]:
    from chaoyang.ops.run_exact78_fresh_frontend_v2 import validate_spec
    spec = {"task_id": TASK, "session_id": SESSION_ID, "task": "poker",
            "source_family": "VST_0915", "source_root": str(SESSION),
            "source_video": inputs["source_files"]["vst.h264"],
            "camera_params": inputs["source_files"]["camera_params.json"],
            "source_index": 1, "source_eye_size": [2048, 1536],
            "frame_count": 150, "source_fps": 30,
            "image_domain": "VST_ENCODED_ALREADY_UNDISTORTED",
            "transform": "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY",
            "map_rule": "NO_MAP_VST_ENCODED", "map_generator_code": "ABSENT",
            "frame_timing": inputs["t0_frame_time_map"],
            "sam_checkpoint": inputs["models"]["sam_checkpoint"],
            "propainter_entry": inputs["propainter_entry"],
            "propainter_weights": inputs["models"]["propainter_weights"],
            "output_root": str(OUT / "candidate")}
    validate_spec(spec)
    return spec


def virtual_json_ref(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def packet_value(inputs: dict[str, Any], spec_ref: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    runtime = [artifact_ref(SCRIPT), *(artifact_ref(REPO_ROOT / name) for name in CODE_RELATIVE)]
    pinned = {REPO_ROOT / name for name in CODE_RELATIVE}
    for vendor in ("vendor/SAM3", "vendor/ProPainter"):
        runtime.extend(artifact_ref(path) for path in sorted((REPO_ROOT / vendor).rglob("*.py"))
                       if path.is_file() and not path.is_symlink() and path not in pinned)
    all_inputs = {**inputs, "frontend_spec": spec_ref}
    config = {"session_id": SESSION_ID, "frame_count": 150, "fps_nominal": 30,
              "time_mode": "TIMEBASE_UNVERIFIED", "qpc_head_sync": "TIMEBASE_UNVERIFIED",
              "physical_left_source_index": 1,
              "stages": list(STAGES), "gpu_lease_count": 1, "training_steps": 0,
              "visual_quality": "PENDING_REVIEW", "geometry": "NOT_IN_T2",
              "sleeve_cable": "MASK_SLEEVE_CABLE_UNVERIFIED"}
    signature = hashlib.sha256(canonical_bytes({"inputs": all_inputs,
        "runtime_code": runtime, "config": config, "command": COMMAND,
        "schema": "exact78-raw-t2-0915-cards001-frontend-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One full 0915 Cards001 raw VST physical-left→fresh Mask/Clean candidate",
        "phase": "EXACT78_RAW_TO_ROBOT_T2_0915_CARDS001_FRONTEND",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_RAW_T2_0915_CARDS001_FRONTEND_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=2), CAMPAIGN_END).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     str((T0 / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
                     str((T0 / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
                     str(T0_CORRECTION.relative_to(REPO_ROOT)), str(BINDING.relative_to(REPO_ROOT)),
                     str(SPEC.relative_to(REPO_ROOT)), str(SCRIPT.relative_to(REPO_ROOT))],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "ONE_GPU_LEASE",
                          "CAPTURE_RAW_ONLY", "PHYSICAL_LEFT_SOURCE_INDEX_1", "TIMEBASE_UNVERIFIED",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["FRONTEND_SPEC.json", *(f"candidate/{name}.json" for name in STAGES),
                             *(f"candidate/SEAL_{name}.json" for name in STAGES),
                             "candidate/propainter_model/frames/masked_in.mp4",
                             "candidate/propainter_model/frames/inpaint_out.mp4",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": 7200, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER", "GPU_LEASE_UNAVAILABLE",
                            "STAGE_SEAL_FAIL", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "0915 Cards001 fresh scene candidate only; timebase, sleeve/cable, Mask/Clean quality, Robot and training unverified",
        "writer": writer, "inputs": all_inputs, "runtime_code": runtime,
        "config": config, "run_signature": signature, "commands": [COMMAND],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_T2_0915_PACKET:" + ";".join(errors))
    return packet


def readiness() -> tuple[bool, str]:
    state = load_json(TASK_STATE_PATH)
    if any(row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in state["tasks"]):
        return False, "ACTIVE_TASK_CONFLICT"
    if state.get("next_task") or any(row.get("execution_allowed") for row in load_json(INDEX)["task_packets"]):
        return False, "ACTIVE_ROUTE_CONFLICT"
    from chaoyang.cli import _maintained_operations
    if OPERATION not in _maintained_operations():
        return False, "T2_OPERATION_NOT_MAINTAINED"
    return True, "READY_FOR_REGISTRATION"


def register(packet: dict[str, Any], spec: dict[str, Any]) -> None:
    ready, reason = readiness()
    if not ready:
        raise RuntimeError(reason)
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        raise RuntimeError("FRESH_T2_0915_TASK_REQUIRED")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    OUT.mkdir(parents=True)
    atomic_json(SPEC, spec)
    if artifact_ref(SPEC) != packet["inputs"]["frontend_spec"]:
        raise RuntimeError("FRONTEND_SPEC_PRECOMPUTED_SHA_DRIFT")
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": SESSION_ID, "updated_at": now, "heartbeat_at": now,
        "pid": packet["writer"]["pid"], "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": 0, "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": SESSION_ID,
        "prerequisites": packet["prerequisites"], "expected_resource": "ONE_GPU_TWO_CPU_THREADS",
        "stop_condition": "one full Cards001 fresh scene terminal result or campaign deadline"}
    errors = validate_task_state(state)
    if errors:
        raise RuntimeError("T2_0915_TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_RAW_T2_0915_CARDS001_FRONTEND_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T2_0915_CARDS001_FRONTEND_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def verify_fence(packet: dict[str, Any]) -> None:
    identity = process_identity(os.getpid())
    writer = packet["writer"]
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("T2_0915_WRITER_FENCE_DRIFT")
    active = [row for row in load_json(TASK_STATE_PATH)["tasks"]
              if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("T2_0915_DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    route = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed")]
    if len(route) != 1 or route[0]["task_id"] != TASK or route[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]:
        raise RuntimeError("T2_0915_PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("T2_0915_CODE_SHA_DRIFT:" + ";".join(errors))


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T2_0915_CARDS001_FRONTEND_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (OUT / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-raw-t2-0915-resume-v1", "elapsed_seconds": round(elapsed, 1),
            "sealed_stages": [name for name in STAGES if (OUT / "candidate" / f"SEAL_{name}.json").is_file()],
            "next_action": "Wait for the fenced single-lease worker or inspect its terminal receipt"})


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
    if remaining < 300:
        raise RuntimeError("T2_0915_REMAINING_BUDGET_TOO_SHORT")
    wait_seconds = min(600, max(0, remaining // 10))
    wall_seconds = min(5400, remaining - wait_seconds - 60)
    command = [str(PYTHON), "-B", "-m", "chaoyang.cli", "run",
        "run_gpu_command_with_v71_lease", "--task-id", TASK,
        "--attempt-id", "attempt_0001_0915_cards001_frontend", "--priority", "CANARY",
        "--gpu-id", "0", "--min-free-mib", "60000", "--wait-seconds", str(wait_seconds),
        "--wall-seconds", str(wall_seconds), "--receipt", str(OUT / "GPU_RECEIPT.json"),
        "--claim-limit", "One 0915 Cards001 fresh scene candidate; no timebase, quality or training authority",
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


def verify_internal_video_records(output: Path) -> None:
    from chaoyang.ops.run_exact78_fresh_frontend_v2 import verify_internal_mp4
    videos = load_json(output / "CLEAN.json")["internal_videos"]
    expected_names = ("masked_in.mp4", "inpaint_out.mp4")
    if set(videos) != set(expected_names):
        raise RuntimeError("INTERNAL_VIDEO_SET")
    for name in expected_names:
        path = output / "propainter_model/frames" / name
        if videos[name] != verify_internal_mp4(path, 150, 30):
            raise RuntimeError(f"INTERNAL_VIDEO_REF_DRIFT:{name}")


def finish(packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    verify_fence(packet)
    stage_refs: dict[str, dict[str, Any]] = {}
    stage_errors: list[str] = []
    previous_ref: dict[str, Any] | str = "ABSENT"
    for name in STAGES:
        path = OUT / "candidate" / f"SEAL_{name}.json"
        if path.is_file():
            seal_ref = artifact_ref(path)
            seal = load_json(path)
            result_ref = seal.get("result")
            if (seal.get("session_id") != SESSION_ID or seal.get("stage") != name
                    or seal.get("frontend_spec") != packet["inputs"]["frontend_spec"]
                    or seal.get("upstream_seal") != previous_ref
                    or seal.get("frame_count") != (0 if name == "MAP" else 150)
                    or not isinstance(result_ref, dict)
                    or validate_artifact_ref(result_ref)):
                stage_errors.append(f"STAGE_SEAL_DRIFT:{name}")
            stage_refs[name] = seal_ref
            previous_ref = seal_ref
        else:
            stage_errors.append(f"STAGE_SEAL_MISSING:{name}")
    if "CLEAN" in stage_refs:
        try:
            verify_internal_video_records(OUT / "candidate")
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            stage_errors.append(f"INTERNAL_VIDEO_VALIDATION:{error}")
    if "RAW" in stage_refs:
        raw = load_json(OUT / "candidate/RAW.json")
        if (raw.get("time_mode") != "TIMEBASE_UNVERIFIED"
                or raw.get("source_family") != "VST_0915"
                or raw.get("source_video") != packet["inputs"]["source_files"]["vst.h264"]):
            stage_errors.append("RAW_VST_TIME_OR_IDENTITY_DRIFT")
    if "MASK" in stage_refs:
        mask = load_json(OUT / "candidate/MASK.json")
        if mask.get("sleeve_cable_status") != "MASK_SLEEVE_CABLE_UNVERIFIED":
            stage_errors.append("SLEEVE_CABLE_CLAIM_DRIFT")
    input_errors = []
    for name, ref in packet["inputs"]["source_files"].items():
        if errors := validate_artifact_ref(ref):
            input_errors.append(f"{name}:{';'.join(errors)}")
    if errors := validate_artifact_ref(packet["inputs"]["models"]["sam_checkpoint"]):
        input_errors.append("sam_checkpoint:" + ";".join(errors))
    for name, ref in packet["inputs"]["models"]["propainter_weights"].items():
        if errors := validate_artifact_ref(ref):
            input_errors.append(f"{name}:{';'.join(errors)}")
    gpu_path = OUT / "GPU_RECEIPT.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    complete = (code == 0 and gpu.get("status") == "PASSED"
                and not stage_errors and not input_errors and reason is None)
    status = "PASSED" if complete else (
        "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL")
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_T2_0915_TERMINAL_RESULT")
    result = {"schema_version": "exact78-raw-t2-0915-cards001-frontend-result-v1",
        "task_id": TASK, "status": status, "execution_complete": complete,
        "worker_returncode": code, "reason": reason, "stage_seal_errors": stage_errors,
        "post_input_sha_errors": input_errors, "source_session": SESSION_ID,
        "source_frames_expected": 150, "time_mode": "TIMEBASE_UNVERIFIED",
        "qpc_head_sync": "TIMEBASE_UNVERIFIED",
        "physical_left_source_index": 1, "frontend_spec": artifact_ref(SPEC),
        "stage_seals": stage_refs, "gpu_receipt": artifact_ref(gpu_path) if gpu_path.is_file() else None,
        "worker_log": artifact_ref(log), "mask_quality": "NOT_YET_ADOPTED",
        "sleeve_cable_status": "MASK_SLEEVE_CABLE_UNVERIFIED",
        "clean_quality": "NOT_YET_ADOPTED", "visual_qualified": False,
        "product_qualified": False, "training_eligible": False, "user_review": "PENDING",
        "source_mutated": False, "files_deleted": False,
        "campaign_deadline_at": CAMPAIGN_END.isoformat(),
        "next_action": "Review fresh sleeve/cable/object masks and synthetic Clean; prove QPC/head clock before synchronization",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-raw-t2-0915-cards001-frontend-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="EXECUTION_ONLY_REVIEW_PENDING" if complete else "RUNTIME_OR_RESOURCE_HOLD",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    errors = validate_task_state(state)
    if errors:
        raise RuntimeError("T2_0915_TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T2_0915_CARDS001_FRONTEND_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_RAW_T2_0915_CARDS001_FRONTEND_{status}",
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
    ready, reason = readiness()
    spec_ref = virtual_json_ref(SPEC, spec)
    packet = packet_value(inputs, spec_ref)
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_READY" if ready else "DRY_RUN_NOT_ROUTABLE",
            "reason": reason, "task_id": TASK, "command": COMMAND,
            "spec_sha256": spec_ref["sha256"], "run_signature": packet["run_signature"],
            "source_frames": 150, "physical_left_source_index": 1,
            "time_mode": "TIMEBASE_UNVERIFIED", "gpu_lease_count": 1}, ensure_ascii=False))
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
    terminal_result = finish(packet, code, log, failure)
    print(json.dumps(terminal_result, ensure_ascii=False), flush=True)
    return 0 if terminal_result["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
