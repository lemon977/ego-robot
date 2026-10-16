"""Register and run one fenced, full Cards042 Raw->HaWoR->Robot canary.

T1-B supersedes the T1-A *candidate code route*, not its immutable evidence.
No production promotion, scene model, training or deletion occurs here.
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
import sys
import time
from typing import Any

from chaoyang.cli.exact78_submit import build_request
from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_raw_to_robot_t1b_cards042_v1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T1B_CARDS042_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SESSION = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_042")
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
T0 = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
T0_CORRECTION = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_correction_v1_20260930/attempts/attempt_0001/RESULT.json"
T1A = REPO_ROOT / "_run/current/exact78_raw_to_robot_t1a_candidate_build_v1_20260930/attempts/attempt_0001"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
CAMPAIGN_END = "2026-10-01T18:47:25.116287+08:00"
PARENT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"
T0_CORRECTION_SHA = "7cd873022e3a72b05fa88701cccb76dec7fda5321c4eed11b2aa14f69b95314c"
T1A_SHA = "264960825aa55f0e5e037419501e7973b5446911dba17004677267cee2f0ad98"
T0_FREEZE_SHA = "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88"
T0_MAP_SHA = "7b68b5b1b64b35e096f4ef8ea16665e73428fede09a5d7b741f8cad8b94de70a"
RUNTIME_RELATIVE = (
    "src/chaoyang/ops/run_exact78_raw_to_robot_v2.py",
    "src/chaoyang/ops/run_exact78_hawor_raw_v3.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/cli/__init__.py",
    "src/chaoyang/cli/exact78_submit.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/current_r3_contracts.py",
)
VENDOR_RELATIVE = (
    "vendor/HaWoR/scripts/scripts_test_video/detect_track_video.py",
    "vendor/HaWoR/scripts/scripts_test_video/hawor_video.py",
    "vendor/HaWoR/lib/pipeline/tools.py",
    "vendor/HaWoR/hawor/utils/process.py",
    "vendor/HaWoR/weights/hawor/model_config.yaml",
)
ASSET_RELATIVE = (
    "vendor/HaWoR/weights/hawor/checkpoints/hawor.ckpt",
    "vendor/HaWoR/weights/external/detector.pt",
    "vendor/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl",
    "vendor/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl",
    "_run/current/environments/hawor-py310-v1/.chaoyang-hardlink-snapshot/manifest.json.gz",
    "assets/robot/ROBOT_ASSET_PIN.json",
    "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json",
    "assets/robot/V2相机固定法兰PRO版本.STL",
)
PYTHON = Path("/usr/local/bin/python")
COMMAND = (f"PYTHONPATH=src {PYTHON} -B -m chaoyang.cli run run_exact78_raw_to_robot_v2 "
           f"--session {SESSION} --output-root {OUT / 'candidate'}")


def index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def checked_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if datetime.fromisoformat(CAMPAIGN_END) <= datetime.now().astimezone():
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    for path, expected in ((PARENT, PARENT_SHA), (T0_CORRECTION, T0_CORRECTION_SHA),
                           (T1A / "RESULT.json", T1A_SHA),
                           (T0 / "INPUT_FREEZE.json", T0_FREEZE_SHA),
                           (T0 / "FRAME_TIME_MAP.json", T0_MAP_SHA)):
        if artifact_ref(path)["sha256"] != expected:
            raise RuntimeError(f"FROZEN_UPSTREAM_SHA_DRIFT:{path}")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_OPERATION_ALREADY_PROMOTED")
    if load_json(T0_CORRECTION)["status"] != "PASSED" or load_json(T1A / "RESULT.json")["status"] != "PASSED":
        raise RuntimeError("T0_OR_T1A_NOT_SEALED")
    request = build_request(session=SESSION)
    if len(request["sources"]) != 1 or request["sources"][0]["session_id"] != SESSION.name:
        raise RuntimeError("CARDS042_IDENTITY_DRIFT")
    t0 = load_json(T0 / "INPUT_FREEZE.json")["selected"][SESSION.name]
    if t0["source_path"] != str(SESSION) or t0["role"] != "development":
        raise RuntimeError("T0_CARDS042_ROLE_DRIFT")
    inputs = {
        "plan": artifact_ref(PLAN), "t0_correction": artifact_ref(T0_CORRECTION),
        "t0_input_freeze": artifact_ref(T0 / "INPUT_FREEZE.json"),
        "t0_frame_time_map": artifact_ref(T0 / "FRAME_TIME_MAP.json"),
        "t1a_result": artifact_ref(T1A / "RESULT.json"),
        "t1a_code_freeze_superseded_by_t1b": artifact_ref(T1A / "CODE_FREEZE.json"),
        "visual_parent": artifact_ref(PARENT), "route": artifact_ref(ROUTE),
        "source_files": request["sources"][0]["files"],
        "submission_input_sha256": request["submission_input_sha256"],
        "assets": [artifact_ref(REPO_ROOT / item) for item in ASSET_RELATIVE],
        "vendor_code": [artifact_ref(path) for path in sorted((REPO_ROOT / "vendor/HaWoR").rglob("*.py"))
                        if path.is_file() and not path.is_symlink()],
        "vendor_config": [artifact_ref(REPO_ROOT / item) for item in VENDOR_RELATIVE],
    }
    return inputs, request


def packet_value(inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    runtime = [artifact_ref(SCRIPT), *(artifact_ref(REPO_ROOT / item) for item in RUNTIME_RELATIVE),
               *inputs["vendor_code"]]
    config = {"session_id": SESSION.name, "session_attempts_max": 1,
              "gpu_count": 1, "cpu_threads_total": 2, "model_calls": "one HaWoR full session",
              "clean": "NOT_IN_T1B", "geometry": "NOT_IN_T1B", "training_steps": 0}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": COMMAND, "schema": "exact78-raw-t1b-cards042-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Cold-start full Cards042 RGB->HaWoR camera predictions->complete Robot diagnostic",
        "phase": "EXACT78_RAW_TO_ROBOT_T1B_CARDS042", "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_RAW_T1B_CARDS042_V1", "execution_mode": "EXACT78_CANDIDATE",
        "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=6), datetime.fromisoformat(CAMPAIGN_END)).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END,
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)),
                     str((T0 / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
                     str((T0 / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
                     str(T0_CORRECTION.relative_to(REPO_ROOT)),
                     str((T1A / "RESULT.json").relative_to(REPO_ROOT)),
                     str(ROUTE.relative_to(REPO_ROOT)), str(SCRIPT.relative_to(REPO_ROOT))],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "T0_T1A_SEALED",
                          "ONE_GPU_LEASE", "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["candidate/RESULT.json",
                             "candidate/hawor/HAWOR_CAMERA_PREDICTIONS.npz",
                             "candidate/hawor/FRAME_PROVENANCE.json",
                             "candidate/HAWOR_STAGE_SEAL.json",
                             "candidate/robot/ROBOT_FRAME_STATES.json",
                             "candidate/robot/render/ego_robot_diagnostic.mp4",
                             "candidate/robot/render/fixed_external_full_robot.mp4",
                             "candidate/ROBOT_STAGE_SEAL.json", "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "training_steps": 0, "wall_seconds_max": 21600, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_SHA_DRIFT", "CODE_OR_ASSET_SHA_DRIFT", "DOUBLE_WRITER",
                            "FORBIDDEN_READ", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "One Cards042 camera-coordinate diagnostic; no visual/product/training PASS or general entry promotion",
        "writer": writer, "inputs": inputs, "runtime_code": runtime, "config": config,
        "run_signature": signature, "commands": [COMMAND],
        "supersedes_candidate_code_freeze": inputs["t1a_code_freeze_superseded_by_t1b"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    return packet


def _verify_fence(packet: dict[str, Any], *, verify_assets: bool = False) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if (len(active) != 1 or active[0]["task_id"] != TASK
            or active[0]["run_signature"] != packet["run_signature"]):
        raise RuntimeError("DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    index = load_json(INDEX)
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if (len(allowed) != 1 or allowed[0]["task_id"] != TASK
            or allowed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]):
        raise RuntimeError("PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        errors = validate_artifact_ref(ref)
        if errors:
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    if verify_assets:
        for ref in (*packet["inputs"]["assets"], *packet["inputs"]["source_files"].values(),
                    *packet["inputs"]["vendor_config"]):
            errors = validate_artifact_ref(ref)
            if errors:
                raise RuntimeError("MODEL_ASSET_SHA_DRIFT:" + ";".join(errors))


def register(packet: dict[str, Any]) -> None:
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        raise RuntimeError("FRESH_T1B_TASK_REQUIRED")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    OUT.mkdir(parents=True)
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
        "stop_condition": "one Cards042 terminal result or six-hour deadline"}
    errors = validate_task_state(state)
    if errors:
        raise RuntimeError("TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_RAW_T1B_CARDS042_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T1B_CARDS042_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    _verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T1B_CARDS042_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    if int(elapsed // 600) > len(list(OUT.glob("RESUME_SUMMARY_*.json"))):
        number = int(elapsed // 600)
        candidate = OUT / "candidate"
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-raw-t1b-resume-v1", "elapsed_seconds": round(elapsed, 1),
            "hawor_result_sealed": (candidate / "hawor/RESULT.json").is_file(),
            "robot_result_sealed": (candidate / "robot/RESULT.json").is_file(),
            "next_action": "Wait for fenced worker; inspect worker.log and per-stage results before downstream use"})


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    _verify_fence(packet, verify_assets=True)
    cache = OUT / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "huggingface"),
        MPLCONFIGDIR=str(cache / "matplotlib"), NUMBA_CACHE_DIR=str(cache / "numba"),
        TORCH_EXTENSIONS_DIR=str(cache / "torch-extensions"), YOLO_CONFIG_DIR=str(cache / "ultralytics"),
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="0")
    command = [str(PYTHON), "-B", "-m", "chaoyang.cli", "run", "run_exact78_raw_to_robot_v2",
               "--session", str(SESSION), "--output-root", str(OUT / "candidate")]
    log = OUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        last_beat = started
        while worker.poll() is None:
            if (datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"])
                    or datetime.now().astimezone() >= datetime.fromisoformat(CAMPAIGN_END)):
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait(timeout=20)
                return 124, log
            if time.monotonic() - last_beat >= 50:
                heartbeat(packet, time.monotonic() - started)
                last_beat = time.monotonic()
            time.sleep(2)
    return worker.returncode, log


def finish(packet: dict[str, Any], code: int, log: Path,
           reason: str | None = None) -> dict[str, Any]:
    _verify_fence(packet)
    asset_errors = [f"{ref['path']}:{';'.join(errors)}"
                    for ref in (*packet["inputs"]["assets"],
                                *packet["inputs"]["source_files"].values(),
                                *packet["inputs"]["vendor_config"])
                    if (errors := validate_artifact_ref(ref))]
    child = OUT / "candidate/RESULT.json"
    child_ref = artifact_ref(child) if child.is_file() else None
    child_value = load_json(child) if child_ref else {}
    passed = (code == 0 and child_value.get("status") == "CAMERA_DIAGNOSTIC_STAGES_COMPLETE"
              and not asset_errors and reason is None)
    status = "PASSED" if passed else "FAILED_RUNTIME_FINAL"
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_TERMINAL_RESULT")
    result = {"schema_version": "exact78-raw-t1b-cards042-result-v1", "task_id": TASK,
        "status": status, "execution_complete": passed, "worker_returncode": code,
        "reason": reason, "post_asset_sha_errors": asset_errors,
        "child_result": child_ref, "worker_log": artifact_ref(log),
        "source_session": SESSION.name, "source_frames_expected": 171,
        "raw_cold_start": "STAGE_LINEAGE_AUDIT_PENDING" if passed else "FAILED_OR_UNVERIFIED",
        "robot_video_complete": bool(child_value.get("robot_video_complete")),
        "stage_status": child_value.get("stage_status", {}),
        "visual_qualified": False, "product_qualified": False, "training_eligible": False,
        "user_review": "PENDING", "general_operation": None,
        "source_mutated": False, "files_deleted": False, "robot100_modified": False,
        "campaign_deadline_at": CAMPAIGN_END,
        "next_action": "Inspect Cards042 stage audit, decoded videos and per-frame failures before next source",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-raw-t1b-cards042-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="CAMERA_DIAGNOSTIC_ONLY" if passed else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    errors = validate_task_state(state)
    if errors:
        raise RuntimeError("TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T1B_CARDS042_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_RAW_T1B_CARDS042_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    inputs, request = checked_inputs()
    packet = packet_value(inputs)
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_PASS", "task_id": TASK,
            "submission_input_sha256": request["submission_input_sha256"],
            "run_signature": packet["run_signature"], "deadline_at": packet["deadline_at"]},
            ensure_ascii=False))
        return 0
    register(packet)
    try:
        code, log = run_worker(packet)
        reason = None
    except Exception as error:
        code = 1
        reason = f"{type(error).__name__}:{error}"
        log = OUT / "worker.log"
        if not log.is_file():
            log = OUT / "executor_error.log"
            with log.open("x", encoding="utf-8") as stream:
                stream.write(reason + "\n")
    terminal = finish(packet, code, log, reason)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)
    return 0 if terminal["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
