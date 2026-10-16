"""One fenced CPU-only Robot retry from T1-B's immutable HaWoR stage.

The failed T1-B Robot partial is deliberately outside this task's read set.
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

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, canonical_bytes, load_json, now_iso,
    process_identity, publish_bundle, validate_artifact_ref, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_raw_to_robot_t1b_robot_retry1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T1B_ROBOT_RETRY1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
SESSION = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_042")
RAW = SESSION / "CameraRecord_play_cards_0902_042.mp4"
PRIOR = REPO_ROOT / "_run/current/exact78_raw_to_robot_t1b_cards042_v1_20260930/attempts/attempt_0001"
HAWOR = PRIOR / "candidate/hawor"
SEAL = PRIOR / "candidate/HAWOR_STAGE_SEAL.json"
NATIVE_AUDIT = PRIOR / "candidate/NATIVE_OPEN_AUDIT.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
T0 = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
CAMPAIGN_END = "2026-10-01T18:47:25.116287+08:00"
EXPECTED = {
    PARENT: "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1",
    T0 / "INPUT_FREEZE.json": "704f9df95705b1939b91525021c9b93b5fcb69ea6be41e8ab68249cef70ccb88",
    T0 / "FRAME_TIME_MAP.json": "7b68b5b1b64b35e096f4ef8ea16665e73428fede09a5d7b741f8cad8b94de70a",
    PRIOR / "RESULT.json": "1197b19d1c0af4f2855eb4e9cca557e45431ab11bf4ef24653ac6c911ff6054f",
    PRIOR / "candidate/RESULT.json": "9c89e55bdf2510a6c3e5f3f138cf78c7aa3ea4a8553b4d02b10c755730476da1",
    HAWOR / "RESULT.json": "85769382755be826a89ddbdb5585691a563f8252de86511593b8c96c3bc98477",
    HAWOR / "HAWOR_CAMERA_PREDICTIONS.npz": "00a417086f9223f0268155f99861d173f7bc229f09ad163b9dae39f2e82fed30",
    HAWOR / "FRAME_PROVENANCE.json": "dfabcb22902000354ec39422b6cdade372582461f6a157aa2d15a7cbf39c6a65",
    HAWOR / "INPUT_AUDIT.json": "7e63aceaf62214bca59ce620cf97f7e2893791eb03b99ddca5317a23f2ccc6ff",
    SEAL: "51d264c334b8fa5ea5299428b52f541e8ee438575c28918672d5dff18c64e18a",
    NATIVE_AUDIT: "cefb3e768a9fb4b12960bb6697bfff0b994ecdaf2a845ff327b221ef0a29f56c",
    RAW: "879cd8f2ec91cd7aecbdff2de83d950d5cad0b6b11bc2fb97c5054df33ca4928",
}
RUNTIME_RELATIVE = (
    "src/chaoyang/ops/run_exact78_robot_from_sealed_hawor_v1.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py",
    "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    "src/chaoyang/pipeline/exact78_motion_state_v1.py",
    "src/chaoyang/pipeline/robot_renderer_eevee_fullchain.py",
    "src/chaoyang/pipeline/robot_renderer_cycles.py",
    "src/chaoyang/pipeline/robot_scene_state_cpu.py",
    "src/chaoyang/pipeline/robot_wrist_kai_adapter.py",
    "src/chaoyang/pipeline/robot_mount_visual_proxy.py",
    "src/chaoyang/pipeline/robot_tool_definition_contract.py",
    "src/chaoyang/governance/exact78_cold_start_guard_v1.py",
    "src/chaoyang/governance/exact78_operation_gate.py",
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/cli/__init__.py",
)
COMMAND = (f"PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run "
           f"run_exact78_robot_from_sealed_hawor_v1 --session {SESSION} "
           f"--sealed-hawor-root {HAWOR} --output-root {OUT / 'candidate/robot'}")


def checked_inputs() -> dict[str, Any]:
    if datetime.now().astimezone() >= datetime.fromisoformat(CAMPAIGN_END):
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    for path, sha in EXPECTED.items():
        if artifact_ref(path)["sha256"] != sha:
            raise RuntimeError(f"FROZEN_SHA_DRIFT:{path}")
    if load_json(PRIOR / "RESULT.json")["status"] != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("PRIOR_NOT_TERMINAL_FAILURE")
    if load_json(SEAL).get("downstream_use") != "SEALED_STAGE_ONLY":
        raise RuntimeError("HAWOR_NOT_SEALED")
    stage = load_json(HAWOR / "RESULT.json")
    if stage.get("model_input_frames") != 171 or stage.get("source_frames_decoded") != 171:
        raise RuntimeError("HAWOR_FRAME_COUNT_DRIFT")
    if load_json(HAWOR / "INPUT_AUDIT.json")["raw_source_ref"] != artifact_ref(RAW):
        raise RuntimeError("RAW_LINEAGE_DRIFT")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_OPERATION_ALREADY_PROMOTED")
    prior_packet = load_json(REPO_ROOT / "tasks/current/exact78_raw_to_robot_t1b_cards042_v1_20260930/TASK_PACKET.json")
    return {
        "plan": artifact_ref(PLAN), "route": artifact_ref(ROUTE),
        "visual_parent": artifact_ref(PARENT),
        "t0_input_freeze": artifact_ref(T0 / "INPUT_FREEZE.json"),
        "t0_frame_time_map": artifact_ref(T0 / "FRAME_TIME_MAP.json"),
        "prior_failed_result": artifact_ref(PRIOR / "RESULT.json"),
        "prior_child_result": artifact_ref(PRIOR / "candidate/RESULT.json"),
        "sealed_hawor": {str(path.relative_to(HAWOR)): artifact_ref(path) for path in
            (HAWOR / "RESULT.json", HAWOR / "HAWOR_CAMERA_PREDICTIONS.npz",
             HAWOR / "FRAME_PROVENANCE.json", HAWOR / "INPUT_AUDIT.json")},
        "hawor_seal": artifact_ref(SEAL), "native_open_audit": artifact_ref(NATIVE_AUDIT),
        "raw_video": artifact_ref(RAW),
        "robot_assets": prior_packet["inputs"]["assets"][5:],
        "submission_input_sha256": prior_packet["inputs"]["submission_input_sha256"],
        "source_session": SESSION.name,
        "forbidden_prior_partial": str(PRIOR / "candidate/robot"),
    }


def packet_value(inputs: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now().astimezone()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    runtime = [artifact_ref(SCRIPT), *(artifact_ref(REPO_ROOT / rel) for rel in RUNTIME_RELATIVE)]
    config = {"session_id": SESSION.name, "cpu_threads_total": 2, "gpu_count": 0,
              "model_calls": 0, "ffmpeg_compat": "vsync_0_for_ffmpeg_4_4",
              "clean": "NOT_IN_RETRY", "geometry": "NOT_IN_RETRY", "training_steps": 0}
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": runtime,
        "config": config, "command": COMMAND, "schema": "exact78-t1b-robot-retry1-v1"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Retry Cards042 Robot camera diagnostic from sealed fresh HaWoR after FFmpeg 4.4 encoder failure",
        "phase": "EXACT78_RAW_TO_ROBOT_T1B_ROBOT_RETRY1",
        "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_RAW_T1B_ROBOT_RETRY1_V1",
        "execution_mode": "EXACT78_CANDIDATE", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=4), datetime.fromisoformat(CAMPAIGN_END)).isoformat(),
        "campaign_deadline_at": CAMPAIGN_END,
        "read_set": [str(PLAN.relative_to(REPO_ROOT)),
                     str((PRIOR / "RESULT.json").relative_to(REPO_ROOT)),
                     str(SEAL.relative_to(REPO_ROOT)),
                     *[str((HAWOR / name).relative_to(REPO_ROOT)) for name in
                       ("RESULT.json", "HAWOR_CAMERA_PREDICTIONS.npz",
                        "FRAME_PROVENANCE.json", "INPUT_AUDIT.json")],
                     str(RAW)],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "SEALED_HAWOR_STAGE",
                          "EXTERNAL_DATA_READ_ONLY", "0916_READ_ONLY", "NO_DELETION"],
        "required_outputs": ["candidate/robot/RESULT.json",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/ROBOT_FRAME_STATES.json",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/ego_robot_diagnostic.mp4",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/fixed_external_full_robot.mp4",
                             "candidate/robot/v1_stage_with_ffmpeg44_encoder/render/raw_motion_source_review.mp4",
                             "RESULT.json", "terminal_receipt"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 14400, "session_attempts_max": 1},
        "stop_conditions": ["SOURCE_SHA_DRIFT", "CODE_OR_ASSET_SHA_DRIFT", "DOUBLE_WRITER",
                            "FORBIDDEN_PRIOR_PARTIAL_READ", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "One Cards042 camera-coordinate Robot diagnostic from sealed RGB HaWoR; no Clean/occlusion/world pose/visual/product/training PASS or production promotion",
        "writer": writer, "inputs": inputs, "runtime_code": runtime, "config": config,
        "run_signature": signature, "commands": [COMMAND],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    return packet


def index_copy() -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def verify_fence(packet: dict[str, Any]) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if (writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]
            or load_json(OUT / "WRITER.json") != writer):
        raise RuntimeError("WRITER_FENCE_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("DOUBLE_WRITER_OR_SIGNATURE_DRIFT")
    allowed = [row for row in load_json(INDEX)["task_packets"] if row.get("execution_allowed") is True]
    if len(allowed) != 1 or allowed[0]["task_id"] != TASK or allowed[0]["packet_sha256"] != artifact_ref(PACKET)["sha256"]:
        raise RuntimeError("PACKET_ROUTING_CONFLICT")
    for ref in packet["runtime_code"]:
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("RUNTIME_CODE_SHA_DRIFT:" + ";".join(errors))
    for ref in (*packet["inputs"]["sealed_hawor"].values(), packet["inputs"]["hawor_seal"],
                packet["inputs"]["native_open_audit"], packet["inputs"]["raw_video"],
                *packet["inputs"]["robot_assets"]):
        if errors := validate_artifact_ref(ref):
            raise RuntimeError("INPUT_SHA_DRIFT:" + ";".join(errors))


def register(packet: dict[str, Any]) -> None:
    if any(path.exists() for path in (OUT, PACKET, RECEIPT)):
        raise RuntimeError("FRESH_RETRY_TASK_REQUIRED")
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    OUT.mkdir(parents=True)
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    stamp = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": SESSION.name, "updated_at": stamp, "heartbeat_at": stamp,
        "pid": packet["writer"]["pid"], "proc_start_ticks": packet["writer"]["proc_start_ticks"],
        "gpu_id": None, "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    state["next_task"] = {"task_id": TASK, "session": SESSION.name,
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "one Robot-only terminal result or deadline"}
    if errors := validate_task_state(state):
        raise RuntimeError("TASK_STATE_PREFLIGHT:" + ";".join(errors))
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_RAW_T1B_ROBOT_RETRY1_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T1B_ROBOT_RETRY1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)


def heartbeat(packet: dict[str, Any], elapsed: float) -> None:
    verify_fence(packet)
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
    publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_RAW_T1B_ROBOT_RETRY1_HEARTBEAT",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT)
    number = int(elapsed // 600)
    if number and not (OUT / f"RESUME_SUMMARY_{number:04d}.json").exists():
        atomic_json(OUT / f"RESUME_SUMMARY_{number:04d}.json", {
            "schema_version": "exact78-robot-retry-resume-v1", "elapsed_seconds": round(elapsed, 1),
            "robot_result_sealed": (OUT / "candidate/robot/RESULT.json").is_file(),
            "next_action": "Wait for fenced Robot-only worker, then inspect decode/frame status"})


def run_worker(packet: dict[str, Any]) -> tuple[int, Path]:
    verify_fence(packet)
    (OUT / "candidate").mkdir()
    cache = OUT / "cache"
    cache.mkdir()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", TMPDIR=str(cache), TMP=str(cache), TEMP=str(cache),
        XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / "matplotlib"),
        NUMBA_CACHE_DIR=str(cache / "numba"), OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    command = ["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "run",
        "run_exact78_robot_from_sealed_hawor_v1", "--session", str(SESSION),
        "--sealed-hawor-root", str(HAWOR), "--output-root", str(OUT / "candidate/robot")]
    log = OUT / "worker.log"
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        worker = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        last_beat = started
        while worker.poll() is None:
            if datetime.now().astimezone() >= datetime.fromisoformat(packet["deadline_at"]):
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


def finish(packet: dict[str, Any], code: int, log: Path, reason: str | None) -> dict[str, Any]:
    verify_fence(packet)
    child = OUT / "candidate/robot/RESULT.json"
    child_ref = artifact_ref(child) if child.is_file() else None
    value = load_json(child) if child_ref else {}
    videos = value.get("verified_videos") or {}
    complete = (code == 0 and value.get("status") == "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
                and value.get("frame_count") == 171 and len(videos) == 3
                and all(item.get("decoded_frames") == 171 for item in videos.values())
                and reason is None)
    status = "PASSED" if complete else "FAILED_RUNTIME_FINAL"
    if (OUT / "RESULT.json").exists() or RECEIPT.exists():
        raise RuntimeError("DUPLICATE_TERMINAL_RESULT")
    result = {"schema_version": "exact78-raw-t1b-robot-retry1-result-v1", "task_id": TASK,
        "status": status, "execution_complete": complete, "worker_returncode": code,
        "reason": reason, "child_result": child_ref, "worker_log": artifact_ref(log),
        "source_session": SESSION.name, "source_frames_expected": 171,
        "hawor_stage_reused_without_model": artifact_ref(SEAL), "model_calls": 0,
        "rendered_frames": value.get("frame_count") if complete else 0,
        "videos": videos if complete else {}, "visual_qualified": False,
        "product_qualified": False, "training_eligible": False,
        "user_review": "PENDING", "general_operation": None,
        "source_mutated": False, "files_deleted": False, "robot100_modified": False,
        "campaign_deadline_at": CAMPAIGN_END,
        "next_action": "Inspect frames/arm reachability and Clean/occlusion separately; no promotion",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "RESULT.json", result)
    atomic_json(RECEIPT, {"schema_version": "exact78-raw-t1b-robot-retry1-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(OUT / "RESULT.json"),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(),
               heartbeat_at=now_iso(), result=artifact_ref(OUT / "RESULT.json"),
               terminal_receipt=artifact_ref(RECEIPT),
               quality="CAMERA_DIAGNOSTIC_ONLY" if complete else "RUNTIME_FAILED",
               adoption="NOT_ADOPTED", review="MACHINE_AUDITED_USER_PENDING")
    state["next_task"] = None
    if errors := validate_task_state(state):
        raise RuntimeError("TERMINAL_STATE_PREFLIGHT:" + ";".join(errors))
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T1B_ROBOT_RETRY1_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_RAW_T1B_ROBOT_RETRY1_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=SCRIPT, task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(OUT / "RESULT.json")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    inputs = checked_inputs()
    packet = packet_value(inputs)
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_PASS", "task_id": TASK,
            "run_signature": packet["run_signature"], "deadline_at": packet["deadline_at"]}, ensure_ascii=False))
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
