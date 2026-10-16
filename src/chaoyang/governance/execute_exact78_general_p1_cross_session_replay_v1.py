#!/usr/bin/env python3
"""Register, execute and terminalize the three-session P1 CPU replay."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    canonical_bytes,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_general_p1_cross_session_replay_v1_20260930"
EXECUTION_REVISION = "EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
ROUTE_ZH = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_ZH.md"
REVIEW = REPO_ROOT / "review/exact78_p1_cross_session_v1"
SOURCE_TASK = "exact78_general_p1_action_canary_v1_retry8_20260930"
SOURCE_ROOT = REPO_ROOT / "_run/current" / SOURCE_TASK / "attempts/attempt_0001"
SELECTED = (
    "play_cards_0902_042",
    "get_potato_chips_0902_103",
    "get_potato_chips_0901_013",
)
ROBOT100_GUARDS = [
    REPO_ROOT / "docs/current/ROBOT100_BATCH_ZH.md",
    REPO_ROOT / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    REPO_ROOT / "src/chaoyang/ops/run_robot100_world_first_v10.py",
    REPO_ROOT / "src/chaoyang/ops/run_robot100_constraint_review_v16_retry1.py",
]


def index_copy() -> dict:
    return {
        key: copy.deepcopy(value)
        for key, value in load_json(INDEX).items()
        if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}
    }


def build_manifest() -> Path:
    source_manifest = load_json(SOURCE_ROOT / "CANARY_MANIFEST.json")
    by_id = {row["session_id"]: row for row in source_manifest["sessions"]}
    rows = []
    for session_id in SELECTED:
        source_row = by_id[session_id]
        session = SOURCE_ROOT / "sessions" / session_id
        hawor = session / "hawor_recovery/HAWOR_RECOVERED_VISUAL_MANO21.npz"
        hawor_result = session / "hawor_recovery/RESULT.json"
        hand_states = session / "hand/HAND_STATES.npz"
        row = {
            "session_id": session_id,
            "task": source_row["task"],
            "fps": 25.0 if "0901_013" in session_id else 30.0,
            "source_video": artifact_ref(Path(source_row["source_video"]["path"])),
            "hawor": artifact_ref(hawor),
            "hawor_result": artifact_ref(hawor_result),
            "hand_states": artifact_ref(hand_states),
            "reuse_policy": "same-session SHA-pinned retry8 state; fresh base/Arm/diagnostics only",
        }
        rows.append(row)
    path = ROOT / "REPLAY_MANIFEST.json"
    atomic_json(
        path,
        {
            "schema_version": "exact78-general-p1-cross-session-replay-manifest-v1",
            "task_id": TASK,
            "source_task": artifact_ref(SOURCE_ROOT / "RESULT.json"),
            "selection": ["Cards042", "Chips103", "native_25fps_Chips013"],
            "sessions": rows,
        },
    )
    return path


def publish_shallow(result: dict) -> dict:
    if REVIEW.exists():
        raise RuntimeError("FRESH_REVIEW_DIRECTORY_REQUIRED")
    REVIEW.mkdir(parents=True)
    outputs = {}
    for session in result.get("sessions", []):
        if session.get("status") != "TERMINAL":
            continue
        session_id = session["session_id"]
        outputs[session_id] = {}
        for key in ("raw_robot_overlay", "independent_whole_robot", "storyboard_12"):
            source = Path(session["outputs"][key]["path"])
            suffix = source.suffix
            destination = REVIEW / f"{session_id}_{key}{suffix}"
            shutil.copyfile(source, destination)
            outputs[session_id][key] = artifact_ref(destination)
    atomic_json(
        REVIEW / "INDEX.json",
        {
            "schema_version": "exact78-p1-cross-session-shallow-review-v1",
            "task_id": TASK,
            "status": "VISUAL_REVIEW_PENDING",
            "outputs": outputs,
            "claim_limit": "Shallow copies for review only; source task RESULT controls metrics and adoption.",
        },
    )
    return {"index": artifact_ref(REVIEW / "INDEX.json"), "outputs": outputs}


def update_route(result_ref: dict, result: dict, shallow: dict) -> None:
    route = load_json(ROUTE)
    route["general_raw_to_robot"]["status"] = "P1_INTEGRATION_REPLAY_COMPLETE_VISUAL_REVIEW_PENDING"
    route["general_raw_to_robot"]["current_runnable_operation"] = None
    route["general_raw_to_robot"]["latest_candidate"] = {
        "task_id": TASK,
        "result": result_ref,
        "four_metrics": result["four_metrics"],
        "shallow_review": shallow["index"],
        "fresh_hawor_executed": False,
        "fresh_base_arm_diagnostics": True,
        "visual_review": "PENDING",
        "adopted": False,
    }
    route["routing_rule"]["next_finite_task"] = (
        "Review the three full videos. If accepted, register one fresh-Raw canary using the same frozen runner; "
        "if rejected, bind the observed frame cluster and change one generic variable only."
    )
    atomic_json(ROUTE, route)
    text = ROUTE_ZH.read_text(encoding="utf-8").rstrip()
    text += f"""

## 2026-09-30 P1 三会话回放进度

任务 `{TASK}` 已完成集成回放；它复用同会话、SHA 固定的 HaWoR/Hand，只重新求解 base/Arm 并重新渲染，所以还不能声称 fresh Raw→Robot 已通过。

- 动作缺口比例：`{result['four_metrics']['motion_completeness']['fraction']}`
- 静态逐帧可达率：`{result['four_metrics']['cross_session_reachability']['fraction']}`
- 平滑轨迹跟随率：`{result['four_metrics']['smoothed_visual_follow']['fraction']}`
- 完成终态：`{result['four_metrics']['terminal_coverage']['terminal']}/3`
- 浅层审阅目录：`review/exact78_p1_cross_session_v1/`

当前仍是“执行完成、视觉审阅待定”，没有晋升通用 operation。
"""
    from chaoyang.governance.common import atomic_write
    atomic_write(ROUTE_ZH, text.encode("utf-8") + b"\n")


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if any(path.exists() for path in (ROOT, PACKET, RECEIPT, REVIEW)):
        raise RuntimeError("FRESH_TASK_REQUIRED")
    for path in (PLAN, ROUTE, ROUTE_ZH, SOURCE_ROOT / "RESULT.json"):
        path.resolve(strict=True)
    guards_before = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    ROOT.mkdir(parents=True)
    (ROOT / "cache").mkdir()
    manifest = build_manifest()
    master_operation = os.environ.get(
        "EXACT78_P1_CROSS_SESSION_OPERATION",
        "run_exact78_general_p1_cross_session_replay_v1",
    )
    arm_operation = os.environ.get(
        "EXACT78_P1_CROSS_SESSION_ARM_OPERATION",
        "run_exact78_session_base_arm_v8",
    )
    runtime_paths = [
        REPO_ROOT / f"src/chaoyang/ops/{master_operation}.py",
        REPO_ROOT / f"src/chaoyang/ops/{arm_operation}.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p1_cross_session_replay_v1.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v8.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_arm_causal_visual_v3.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_arm_causal_visual_v2.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v7.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v6.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v5.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v4.py",
        REPO_ROOT / "src/chaoyang/ops/run_exact78_session_base_arm_v1.py",
        REPO_ROOT / "src/chaoyang/ops/render_exact78_action_diagnostics_v2.py",
        REPO_ROOT / "src/chaoyang/ops/render_exact78_action_diagnostics_v1.py",
        REPO_ROOT / "src/chaoyang/pipeline/exact78_robot_raster_v1.py",
    ]
    writer = {
        "pid": os.getpid(),
        "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1,
        "fencing_token": secrets.token_hex(24),
        "scope": TASK,
    }
    inputs = {
        "execution_plan": artifact_ref(PLAN),
        "manifest": artifact_ref(manifest),
        "route_before": artifact_ref(ROUTE),
        "source_retry8": artifact_ref(SOURCE_ROOT / "RESULT.json"),
    }
    runtime_code = [artifact_ref(path) for path in runtime_paths]
    config = {
        "sessions": 3,
        "gpu_count": 0,
        "cpu_threads_total": 2,
        "wall_seconds_max": 14400,
        "position_gate_mm": 10.0,
        "rotation_gate_deg": 5.0,
        "velocity_rad_per_frame_max": 0.12,
        "acceleration_rad_per_frame2_max": 0.06,
        "models_changed": False,
        "numeric_thresholds_changed": False,
        "fresh_hawor": False,
    }
    signature = hashlib.sha256(
        canonical_bytes({"inputs": inputs, "runtime_code": runtime_code, "config": config})
    ).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK,
        "objective": "Replay the current corrected base/Arm/diagnostic chain on Cards042, Chips103 and native-25fps Chips013 with a separated metric contract",
        "phase": "EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY",
        "plan_revision": "EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
        "execution_revision": EXECUTION_REVISION,
        "deadline_at": (datetime.now().astimezone() + timedelta(hours=4)).isoformat(),
        "read_set": [
            "AGENTS.md",
            "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md",
            "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
        ],
        "write_set": [str(ROOT), str(PACKET), str(RECEIPT), str(REVIEW), str(ROUTE), str(ROUTE_ZH)],
        "prerequisites": [
            "GOVERNANCE_PASS",
            "NO_ACTIVE_TASK",
            "FROZEN_SAME_SESSION_HAWOR_HAND",
            "NO_MODEL_OR_THRESHOLD_CHANGE",
            "ROBOT100_READ_ONLY",
        ],
        "required_outputs": [
            "three_terminal_results",
            "raw_robot_overlay_per_session",
            "independent_robot_view_per_session",
            "four_metric_report",
            "problem_table",
            "shallow_review_index",
        ],
        "budgets": {
            "workers_max": 1,
            "cpu_threads_total": 2,
            "gpu_count": 0,
            "training_steps": 0,
            "wall_seconds_max": 14400,
            "session_attempts_max": 1,
            "quality_modification_rounds_max": 0,
        },
        "stop_conditions": [
            "source_sha_drift",
            "writer_conflict",
            "deadline",
            "Robot100_guard_drift",
            "one_attempt_terminal",
        ],
        "claim_limit": "Three-session CPU integration replay only; no fresh HaWoR, visual adoption, training, deletion, control, deployment or Robot100 authority.",
        "writer": writer,
        "inputs": inputs,
        "runtime_code": runtime_code,
        "config": config,
        "run_signature": signature,
        "commands": [f"chaoyang run {master_operation} --manifest {manifest}"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    atomic_json(ROOT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "Cards042_Chips103_Chips013", "updated_at": now, "heartbeat_at": now,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature,
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "PENDING",
    })
    state["next_task"] = {
        "task_id": TASK, "session": "Cards042_Chips103_Chips013",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_TWO_THREADS",
        "stop_condition": "three terminal sessions or four-hour deadline",
    }
    index["task_packets"].append({
        "task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
    })
    index.update(
        packet_revision=f"{EXECUTION_REVISION}_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX),
    )
    publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX, task_packet_index_value=index,
    )

    environment = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(ROOT / "cache"),
        "XDG_CACHE_HOME": str(ROOT / "cache"),
        "MPLCONFIGDIR": str(ROOT / "cache"),
        "NUMBA_CACHE_DIR": str(ROOT / "cache"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    command = [
        sys.executable, "-B", "-m", "chaoyang.cli", "run",
        master_operation, "--manifest", str(manifest),
    ]
    with (ROOT / "worker.log").open("x") as log:
        worker = subprocess.Popen(
            command, cwd=REPO_ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        started = time.monotonic()
        last_heartbeat = started
        while True:
            try:
                code = worker.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                current = copy.deepcopy(load_json(TASK_STATE_PATH))
                elapsed = time.monotonic() - started
                if ((current.get("next_task") or {}).get("task_id") != TASK
                        or elapsed > config["wall_seconds_max"]):
                    os.killpg(worker.pid, signal.SIGTERM)
                    code = worker.wait(timeout=30)
                    break
                if time.monotonic() - last_heartbeat >= 120:
                    row = next(item for item in current["tasks"] if item["task_id"] == TASK)
                    row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                    publish_bundle(
                        load_json(AUTHORITY_PATH), current,
                        event_type="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_HEARTBEAT",
                        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                        generator_path=Path(__file__), task_packet_index_path=INDEX,
                        task_packet_index_value=index_copy(),
                    )
                    last_heartbeat = time.monotonic()

    result_path = ROOT / "RESULT.json"
    if not result_path.is_file():
        atomic_json(result_path, {
            "status": "FAILED_RUNTIME_FINAL", "returncode": code,
            "execution_complete": False, "visual_qualified": False,
            "quality_passed": False, "training_eligible": False,
            "deletion_authorized": False,
        })
    result = load_json(result_path)
    shallow = publish_shallow(result)
    update_route(artifact_ref(result_path), result, shallow)
    guards_after = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    if guards_before != guards_after:
        raise RuntimeError("ROBOT100_GUARD_CHANGED")
    atomic_json(ROOT / "ROBOT100_GUARD.json", {"unchanged": True, "files": guards_after})
    terminal = "PASSED" if code == 0 and result.get("execution_complete") else "FAILED_RUNTIME_FINAL"
    atomic_json(RECEIPT, {
        "schema_version": "exact78-general-p1-cross-session-replay-receipt-v1",
        "task_id": TASK, "status": terminal, "result": artifact_ref(result_path),
        "shallow_review": shallow["index"], "route": artifact_ref(ROUTE),
        "claim_limit": packet["claim_limit"], "deletion_authorized": False,
    })
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(
        status=terminal, pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
        result=artifact_ref(result_path), terminal_receipt=artifact_ref(RECEIPT),
        quality="EXECUTION_COMPLETE_VISUAL_REVIEW_PENDING" if terminal == "PASSED" else terminal,
        adoption="NOT_ADOPTED", review="PENDING_HUMAN_FULL_VIDEO",
    )
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(
        packet_revision=f"{EXECUTION_REVISION}_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX),
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="EXACT78_GENERAL_P1_CROSS_SESSION_REPLAY_V1_TERMINAL",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__), task_packet_index_path=INDEX, task_packet_index_value=index,
    )
    validation = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True,
    )
    if validation.returncode:
        raise RuntimeError(validation.stdout[-4000:] + validation.stderr[-4000:])
    print(json.dumps({
        "status": terminal, "governance_revision": published["governance_revision"],
        "result": artifact_ref(result_path), "review": shallow["index"],
    }, ensure_ascii=False, indent=2))
    return 0 if terminal == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
