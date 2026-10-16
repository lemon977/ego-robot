"""Register, execute and terminalize Exact78 general baseline P0."""
from __future__ import annotations

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

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_general_p0_preflight_v1_20260929"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md"
COHORT = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
          / "20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json")
PARENT_RESULT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
V2_ARM = (REPO_ROOT / "_run/current/exact78_cards042_visual_baseline_v2_publish_20260929"
          / "attempts/attempt_0001/dependencies/selected_motion_bundle/inputs/ARM.npz")
HISTORICAL_ROBOT = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
                    / "20260908_two_task_e2e_baseline_v1/robot_mount_proxy_baseline_v1/task_action_canary_v1"
                    / "play_cards_0902_042_same_side_world_temporal48_v3/WORLD_FIRST_SAME_SIDE_48FRAME_STATES.npz")
HISTORICAL_HAWOR = (REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
                    / "20260908_two_task_e2e_baseline_v1/hawor_fresh_bounded_v2_v1"
                    / "play_cards_0902_042/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz")
RETRY4 = REPO_ROOT / "tasks/current/exact78_general_raw_to_robot_v1_retry4_20260929/TASK_PACKET.json"
ROBOT100_GUARDS = [
    REPO_ROOT / "docs/current/ROBOT100_BATCH_ZH.md",
    REPO_ROOT / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    REPO_ROOT / "src/chaoyang/ops/run_robot100_world_first_v10.py",
]


def index_copy() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    revision = load_json(RECEIPT_PATH)["governance_revision"]
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")
    for path in (PLAN, COHORT, PARENT_RESULT, V2_ARM, HISTORICAL_ROBOT, HISTORICAL_HAWOR, RETRY4):
        path.resolve(strict=True)
    root_parent = ROOT.parent.resolve()
    root_parent.relative_to((REPO_ROOT / "_run/current").resolve())
    ROOT.mkdir(parents=True)
    cache = ROOT / "cache"
    cache.mkdir()
    runner = REPO_ROOT / "src/chaoyang/ops/run_exact78_general_p0_preflight_v1.py"
    robot100_before = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    atomic_json(ROOT / "ROBOT100_GUARD_BEFORE.json", robot100_before)
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {
        "execution_plan": artifact_ref(PLAN), "frozen_cohort": artifact_ref(COHORT),
        "parent_robot_result": artifact_ref(PARENT_RESULT), "v2_retained_arm": artifact_ref(V2_ARM),
        "historical_accepted_robot": artifact_ref(HISTORICAL_ROBOT),
        "historical_accepted_hawor": artifact_ref(HISTORICAL_HAWOR),
        "retry4_packet": artifact_ref(RETRY4),
    }
    runtime_code = [artifact_ref(runner)]
    signature = hashlib.sha256(canonical_bytes({
        "inputs": inputs, "runtime_code": runtime_code,
        "config": {"workers": 2, "cpu_threads_total": 2, "gpu_count": 0, "wall_seconds_max": 3600,
                   "external_data": "READ_ONLY", "total_frames_expected": 56663},
        "schema": "exact78-general-p0-preflight-v1",
    })).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "CPU preflight the frozen 156 Exact78 inputs, runtime dependency closure and Cards042 template leakage before P1",
        "phase": "EXACT78_GENERAL_P0_PREFLIGHT", "plan_revision": "EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
        "execution_revision": "EXACT78_GENERAL_P0_PREFLIGHT_V1",
        "deadline_at": (datetime.now().astimezone() + timedelta(hours=1)).isoformat(),
        "read_set": ["AGENTS.md", "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md",
                     "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json"],
        "write_set": [str(ROOT)],
        "prerequisites": ["NO_ACTIVE_WRITER", "FROZEN_COHORT_SHA", "PARENT_ROBOT_RESULT_SHA", "EXTERNAL_DATA_READ_ONLY"],
        "required_outputs": ["156_session_level_ceiling", "runtime_dependency_sha_audit", "template_leakage_audit",
                             "P2_P3_runtime_estimate", "four_metric_report"],
        "budgets": {"workers_max": 2, "cpu_threads_total": 2, "gpu_count": 0, "training_steps": 0,
                    "wall_seconds_max": 3600, "session_attempts_max": 1},
        "stop_conditions": ["source_sha_drift", "writer_conflict", "deadline", "external_write_attempt"],
        "claim_limit": "CPU-only P0 input and dependency preflight; no model inference, rendering, visual quality adoption, deletion, training or Robot100 authority.",
        "writer": writer, "inputs": inputs, "runtime_code": runtime_code, "run_signature": signature,
        "commands": [f"chaoyang run run_exact78_general_p0_preflight_v1 --packet {PACKET} --output-root {ROOT}"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError(str(errors))
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({
        "task_id": TASK, "phase": packet["phase"], "attempt": 1, "status": "RUNNING",
        "session": "exact78_frozen_156", "updated_at": now, "heartbeat_at": now,
        "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": signature, "quality": "PREFLIGHT_PENDING",
        "adoption": "NOT_ADOPTED", "review": "NOT_APPLICABLE",
    })
    state["next_task"] = {"task_id": TASK, "session": "exact78_frozen_156",
                          "prerequisites": packet["prerequisites"], "expected_resource": "TWO_CPU_THREADS_NO_GPU",
                          "stop_condition": "one terminal attempt or one-hour deadline"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"],
                                  "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_GENERAL_P0_PREFLIGHT_V1_RUNNING", plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_GENERAL_P0_PREFLIGHT_REGISTERED",
                   expected_revision=revision, generator_path=Path(__file__), task_packet_index_path=INDEX,
                   task_packet_index_value=index)
    environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                   "TMPDIR": str(cache), "XDG_CACHE_HOME": str(cache), "MPLCONFIGDIR": str(cache),
                   "NUMBA_CACHE_DIR": str(cache), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
    command = [sys.executable, "-B", "-m", "chaoyang.cli", "run", "run_exact78_general_p0_preflight_v1",
               "--packet", str(PACKET), "--output-root", str(ROOT)]
    with (ROOT / "worker.log").open("x") as log:
        worker = subprocess.Popen(command, cwd=REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        started = time.monotonic()
        while True:
            try:
                code = worker.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                current = copy.deepcopy(load_json(TASK_STATE_PATH))
                if (current.get("next_task") or {}).get("task_id") != TASK or time.monotonic() - started > 3600:
                    os.killpg(worker.pid, signal.SIGTERM)
                    code = worker.wait(timeout=30)
                    break
                current_row = next(item for item in current["tasks"] if item["task_id"] == TASK)
                current_row.update(heartbeat_at=now_iso(), updated_at=now_iso())
                publish_bundle(load_json(AUTHORITY_PATH), current, event_type="EXACT78_GENERAL_P0_PREFLIGHT_HEARTBEAT",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=index_copy())
    result_path = ROOT / "RESULT.json"
    if not result_path.is_file():
        atomic_json(result_path, {"status": "FAILED_RUNTIME_FINAL", "returncode": code,
                                  "execution_complete": False, "visual_qualified": False, "quality_passed": False,
                                  "deletion_authorized": False, "next_action": "DIAGNOSE_P0_RUNTIME"})
    result = load_json(result_path)
    robot100_after = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    guard_ok = robot100_before == robot100_after
    atomic_json(ROOT / "ROBOT100_GUARD_AFTER.json", {"unchanged": guard_ok, "files": robot100_after})
    if not guard_ok:
        raise RuntimeError("ROBOT100_GUARD_CHANGED")
    terminal = "PASSED" if code == 0 and result.get("execution_complete") else "FAILED_RUNTIME_FINAL"
    receipt_path = ROOT / "TERMINAL_RECEIPT.json"
    atomic_json(receipt_path, {"task_id": TASK, "status": terminal, "result": artifact_ref(result_path),
                               "claim_limit": packet["claim_limit"], "deletion_authorized": False})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    if (state.get("next_task") or {}).get("task_id") != TASK:
        raise RuntimeError("FINALIZE_ROUTE_CONFLICT")
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=terminal, updated_at=now_iso(), heartbeat_at=now_iso(), pid=None, proc_start_ticks=None,
               result=artifact_ref(result_path), terminal_receipt=artifact_ref(receipt_path),
               quality="PREFLIGHT_COMPLETE" if terminal == "PASSED" else terminal)
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_GENERAL_P0_PREFLIGHT_V1_TERMINAL", plan_revision=packet["plan_revision"],
                 execution_revision=packet["execution_revision"], status="PASS_NO_ACTIVE_TASKS",
                 supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_GENERAL_P0_PREFLIGHT_TERMINAL",
                               expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=index)
    print(json.dumps({"status": terminal, "revision": published["governance_revision"],
                      "result": artifact_ref(result_path)}, ensure_ascii=False))
    return 0 if terminal == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
