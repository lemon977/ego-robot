"""Bounded CPU development registration using the existing governance publisher.

This is not a model/training entry point. The live process owns publication;
send ``finish <absolute result path>`` on stdin after development verification.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH, artifact_ref,
    atomic_json, canonical_bytes, load_json, now_iso, process_identity,
    publish_bundle, validate_task_state,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "pick_place_humanego_ict_revo2_development_v2_20261010"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ACTIVE = {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}


def publish(state: dict, index: dict, event: str) -> None:
    revision = int(load_json(RECEIPT_PATH)["governance_revision"])
    index["supersedes_index"] = artifact_ref(INDEX)
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type=event,
                   expected_revision=revision, generator_path=Path(__file__),
                   task_packet_index_path=INDEX, task_packet_index_value=index)


def main() -> int:
    state, index = copy.deepcopy(load_json(TASK_STATE_PATH)), copy.deepcopy(load_json(INDEX))
    if (state.get("next_task") or any(r.get("status") in ACTIVE for r in state["tasks"])
            or any(r.get("execution_allowed") for r in index["task_packets"])):
        raise RuntimeError("ACTIVE_WRITER_CONFLICT")
    if ROOT.exists() or PACKET.exists():
        raise RuntimeError("FRESH_ATTEMPT_REQUIRED")
    proc_stat = Path("/proc/self/stat").read_text()
    proc_pid = int(proc_stat.split(" ", 1)[0])
    proc_ticks = int(proc_stat[proc_stat.rfind(")") + 2:].split()[19])
    if process_identity(proc_pid)["start_ticks"] != proc_ticks:
        raise RuntimeError("PROCFS_WRITER_IDENTITY_UNVERIFIED")
    writer = {"pid": proc_pid, "proc_start_ticks": proc_ticks,
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    inputs = {
        "raw_root": "/path/to/private_storage/datasets/ego/pick_place_1008",
        "robotized_root": "/path/to/private_storage/datasets/ego/pick_place_1008_raw_to_robot_v9_original_execution_v1",
        "target_hardware": "REVO2_6_HAND_CHANNELS_PER_SIDE",
        "reference_model": artifact_ref(REPO_ROOT / "src/chaoyang/human_ego/training/FlowMatchingModel.py"),
        "existing_trainer": artifact_ref(REPO_ROOT / "src/chaoyang/human_ego/training/FlowMatchingTrainer.py"),
    }
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Build and CPU-test a fail-closed paired ICT training handoff; no formal training or physical deployment",
        "phase": "HUMANEGO_ICT_REVO2_ENGINEERING", "plan_revision": "PICK_PLACE_HUMANEGO_ICT_REVO2_V1",
        "execution_revision": TASK.upper(), "execution_mode": "CPU_DEVELOPMENT_ONLY",
        "read_set": ["AGENTS.md", "docs/governance/CURRENT_STATUS_RECEIPT.json",
                     "docs/governance/ALGORITHM_CONTRACT.json", "docs/governance/DOC_AUTHORITY_MAP.json",
                     "src/chaoyang/human_ego/training/FlowMatchingModel.py",
                     "src/chaoyang/human_ego/training/FlowMatchingTrainer.py",
                     "src/chaoyang/governance/common.py", "tasks/current/INDEX.json"],
        "read_set_expansions": ["raw session manifests and dataset.hdf schemas", "NAS portable export manifests and source/mapping/state arrays",
                                "existing HumanEgo model/transfer public APIs", "OpenPI Revo2 input/action schema only"],
        "write_set": ["src/chaoyang/humanego_ict", "scripts/humanego_ict",
                      "scripts/maintenance/pick_place_ict_development_task_v1.py",
                      "tests/humanego_ict", "docs/plans/PICK_PLACE_HUMANEGO_ICT_REVO2_V1",
                      str(ROOT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      "tasks/receipts/" + TASK.upper() + ".json", "tasks/current/INDEX.json",
                      "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "REVO2_TARGET_CONFIRMED_BY_USER"],
        "required_outputs": ["candidate_adapter_and_training_entry", "fixed_global_batch_DDP_and_20000_update_support",
                             "CPU_TEST_RESULT.json", "DATA_READINESS.json", "HANDOFF_ZH.md", "RESULT.json"],
        "budgets": {"wall_seconds_max": 7200, "gpu_count": 0, "formal_training_steps": 0,
                    "subagents_max": 2, "CPU_threads_per_process": 2},
        "stop_conditions": ["writer conflict", "source SHA drift", "deadline", "missing eligibility preserves BLOCKED status"],
        "claim_limit": "Engineering candidate only; no causal-input, real control label, trained checkpoint, baseline adoption or deployment qualification",
        "inputs": inputs, "writer": writer,
        "run_signature": hashlib.sha256(canonical_bytes(inputs)).hexdigest(),
        "deadline_at": (datetime.now(ZoneInfo("Asia/Shanghai")) + timedelta(seconds=7200)).isoformat(),
        "commands": ["CPU editing and tests only; algorithm/model invocation not authorized"],
        "development_code_is_output_not_frozen_algorithm": True,
    }
    if errors := _validate_packet(packet):
        raise RuntimeError(str(errors))
    ROOT.mkdir(parents=True)
    PACKET.parent.mkdir(parents=True)
    atomic_json(ROOT / "WRITER.json", writer)
    atomic_json(PACKET, packet)
    stamp = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
                           "status": "RUNNING", "updated_at": stamp, "heartbeat_at": stamp,
                           "pid": writer["pid"], "proc_start_ticks": writer["proc_start_ticks"],
                           "gpu_id": None, "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"]})
    state["next_task"] = {"task_id": TASK, "expected_resource": "CPU_DEVELOPMENT_NO_MODELS"}
    if errors := validate_task_state(state):
        raise RuntimeError(str(errors))
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
                                  "packet_sha256": artifact_ref(PACKET)["sha256"], "execution_allowed": True,
                                  "execution_class": "CPU_DEVELOPMENT_NO_ALGORITHM_EXECUTION"})
    index.update(packet_revision=TASK.upper(), execution_revision=TASK.upper(), plan_revision=packet["plan_revision"], status="PASS")
    publish(state, index, "HUMANEGO_ICT_REVO2_DEVELOPMENT_REGISTERED")
    print(json.dumps({"status": "REGISTERED", "task_id": TASK, "writer": writer}), flush=True)
    deadline = time.monotonic() + 7200
    last = time.monotonic()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 10)
        if ready:
            command = sys.stdin.readline().strip()
            if command.startswith("finish "):
                result_path = Path(command[7:]).resolve(strict=True)
                if not result_path.is_relative_to(ROOT):
                    raise RuntimeError("RESULT_OUTSIDE_ATTEMPT")
                result = load_json(result_path)
                if result.get("task_id") != TASK:
                    raise RuntimeError("RESULT_TASK_MISMATCH")
                state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
                task = next(r for r in state["tasks"] if r["task_id"] == TASK)
                task.update(status="PASSED" if result["engineering_status"] == "PASSED" else "BLOCKED_PREREQ",
                            updated_at=now_iso(), result=artifact_ref(result_path))
                state["next_task"] = None
                next(r for r in index["task_packets"] if r["task_id"] == TASK)["execution_allowed"] = False
                receipt_path = REPO_ROOT / "tasks/receipts" / (TASK.upper() + ".json")
                atomic_json(receipt_path, {"task_id": TASK, "status": task["status"], "result": artifact_ref(result_path),
                                          "claim_limit": packet["claim_limit"]})
                task["receipt"] = artifact_ref(receipt_path)
                publish(state, index, "HUMANEGO_ICT_REVO2_DEVELOPMENT_SEALED")
                print(json.dumps({"status": task["status"], "receipt": str(receipt_path)}), flush=True)
                return 0
        if time.monotonic() - last >= 60:
            state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
            task = next(r for r in state["tasks"] if r["task_id"] == TASK)
            if (task["status"] != "RUNNING" or task["pid"] != writer["pid"]
                    or process_identity(writer["pid"])["start_ticks"] != writer["proc_start_ticks"]):
                raise RuntimeError("WRITER_FENCE_DRIFT")
            task.update(heartbeat_at=now_iso(), updated_at=now_iso())
            publish(state, index, "HUMANEGO_ICT_REVO2_DEVELOPMENT_HEARTBEAT")
            last = time.monotonic()
    raise RuntimeError("DEVELOPMENT_DEADLINE_REACHED")


if __name__ == "__main__":
    raise SystemExit(main())
