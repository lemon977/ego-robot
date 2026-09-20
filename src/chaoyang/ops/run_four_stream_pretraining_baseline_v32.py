#!/usr/bin/env python3
"""Initialize or refresh the isolated V3.2 four-stream parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from chaoyang.governance.build_four_stream_status_v32 import build_status
from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    sha256_file,
)


TASK_ID = "four_stream_pretraining_baseline_v32"
CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TASK_PACKET = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
ATTEMPT_ROOT = REPO_ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
STATUS_PATH = REPO_ROOT / "docs/current/STATUS.json"
LANES = ("exact78", "ai1", "ai2", "ai4_huro")


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def _fencing_token(lane: str, created: str) -> str:
    return hashlib.sha256(f"{TASK_ID}\0{lane}\0{created}\0{os.getpid()}".encode()).hexdigest()


def _validate_route() -> dict[str, Any]:
    index = load_json(CURRENT_INDEX)
    entries = index.get("task_packets", [])
    if len(entries) != 1:
        raise RuntimeError("V3.2 requires exactly one routed parent")
    entry = entries[0]
    if entry.get("task_id") != TASK_ID or entry.get("execution_allowed") is not True:
        raise RuntimeError("V3.2 parent is not execution_allowed")
    if not TASK_PACKET.is_file() or sha256_file(TASK_PACKET) != entry.get("packet_sha256"):
        raise RuntimeError("V3.2 task packet SHA mismatch")
    state = load_json(TASK_STATE_PATH)
    next_task = state.get("next_task")
    if not isinstance(next_task, dict) or next_task.get("task_id") != TASK_ID:
        raise RuntimeError("V3.2 parent is not the ledger successor")
    return state


def _lane_state(lane: str, created: str) -> dict[str, Any]:
    return {
        "schema_version": "chaoyang-four-stream-lane-state-v1",
        "parent_task_id": TASK_ID,
        "lane": lane,
        "status": "READY_CPU_PREFLIGHT",
        "current_action": "CPU_PREFLIGHT_NOT_STARTED",
        "latest_artifacts": [],
        "blocker": None,
        "next_step": "RUN_FROZEN_FIRST_MILESTONE",
        "writer": {
            "pid": None,
            "proc_start_ticks": None,
            "executor_epoch": 1,
            "fencing_token": _fencing_token(lane, created),
            "writer_root": str((ATTEMPT_ROOT / "lanes" / lane).resolve()),
        },
        "authority": "DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE",
        "updated_at": created,
        "claims": {
            "PIPELINE_COMPLETE": False,
            "NUMERIC_QUALITY_PASS": False,
            "VISUAL_REVIEW_STATUS": "NOT_REVIEWED",
            "TRAINING_COMPLETE": False,
            "TRAINING_ELIGIBLE": False,
            "CONTROL_GROUND_TRUTH": False,
            "PHYSICAL_DEPLOYABLE": False,
        },
    }


def initialize(expected_revision: int) -> dict[str, Any]:
    state = _validate_route()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != expected_revision:
        raise RuntimeError("CAS revision mismatch before V3.2 initialization")
    if ATTEMPT_ROOT.exists() or ATTEMPT_ROOT.is_symlink():
        raise RuntimeError(f"fresh V3.2 attempt required: {ATTEMPT_ROOT}")
    created = now_iso()
    ATTEMPT_ROOT.mkdir(parents=True)
    _write_once(
        ATTEMPT_ROOT / "RUN_SIGNATURE.json",
        {
            "schema_version": "chaoyang-four-stream-run-signature-v1",
            "task_id": TASK_ID,
            "attempt": 1,
            "t0": created,
            "task_packet": artifact_ref(TASK_PACKET),
            "plan": artifact_ref(REPO_ROOT / "docs/current/PLAN.md"),
            "lane_documents": {
                "exact78": artifact_ref(REPO_ROOT / "docs/current/EXACT78.md"),
                "ai1": artifact_ref(REPO_ROOT / "docs/current/AI1.md"),
                "ai2": artifact_ref(REPO_ROOT / "docs/current/AI2.md"),
                "ai4_huro": artifact_ref(REPO_ROOT / "docs/current/AI4_HURO.md"),
            },
            "weights": "ABSENT_AT_PARENT_CHILD_CALLS_PIN_ONE_WEIGHT_OR_ABSENT",
            "source_data_write_authorized": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
        },
    )
    lane_refs: dict[str, Any] = {}
    for lane in LANES:
        lane_root = ATTEMPT_ROOT / "lanes" / lane
        lane_root.mkdir(parents=True)
        state_path = lane_root / "STATE.json"
        _write_once(state_path, _lane_state(lane, created))
        lane_refs[lane] = artifact_ref(state_path)
    _write_once(
        ATTEMPT_ROOT / "LANE_LEDGER.json",
        {
            "schema_version": "chaoyang-four-stream-lane-ledger-v1",
            "task_id": TASK_ID,
            "created_at": created,
            "cpu_threads_soft_cap": 8,
            "single_publisher": True,
            "cross_lane_writes_allowed": False,
            "lanes": lane_refs,
        },
    )
    _write_once(
        ATTEMPT_ROOT / "GPU_QUEUE_LEDGER.json",
        {
            "schema_version": "chaoyang-four-stream-gpu-queue-v1",
            "task_id": TASK_ID,
            "created_at": created,
            "concurrent_owners_max": 1,
            "lease_required": True,
            "first_cycle_gpu_seconds_cap": 72900,
            "allocations_seconds": {
                "exact78_aggregate": 43200,
                "necessary_observation": 14400,
                "ai4_huro": 14400,
                "stereo": 900,
            },
            "consumed_seconds": {"exact78_aggregate": 0, "necessary_observation": 0, "ai4_huro": 0, "stereo": 0},
            "current_owner": None,
            "queue_policy": [
                "UNLOCK_EXACT78_FIRST_REAL_PAIR_AND_UPDATE",
                "AI1_AI2_AI4_FIRST_MILESTONES_GET_BOUNDED_SLOTS",
                "EXACT_PAIRED_TRAINING_ROTATES_AT_SAFE_CHECKPOINTS",
                "STEREO_NEVER_BLOCKS_TRAINING",
            ],
        },
    )
    task = next(row for row in state["tasks"] if row.get("task_id") == TASK_ID)
    task.update(
        attempt=1,
        status="PENDING",
        updated_at=created,
        heartbeat_at=None,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        attempt_root=str(ATTEMPT_ROOT),
    )
    state["recent_events"] = (
        state.get("recent_events", [])
        + [{
            "task_id": TASK_ID,
            "session": None,
            "attempt": 1,
            "status": "PENDING",
            "created_at": created,
            "message": "Initialized four isolated lanes and the single-owner GPU queue; algorithms not yet run.",
            "run_signature": artifact_ref(ATTEMPT_ROOT / "RUN_SIGNATURE.json"),
        }]
    )[-100:]
    projected = ATTEMPT_ROOT / "TASK_STATE_INITIALIZED.json"
    _write_once(projected, state)
    atomic_json(
        STATUS_PATH,
        build_status(
            task_state=state,
            generated_at=created,
            source_task_state_path=projected,
        ),
    )
    result = {
        "schema_version": "chaoyang-four-stream-initialization-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "created_at": created,
        "lane_ledger": artifact_ref(ATTEMPT_ROOT / "LANE_LEDGER.json"),
        "gpu_queue": artifact_ref(ATTEMPT_ROOT / "GPU_QUEUE_LEDGER.json"),
        "algorithm_execution_started": False,
    }
    _write_once(ATTEMPT_ROOT / "INITIALIZATION_RESULT.json", result)
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="FOUR_STREAM_PRETRAINING_BASELINE_V32_INITIALIZED",
        expected_revision=expected_revision,
        generator_path=Path(__file__),
    )
    return {
        "status": "PASSED",
        "task_id": TASK_ID,
        "attempt_root": str(ATTEMPT_ROOT),
        "governance_revision": published["governance_revision"],
    }


def refresh(expected_revision: int) -> dict[str, Any]:
    state = _validate_route()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt.get("governance_revision", -1)) != expected_revision:
        raise RuntimeError("CAS revision mismatch before V3.2 refresh")
    if not ATTEMPT_ROOT.is_dir():
        raise RuntimeError("V3.2 attempt not initialized")
    created = now_iso()
    snapshots = ATTEMPT_ROOT / "status_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    projected = snapshots / f"TASK_STATE_BEFORE_REV_{expected_revision + 1:06d}.json"
    _write_once(projected, state)
    atomic_json(
        STATUS_PATH,
        build_status(task_state=state, generated_at=created, source_task_state_path=projected),
    )
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="FOUR_STREAM_PRETRAINING_BASELINE_V32_STATUS_REFRESHED",
        expected_revision=expected_revision,
        generator_path=Path(__file__),
    )
    return {"status": "PASSED", "governance_revision": published["governance_revision"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("initialize", "refresh-status"), required=True)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    result = initialize(args.expected_revision) if args.mode == "initialize" else refresh(args.expected_revision)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
