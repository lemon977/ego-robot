#!/usr/bin/env python3
"""Initialize the isolated recovery attempt after CAS registration."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "four_stream_full_pipeline_v4_recovery_v1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PACKET = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
ATT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LANES = ("exact78", "controller_manus", "hawor_retarget", "huro")


def token(lane: str, created: str) -> str:
    return hashlib.sha256(f"{TASK}\0{lane}\0{created}\0{os.getpid()}".encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-revision", type=int, required=True)
    args = ap.parse_args()
    receipt = load_json(RECEIPT_PATH)
    rev = int(receipt["governance_revision"])
    if rev != args.expected_revision:
        raise RuntimeError(f"CAS mismatch: expected {args.expected_revision}, current {rev}")
    index = load_json(INDEX)
    entry = index.get("task_packets", [{}])[0]
    if entry.get("task_id") != TASK or entry.get("execution_allowed") is not True:
        raise RuntimeError("recovery route is not executable")
    state = load_json(TASK_STATE_PATH)
    row = next((x for x in state.get("tasks", []) if x.get("task_id") == TASK), None)
    if row is None or row.get("status") != "PENDING":
        raise RuntimeError("recovery task is not pending")
    if (ATT / "INITIALIZATION_RESULT.json").exists():
        raise RuntimeError(f"attempt already initialized: {ATT}")
    created = now_iso()
    ATT.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": "chaoyang-v4-recovery-run-signature-v1",
        "task_id": TASK,
        "attempt": 1,
        "t0": created,
        "deadline_at": row.get("deadline_at"),
        "task_packet": artifact_ref(PACKET),
        "source_data_write_authorized": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "image_domain_policy": "per_session_explicit; no global sourceIndex assumption",
        "publisher_pid": os.getpid(),
    }
    atomic_json(ATT / "RUN_SIGNATURE.json", signature)
    refs = {}
    for lane in LANES:
        root = ATT / "lanes" / lane
        root.mkdir(parents=True, exist_ok=True)
        state_value = {
            "schema_version": "chaoyang-v4-recovery-lane-state-v1",
            "parent_task_id": TASK,
            "lane": lane,
            "status": "READY_CPU_PREFLIGHT",
            "current_action": "RECOVERY_CONTRACT_PREFLIGHT_PENDING",
            "latest_artifacts": [],
            "blocker": None,
            "next_step": "RECOVERY_CONTRACT_PREFLIGHT",
            "writer": {
                "pid": None,
                "proc_start_ticks": None,
                "executor_epoch": 1,
                "fencing_token": token(lane, created),
                "writer_root": str(root.resolve()),
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
        atomic_json(root / "STATE.json", state_value)
        refs[lane] = artifact_ref(root / "STATE.json")
    atomic_json(ATT / "LANE_LEDGER.json", {
        "schema_version": "chaoyang-v4-recovery-lane-ledger-v1",
        "task_id": TASK,
        "created_at": created,
        "cpu_threads_soft_cap": 8,
        "single_publisher": True,
        "cross_lane_writes_allowed": False,
        "lanes": refs,
    })
    atomic_json(ATT / "GPU_QUEUE_LEDGER.json", {
        "schema_version": "chaoyang-v4-recovery-gpu-queue-v1",
        "task_id": TASK,
        "created_at": created,
        "concurrent_owners_max": 1,
        "lease_required": True,
        "current_owner": None,
        "allocations_seconds": {"exact78": 7200, "observation": 7200, "huro": 7200, "stereo": 900},
        "consumed_seconds": {"exact78": 0, "observation": 0, "huro": 0, "stereo": 0},
    })
    row.update(attempt=1, status="PENDING", attempt_root=str(ATT), updated_at=created, heartbeat_at=None)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "READY", "created_at": created,
        "message": "Recovery attempt initialized; lane writers remain unclaimed.",
        "run_signature": artifact_ref(ATT / "RUN_SIGNATURE.json"),
    }])[-100:]
    atomic_json(ATT / "TASK_STATE_INITIALIZED.json", state)
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_FULL_PIPELINE_V4_RECOVERY_INITIALIZED",
        expected_revision=rev,
        generator_path=Path(__file__),
    )
    atomic_json(ATT / "INITIALIZATION_RESULT.json", {
        "schema_version": "chaoyang-v4-recovery-initialization-result-v1",
        "task_id": TASK,
        "status": "PASSED",
        "created_at": created,
        "governance_revision": published["governance_revision"],
    })
    print({"status": "PASSED", "task_id": TASK, "attempt_root": str(ATT), "governance_revision": published["governance_revision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
