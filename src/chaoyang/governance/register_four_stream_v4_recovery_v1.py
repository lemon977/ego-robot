#!/usr/bin/env python3
"""Register a bounded successor after the V4 coordinator-only terminal."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK_ID = "four_stream_full_pipeline_v4_recovery_v1"
PLAN = "FOUR_STREAM_FULL_PIPELINE_V4_RECOVERY_V1"
EXEC = "FOUR_STREAM_FULL_PIPELINE_V4_RECOVERY_V1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REG = REPO_ROOT / f"_run/current/{TASK_ID}/registration_0001"
LANES = ("exact78", "controller_manus", "hawor_retarget", "huro")
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-revision", type=int, required=True)
    args = ap.parse_args()

    receipt = load_json(RECEIPT_PATH)
    rev = int(receipt["governance_revision"])
    if rev != args.expected_revision:
        raise RuntimeError(f"CAS mismatch: expected {args.expected_revision}, current {rev}")
    state = load_json(TASK_STATE_PATH)
    if state.get("next_task") is not None or any(r.get("status") in LIVE for r in state.get("tasks", [])):
        raise RuntimeError("an active task already exists")
    current = load_json(INDEX)
    if current.get("status") != "PASS_NO_ACTIVE_TASKS" or current.get("task_packets") != []:
        raise RuntimeError("current task index is not terminal and empty")
    if REG.exists():
        raise RuntimeError(f"registration already exists: {REG}")

    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK_ID,
        "objective": "Recover the V4 execution contract and produce one reproducible first milestone per isolated lane; preserve all V4 audit evidence and keep development-only authority.",
        "phase": EXEC,
        "plan_revision": PLAN,
        "execution_revision": EXEC,
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/governance/ALGORITHM_CONTRACT.json",
            "docs/current/PLAN.md",
            "docs/current/FOUR_STREAM_FULL_PIPELINE_V4_HANDOFF_ZH.md",
            "tasks/current/INDEX.json",
            "_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001/V4_ROUTE_CONTRACT_AUDIT_20260922_ZH.md",
            "_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001/V4_RUNTIME_TERMINAL_20260922.json",
            "_run/current/four_stream_full_pipeline_v4/attempts/attempt_0001/V3_REUSE_MANIFEST.json",
        ],
        "write_set": [
            f"_run/current/{TASK_ID}",
            "docs/current/visuals/FOUR_STREAM_FULL_PIPELINE_V4_RECOVERY_V1",
            f"tasks/receipts/{TASK_ID.upper()}_RESULT.json",
        ],
        "prerequisites": [
            "governance_PASS_FRESH",
            "no_active_current_task",
            "source_processed_archive_sealed_READ_ONLY",
            "single_publisher",
            "four_lane_writer_roots_ISOLATED",
            "single_GPU_lease_owner",
            "per_session_image_domain_explicit",
            "development_only_non_control_non_deployable",
        ],
        "required_outputs": [
            *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
            "attempts/attempt_0001/RUN_SIGNATURE.json",
            "attempts/attempt_0001/PROGRESS_2H.json",
            "attempts/attempt_0001/RESULT.json",
        ],
        "budgets": {
            "wall_seconds_max": 21600,
            "cpu_threads_soft_cap": 8,
            "gpu_concurrent_owners_max": 1,
            "progress_hours": [2, 4, 6],
            "same_signature_retry_max": 1,
            "candidate_max_per_method": 1,
        },
        "stop_conditions": [
            "contract_or_claim_conflict_fails_closed",
            "deadline_or_budget_saves_resumable_state",
            "quality_failure_no_same_signature_retry",
            "no_source_or_sealed_writes",
            "control_ground_truth_false",
            "physical_deployable_false",
        ],
        "weights": "ABSENT",
        "child_weights_policy": "EACH_CHILD_BINDS_EXACTLY_ONE_PINNED_WEIGHT_OR_ABSENT",
        "calibration_or_absent": "PER_LANE_PINNED_EVIDENCE_OR_ABSENT",
        "external_metric_authority": False,
        "expected_resource": "FOUR_ISOLATED_CPU_LANES_PLUS_ONE_GOVERNED_GPU_LEASE",
        "executor_epoch": 1,
        "fencing": {
            "pid_startticks_required": True,
            "immutable_final": True,
            "unique_primary_writer": True,
            "lane_writer_roots": {
                lane: f"_run/current/{TASK_ID}/attempts/attempt_0001/lanes/{lane}"
                for lane in LANES
            },
        },
        "attempt_max": 1,
        "claim_limit": "Execution recovery only. No algorithm result is adopted; no quality, control, deployment, or external metric authority is granted.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid recovery packet: " + "; ".join(errors))

    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(seconds=21600)).isoformat(timespec="seconds")
    REG.mkdir(parents=True)
    shutil.copyfile(INDEX, REG / "PREDECESSOR_TASK_PACKET_INDEX.json")
    packet_path = REPO_ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    predecessor_ref = artifact_ref(REG / "PREDECESSOR_TASK_PACKET_INDEX.json")
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{EXEC}_ROUTABLE",
        "plan_revision": PLAN,
        "execution_revision": EXEC,
        "status": "PASS",
        "supersedes_index": predecessor_ref,
        "task_packets": [{
            "task_id": TASK_ID,
            "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
            "weights": "ABSENT",
        }],
        "claim_limit": "One bounded recovery successor; four lanes isolated and GPU serialized.",
    }
    state["tasks"].append({
        "task_id": TASK_ID,
        "phase": EXEC,
        "plan_execution_revision": EXEC,
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": packet_ref,
        "deadline_at": deadline,
        "t0": created,
    })
    state["next_task"] = {
        "task_id": TASK_ID,
        "session": "four_lane_recovery",
        "prerequisites": packet["prerequisites"],
        "expected_resource": packet["expected_resource"],
        "stop_condition": "Six-hour contract-recovered first milestone or resumable blocker receipts.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK_ID,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered bounded recovery successor after coordinator-only V4 terminal.",
        "task_packet": packet_ref,
        "predecessor_terminal": "four_stream_full_pipeline_v4",
    }])[-100:]
    atomic_json(REG / "PROJECTED_TASK_STATE.json", state)
    atomic_json(REG / "RESULT.json", {
        "schema_version": "chaoyang-v4-recovery-registration-result-v1",
        "task_id": TASK_ID,
        "status": "REGISTERED",
        "registered_at": created,
        "deadline_at": deadline,
        "task_packet": packet_ref,
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_FULL_PIPELINE_V4_RECOVERY_REGISTERED",
        expected_revision=rev,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor,
    )
    print(json.dumps({"status": "PASSED", "task_id": TASK_ID, "governance_revision": published["governance_revision"], "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
