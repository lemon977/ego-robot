#!/usr/bin/env python3
"""Fail-closed terminal publisher for Human-to-Robot quality closure S1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
)

TASK = "human_to_robot_quality_closure_s1_20260923"
ROUTE = "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TASK_RECEIPT = REPO_ROOT / "tasks/receipts/HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_20260923_RESULT.json"
LANES = ("scene_evidence", "geometry_contact", "assembly_product", "compare")


def pid_is_live(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()

    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") != "RUNNING":
        raise RuntimeError("S1_NOT_RUNNING")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("S1_NOT_SOLE_CURRENT_TASK")
    packets = load_json(INDEX).get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("S1_PACKET_NOT_SOLE_CURRENT")
    if (ATTEMPT / "RESULT.json").exists() or TASK_RECEIPT.exists():
        raise RuntimeError("IMMUTABLE_FINAL_ALREADY_EXISTS")

    validation_path = ATTEMPT / "FINAL_VALIDATION.json"
    validation = load_json(validation_path)
    if validation.get("status") != "PASS_STRUCTURE_WITH_QUALITY_GAPS":
        raise RuntimeError("FINAL_VALIDATION_NOT_ACCEPTABLE")
    if validation.get("video_count") != 15 or not validation.get("video_full_decode_pass"):
        raise RuntimeError("FINAL_MEDIA_VALIDATION_INCOMPLETE")
    if validation.get("pytest", {}).get("passed", 0) < 1493:
        raise RuntimeError("REGRESSION_EVIDENCE_NOT_BOUND")

    for lane in LANES:
        value = load_json(ATTEMPT / "lanes" / lane / "STATE.json")
        if pid_is_live(value.get("writer", {}).get("pid")):
            raise RuntimeError(f"ACTIVE_LANE_WRITER:{lane}")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") == "ACQUIRED" and lease.get("task_id") == TASK:
        raise RuntimeError("S1_GPU_LEASE_ACTIVE")

    time = now_iso()
    lane_refs = {lane: artifact_ref(ATTEMPT / "lanes" / lane / "STATE.json") for lane in LANES}
    result = {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_RESULT_V1",
        "task_id": TASK,
        "route": ROUTE,
        "status": "TERMINAL_WITH_QUALITY_GAPS",
        "task_terminal_status": "REJECTED_QUALITY",
        "release_status": "INCOMPLETE",
        "terminal_at": time,
        "execution_summary": {
            "local_attachment_track_sessions": 3,
            "local_attachment_clean_executed": 2,
            "formal_clean_quality_pass": 0,
            "formal_clean_adopted": 0,
            "stereo_preflight_pass": 1,
            "depth_full_pass": 1,
            "object6d_observability_complete": 1,
            "strict_contact_windows": 0,
            "robot_r1_executed": 0,
            "robot_r1_adopted": 0,
            "adapter_candidate_mesh_loaded": True,
            "measured_installation": "ABSENT",
            "preserved_numeric_compare_sessions": 2,
            "method_winner": None,
        },
        "quality_axes": {
            "execution": "EXECUTED",
            "structure": "PASS",
            "quality": "REJECTED_QUALITY",
            "adoption": "NOT_ADOPTED",
        },
        "reason_codes": [
            "FULL_TIMELINE_DEVICE_CABLE_EVIDENCE_INCOMPLETE",
            "LOCAL_CLEAN_VISUAL_QUALITY_REJECTED",
            "STRICT_CONTACT_DIRECT_OBSERVATION_ABSENT",
            "007_STEREO_PREFLIGHT_BLOCKED",
            "0902_CURRENT_ENCODED_DOMAIN_ABSENT_OR_WITHDRAWN",
            "MEASURED_ASSEMBLY_TRANSFORMS_ABSENT",
            "ADOPTED_CLEAN_ABSENT_FOR_FINAL_COMPARE",
        ],
        "final_validation": artifact_ref(validation_path),
        "progress_final": artifact_ref(ATTEMPT / "PROGRESS_FINAL.json"),
        "lane_states": lane_refs,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "S1 produced valid local attachment tracks, two real local Clean executions, and a valid "
            "031 encoded-domain Depth/Object6D chain. Both Clean canaries were rejected visually; no "
            "strict Contact window, Robot R1, measured assembly, final method winner, or deployment authority exists."
        ),
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    atomic_json(TASK_RECEIPT, {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_RECEIPT_V1",
        "task_id": TASK,
        "status": "REJECTED_QUALITY",
        "result": result_ref,
        "videos_full_decode": 15,
        "tests_passed_preterminal": validation["pytest"]["passed"],
        "tests_skipped": validation["pytest"]["skipped"],
    })

    task.update(
        status="REJECTED_QUALITY",
        phase="HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_TERMINAL",
        attempt=1,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        heartbeat_at=None,
        updated_at=time,
        result=result_ref,
        last_attempt_terminal="REJECTED_QUALITY",
        last_attempt_reason="STRUCTURE_COMPLETE_BUT_FORMAL_CLEAN_CONTACT_ASSEMBLY_AND_COMPARE_QUALITY_UNMET",
    )
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK,
        "attempt": 1,
        "status": "REJECTED_QUALITY",
        "created_at": time,
        "message": "S1 terminal: valid local evidence and 031 geometry, but formal quality/adoption remains unmet.",
        "result": result_ref,
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_20260923",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "S1 is terminal rejected-quality; further algorithm work requires a new finite task.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_QUALITY_CLOSURE_S1_TERMINAL_QUALITY_GAPS",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor,
    )
    atomic_json(REPO_ROOT / "docs/current/STATUS.json", {
        "schema_version": "HUMAN_TO_ROBOT_S1_TERMINAL_STATUS_V1",
        "generated_at": now_iso(),
        "governance_revision": published["governance_revision"],
        "status": "PASS_NO_ACTIVE_TASKS",
        "active_task": None,
        "latest_task": TASK,
        "latest_terminal_status": "REJECTED_QUALITY",
        "result": result_ref,
        "final_validation": artifact_ref(validation_path),
        "quality_axes": result["quality_axes"],
        "execution_summary": result["execution_summary"],
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    })
    print(json.dumps({
        "status": "REJECTED_QUALITY",
        "governance_revision": published["governance_revision"],
        "result": result_ref,
        "videos_full_decode": 15,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
