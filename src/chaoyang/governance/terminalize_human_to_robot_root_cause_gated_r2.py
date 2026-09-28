#!/usr/bin/env python3
"""Fail-closed terminal publisher for the bounded Human-to-Robot R2 run."""
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

TASK = "human_to_robot_root_cause_gated_r2_20260923"
ROUTE = "HUMAN_TO_ROBOT_BASELINE_V1_R2"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TASK_RECEIPT = REPO_ROOT / f"tasks/receipts/{TASK.upper()}_RESULT.json"
LANES = ("lane1_scene", "lane2_motion", "lane3_sensor", "lane4_compare")


def _pid_is_live(pid: object) -> bool:
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
        raise RuntimeError("R2_NOT_RUNNING")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("R2_NOT_SOLE_CURRENT_TASK")
    packets = load_json(INDEX).get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK:
        raise RuntimeError("R2_PACKET_NOT_SOLE_CURRENT")
    if (ATTEMPT / "RESULT.json").exists() or TASK_RECEIPT.exists():
        raise RuntimeError("IMMUTABLE_FINAL_ALREADY_EXISTS")

    validation_path = ATTEMPT / "FINAL_VALIDATION.json"
    validation = load_json(validation_path)
    if validation.get("status") != "PASS_STRUCTURE_WITH_QUALITY_GAPS":
        raise RuntimeError("FINAL_VALIDATION_NOT_ACCEPTABLE")
    if validation.get("video_count") != 13 or not validation.get("video_full_decode_pass"):
        raise RuntimeError("FINAL_MEDIA_VALIDATION_INCOMPLETE")
    if validation.get("pytest", {}).get("passed") != 1489:
        raise RuntimeError("FULL_REGRESSION_NOT_BOUND")

    for lane in LANES:
        path = ATTEMPT / "lanes" / lane / "STATE.json"
        value = load_json(path)
        writer = value.get("writer", {})
        if _pid_is_live(writer.get("pid")):
            raise RuntimeError(f"ACTIVE_LANE_WRITER:{lane}:{writer['pid']}")
        value["writer"] = {"pid": None, "proc_start_ticks": None}
        value["status"] = "TERMINAL_WITH_QUALITY_GAPS"
        value["updated_at"] = now_iso()
        atomic_json(path, value)

    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") == "ACQUIRED" and str(lease.get("task_id", "")).startswith(TASK):
        raise RuntimeError("R2_GPU_LEASE_ACTIVE")

    time = now_iso()
    lane_refs = {lane: artifact_ref(ATTEMPT / "lanes" / lane / "STATE.json") for lane in LANES}
    result = {
        "schema_version": "HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_RESULT_V1",
        "task_id": TASK,
        "route": ROUTE,
        "status": "TERMINAL_WITH_QUALITY_GAPS",
        "task_terminal_status": "REJECTED_QUALITY",
        "release_status": "INCOMPLETE",
        "terminal_at": time,
        "execution_summary": {
            "scene_executed": 4,
            "scene_structure_pass": 4,
            "scene_quality_pass": 0,
            "scene_adopted": 0,
            "product_candidates": 4,
            "product_quality_pass": 0,
            "product_adopted": 0,
            "strict_resume_pass": 4,
            "sensor_common_backend": 3,
            "sensor_authority": "KINEMATIC_ONLY",
            "robot_r1_eligible_windows": 0,
            "robot_r1_executed": 0,
            "robot_r1_adopted": 0,
            "local_huro_numeric_sessions": 2,
            "local_huro_winner": None,
            "real_session_occlusion": "UNKNOWN",
            "adapter_candidate_mesh_loaded": True,
            "measured_installation": "ABSENT",
        },
        "quality_axes": {
            "execution": "EXECUTED",
            "structure": "PASS",
            "quality": "REJECTED_QUALITY",
            "adoption": "NOT_ADOPTED",
        },
        "reason_codes": [
            "SCENE_DEVICE_AND_CABLE_EVIDENCE_INCOMPLETE",
            "VISIBLE_OBJECT_PROTECTION_QUALITY_REJECTED",
            "REAL_SESSION_OCCLUSION_UNKNOWN",
            "STRICT_METRIC_CONTACT_AUTHORITY_ABSENT",
            "MEASURED_ASSEMBLY_TRANSFORMS_ABSENT",
            "LOCAL_HURO_EXTERNAL_TRUTH_ABSENT",
        ],
        "final_validation": artifact_ref(validation_path),
        "progress_2h": artifact_ref(ATTEMPT / "PROGRESS_2H.json"),
        "lane_states": lane_refs,
        "contact_r1": artifact_ref(ATTEMPT / "lanes/lane1_scene/contact_r1_funnel_wave9/RESULT.json"),
        "assembly": artifact_ref(ATTEMPT / "lanes/lane2_motion/assembly_scene_canary_wave11/RESULT.json"),
        "visual_index": artifact_ref(REPO_ROOT / "docs/current/visuals/HUMAN_TO_ROBOT_R2/README_ZH.md"),
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Four Scene and four Robot videos are rejected-quality offline candidates. "
            "Three Sensor results are kinematic-only. No formal Clean, metric Contact, "
            "Robot R1, physical calibration, method winner, control, or deployment authority was obtained."
        ),
        "next_prerequisite": (
            "A newly authorized finite task must add independently supported device/cable evidence, "
            "directly visible object protection, registered real-session depth/Object6D, and measured "
            "assembly transforms where their consumers require them; no same-signature quality retry."
        ),
    }
    atomic_json(ATTEMPT / "RESULT.json", result)
    result_ref = artifact_ref(ATTEMPT / "RESULT.json")
    atomic_json(
        TASK_RECEIPT,
        {
            "schema_version": "HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_RECEIPT_V1",
            "task_id": TASK,
            "status": "REJECTED_QUALITY",
            "result": result_ref,
            "scene_quality_pass": 0,
            "product_quality_pass": 0,
            "robot_r1_adopted": 0,
            "videos_full_decode": 13,
            "tests_passed": 1489,
            "tests_skipped": 1,
        },
    )

    task.update(
        status="REJECTED_QUALITY",
        phase="HUMAN_TO_ROBOT_BASELINE_V1_R2_TERMINAL",
        attempt=1,
        pid=None,
        proc_start_ticks=None,
        gpu_id=None,
        heartbeat_at=None,
        updated_at=time,
        result=result_ref,
        last_attempt_terminal="REJECTED_QUALITY",
        last_attempt_reason="STRUCTURE_COMPLETE_BUT_PRODUCT_QUALITY_AND_AUTHORITIES_UNMET",
    )
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK,
        "attempt": 1,
        "status": "REJECTED_QUALITY",
        "created_at": time,
        "message": "R2 terminal with structural evidence and explicit quality gaps; no formal product adoption.",
        "result": result_ref,
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_TERMINAL",
        "plan_revision": "HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_20260923",
        "execution_revision": "HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_20260923",
        "status": "PASS_NO_ACTIVE_TASKS",
        "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "R2 is terminal rejected-quality; a new algorithm candidate requires a new finite task packet.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_TERMINAL_QUALITY_GAPS",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=successor,
    )
    print(json.dumps({
        "status": "REJECTED_QUALITY",
        "governance_revision": published["governance_revision"],
        "result": result_ref,
        "videos_full_decode": 13,
        "tests_passed": 1489,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
