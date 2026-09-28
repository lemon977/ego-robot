#!/usr/bin/env python3
"""Register the bounded R2 successor after R1 is terminal."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from chaoyang.governance.common import (AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT,
    TASK_STATE_PATH, artifact_ref, atomic_json, load_json, now_iso, publish_bundle)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "human_to_robot_root_cause_gated_r2_20260923"
R1 = "human_to_robot_completion_r1_20260922"
OLD = "four_stream_visual_delivery_v5"
PLAN = "HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_20260923"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN_ROOT = REPO_ROOT / "docs/plans" / PLAN
RUN = REPO_ROOT / "_run/current" / TASK
ATTEMPT = RUN / "attempts/attempt_0001"
LANES = ("lane1_scene", "lane2_motion", "lane3_sensor", "lane4_compare")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    index = load_json(INDEX)
    if state.get("next_task") is not None or any(t.get("status") in {"PENDING","READY","CLAIMED","RUNNING","WAIT_GPU_RESOURCE"} for t in state.get("tasks", [])):
        raise RuntimeError("an active task exists")
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("current task packet index is not terminal empty")
    if any(t.get("task_id") == TASK for t in state.get("tasks", [])):
        raise RuntimeError("R2 already registered")
    r1_result = REPO_ROOT / f"_run/current/{R1}/attempts/attempt_0001/RESULT.json"
    v5_result = REPO_ROOT / f"_run/current/{OLD}/attempts/attempt_0001/RESULT.json"
    if not r1_result.is_file() or not v5_result.is_file():
        raise RuntimeError("sealed predecessor result missing")
    plan_path = PLAN_ROOT / "00_R2_EXECUTION.md"
    if not plan_path.is_file():
        raise RuntimeError("R2 plan file missing")
    t0 = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=12)).isoformat(timespec="seconds")
    RUN.mkdir(parents=True); ATTEMPT.mkdir(parents=True)
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Run R2 root-cause-first, per-session Human-to-Robot baseline: Scene/Motion/Sensor first, gated Product and Local-vs-HuRo comparison.",
        "phase": "HUMAN_TO_ROBOT_BASELINE_V1_R2", "plan_revision": PLAN, "execution_revision": PLAN,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1_R2",
        "read_set": ["tasks/current/INDEX.json", "docs/current/STATUS.json", str(plan_path.relative_to(REPO_ROOT)),
                      f"_run/current/{R1}/attempts/attempt_0001/RESULT.json", f"_run/current/{OLD}/attempts/attempt_0001/RESULT.json"],
        "write_set": [f"_run/current/{TASK}", f"tasks/receipts/{TASK.upper()}_RESULT.json", "docs/current/PLAN.md", "docs/current/README_ZH.md", "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/STATUS.json"],
        "prerequisites": ["governance_PASS_FRESH", "exclusive_current_index", "sealed_v5_and_r1_predecessors", "read_only_source_processed_archive_sealed", "single_publisher", "single_GPU_lease_owner", "cpu_total_soft_cap_8", "noncommercial_offline_visual_only"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json", "attempts/attempt_0001/PROGRESS_2H.json", *(f"attempts/attempt_0001/lanes/{x}/STATE.json" for x in LANES), "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 43200, "cpu_threads_soft_cap": 8, "gpu_concurrent_owners_max": 1, "max_evidence_driven_repairs_per_fault_package": 2, "same_signature_quality_retry_max": 0, "candidate_max_per_fault_package": 1},
        "stop_conditions": ["terminal_with_gaps_at_deadline", "quality_failure_does_not_auto_retry", "writer_or_gpu_lease_violation_fails_closed", "no_source_processed_archive_or_sealed_writes", "training_eligible_false", "control_ground_truth_false", "physical_deployable_false"],
        "target_sessions": {"main_products": ["get_potato_chips_0915_007", "play_cards_0915_031"], "regressions": ["get_potato_chips_0902_103", "play_cards_0902_042"], "sensor": ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"], "frame_counts": {"get_potato_chips_0915_007":378,"play_cards_0915_031":149,"get_potato_chips_0902_103":284,"play_cards_0902_042":171,"play_cards_0916_097":165,"play_cards_0916_098":179,"play_cards_0916_101":122}},
        "fault_packages": {f"F{i:02d}": 2 for i in range(1, 9)},
        "predecessors": {"r1": artifact_ref(r1_result), "v5": artifact_ref(v5_result)}, "plan": artifact_ref(plan_path),
        "weights": "ABSENT_OR_EXISTING_PINNED_ONLY", "calibration_or_absent": "PER_LANE_PINNED_OR_ABSENT", "external_metric_authority": False,
        "expected_resource": "FOUR_CPU_LANES_PLUS_ONE_GOVERNED_GPU_LEASE", "attempt_max": 1, "executor_epoch": 5,
        "fencing": {"pid_startticks_required": True, "unique_primary_writer": True, "immutable_final": True, "lane_writer_roots": {x: f"_run/current/{TASK}/attempts/attempt_0001/lanes/{x}" for x in LANES}},
        "claim_limit": "Offline visual candidate and reproducibility evidence only; no training, contact truth, physical metric authority, control or deployment.",
    }
    errors = _validate_packet(packet)
    if errors: raise RuntimeError("invalid R2 packet: " + "; ".join(errors))
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True); atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    for lane in LANES:
        p = ATTEMPT / "lanes" / lane; p.mkdir(parents=True)
        atomic_json(p / "STATE.json", {"schema_version":"human-to-robot-r2-lane-state-v1","task_id":TASK,"lane":lane,"status":"PENDING","execution":"NOT_STARTED","structure":"NOT_EVALUATED","quality":"NOT_EVALUATED","adoption":"NOT_ADOPTED","blocker":None,"updated_at":t0,"writer":{"pid":None,"proc_start_ticks":None,"executor_epoch":5},"training_eligible":False})
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {"schema_version":"human-to-robot-r2-run-signature-v1","task_id":TASK,"plan_revision":PLAN,"predecessors":{"r1":artifact_ref(r1_result),"v5":artifact_ref(v5_result)},"status":"REGISTERED_NOT_STARTED"})
    atomic_json(RUN / "registration_0001" / "RESULT.json", {"schema_version":"human-to-robot-r2-registration-v1","task_id":TASK,"status":"REGISTERED","t0":t0,"deadline_at":deadline,"task_packet":packet_ref})
    state["tasks"].append({"task_id":TASK,"phase":packet["phase"],"plan_execution_revision":PLAN,"attempt":0,"status":"PENDING","updated_at":t0,"heartbeat_at":None,"session":"r2_root_cause_gated","pid":None,"proc_start_ticks":None,"gpu_id":None,"task_packet":packet_ref,"deadline_at":deadline,"t0":t0})
    state["next_task"]={"task_id":TASK,"session":"r2_root_cause_gated","prerequisites":packet["prerequisites"],"expected_resource":packet["expected_resource"],"stop_condition":"12-hour bounded R2 execution or terminal_with_gaps."}
    state["recent_events"]=(state.get("recent_events",[])+[{"task_id":TASK,"attempt":0,"status":"PENDING","created_at":t0,"message":"Registered R2 root-cause-first successor; V5 and R1 preserved."}])[-100:]
    successor={"schema_version":"chaoyang-v71-task-packet-index-v3","packet_revision":f"{PLAN}_ROUTABLE","plan_revision":PLAN,"execution_revision":PLAN,"status":"PASS","supersedes_index":artifact_ref(INDEX),"task_packets":[{"task_id":TASK,"packet_path":str(packet_path.relative_to(REPO_ROOT)),"packet_sha256":packet_ref["sha256"],"execution_class":"CURRENT_LEDGER_ROUTABLE","execution_allowed":True,"weights":"ABSENT_OR_EXISTING_PINNED_ONLY"}],"claim_limit":"R2 routing only; quality, product adoption and improvement remain receipt-bound."}
    published=publish_bundle(load_json(AUTHORITY_PATH),state,event_type="HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_REGISTERED",expected_revision=args.expected_revision,generator_path=Path(__file__),task_packet_index_path=INDEX,task_packet_index_value=successor)
    print(json.dumps({"status":"REGISTERED","task_id":TASK,"governance_revision":published["governance_revision"],"t0":t0,"deadline_at":deadline},ensure_ascii=False))


if __name__ == "__main__": raise SystemExit(main())
