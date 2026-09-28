"""Register one bounded four-lane Human-to-Robot consumer-delivery successor."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

TASK = "human_to_robot_10h_delivery_20260924"
REVISION = "HUMAN_TO_ROBOT_10H_DELIVERY_20260924"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{REVISION}/00_EXECUTION.md"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
BASELINE = REPO_ROOT / "_run/current/human_to_robot_007_attachment_clean_canary_20260923/attempts/attempt_0001/RESULT.json"
PRODUCT_FIRST = REPO_ROOT / "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001/RESULT.json"
LANES = ("scene", "sensor", "motion_product", "compare")
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("INDEX_NOT_EMPTY")
    if ATTEMPT.exists() or any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_EXISTS")
    for path in (PLAN, BASELINE, PRODUCT_FIRST):
        if not path.is_file():
            raise FileNotFoundError(path)
    if load_json(BASELINE).get("quality_axes", {}).get("quality") != "REJECTED_QUALITY":
        raise RuntimeError("BASELINE_007_NOT_FROZEN_REJECTED")
    if load_json(PRODUCT_FIRST).get("execution_summary", {}).get("products_quality") != "0/4":
        raise RuntimeError("PRODUCT_FIRST_BASELINE_DRIFT")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=10)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "One 007 fixed-center temporal-context Clean candidate and four-lane consumer, review, and delivery closure.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(BASELINE.relative_to(REPO_ROOT)), str(PRODUCT_FIRST.relative_to(REPO_ROOT)),
                     "docs/current/STATUS.json"],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}", "tasks/current/INDEX.json",
                      "docs/current/PLAN.md", "docs/current/AI_WORK_ENTRY_ZH.md",
                      "docs/current/V5_SCENE.md", "docs/current/V5_SENSOR.md",
                      "docs/current/V5_MOTION.md", "docs/current/V5_HURO.md",
                      "docs/current/STATUS.json", "docs/current/visuals/HUMAN_TO_ROBOT_10H_DELIVERY",
                      "src/chaoyang/ops/run_human_to_robot_10h_007_context.py",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "tests/test_human_to_robot_10h_007_context.py"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only",
                          "all_writes_inside_repo", "one_GPU_owner", "no_same_signature_quality_retry"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 36000, "cpu_threads_soft_cap": 8,
                    "cpu_heavy_jobs_max": 2, "gpu_concurrent_owners_max": 1,
                    "same_signature_quality_retry_max": 0, "same_signature_runtime_retry_max": 1,
                    "new_031_ik_attempts_max": 0},
        "stop_conditions": ["all_authorized_branches_terminal_or_10h_deadline",
                            "one_007_context_candidate_only", "quality_failure_does_not_stop_other_ready_lanes",
                            "no_old_output_overwrite", "training_eligible_false", "control_ground_truth_false",
                            "physical_deployable_false", "external_metric_authority_false"],
        "frozen_gates": {"session": "get_potato_chips_0915_007", "input_frames": [160, 216],
                         "evaluation_frames": [181, 196], "evaluation_local_indices": [21, 36],
                         "left_context": 21, "right_context": 20, "center_step": 16,
                         "full_frame_count": 378, "new_031_ik_attempts_max": 0},
        "baseline": artifact_ref(BASELINE), "product_first": artifact_ref(PRODUCT_FIRST),
        "plan": artifact_ref(PLAN), "weights": "PINNED_EXISTING_PROPAINTER_ONLY",
        "calibration_or_absent": "ABSENT_PIXEL_ONLY_FOR_CLEAN;DEVELOPMENT_MOUNT_FOR_ROBOT",
        "expected_resource": "FOUR_LANES_SINGLE_GPU_LEASE", "attempt_max": 1,
        "executor_epoch": 25,
        "claim_limit": "Offline candidate and review only; no Clean/product, Contact, training, external metric, control, or physical authority without their own gates.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    for lane in LANES:
        path = ATTEMPT / "lanes" / lane / "STATE.json"
        path.parent.mkdir(parents=True)
        atomic_json(path, {"schema_version": "HUMAN_TO_ROBOT_10H_LANE_STATE_V1",
                           "task_id": TASK, "lane": lane, "status": "PENDING", "execution": "NOT_STARTED",
                           "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED",
                           "improvement": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
                           "next_action": "BIND_INPUTS_AND_RUN_ACTUAL_CONSUMER",
                           "dependencies": [], "blocker": None,
                           "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 25},
                           "updated_at": created, "training_eligible": False,
                           "control_ground_truth": False, "physical_deployable": False,
                           "external_metric_authority": False})
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_10H_RUN_SIGNATURE_V1", "task_id": TASK,
        "plan": artifact_ref(PLAN), "baseline": artifact_ref(BASELINE),
        "product_first": artifact_ref(PRODUCT_FIRST), "task_packet": packet_ref,
        "status": "REGISTERED_NOT_STARTED", "t0": created, "deadline_at": deadline,
    })
    state["tasks"].append({"task_id": TASK, "phase": REVISION, "plan_execution_revision": REVISION,
                           "attempt": 0, "status": "PENDING", "updated_at": created,
                           "heartbeat_at": None, "session": "human_to_robot_10h_delivery",
                           "pid": None, "proc_start_ticks": None, "gpu_id": None,
                           "task_packet": packet_ref, "deadline_at": deadline, "t0": created})
    state["next_task"] = {"task_id": TASK, "session": "human_to_robot_10h_delivery",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": packet["expected_resource"],
                          "stop_condition": "Four lanes complete or 10-hour deadline, not one candidate failure."}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered bounded consumer-delivery successor; prior attempts remain immutable.",
    }])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
                 "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
                 "task_packets": [{"task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                                   "packet_sha256": packet_ref["sha256"],
                                   "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                                   "weights": "PINNED_EXISTING_PROPAINTER_ONLY"}],
                 "claim_limit": packet["claim_limit"]}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_10H_DELIVERY_REGISTERED",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
