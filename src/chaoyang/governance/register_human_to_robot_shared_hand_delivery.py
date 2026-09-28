"""Register the bounded shared-hand/Robot/Clean delivery task."""
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

TASK = "human_to_robot_shared_hand_delivery_20260924"
REVISION = "HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_20260924"
PLAN = REPO_ROOT / "docs/current/SHARED_HAND_DELIVERY_CARD_ZH.md"
BASELINE = REPO_ROOT / "_run/current/human_to_robot_result_breakthrough_20260924/attempts/attempt_0001/RESULT.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
LANES = ("hand_data", "robot", "clean", "delivery")
ACTIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if state.get("next_task") is not None or any(row.get("status") in ACTIVE for row in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if index.get("task_packets") or any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_REGISTERED")
    if ATTEMPT.exists() or not PLAN.is_file():
        raise RuntimeError("ATTEMPT_EXISTS_OR_PLAN_MISSING")
    baseline = load_json(BASELINE)
    if baseline.get("counts", {}).get("products_quality") != "0/4":
        raise RuntimeError("BASELINE_QUALITY_CHANGED")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=10)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Repair shared local hand targets, evaluate one joint arm-trajectory candidate, localize and repair one Clean failure path, and deliver real consumers.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(BASELINE.relative_to(REPO_ROOT)), "docs/current/STATUS.json",
                     "docs/current/RESULT_BREAKTHROUGH_VIDEO_INDEX_ZH.md"],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}", "tasks/current/INDEX.json",
                      "docs/current", "docs/governance", "src/chaoyang", "tests", "_run/cache"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_receipts_read_only",
                          "all_writes_inside_repo", "one_GPU_owner", "optimizer_steps_zero"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 36000, "cpu_threads_soft_cap": 8,
                    "cpu_heavy_jobs_max": 2, "gpu_concurrent_owners_max": 1,
                    "new_huro_solver_runs_max": 0, "new_031_ik_attempts_max": 0,
                    "optimizer_steps_max": 0, "hand_recipe_max": 1,
                    "robot_recipe_max": 1, "clean_recipe_max": 1,
                    "implementation_repairs_per_root_cause_max": 2},
        "stop_conditions": ["all_authorized_actions_terminal_or_10h_deadline",
                            "one_branch_rejection_does_not_stop_other_ready_work",
                            "no_automatic_successor"],
        "frozen_sessions": {"poker": "play_cards_0902_042", "poker_frames": 171,
                            "sensor_development": "play_cards_0916_097",
                            "sensor_regression": ["play_cards_0916_098", "play_cards_0916_101"],
                            "clean_window": [76, 91], "robot_window": [80, 111],
                            "product_denominator": ["get_potato_chips_0915_007", "play_cards_0915_031",
                                                    "get_potato_chips_0902_103", "play_cards_0902_042"]},
        "frozen_gates": {"root_position_m_max": 0.020, "root_rotation_deg_max": 15.0,
                         "local_bone_direction_deg_max": 15.0,
                         "normalized_pinch_error_max": 0.10,
                         "h50_source_frames": 51},
        "baseline": artifact_ref(BASELINE), "plan": artifact_ref(PLAN),
        "weights": "EXISTING_PINNED_ONLY;LAMA_EXISTING_ASSET_ALLOWED;NO_NEW_WEIGHT_DOWNLOAD",
        "calibration_or_absent": "PINNED_DEVELOPMENT_MOUNT_OR_ABSENT_PER_FIELD",
        "expected_resource": "THREE_LANES_CPU8_GPU_SINGLE_OWNER",
        "attempt_max": 1, "executor_epoch": 30,
        "claim_limit": "Offline local q22 learning eligibility and Robot/Clean development quality only; no training, control, deployment or external metric authority.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    next_actions = {
        "hand_data": "IMPLEMENT_SHARED_LOCAL_TARGET_AND_RUN_SYNTHETIC_TESTS",
        "robot": "IMPLEMENT_JOINT_WINDOW_TRAJECTORY_SOLVER_AND_RUN_080_111",
        "clean": "INSTRUMENT_PROPainter_INTERNALS_AND_CLASSIFY_FAILURE_LAYER",
        "delivery": "WAIT_FOR_LANE_ARTIFACTS_THEN_EVALUATE_AND_PUBLISH",
    }
    for lane in LANES:
        path = ATTEMPT / "lanes" / lane / "STATE.json"
        path.parent.mkdir(parents=True)
        atomic_json(path, {"schema_version": "HUMAN_TO_ROBOT_SHARED_DELIVERY_LANE_STATE_V1",
                           "task_id": TASK, "lane": lane, "status": "READY" if lane != "delivery" else "PENDING",
                           "execution": "NOT_STARTED", "structure": "NOT_EVALUATED",
                           "quality": "NOT_EVALUATED", "improvement": "NOT_EVALUATED",
                           "consumer_qualification": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
                           "next_action": next_actions[lane], "dependencies": [], "evidence": [],
                           "blocker": None, "updated_at": created})
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_SHARED_DELIVERY_RUN_SIGNATURE_V1",
        "task_id": TASK, "plan": artifact_ref(PLAN), "baseline": artifact_ref(BASELINE),
        "task_packet": packet_ref, "t0": created, "deadline_at": deadline,
        "status": "REGISTERED_NOT_STARTED"})
    state["tasks"].append({"task_id": TASK, "phase": REVISION,
                           "plan_execution_revision": REVISION, "attempt": 0,
                           "status": "PENDING", "updated_at": created, "heartbeat_at": None,
                           "session": "human_to_robot_shared_hand_delivery", "pid": None,
                           "proc_start_ticks": None, "gpu_id": None, "task_packet": packet_ref,
                           "deadline_at": deadline, "t0": created})
    state["next_task"] = {"task_id": TASK, "session": "human_to_robot_shared_hand_delivery",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": packet["expected_resource"],
                          "stop_condition": "All bounded branches terminal or ten-hour deadline; local rejection does not stop independent READY work."}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered shared-hand, joint-trajectory, and Clean-layer delivery; predecessor immutable."}])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
                 "execution_revision": REVISION, "status": "PASS",
                 "supersedes_index": artifact_ref(INDEX),
                 "task_packets": [{"task_id": TASK,
                                   "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                                   "packet_sha256": packet_ref["sha256"],
                                   "execution_class": "CURRENT_LEDGER_ROUTABLE",
                                   "execution_allowed": True, "weights": packet["weights"]}],
                 "claim_limit": packet["claim_limit"]}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_REGISTERED",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
