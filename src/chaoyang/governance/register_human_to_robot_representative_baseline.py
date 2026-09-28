"""Register the bounded 10-hour representative Human-to-Robot baseline task."""
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

TASK = "human_to_robot_representative_baseline_20260924"
REVISION = "HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE_20260924"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{REVISION}/00_EXECUTION.md"
BASELINE = REPO_ROOT / "_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/RESULT.json"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LANES = ("scene", "sensor", "motion_product", "compare")
ACTIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if state.get("next_task") is not None or any(row.get("status") in ACTIVE for row in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("INDEX_NOT_EMPTY")
    if ATTEMPT.exists() or any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_EXISTS")
    for path in (PLAN, BASELINE):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = load_json(BASELINE)
    if (baseline.get("product_structure"), baseline.get("product_quality"), baseline.get("product_adoption")) != ("4/4", "0/4", "0/4"):
        raise RuntimeError("BASELINE_STATUS_DRIFT")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=10)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Select two representative sessions; deliver stable Robot review, bounded Clean, Sensor field quality, frozen HuRo comparison and genuine Flow Matching consumer check.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(BASELINE.relative_to(REPO_ROOT)), "docs/current/STATUS.json"],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}", "tasks/current/INDEX.json",
                      "docs/current/PLAN.md", "docs/current/AI_WORK_ENTRY_ZH.md",
                      "docs/current/V5_SCENE.md", "docs/current/V5_SENSOR.md",
                      "docs/current/V5_MOTION.md", "docs/current/V5_HURO.md",
                      "docs/current/STATUS.json", "docs/current/visuals/HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE",
                      f"docs/plans/{REVISION}",
                      "src/chaoyang/governance/register_human_to_robot_representative_baseline.py",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
                      "src/chaoyang/ops/run_human_to_robot_representative_baseline.py",
                      "src/chaoyang/pipeline/full_robot_review_v2.py",
                      "src/chaoyang/human_ego/preprocess/retarget_labels/schema.py",
                      "src/chaoyang/human_ego/training/FlowMatchingDataloader.py",
                      "src/chaoyang/human_ego/training/FlowMatchingModel.py",
                      "src/chaoyang/human_ego/training/FlowMatchingTrainer.py",
                      "tests/test_human_to_robot_representative_baseline.py"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only",
                          "all_writes_inside_repo", "one_GPU_owner", "no_same_signature_quality_retry"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 36000, "cpu_threads_soft_cap": 8,
                    "cpu_heavy_jobs_max": 2, "gpu_concurrent_owners_max": 1,
                    "same_signature_quality_retry_max": 0, "new_031_ik_attempts_max": 0,
                    "new_huro_solver_runs_max": 0, "optimizer_steps_max": 0},
        "stop_conditions": ["all_authorized_lanes_terminal_or_10h_deadline",
                            "quality_failure_is_local", "no_old_output_overwrite",
                            "no_new_model_training", "control_ground_truth_false",
                            "physical_deployable_false", "external_metric_authority_false"],
        "frozen_gates": {"chips_candidates": ["get_potato_chips_0915_042", "get_potato_chips_0902_103", "get_potato_chips_0915_007"],
                         "poker_candidates": ["play_cards_0902_042", "play_cards_0915_119"],
                         "horizon_frames": 50, "new_031_ik_attempts_max": 0,
                         "new_huro_solver_runs_max": 0},
        "baseline": artifact_ref(BASELINE), "plan": artifact_ref(PLAN),
        "weights": "EXISTING_PINNED_WEIGHTS_ONLY_NO_DOWNLOAD",
        "calibration_or_absent": "PINNED_DEVELOPMENT_MOUNT_OR_ABSENT_PER_FIELD",
        "expected_resource": "FOUR_LANES_CPU8_GPU_SINGLE_OWNER", "attempt_max": 1,
        "executor_epoch": 27,
        "claim_limit": "Offline visual Robot/Clean/Sensor diagnostics and field-scoped learning interface check; no learning benefit, real action, external metric, control or deployment authority.",
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
        atomic_json(path, {"schema_version": "HUMAN_TO_ROBOT_REPRESENTATIVE_LANE_STATE_V1",
                           "task_id": TASK, "lane": lane, "status": "PENDING", "execution": "NOT_STARTED",
                           "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED",
                           "review": "NOT_EVALUATED", "improvement": "NOT_EVALUATED",
                           "consumer_eligible": False, "adoption": "NOT_ADOPTED",
                           "next_action": "BIND_FROZEN_INPUTS_AND_RUN_CONSUMER", "dependencies": [],
                           "blocker": None, "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 27},
                           "updated_at": created, "training_eligible": False,
                           "control_ground_truth": False, "physical_deployable": False,
                           "external_metric_authority": False})
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_REPRESENTATIVE_RUN_SIGNATURE_V1",
        "task_id": TASK, "plan": artifact_ref(PLAN), "baseline": artifact_ref(BASELINE),
        "task_packet": packet_ref, "t0": created, "deadline_at": deadline,
        "status": "REGISTERED_NOT_STARTED"})
    state["tasks"].append({"task_id": TASK, "phase": REVISION,
                           "plan_execution_revision": REVISION, "attempt": 0, "status": "PENDING",
                           "updated_at": created, "heartbeat_at": None,
                           "session": "human_to_robot_representative_baseline",
                           "pid": None, "proc_start_ticks": None, "gpu_id": None,
                           "task_packet": packet_ref, "deadline_at": deadline, "t0": created})
    state["next_task"] = {"task_id": TASK, "session": "human_to_robot_representative_baseline",
                          "prerequisites": packet["prerequisites"],
                          "expected_resource": packet["expected_resource"],
                          "stop_condition": "Four lanes complete or 10-hour deadline; one branch failure is local."}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered representative consumer baseline; prior attempts immutable.",
    }])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
                 "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
                 "task_packets": [{"task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                                   "packet_sha256": packet_ref["sha256"],
                                   "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                                   "weights": packet["weights"]}], "claim_limit": packet["claim_limit"]}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE_REGISTERED",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
