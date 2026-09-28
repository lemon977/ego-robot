"""Register one bounded CPU spatial-support check for frozen 007 donor pairs."""
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

TASK = "human_to_robot_007_spatial_donor_check_20260923"
REVISION = "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_CHECK_20260923"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{REVISION}/00_EXECUTION.md"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PRIOR = REPO_ROOT / "_run/current/human_to_robot_007_late_donor_probe_20260923/attempts/attempt_0001/RESULT.json"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    if state.get("next_task") is not None or any(row.get("status") in LIVE for row in state["tasks"]):
        raise RuntimeError("ACTIVE_TASK_EXISTS")
    if index.get("status") != "PASS_NO_ACTIVE_TASKS" or index.get("task_packets") != []:
        raise RuntimeError("CURRENT_INDEX_NOT_EMPTY")
    if any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_REGISTERED")
    prior = load_json(PRIOR)
    if prior.get("task_terminal_status") != "PASSED" or not prior.get("full_cohort_evaluated"):
        raise RuntimeError("PRIOR_SOURCE_SEARCH_NOT_COMPLETE")
    if not PLAN.is_file() or ATTEMPT.exists():
        raise RuntimeError("PLAN_MISSING_OR_ATTEMPT_EXISTS")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Check target and donor inlier-hull interpolation support for twelve frozen 007 pairs.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(PRIOR.relative_to(REPO_ROOT)),
                     "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw",
                     "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/masks_007_recovered_join_v1/human",
                     "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1",
                     "_run/current/human_to_robot_007_attachment_clean_canary_20260923/attempts/attempt_0001/lanes/scene/attachment_clean_window_v1",
                     "src/chaoyang/ops/run_human_to_robot_007_device_donor_probe.py"],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}", "tasks/current/INDEX.json",
                      "docs/current/PLAN.md", "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/V5_SCENE.md",
                      "docs/current/STATUS.json", "src/chaoyang/ops/run_human_to_robot_007_spatial_donor_check.py",
                      "src/chaoyang/ops/finalize_human_to_robot_007_spatial_donor_check.py",
                      "src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "tests/test_human_to_robot_007_spatial_donor_check.py"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only",
                          "all_writes_inside_repo", "CPU_TOTAL_SOFT_LIMIT_8"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             "attempts/attempt_0001/lanes/scene/spatial_donor_check_v1/RESULT.json",
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 3600, "gpu_seconds_max": 0,
                    "same_signature_quality_retry_max": 0, "same_signature_runtime_retry_max": 1},
        "stop_conditions": ["twelve_frozen_pairs_terminal_or_budget_exhausted", "no_old_output_overwrite",
                            "no_Clean_or_product_promotion", "training_eligible_false", "control_ground_truth_false"],
        "frozen_gates": {"target_frames": [192, 196], "donor_pairs": {
            "192": [115, 135, 170, 175, 281, 282, 336, 363],
            "196": [48, 51, 197, 377]}, "fit_inlier_hulls_both_sides_required": True},
        "prior": artifact_ref(PRIOR), "plan": artifact_ref(PLAN),
        "weights": "ABSENT_CPU_ONLY", "calibration_or_absent": "ABSENT_PIXEL_ONLY",
        "expected_resource": "CPU_ONLY_ONE_HOUR", "attempt_max": 1, "executor_epoch": 20,
        "claim_limit": "Spatial interpolation diagnostic only; no Clean, physical background or product claim.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    lane = ATTEMPT / "lanes/scene"
    lane.mkdir(parents=True)
    atomic_json(lane / "STATE.json", {
        "schema_version": "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_LANE_STATE_V1", "task_id": TASK,
        "lane": "scene", "status": "PENDING", "execution": "NOT_STARTED",
        "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
        "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 20},
        "updated_at": created, "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_RUN_SIGNATURE_V1", "task_id": TASK,
        "plan": artifact_ref(PLAN), "prior": artifact_ref(PRIOR),
        "status": "REGISTERED_NOT_STARTED", "t0": created, "deadline_at": deadline,
    })
    state["tasks"].append({
        "task_id": TASK, "phase": REVISION, "plan_execution_revision": REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": created,
        "heartbeat_at": None, "session": "human_to_robot_007_spatial_donor_check",
        "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": created,
    })
    state["next_task"] = {
        "task_id": TASK, "session": "human_to_robot_007_spatial_donor_check",
        "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
        "stop_condition": "Twelve frozen donor pairs or one-hour wall budget exhausted.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered 007 donor spatial interpolation check.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
        "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{"task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                          "packet_sha256": packet_ref["sha256"],
                          "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                          "weights": "ABSENT_CPU_ONLY"}],
        "claim_limit": "Spatial donor support only; no Clean or product quality promotion.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_007_SPATIAL_DONOR_REGISTERED",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
