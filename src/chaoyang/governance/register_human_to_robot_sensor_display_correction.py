"""Register a finite CPU-only re-render of the three saved Sensor sessions."""
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

TASK = "human_to_robot_sensor_display_correction_20260923"
REVISION = "HUMAN_TO_ROBOT_SENSOR_DISPLAY_CORRECTION_20260923"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / f"docs/plans/{REVISION}/00_EXECUTION.md"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OLD = REPO_ROOT / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001/lanes/sensor/review_v1/RESULT.json"
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
    old = load_json(OLD)
    if ([row["session_id"] for row in old["sessions"]] !=
            ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"]
            or [row["frames"] for row in old["sessions"]] != [165, 179, 122]):
        raise RuntimeError("FROZEN_SENSOR_INPUTS_NOT_READY")
    if not PLAN.is_file() or ATTEMPT.exists():
        raise RuntimeError("PLAN_MISSING_OR_ATTEMPT_EXISTS")
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=2)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Re-render 466 saved Sensor frames with honest equal metric world and fixed shared local scales.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(OLD.relative_to(REPO_ROOT)),
                     "src/chaoyang/ops/run_human_to_robot_convergence_sensor_review.py"],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}",
                      "tasks/current/INDEX.json", "docs/current/PLAN.md", "docs/current/V5_SENSOR.md",
                      "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/STATUS.json",
                      "src/chaoyang/ops/run_human_to_robot_sensor_display_correction.py",
                      "src/chaoyang/ops/finalize_human_to_robot_sensor_display_correction.py",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "tests/test_human_to_robot_sensor_display_correction.py"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only",
                          "all_writes_inside_repo", "CPU_TOTAL_SOFT_LIMIT_8"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             "attempts/attempt_0001/lanes/sensor/review_metric_v2/RESULT.json",
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 7200, "gpu_seconds_max": 0,
                    "same_signature_quality_retry_max": 0, "same_signature_runtime_retry_max": 1},
        "stop_conditions": ["three_full_replays_terminal_or_budget_exhausted", "no_old_output_overwrite",
                            "no_sensor_alignment_or_product_quality_promotion", "training_eligible_false",
                            "control_ground_truth_false", "physical_deployable_false"],
        "frozen_gates": {"sessions": ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"],
                         "frames": [165, 179, 122], "new_solver_invocations": 0,
                         "world_xz_equal_pixels_per_m": True, "local_scale_shared_fixed_per_session": True},
        "old_review": artifact_ref(OLD), "plan": artifact_ref(PLAN),
        "weights": "ABSENT_CPU_ONLY", "calibration_or_absent": "REUSED_DEVELOPMENT_SENSOR",
        "expected_resource": "CPU_ONLY_TWO_HOURS", "attempt_max": 1, "executor_epoch": 18,
        "claim_limit": "Display correction only; no new calibration, wrist accuracy, robot product or deployment authority.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    lane = ATTEMPT / "lanes/sensor"
    lane.mkdir(parents=True)
    atomic_json(lane / "STATE.json", {
        "schema_version": "HUMAN_TO_ROBOT_SENSOR_DISPLAY_LANE_STATE_V1",
        "task_id": TASK, "lane": "sensor", "status": "PENDING",
        "execution": "NOT_STARTED", "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED",
        "adoption": "NOT_ADOPTED", "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 18},
        "updated_at": created, "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_SENSOR_DISPLAY_RUN_SIGNATURE_V1",
        "task_id": TASK, "plan": artifact_ref(PLAN), "old_review": artifact_ref(OLD),
        "status": "REGISTERED_NOT_STARTED", "t0": created, "deadline_at": deadline,
    })
    state["tasks"].append({
        "task_id": TASK, "phase": REVISION, "plan_execution_revision": REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": created,
        "heartbeat_at": None, "session": "human_to_robot_sensor_display_correction",
        "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": created,
    })
    state["next_task"] = {
        "task_id": TASK, "session": "human_to_robot_sensor_display_correction",
        "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
        "stop_condition": "Three saved-array full-session reviews or two-hour wall budget exhausted.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered Sensor display-only metric scale correction.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
        "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{"task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                          "packet_sha256": packet_ref["sha256"],
                          "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                          "weights": "ABSENT_CPU_ONLY"}],
        "claim_limit": "Sensor display-only correction routing, no calibration or product promotion.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_SENSOR_DISPLAY_REGISTERED",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
