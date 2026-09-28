"""Register one bounded CPU diagnostic of 031 MANO/Stereo visible-surface Z."""
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
from chaoyang.ops.run_human_to_robot_031_surface_depth_diagnostic import (
    TASK, ATTEMPT, SOURCE, DOMAIN, MASK_ROOT, DEPTH_RESULT, DEPTH_ROOT,
)

REVISION = "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_DIAGNOSTIC_20260923"
PLAN = REPO_ROOT / f"docs/plans/{REVISION}/00_EXECUTION.md"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
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
        raise RuntimeError("INDEX_NOT_EMPTY")
    if ATTEMPT.exists() or any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_EXISTS")
    for path in (PLAN, SOURCE, DOMAIN, DEPTH_RESULT, MASK_ROOT, DEPTH_ROOT):
        if not path.exists():
            raise FileNotFoundError(path)
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Measure fixed-frame same-pixel MANO visible-surface versus Stereo optical-Z differences in 031.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(SOURCE.relative_to(REPO_ROOT)), str(DOMAIN.relative_to(REPO_ROOT)),
                     str(MASK_ROOT.relative_to(REPO_ROOT)), str(DEPTH_RESULT.relative_to(REPO_ROOT)),
                     str((DEPTH_ROOT / "frames").relative_to(REPO_ROOT))],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}", "tasks/current/INDEX.json",
                      "docs/current/PLAN.md", "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/V5_MOTION.md",
                      "docs/current/STATUS.json", "src/chaoyang/ops/run_human_to_robot_031_surface_depth_diagnostic.py",
                      "src/chaoyang/ops/finalize_human_to_robot_031_surface_depth_diagnostic.py",
                      "src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "tests/test_human_to_robot_031_surface_depth_diagnostic.py"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only",
                          "all_writes_inside_repo", "CPU_TOTAL_SOFT_LIMIT_8"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             "attempts/attempt_0001/lanes/motion/surface_depth_v1/RESULT.json",
                             "attempts/attempt_0001/lanes/motion/surface_depth_v1/031_MANO_STEREO_SURFACE_Z_16FRAME_REVIEW.mp4",
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 3600, "gpu_seconds_max": 0,
                    "same_signature_quality_retry_max": 0, "same_signature_runtime_retry_max": 1},
        "stop_conditions": ["16_fixed_frames_diagnosed_or_wall_budget_exhausted",
                            "no_model_reinference_or_calibration_fit", "no_product_quality_promotion"],
        "frozen_gates": {"session": "play_cards_0915_031", "sampled_frames": 16,
                         "time_frame_count": 149, "depth_discontinuity_m": 0.025},
        "plan": artifact_ref(PLAN), "weights": "ABSENT_CPU_ONLY_FROZEN_OUTPUTS",
        "calibration_or_absent": "LOCAL_STEREO_METRIC_DEV_NOT_EXTERNAL",
        "expected_resource": "CPU_ONLY_ONE_HOUR", "attempt_max": 1, "executor_epoch": 23,
        "claim_limit": "Surface-Z diagnostic only; not anatomical wrist truth, correction or product quality.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_TASK_PACKET:" + ";".join(errors))
    ATTEMPT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    packet_path.parent.mkdir(parents=True)
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    lane = ATTEMPT / "lanes/motion"
    lane.mkdir(parents=True)
    atomic_json(lane / "STATE.json", {
        "schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_LANE_STATE_V1", "task_id": TASK,
        "lane": "motion", "status": "PENDING", "execution": "NOT_STARTED", "structure": "NOT_EVALUATED",
        "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
        "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 23},
        "updated_at": created, "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_RUN_SIGNATURE_V1", "task_id": TASK,
        "plan": artifact_ref(PLAN), "config": packet_ref,
        "code": artifact_ref(REPO_ROOT / "src/chaoyang/ops/run_human_to_robot_031_surface_depth_diagnostic.py"),
        "input_manifest": artifact_ref(DOMAIN), "input_model": artifact_ref(SOURCE),
        "input_stereo_receipt": artifact_ref(DEPTH_RESULT), "weights": "ABSENT_CPU_ONLY_FROZEN_OUTPUTS",
        "calibration": "LOCAL_STEREO_METRIC_DEV_NOT_EXTERNAL", "schema": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_DIAGNOSTIC_V1",
        "status": "REGISTERED_NOT_STARTED", "t0": created, "deadline_at": deadline,
    })
    state["tasks"].append({
        "task_id": TASK, "phase": REVISION, "plan_execution_revision": REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": created, "heartbeat_at": None,
        "session": "play_cards_0915_031", "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": created,
    })
    state["next_task"] = {"task_id": TASK, "session": "play_cards_0915_031",
                          "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
                          "stop_condition": "Fixed 16-frame surface diagnostic or one-hour wall budget."}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered 031 same-pixel MANO/Stereo surface diagnostic.",
    }])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
                 "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
                 "task_packets": [{"task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                                   "packet_sha256": packet_ref["sha256"],
                                   "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                                   "weights": "ABSENT_CPU_ONLY_FROZEN_OUTPUTS"}],
                 "claim_limit": packet["claim_limit"]}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_031_SURFACE_DEPTH_REGISTERED",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
