"""Register a bounded CPU correction of the 031 region-observability ledger."""
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
from chaoyang.ops.run_human_to_robot_031_observability_dtype_fix import (
    TASK, ATTEMPT, MANIFEST, SPEC, DOMAIN, MOTION, OLD,
)

REVISION = "HUMAN_TO_ROBOT_031_OBSERVABILITY_DTYPE_FIX_20260923"
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
    if any(row.get("task_id") == TASK for row in state["tasks"]):
        raise RuntimeError("TASK_ALREADY_REGISTERED")
    if ATTEMPT.exists() or not PLAN.is_file():
        raise RuntimeError("PLAN_MISSING_OR_ATTEMPT_EXISTS")
    for path in (MANIFEST, SPEC, DOMAIN, MOTION, OLD):
        if not path.is_file():
            raise FileNotFoundError(path)
    created = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=1)).isoformat(timespec="seconds")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Correct the uint16 foreground-one 031 SAM-region denominator and render its full-timeline evidence.",
        "phase": REVISION, "plan_revision": REVISION, "execution_revision": REVISION,
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "read_set": ["tasks/current/INDEX.json", str(PLAN.relative_to(REPO_ROOT)),
                     str(MANIFEST.relative_to(REPO_ROOT)), str(SPEC.relative_to(REPO_ROOT)),
                     str(DOMAIN.relative_to(REPO_ROOT)), str(MOTION.relative_to(REPO_ROOT)),
                     str(OLD.relative_to(REPO_ROOT)),
                     str((MANIFEST.parent / "human").relative_to(REPO_ROOT))],
        "write_set": [f"_run/current/{TASK}", f"tasks/current/{TASK}", "tasks/current/INDEX.json",
                      "docs/current/PLAN.md", "docs/current/AI_WORK_ENTRY_ZH.md", "docs/current/V5_MOTION.md",
                      "docs/current/STATUS.json", "src/chaoyang/ops/run_human_to_robot_031_observability_dtype_fix.py",
                      "src/chaoyang/ops/finalize_human_to_robot_031_observability_dtype_fix.py",
                      "src/chaoyang/governance/build_human_to_robot_r2_terminal_status.py",
                      "src/chaoyang/governance/current_r3_contracts.py",
                      "tests/test_human_to_robot_031_observability_dtype_fix.py"],
        "prerequisites": ["governance_PASS_FRESH", "single_publisher", "old_outputs_read_only",
                          "all_writes_inside_repo", "CPU_TOTAL_SOFT_LIMIT_8"],
        "required_outputs": ["attempts/attempt_0001/RUN_SIGNATURE.json",
                             "attempts/attempt_0001/lanes/motion/observability_dtype_fix_v1/RESULT.json",
                             "attempts/attempt_0001/lanes/motion/observability_dtype_fix_v1/031_REGION_PROXY_FULL_REVIEW.mp4",
                             "attempts/attempt_0001/RESULT.json"],
        "budgets": {"wall_seconds_max": 3600, "gpu_seconds_max": 0,
                    "same_signature_quality_retry_max": 0, "same_signature_runtime_retry_max": 1},
        "stop_conditions": ["149_frames_decoded_and_old_denominator_corrected_or_budget_exhausted",
                            "no_HaWoR_or_product_quality_promotion", "no_old_output_overwrite"],
        "frozen_gates": {"frames": 149, "mask_dtype": "uint16", "foreground_value": 1,
                         "anatomical_side": "UNKNOWN", "review_decoded_frames": 149},
        "plan": artifact_ref(PLAN), "weights": "ABSENT_CPU_ONLY",
        "calibration_or_absent": "ABSENT_REGION_ONLY", "expected_resource": "CPU_ONLY_ONE_HOUR",
        "attempt_max": 1, "executor_epoch": 22,
        "claim_limit": "Upstream SAM region denominator correction, not independent HaWoR pose accuracy or product quality.",
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
        "schema_version": "HUMAN_TO_ROBOT_031_REGION_DTYPE_LANE_STATE_V1", "task_id": TASK,
        "lane": "motion", "status": "PENDING", "execution": "NOT_STARTED",
        "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
        "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 22},
        "updated_at": created, "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    atomic_json(ATTEMPT / "RUN_SIGNATURE.json", {
        "schema_version": "HUMAN_TO_ROBOT_031_REGION_DTYPE_RUN_SIGNATURE_V1", "task_id": TASK,
        "plan": artifact_ref(PLAN), "config": packet_ref,
        "code": artifact_ref(REPO_ROOT / "src/chaoyang/ops/run_human_to_robot_031_observability_dtype_fix.py"),
        "input_manifest": artifact_ref(MANIFEST), "input_motion": artifact_ref(MOTION),
        "input_domain": artifact_ref(DOMAIN), "weights": "ABSENT_CPU_ONLY",
        "calibration": "ABSENT_REGION_ONLY", "schema": "HUMAN_TO_ROBOT_031_REGION_DTYPE_CORRECTION_V1",
        "status": "REGISTERED_NOT_STARTED", "t0": created, "deadline_at": deadline,
    })
    state["tasks"].append({
        "task_id": TASK, "phase": REVISION, "plan_execution_revision": REVISION,
        "attempt": 0, "status": "PENDING", "updated_at": created,
        "heartbeat_at": None, "session": "play_cards_0915_031",
        "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": created,
    })
    state["next_task"] = {"task_id": TASK, "session": "play_cards_0915_031",
                          "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
                          "stop_condition": "Correct 149 masks and decode full review or one-hour wall budget."}
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 0, "status": "PENDING", "created_at": created,
        "message": "Registered fixed 031 uint16 mask decoding correction.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{REVISION}_ROUTABLE", "plan_revision": REVISION,
        "execution_revision": REVISION, "status": "PASS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [{"task_id": TASK, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
                          "packet_sha256": packet_ref["sha256"],
                          "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
                          "weights": "ABSENT_CPU_ONLY"}],
        "claim_limit": "031 region-denominator correction only; no pose or product authority.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_031_REGION_DTYPE_REGISTERED",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "REGISTERED", "task_id": TASK,
                      "governance_revision": published["governance_revision"],
                      "t0": created, "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
