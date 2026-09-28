#!/usr/bin/env python3
"""Close the unowned V4 task and CAS-register one twelve-hour product V5 route."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.governance.register_single_task_packet import _validate_packet

OLD = "four_stream_full_pipeline_v4_takeover_v2"
NEW = "four_stream_visual_delivery_v5"
LANES = ("scene", "sensor", "motion", "huro")
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
OLD_ATT = REPO_ROOT / f"_run/current/{OLD}/attempts/attempt_0001"
NEW_REG = REPO_ROOT / f"_run/current/{NEW}/registration_0001"
NEW_ATT = REPO_ROOT / f"_run/current/{NEW}/attempts/attempt_0001"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    old = next(row for row in state["tasks"] if row.get("task_id") == OLD)
    if old.get("status") != "PENDING" or old.get("pid") is not None:
        raise RuntimeError("predecessor is not unowned PENDING")
    if state.get("next_task", {}).get("task_id") != OLD:
        raise RuntimeError("predecessor is not current")
    index = load_json(INDEX)
    entries = index.get("task_packets", [])
    if len(entries) != 1 or entries[0].get("task_id") != OLD or entries[0].get("execution_allowed") is not True:
        raise RuntimeError("predecessor is not sole executable route")
    if NEW_REG.exists() or NEW_ATT.exists() or (OLD_ATT / "RESULT.json").exists():
        raise RuntimeError("immutable registration or predecessor terminal already exists")
    huro_path = OLD_ATT / "lanes/huro/wrist_pose_candidate_v1/get_potato_chips_0915_007/RESULT.json"
    huro = load_json(huro_path)
    if huro.get("session_id") != "get_potato_chips_0915_007" or huro.get("quality_adopted") is not False:
        raise RuntimeError("predecessor HuRo result changed")
    time = now_iso()
    deadline = (datetime.now().astimezone() + timedelta(hours=12)).isoformat(timespec="seconds")
    NEW_REG.mkdir(parents=True)
    NEW_ATT.mkdir(parents=True)
    atomic_json(NEW_REG / "PREDECESSOR_INDEX.json", index)
    old_result = {
        "schema_version": "chaoyang-v4-to-v5-supersession-v1",
        "task_id": OLD, "status": "CANCELLED",
        "reason_code": "USER_REPLACED_OBJECTIVE_WITH_SHARED_HUMAN_TO_ROBOT_PRODUCT",
        "quality_evaluated_for_all_lanes": False,
        "partial_evidence_preserved": True,
        "huro_development_candidate": artifact_ref(huro_path),
        "successor_task_id": NEW,
        "claim_limit": "Administrative supersession only; the preserved candidate is not a quality pass.",
    }
    atomic_json(OLD_ATT / "RESULT.json", old_result)
    old_ref = artifact_ref(OLD_ATT / "RESULT.json")
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": NEW,
        "objective": "Twelve-hour shared Scene plus motion Human-to-Robot product on four frozen visual sessions, independent sensor input on three sessions, and bounded same-input HuRo comparison.",
        "phase": "HUMAN_TO_ROBOT_BASELINE_V1",
        "plan_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5",
        "execution_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5",
        "read_set": [
            "docs/governance/CURRENT_STATUS_RECEIPT.json",
            "docs/governance/DOC_AUTHORITY_MAP.json",
            "docs/governance/ALGORITHM_CONTRACT.json",
            "tasks/current/INDEX.json",
            f"_run/current/{OLD}/attempts/attempt_0001/RESULT.json",
            "docs/current/AI_WORK_ENTRY_ZH.md",
            "docs/current/PLAN.md",
            "docs/current/FOUR_STREAM_FULL_PIPELINE_V4_HANDOFF_ZH.md",
        ],
        "write_set": [
            f"_run/current/{NEW}",
            "docs/current/visuals/FOUR_STREAM_V5",
            f"tasks/receipts/{NEW.upper()}_RESULT.json",
        ],
        "prerequisites": [
            "governance_PASS_FRESH", "exclusive_current_index", "registered_chaoyang_run_entry",
            "read_only_source_processed_archive", "per_session_image_domain_and_SHA",
            "four_isolated_lane_roots", "single_GPU_lease_owner",
        ],
        "required_outputs": [
            "attempts/attempt_0001/ROUTE_MANIFEST.json",
            *(f"attempts/attempt_0001/lanes/{lane}/STATE.json" for lane in LANES),
            "attempts/attempt_0001/RESULT.json",
        ],
        "budgets": {
            "wall_seconds_max": 43200, "cpu_threads_soft_cap": 8,
            "gpu_concurrent_owners_max": 1, "candidate_max_per_lane": 1,
            "engineering_repair_batch_max": 1, "same_signature_quality_retry_max": 0,
        },
        "stop_conditions": [
            "deadline_terminal_receipt", "quality_C_no_automatic_retry", "missing_input_BLOCKED_PREREQ",
            "no_source_or_sealed_writes", "no_clean_to_geometry",
            "control_ground_truth_false", "physical_deployable_false",
        ],
        "attempt_max": 1,
        "weights": "PER_LANE_PINNED_OR_ABSENT",
        "calibration_or_absent": "PER_LANE_PINNED_OR_ABSENT",
        "external_metric_authority": False,
        "expected_resource": "CPU_PARALLEL_PLUS_ONE_GOVERNED_GPU_LEASE",
        "executor_epoch": 3,
        "fencing": {
            "pid_startticks_required": True,
            "unique_primary_writer": True,
            "immutable_final": True,
            "lane_writer_roots": {
                lane: f"_run/current/{NEW}/attempts/attempt_0001/lanes/{lane}" for lane in LANES
            },
        },
        "target_sessions": {
            "scene": ["get_potato_chips_0915_007", "play_cards_0915_031", "get_potato_chips_0902_103", "play_cards_0902_042"],
            "sensor": ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"],
            "motion": ["get_potato_chips_0915_007", "play_cards_0915_031"],
            "huro": "same_frozen_HandMotion_R0_scene_assets_mount_placement",
        },
        "image_domain_policy": "PER_SESSION_EXPLICIT_NO_GLOBAL_SOURCE_INDEX",
        "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "claim_limit": "Offline visual candidate only. No causal training, measured metric accuracy, physical contact truth, control ground truth or deployment authority.",
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("invalid V5 packet: " + "; ".join(errors))
    packet_path = REPO_ROOT / f"tasks/current/{NEW}/TASK_PACKET.json"
    atomic_json(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    old.update(status="CANCELLED", phase="SUPERSEDED_BY_V5_PRODUCT_PLAN", updated_at=time,
               heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None,
               result=old_ref, last_attempt_terminal="CANCELLED",
               last_attempt_reason="USER_REPLACED_OBJECTIVE_WITH_SHARED_HUMAN_TO_ROBOT_PRODUCT")
    state["tasks"].append({
        "task_id": NEW, "phase": "HUMAN_TO_ROBOT_BASELINE_V1",
        "plan_execution_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5",
        "attempt": 0, "status": "PENDING", "updated_at": time, "heartbeat_at": None,
        "session": None, "pid": None, "proc_start_ticks": None, "gpu_id": None,
        "task_packet": packet_ref, "deadline_at": deadline, "t0": time,
    })
    state["next_task"] = {
        "task_id": NEW, "session": "shared_scene_motion_product",
        "prerequisites": packet["prerequisites"], "expected_resource": packet["expected_resource"],
        "stop_condition": "Twelve-hour bounded V5 product or explicit terminal blockers.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [
        {"task_id": OLD, "attempt": 1, "status": "CANCELLED", "created_at": time,
         "message": "User superseded V4 with shared product plan; HuRo evidence preserved.", "result": old_ref},
        {"task_id": NEW, "attempt": 0, "status": "PENDING", "created_at": time,
         "message": "Registered twelve-hour four-lane shared Human-to-Robot baseline V1.", "task_packet": packet_ref},
    ])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5_ROUTABLE",
        "plan_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5",
        "execution_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5",
        "status": "PASS",
        "supersedes_index": artifact_ref(NEW_REG / "PREDECESSOR_INDEX.json"),
        "task_packets": [{
            "task_id": NEW, "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True,
            "weights": "PER_LANE_PINNED_OR_ABSENT",
        }],
        "claim_limit": "Only V5 shared product route is executable; algorithm quality remains receipt-bound.",
    }
    for lane in LANES:
        lane_dir = NEW_ATT / "lanes" / lane
        lane_dir.mkdir(parents=True)
        atomic_json(lane_dir / "STATE.json", {
            "schema_version": "chaoyang-v5-lane-state-v1", "task_id": NEW,
            "lane": lane, "status": "PENDING", "updated_at": time,
            "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 3},
            "quality_pass": False, "control_ground_truth": False,
        })
    atomic_json(NEW_REG / "RESULT.json", {
        "schema_version": "chaoyang-v5-registration-v1", "task_id": NEW,
        "status": "REGISTERED", "registered_at": time, "deadline_at": deadline,
        "predecessor_terminal": old_ref, "task_packet": packet_ref,
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_VISUAL_DELIVERY_V5_REGISTERED",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=successor,
    )
    print(json.dumps({"status": "REGISTERED", "task_id": NEW,
                      "revision": published["governance_revision"], "deadline_at": deadline}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
