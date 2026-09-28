#!/usr/bin/env python3
"""Run one bounded V4 takeover input/route audit and release the sole writer.

This is the maintained first execution step, not a GPU algorithm or quality pass.
It keeps the V2 task pending for one frozen candidate per eligible lane.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, process_identity, publish_bundle,
)


TASK = "four_stream_full_pipeline_v4_takeover_v2"
ATT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OLD_ATT = REPO_ROOT / "_run/current/four_stream_full_pipeline_v4_recovery_v1/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PROCESSED = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916/cleaned/playing_cards")
SESSIONS = ("play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101")
LANES = ("exact78", "controller_manus", "hawor_retarget", "huro")


def claim(expected: int) -> None:
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != expected:
        raise RuntimeError("governance CAS mismatch")
    index = load_json(INDEX)
    rows = index.get("task_packets", [])
    if len(rows) != 1 or rows[0].get("task_id") != TASK or rows[0].get("execution_allowed") is not True:
        raise RuntimeError("takeover is not the exclusive executable route")
    state = load_json(TASK_STATE_PATH)
    row = next(x for x in state["tasks"] if x.get("task_id") == TASK)
    if row.get("status") != "PENDING" or row.get("pid") is not None:
        raise RuntimeError("takeover is not unowned PENDING")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("takeover is not current")
    ident = process_identity(os.getpid())
    time = now_iso()
    row.update(status="RUNNING", phase="V2_ROUTE_AND_INPUT_AUDIT", attempt=1,
               session="four_lane_takeover", pid=os.getpid(), proc_start_ticks=ident["start_ticks"],
               updated_at=time, heartbeat_at=time)
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "RUNNING", "created_at": time,
        "message": "Maintained V2 route/input audit claimed by sole writer.",
        "pid": os.getpid(), "proc_start_ticks": ident["start_ticks"],
    }])[-100:]
    publish_bundle(load_json(AUTHORITY_PATH), state,
                   event_type="FOUR_STREAM_V4_TAKEOVER_V2_AUDIT_CLAIMED",
                   expected_revision=expected, generator_path=Path(__file__))


def audit() -> dict:
    if ATT.exists():
        raise RuntimeError("takeover attempt already exists; do not overwrite")
    ATT.mkdir(parents=True)
    packet_path = REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json"
    code_path = Path(__file__).resolve()
    old_exact = OLD_ATT / "lanes/exact78/pair_production_0001/RESULT.json"
    exact_result = load_json(old_exact)
    if exact_result.get("status") != "BLOCKED_INPUTS" or exact_result.get("training_started") is not False:
        raise RuntimeError("exact78 predecessor result changed")
    session_rows = []
    for session in SESSIONS:
        root = PROCESSED / session / "preprocess/all_data"
        records = sorted(root.glob("*/training_data.json"))
        session_rows.append({
            "session_id": session, "root": str(root), "record_count": len(records),
            "first_record": artifact_ref(records[0]) if records else None,
            "last_record": artifact_ref(records[-1]) if records else None,
            "present": bool(records),
        })
    ai1_ready = all(row["present"] for row in session_rows)
    ai1_source = REPO_ROOT / "src/chaoyang/ops/run_wiyh_ai1_static_wrist_candidate_v32.py"
    ai1_producer = REPO_ROOT / "src/chaoyang/ops/build_wiyh_wrist_dual_input_v1.py"
    ai1_experiment_candidates = [
        REPO_ROOT / "_run/current/world_in_your_hands_hawor_scale_adaptive_blind_v44",
        REPO_ROOT / "tasks/control/runs/world_in_your_hands_hawor_scale_adaptive_blind_v44",
    ]
    ai1_experiment_present = any(x.exists() for x in ai1_experiment_candidates)
    hawor_worker = REPO_ROOT / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"
    hawor_npz = [
        REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai2/hawor_full007_roi_c2/get_potato_chips_0915_007/HAWOR_CAMERA_SOURCE_V3.npz",
        REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz",
    ]
    hawor_refs = [artifact_ref(path) if path.is_file() else None for path in hawor_npz]
    huro_worker = REPO_ROOT / "src/chaoyang/ops/run_huro_common_review_v2.py"
    old_ai1_probe = OLD_ATT / "lanes/controller_manus/AI1_INPUT_PREFLIGHT.json"
    result = {
        "schema_version": "chaoyang-v4-takeover-route-input-audit-v1",
        "task_id": TASK, "generated_at": now_iso(),
        "source_archive_or_processed_mutated": False,
        "per_session_image_domain_required": True,
        "lanes": {
            "exact78": {
                "status": "BLOCKED_PREREQ", "reason_codes": exact_result["blockers"],
                "pair_count": 0, "training_started": False, "predecessor_result": artifact_ref(old_exact),
                "next_step": "Register a legal same-session causal Raw/Robotized candidate index or keep blocked; no Raw copy fallback.",
            },
            "controller_manus": {
                "status": "INPUT_PRESENT_PRODUCER_UNVERIFIED" if ai1_ready else "BLOCKED_PREREQ",
                "old_false_blocker": "MISSING_PROCESSED_0916_SESSIONS",
                "correction": "WRONG_PROCESSED_ROOT_USED",
                "old_probe": artifact_ref(old_ai1_probe),
                "session_rows": session_rows,
                "candidate_op": artifact_ref(ai1_source), "input_builder": artifact_ref(ai1_producer),
                "experiment_root_candidates_checked": [str(x) for x in ai1_experiment_candidates],
                "experiment_root_found_in_checked_locations": ai1_experiment_present,
                "next_step": "Locate and SHA-bind exact HaWoR experiment inputs; then run fixed M0/M1 without fitting regression session.",
            },
            "hawor_retarget": {
                "status": "BLOCKED_TARGET_PRODUCER",
                "v3_read_only_input_refs": hawor_refs,
                "old_worker": artifact_ref(hawor_worker),
                "reason_code": "FIXED_W0_WORKER_NOT_007_031_TARGET_PRODUCER",
                "next_step": "Build/register target-specific producer or verify exact pinned V3 input as development-only; no V4 quality claim.",
            },
            "huro": {
                "status": "BLOCKED_TARGET_RUNNER",
                "old_worker": artifact_ref(huro_worker),
                "reason_code": "V2_OUTPUT_ROOT_AND_FIXTURE_PATH_NOT_REAL_SESSION_V4_CANARY",
                "next_step": "Register shared-input real-session wrist-objective canary with same Robot target and numeric gates.",
            },
        },
        "authority": "DEVELOPMENT_ONLY_NON_CONTROL_NON_DEPLOYABLE",
        "quality_pass": False, "training_complete": False, "control_ground_truth": False,
        "claim_limit": "Input and producer routing audit; does not evaluate Mask, Robot, HuRo, M0/M1 quality or training.",
    }
    signature = {
        "schema_version": "chaoyang-v4-takeover-run-signature-v1", "task_id": TASK,
        "task_packet": artifact_ref(packet_path), "code": artifact_ref(code_path),
        "predecessor_terminal": artifact_ref(OLD_ATT / "RECOVERY_RUNTIME_TERMINAL_20260922.json"),
        "old_exact78_result": artifact_ref(old_exact), "old_ai1_probe": artifact_ref(old_ai1_probe),
        "image_domain_policy": "PER_SESSION_EXPLICIT_NO_GLOBAL_SOURCE_INDEX",
        "weights": "ABSENT_FOR_ROUTE_AUDIT", "calibration": "ABSENT_FOR_ROUTE_AUDIT",
        "schema": "chaoyang-v4-takeover-route-input-audit-v1",
        "control_ground_truth": False,
    }
    atomic_json(ATT / "RUN_SIGNATURE.json", signature)
    atomic_json(ATT / "ROUTE_AND_INPUT_AUDIT.json", result)
    for lane in LANES:
        root = ATT / "lanes" / lane
        root.mkdir(parents=True)
        atomic_json(root / "STATE.json", {
            "schema_version": "chaoyang-v4-takeover-lane-state-v1", "task_id": TASK,
            "lane": lane, "status": result["lanes"][lane]["status"],
            "route_audit": artifact_ref(ATT / "ROUTE_AND_INPUT_AUDIT.json"),
            "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 2},
            "quality_pass": False, "control_ground_truth": False,
        })
    return result


def release() -> int:
    receipt = load_json(RECEIPT_PATH)
    state = load_json(TASK_STATE_PATH)
    row = next(x for x in state["tasks"] if x.get("task_id") == TASK)
    ident = process_identity(os.getpid())
    if row.get("status") != "RUNNING" or row.get("pid") != os.getpid() or row.get("proc_start_ticks") != ident["start_ticks"]:
        raise RuntimeError("writer identity changed before release")
    time = now_iso()
    row.update(status="PENDING", phase="V2_FROZEN_CANDIDATE_SELECTION", pid=None,
               proc_start_ticks=None, heartbeat_at=None, updated_at=time,
               route_audit=artifact_ref(ATT / "ROUTE_AND_INPUT_AUDIT.json"))
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PENDING", "created_at": time,
        "message": "Corrected route/input audit complete; sole writer released for bounded lane candidates.",
        "result": artifact_ref(ATT / "ROUTE_AND_INPUT_AUDIT.json"),
    }])[-100:]
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="FOUR_STREAM_V4_TAKEOVER_V2_INPUT_AUDIT_PUBLISHED",
                               expected_revision=int(receipt["governance_revision"]),
                               generator_path=Path(__file__))
    return int(published["governance_revision"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    claim(args.expected_revision)
    result = audit()
    revision = release()
    print(json.dumps({"status": "AUDIT_PUBLISHED_TASK_PENDING", "revision": revision,
                      "lane_status": {k: v["status"] for k, v in result["lanes"].items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
