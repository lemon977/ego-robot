#!/usr/bin/env python3
"""Freeze the four visual inputs for the sole V5 product route.

Usage: chaoyang run run_four_stream_v5_setup --expected-revision N
This is an input audit, not a Mask, Clean, Robot, or quality result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "four_stream_visual_delivery_v5"
ROUTE = "HUMAN_TO_ROBOT_BASELINE_V1"
ATT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
BASE = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78"
SESSIONS = (
    ("get_potato_chips_0915_007", "prepare_full_v1", 378, "chips"),
    ("play_cards_0915_031", "prepare_full_v1", 149, "poker"),
    ("get_potato_chips_0902_103", "exact_domain_v1", 284, "chips"),
    ("play_cards_0902_042", "exact_domain_v1", 171, "poker"),
)


def build_manifest() -> dict:
    rows = []
    for sid, group, count, task in SESSIONS:
        path = BASE / group / sid / "DOMAIN_MANIFEST.json"
        domain = load_json(path)
        frames = domain.get("frames", [])
        if domain.get("session_id") != sid or domain.get("frame_count") != count or len(frames) != count:
            raise RuntimeError(f"domain frame count mismatch: {sid}")
        ids = [f.get("frame_id") for f in frames]
        if ids != list(range(count)):
            raise RuntimeError(f"non-contiguous presentation frame IDs: {sid}")
        for frame in frames:
            if not Path(frame["rgb"]).is_file():
                raise RuntimeError(f"missing frozen RGB: {sid}/{frame['frame_id']}")
        first_ts = frames[0].get("capture_time", {}).get("original_time_fields", {}).get("ts")
        last_ts = frames[-1].get("capture_time", {}).get("original_time_fields", {}).get("ts")
        if not isinstance(first_ts, int) or not isinstance(last_ts, int) or last_ts <= first_ts:
            raise RuntimeError(f"missing or invalid source timestamps: {sid}")
        rows.append({
            "session_id": sid, "task": task, "frame_count": count,
            "domain_manifest": artifact_ref(path),
            "image_domain": domain["image_domain"],
            "source_index": domain["source_index"],
            "first_source_timestamp_ns": first_ts,
            "last_source_timestamp_ns": last_ts,
            "camera_source": domain.get("source_camera"),
            "consumer_rule": "EXACT_DOMAIN_FRAME_AND_CAMERA_BINDING_ONLY",
        })
    return {
        "schema_version": "chaoyang-human-to-robot-route-v1",
        "route_id": ROUTE,
        "task_id": TASK,
        "created_at": now_iso(),
        "status": "FROZEN_INPUT_BINDING_ONLY",
        "sessions": rows,
        "stage_edges": [
            "Raw->Scene->Clean->Compositor",
            "Raw->Depth/Object/ContactHints",
            "Raw->HandMotion->Robot_R0->Compositor",
            "Robot_R0+VALID_GEOMETRY->OPTIONAL_R1",
        ],
        "forbidden_edges": ["Clean->Depth", "Clean->Object", "Clean->Contact", "Clean->Calibration", "Clean->IK"],
        "scene_owner": "lane_scene", "motion_owner": "lane_motion",
        "sensor_owner": "lane_sensor", "comparison_owner": "lane_huro",
        "output_authority": {
            "mode": "OFFLINE_VISUAL", "training_eligible": False,
            "control_ground_truth": False, "physical_deployable": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    receipt = load_json(RECEIPT_PATH)
    if int(receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    index = load_json(INDEX)
    rows = index.get("task_packets", [])
    if len(rows) != 1 or rows[0].get("task_id") != TASK or rows[0].get("execution_allowed") is not True:
        raise RuntimeError("V5 is not the sole executable task")
    state = load_json(TASK_STATE_PATH)
    row = next(x for x in state["tasks"] if x.get("task_id") == TASK)
    if row.get("status") != "PENDING" or row.get("pid") is not None:
        raise RuntimeError("V5 not unowned PENDING")
    target = ATT / "ROUTE_MANIFEST.json"
    if target.exists():
        raise RuntimeError("immutable route manifest already exists")
    manifest = build_manifest()
    atomic_json(target, manifest)
    report = {
        "schema_version": "chaoyang-v5-route-setup-v1", "route_id": ROUTE,
        "task_id": TASK, "status": "PASSED_INPUT_AUDIT",
        "route_manifest": artifact_ref(target), "visual_source_frames": 982,
        "algorithm_quality_pass": False,
    }
    atomic_json(ATT / "SETUP_RESULT.json", report)
    row.update(updated_at=now_iso(), phase="V5_INPUTS_FROZEN_LANES_PENDING")
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PENDING", "created_at": now_iso(),
        "message": "Four visual source domains frozen; algorithm lanes not yet evaluated.",
        "result": artifact_ref(ATT / "SETUP_RESULT.json"),
    }])[-100:]
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_V5_INPUTS_FROZEN",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
    )
    print(json.dumps({"status": report["status"], "revision": published["governance_revision"],
                      "route_manifest": str(target)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
