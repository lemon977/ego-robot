#!/usr/bin/env python3
"""Fail-closed V5 terminal publisher after the one-candidate quality gate.

Usage: PYTHONPATH=src python -B -m \
  chaoyang.governance.terminalize_four_stream_visual_delivery_v5 \
  --expected-revision <validated current revision>

This creates immutable product-preflight, result and receipt files, verifies
all delivered media, and CAS-closes the current task. It never promotes video
playability to Clean, Robot, contact or training quality.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)

TASK = "four_stream_visual_delivery_v5"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
SHALLOW = REPO_ROOT / "docs/current/visuals/FOUR_STREAM_V5"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
TASK_RECEIPT = REPO_ROOT / "tasks/receipts/FOUR_STREAM_VISUAL_DELIVERY_V5_RESULT.json"
SCENE_IDS = (("007", "get_potato_chips_0915_007", 378),
             ("031", "play_cards_0915_031", 149),
             ("103", "get_potato_chips_0902_103", 284),
             ("042", "play_cards_0902_042", 171))


def checked_ref(item: dict) -> dict:
    observed = artifact_ref(Path(item["path"]))
    if any(observed[key] != item[key] for key in ("path", "bytes", "sha256")):
        raise RuntimeError(f"SHA_BINDING_MISMATCH:{item['path']}")
    return observed


def media(source: dict, sid: str, shallow_name: str, count: int) -> dict:
    deep = checked_ref(source)
    target = SHALLOW / shallow_name
    if target.is_symlink() or not target.is_file():
        raise RuntimeError(f"SHALLOW_VIDEO_ABSENT_OR_SYMLINK:{target}")
    shallow = artifact_ref(target)
    if shallow["sha256"] != deep["sha256"] or shallow["bytes"] != deep["bytes"]:
        raise RuntimeError(f"SHALLOW_VIDEO_SHA_MISMATCH:{target}")
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
               "-show_entries", "stream=nb_read_frames",
               "-of", "default=noprint_wrappers=1:nokey=1", str(target)]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if process.returncode or process.stdout.strip() != str(count):
        raise RuntimeError(f"SHALLOW_FULL_DECODE_MISMATCH:{target}:{process.stdout.strip()}")
    return {"session_id": sid, "expected_frames": count,
            "decoded_frames": count, "deep": deep, "shallow": shallow}


def run_preflight(route: dict, *, dry_run: bool = False) -> dict:
    rows = []
    for row in route["sessions"]:
        sid = row["session_id"]
        checked_ref(row["camera_source"])
        input_dir = str(Path(row["camera_source"]["path"]).parent)
        command = [sys.executable, "-B", "-m", "chaoyang.cli", "run",
                   "run_human_to_robot_baseline_v1", "--input", input_dir,
                   "--motion-source", "hawor", "--output", str(ROOT / "products" / sid),
                   "--dry-run"]
        process = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                                 timeout=30, env=os.environ.copy())
        try:
            report = json.loads(process.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(f"PRODUCT_PREFLIGHT_INVALID_JSON:{sid}:{process.stdout}") from exc
        if (process.returncode != 2 or report.get("status") != "BLOCKED_PREREQ"
                or not report.get("reason", "").startswith("STAGE_BINDING_MISSING:")):
            raise RuntimeError(f"PRODUCT_PREFLIGHT_UNEXPECTED:{sid}:{process.returncode}:{report}")
        rows.append({"session_id": sid, "command": command, "returncode": process.returncode,
                     "stdout": report, "stderr_tail": process.stderr[-1000:]})
    value = {"schema_version": "chaoyang-v5-product-preflight-v1", "task_id": TASK,
             "status": "BLOCKED_PREREQ", "tested_session_count": len(rows),
             "product_video_count": 0, "rows": rows,
             "reason": "No accepted Scene Clean manifest or same-session product binding was published.",
             "control_ground_truth": False, "training_eligible": False}
    if dry_run:
        return value
    path = ROOT / "PRODUCT_PREFLIGHT.json"
    if path.exists():
        raise RuntimeError("PRODUCT_PREFLIGHT_ALREADY_EXISTS")
    atomic_json(path, value)
    return artifact_ref(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if int(load_json(RECEIPT_PATH)["governance_revision"]) != args.expected_revision:
        raise RuntimeError("governance CAS mismatch")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") != "PENDING" or task.get("pid") is not None:
        raise RuntimeError("V5_TASK_NOT_UNOWNED_PENDING")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("V5_NOT_SOLE_CURRENT_TASK")
    current_index = load_json(INDEX)
    packet_rows = current_index.get("task_packets", [])
    if len(packet_rows) != 1 or packet_rows[0].get("task_id") != TASK:
        raise RuntimeError("V5_PACKET_NOT_SOLE_CURRENT")
    if (ROOT / "RESULT.json").exists() or TASK_RECEIPT.exists():
        raise RuntimeError("IMMUTABLE_FINAL_ALREADY_EXISTS")
    for lane in ("scene", "sensor", "motion", "huro"):
        value = load_json(ROOT / "lanes" / lane / "STATE.json")
        if value.get("writer", {}).get("pid") is not None:
            raise RuntimeError(f"ACTIVE_REGISTERED_WRITER:{lane}")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") == "ACQUIRED" and str(lease.get("task_id", "")).startswith(TASK):
        raise RuntimeError("V5_GPU_LEASE_ACTIVE")

    scene_ref = artifact_ref(ROOT / "lanes/scene/SCENE_FINAL_ASSESSMENT.json")
    scene = load_json(Path(scene_ref["path"]))
    motion_ref = artifact_ref(ROOT / "lanes/motion/MOTION_HANDOFF_V5.json")
    motion = load_json(Path(motion_ref["path"]))
    sensor_ref = artifact_ref(ROOT / "lanes/sensor/run_0001/RESULT.json")
    sensor = load_json(Path(sensor_ref["path"]))
    huro_ref = artifact_ref(ROOT / "lanes/huro/HURO_TERMINAL.json")
    huro = load_json(Path(huro_ref["path"]))
    if (scene.get("status") != "FAILED_QUALITY_C" or scene.get("product_clean_allowed") is not False
            or scene.get("propainter_executed") is not False or scene.get("fresh_clean_frame_count") != 0):
        raise RuntimeError("SCENE_TERMINAL_SEMANTICS_CHANGED")
    if (motion.get("claim_limits", {}).get("numeric_quality_pass") is not False
            or len(motion.get("results", [])) != 4):
        raise RuntimeError("MOTION_TERMINAL_SEMANTICS_CHANGED")
    if len(sensor.get("sessions", [])) != 3 or sensor.get("quality_pass") is not False:
        raise RuntimeError("SENSOR_TERMINAL_SEMANTICS_CHANGED")
    if huro.get("status") != "FAILED_QUALITY_LIMITS" or huro.get("quality_pass") is not False:
        raise RuntimeError("HURO_TERMINAL_SEMANTICS_CHANGED")

    reviews = []
    issues = []
    for short, sid, count in SCENE_IDS:
        receipt = load_json(ROOT / f"lanes/scene/review_{short}_v1/RESULT.json")
        if receipt.get("session_id") != sid or receipt.get("frame_count") != count:
            raise RuntimeError(f"SCENE_REVIEW_ID_OR_COUNT:{short}")
        reviews.append(media(receipt["review_video"], sid, f"{sid}_ROLE_MASK_FAILED_REVIEW.mp4", count))
        source_issue = checked_ref(receipt["fixed_frame_sheet"])
        target = SHALLOW / f"ISSUE_{short}_DEVICE_OBJECT.jpg"
        if target.is_symlink() or artifact_ref(target)["sha256"] != source_issue["sha256"]:
            raise RuntimeError(f"ISSUE_COLLAGE_SHA:{short}")
        issues.append(artifact_ref(target))
    for item in sensor["sessions"]:
        sid, count = item["session_id"], item["frames"]
        if item["decoded_frames"] != count:
            raise RuntimeError(f"SENSOR_DECODE_RECEIPT:{sid}")
        reviews.append(media(item["video"], sid, f"{sid}_SENSOR_REVIEW.mp4", count))
    for sid, short, count in (("get_potato_chips_0902_103", "103", 284),
                              ("play_cards_0902_042", "042", 171)):
        receipt = load_json(ROOT / f"lanes/motion/review_0902_{short}_v1/RESULT.json")
        if receipt.get("session_id") != sid or receipt.get("decoded_frames") != count:
            raise RuntimeError(f"MOTION_REVIEW_RECEIPT:{sid}")
        reviews.append(media(receipt["video"], sid, f"{sid}_R0_RAW_DIAGNOSTIC.mp4", count))
    for sid, folder, count in (("get_potato_chips_0915_007", "review_get_potato_chips_0915_007", 378),
                               ("play_cards_0915_031", "review_play_cards_0915_031", 149)):
        receipt = load_json(ROOT / "lanes/huro" / folder / "RESULT.json")
        if receipt.get("session_id") != sid or receipt.get("decoded_frames") != count:
            raise RuntimeError(f"HURO_REVIEW_RECEIPT:{sid}")
        reviews.append(media(receipt["video"], sid, f"{sid}_R0_VS_HURO_RAW_DIAGNOSTIC.mp4", count))
    actual_mp4 = sorted(path.name for path in SHALLOW.glob("*.mp4"))
    if len(reviews) != 11 or len(actual_mp4) != 11 or len(issues) != 4:
        raise RuntimeError("SHALLOW_DELIVERY_COUNT_MISMATCH")
    if any((ROOT / "products" / sid / "robot.mp4").exists() for _, sid, _ in SCENE_IDS):
        raise RuntimeError("UNEXPECTED_PRODUCT_VIDEO")

    route_ref = artifact_ref(ROOT / "ROUTE_MANIFEST.json")
    route = load_json(Path(route_ref["path"]))
    if route.get("route_id") != "HUMAN_TO_ROBOT_BASELINE_V1" or len(route.get("sessions", [])) != 4:
        raise RuntimeError("ROUTE_CHANGED")
    preflight = run_preflight(route, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps({"status": "READY_TO_TERMINALIZE", "review_videos": len(reviews),
                          "issue_collages": len(issues),
                          "product_preflight": [row["stdout"] for row in preflight["rows"]]},
                         ensure_ascii=False))
        return 0
    preflight_ref = preflight
    time = now_iso()
    result = {
        "schema_version": "chaoyang-human-to-robot-baseline-v1-terminal-v1",
        "task_id": TASK, "route_id": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "FAILED_QUALITY_C", "release_status": "INCOMPLETE",
        "terminal_at": time, "reason_codes": ["SCENE_DEVICE_MASK_FAILED",
                                               "TRUSTED_VISIBLE_OBJECT_PROTECTION_UNVERIFIED",
                                               "MOTION_NUMERIC_QUALITY_FAILED",
                                               "HURO_JOINT_LIMIT_FAILED"],
        "review_videos_complete": 11, "issue_collages_complete": 4,
        "clean_frames_adopted": 0, "product_robot_videos": 0,
        "algorithm_quality_pass": False, "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "contact_authority": False,
        "scene": scene_ref, "motion": motion_ref, "sensor": sensor_ref,
        "huro": huro_ref, "route": route_ref,
        "product_preflight": preflight_ref,
        "shallow_visual_index": artifact_ref(SHALLOW / "INDEX_ZH.md"),
        "review_media": reviews, "issue_media": issues,
        "claim_limit": "Eleven full diagnostic/review videos are not four human-removed Robot products. No Clean, metric contact, causal training or control authority was obtained.",
        "next_prerequisite": "New finite task packet with independently verified wearable/cable deletion and same-frame visible-object protection; no same-signature quality retry.",
    }
    atomic_json(ROOT / "RESULT.json", result)
    result_ref = artifact_ref(ROOT / "RESULT.json")
    atomic_json(TASK_RECEIPT, {"schema_version": "chaoyang-v5-terminal-receipt-v1",
                               "task_id": TASK, "status": "FAILED_QUALITY_C",
                               "release_status": "INCOMPLETE", "result": result_ref,
                               "review_videos_complete": 11, "product_robot_videos": 0})
    task.update(status="FAILED_QUALITY_C", phase="HUMAN_TO_ROBOT_BASELINE_V1_TERMINAL",
                attempt=1, pid=None, proc_start_ticks=None, gpu_id=None,
                heartbeat_at=None, updated_at=time, result=result_ref,
                last_attempt_terminal="FAILED_QUALITY_C",
                last_attempt_reason="SCENE_CLEAN_AND_ROBOT_PRODUCT_QUALITY_GATES_UNMET")
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "FAILED_QUALITY_C",
        "created_at": time, "message": "11 review MP4 and four issue collages delivered; no accepted Clean or Robot product.",
        "result": result_ref,
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5_TERMINAL",
        "plan_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5",
        "execution_revision": "FOUR_STREAM_VISUAL_DELIVERY_V5",
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [],
        "claim_limit": "V5 is terminal quality C. New algorithm candidate requires a newly registered finite task packet.",
    }
    published = publish_bundle(
        load_json(AUTHORITY_PATH), state,
        event_type="FOUR_STREAM_VISUAL_DELIVERY_V5_TERMINAL_QUALITY_C",
        expected_revision=args.expected_revision, generator_path=Path(__file__),
        task_packet_index_path=INDEX, task_packet_index_value=successor,
    )
    print(json.dumps({"status": "FAILED_QUALITY_C", "revision": published["governance_revision"],
                      "review_videos": 11, "product_videos": 0,
                      "result": str(ROOT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
