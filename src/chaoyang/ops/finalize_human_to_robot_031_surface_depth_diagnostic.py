"""Seal the 031 fixed-frame surface-depth diagnostic without product promotion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.ops.run_human_to_robot_031_surface_depth_diagnostic import TASK, ATTEMPT, DEST, FRAMES

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVISION = "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_DIAGNOSTIC_20260923"


def verify_result(result: dict) -> None:
    rows = result.get("rows", [])
    if (result.get("task_id") != TASK or result.get("sampled_frames") != 16
            or [row["source_frame_id"] for row in rows] != list(FRAMES)
            or result.get("quality") != "DEVELOPMENT_DIAGNOSTIC_ONLY"
            or result.get("adoption") != "NOT_ADOPTED"):
        raise RuntimeError("SURFACE_RESULT_SEMANTICS_INVALID")
    if sum(row["same_pixel_surface_samples"] for row in rows) != result.get("total_surface_samples"):
        raise RuntimeError("SURFACE_SAMPLE_DENOMINATOR_INVALID")
    retry = result.get("runtime_retry_signature")
    if not retry or artifact_ref(Path(retry["path"])) != retry:
        raise RuntimeError("RUNTIME_RETRY_SIGNATURE_INVALID")
    if artifact_ref(Path(load_json(Path(retry["path"]))["runner"]["path"])) != load_json(Path(retry["path"]))["runner"]:
        raise RuntimeError("RUNTIME_RETRY_CODE_SHA_DRIFT")
    for ref in result["inputs"].values():
        if artifact_ref(Path(ref["path"])) != ref:
            raise RuntimeError("FROZEN_INPUT_SHA_DRIFT")
    for row in rows:
        for name in ("mask", "stereo", "raw"):
            if artifact_ref(Path(row[name]["path"])) != row[name]:
                raise RuntimeError("FRAME_INPUT_SHA_DRIFT")
    video_ref = result["review_video"]
    video = Path(video_ref["path"])
    if artifact_ref(video) != {k: video_ref[k] for k in ("path", "bytes", "sha256")}:
        raise RuntimeError("VIDEO_SHA_DRIFT")
    capture = cv2.VideoCapture(str(video))
    count = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        if image.shape != (480, 1280, 3):
            raise RuntimeError("VIDEO_FRAME_DOMAIN_INVALID")
        count += 1
    capture.release()
    if count != 16 or video_ref.get("decoded_frames") != 16:
        raise RuntimeError("VIDEO_DECODE_COUNT_INVALID")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()
    if load_json(RECEIPT_PATH)["governance_revision"] != args.expected_revision:
        raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
    state = load_json(TASK_STATE_PATH)
    task = next((row for row in state["tasks"] if row.get("task_id") == TASK), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "RUNNING", "CLAIMED"}:
        raise RuntimeError("TASK_NOT_ACTIVE")
    if state.get("next_task", {}).get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_CURRENT")
    index = load_json(INDEX)
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_SOLE_INDEX_ENTRY")
    source = DEST / "RESULT.json"
    result = load_json(source)
    verify_result(result)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "TMPDIR": str(REPO_ROOT / ".cache/tmp"),
           "XDG_CACHE_HOME": str(REPO_ROOT / ".cache/xdg")}
    tests = subprocess.run(["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests/test_human_to_robot_031_surface_depth_diagnostic.py"],
                           cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False)
    if tests.returncode:
        raise RuntimeError("TARGETED_TESTS_FAILED:" + tests.stdout[-500:] + tests.stderr[-500:])
    log = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    log.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation, {"schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_FINAL_VALIDATION_V1",
                             "task_id": TASK, "status": "PASS_FOR_SAME_PIXEL_SURFACE_DIAGNOSTIC_ONLY",
                             "source_result": artifact_ref(source), "targeted_tests": artifact_ref(log),
                             "claim_limit": "No exact correspondence, anatomical wrist centre, correction or product promotion."})
    lane_path = ATTEMPT / "lanes/motion/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_DEVELOPMENT_DIAGNOSTIC", execution="EXECUTED", structure="PASS",
                quality="DEVELOPMENT_DIAGNOSTIC_ONLY", adoption="NOT_ADOPTED",
                evidence=[artifact_ref(source)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    output = ATTEMPT / "RESULT.json"
    atomic_json(output, {
        "schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_TERMINAL_V1", "task_id": TASK,
        "route": "HUMAN_TO_ROBOT_BASELINE_V1", "status": "TERMINAL_DEVELOPMENT_DIAGNOSTIC",
        "task_terminal_status": "PASSED", "terminal_at": now_iso(),
        "source_result": artifact_ref(source), "final_validation": artifact_ref(validation),
        "sampled_frames": result["sampled_frames"],
        "frames_with_surface_samples": result["frames_with_surface_samples"],
        "total_surface_samples": result["total_surface_samples"],
        "frame_median_of_signed_medians_m": result["frame_median_of_signed_medians_m"],
        "review_video": result["review_video"], "products_structure": "4/4",
        "products_quality": "0/4", "products_adopted": "0/4",
        "claim_limit": result["claim_limit"], "training_eligible": False,
        "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    })
    task.update(status="PASSED", result=artifact_ref(output), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": "PASSED", "created_at": now_iso(),
        "message": "Sealed fixed 031 same-pixel MANO/Stereo surface-Z diagnostic without product authority.",
    }])[-100:]
    successor = {"schema_version": "chaoyang-v71-task-packet-index-v3",
                 "packet_revision": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_TERMINAL",
                 "plan_revision": REVISION, "execution_revision": REVISION,
                 "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
                 "task_packets": [], "claim_limit": "031 surface-depth diagnostic terminal; no active algorithm task."}
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_031_SURFACE_DEPTH_TERMINAL",
                               expected_revision=args.expected_revision, generator_path=Path(__file__),
                               task_packet_index_path=INDEX, task_packet_index_value=successor)
    print(json.dumps({"status": "PASSED", "task_id": TASK,
                      "result": str(output), "governance_revision": published["governance_revision"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
