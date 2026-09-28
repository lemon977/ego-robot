"""Seal three saved-array Sensor replays after fixed-sample visual inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, load_json, now_iso, publish_bundle,
)
from chaoyang.ops.run_human_to_robot_sensor_display_correction import TASK, ATTEMPT, DEST

INDEX = REPO_ROOT / "tasks/current/INDEX.json"
REVISION = "HUMAN_TO_ROBOT_SENSOR_DISPLAY_CORRECTION_20260923"


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
    path = DEST / "RESULT.json"
    result = load_json(path)
    rows = result.get("sessions", [])
    if (result.get("task_id") != TASK or result.get("new_solver_invocations") != 0
            or [row.get("session_id") for row in rows] !=
            ["play_cards_0916_097", "play_cards_0916_098", "play_cards_0916_101"]
            or [row.get("frames") for row in rows] != [165, 179, 122]):
        raise RuntimeError("SENSOR_RESULT_INCOMPLETE")
    for row in rows:
        if (row["world_pixels_per_m_x"] != row["world_pixels_per_m_z"]
                or row["local_pixels_per_m_manus"] != row["local_pixels_per_m_robot"]):
            raise RuntimeError(f"DISPLAY_SCALE_NOT_EQUAL:{row['session_id']}")
        for key in ("source_motion", "source_robot_fk", "source_video", "video"):
            ref = row[key]
            if artifact_ref(Path(ref["path"])) != {k: ref[k] for k in ("path", "bytes", "sha256")}:
                raise RuntimeError(f"SOURCE_OR_VIDEO_SHA_MISMATCH:{row['session_id']}:{key}")
        cap = cv2.VideoCapture(row["video"]["path"])
        decoded = 0
        while cap.read()[0]:
            decoded += 1
        cap.release()
        if decoded != row["frames"] or row["video"]["decoded_frames"] != decoded:
            raise RuntimeError(f"REVIEW_DECODE_MISMATCH:{row['session_id']}")
        if len(row["samples"]) != 3 or row["sample_frame_ids"] != [0, decoded // 2, decoded - 1]:
            raise RuntimeError(f"SAMPLE_FRAME_MAP_MISMATCH:{row['session_id']}")
        for ref in row["samples"]:
            frame = cv2.imread(ref["path"], cv2.IMREAD_COLOR)
            if frame is None or frame.shape != (1080, 1920, 3) or artifact_ref(Path(ref["path"])) != ref:
                raise RuntimeError(f"SAMPLE_UNREADABLE_OR_CHANGED:{row['session_id']}")
    visual_path = ATTEMPT / "AI_VISUAL_REVIEW.json"
    visual = load_json(visual_path)
    if (visual.get("task_id") != TASK or visual.get("review_authority") != "AI_REVIEW_PROXY_NOT_HUMAN_GT"
            or visual.get("quality") not in {"PASS_DISPLAY_ONLY", "REJECTED_DISPLAY"}
            or visual.get("session_video_sha256") != {row["session_id"]: row["video"]["sha256"] for row in rows}):
        raise RuntimeError("INDEPENDENT_VISUAL_REVIEW_INVALID")
    tests = subprocess.run(
        ["/usr/local/bin/python", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_human_to_robot_sensor_display_correction.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode:
        raise RuntimeError(f"TARGETED_TESTS_FAILED:{tests.stdout[-700:]}:{tests.stderr[-700:]}")
    test_path = ATTEMPT / "FINAL_TARGETED_TESTS.log"
    test_path.write_text(tests.stdout + tests.stderr, encoding="utf-8")
    validation = ATTEMPT / "FINAL_VALIDATION.json"
    atomic_json(validation, {
        "schema_version": "HUMAN_TO_ROBOT_SENSOR_DISPLAY_FINAL_VALIDATION_V1",
        "task_id": TASK, "status": "PASS_FOR_DISPLAY_STRUCTURE_NOT_SENSOR_ALIGNMENT",
        "targeted_tests": artifact_ref(test_path),
        "pytest": {"exit_code": 0, "summary": tests.stdout.strip().splitlines()[-1]},
        "model_result": artifact_ref(path), "visual_review": artifact_ref(visual_path),
        "claim_limit": "Three full saved-array videos and fixed scales checked; no measured sensor-to-RGB alignment.",
    })
    lane_path = ATTEMPT / "lanes/sensor/STATE.json"
    lane = load_json(lane_path)
    lane.update(status="TERMINAL_466_FRAME", execution="EXECUTED", structure="PASS",
                quality=visual["quality"], adoption="NOT_ADOPTED",
                evidence=[artifact_ref(path), artifact_ref(visual_path)], updated_at=now_iso())
    atomic_json(lane_path, lane)
    terminal = "PASSED" if visual["quality"] == "PASS_DISPLAY_ONLY" else "REJECTED_QUALITY"
    result_path = ATTEMPT / "RESULT.json"
    atomic_json(result_path, {
        "schema_version": "HUMAN_TO_ROBOT_SENSOR_DISPLAY_RESULT_V1",
        "task_id": TASK, "route": "HUMAN_TO_ROBOT_BASELINE_V1",
        "status": "TERMINAL_466_FRAME", "task_terminal_status": terminal,
        "terminal_at": now_iso(), "model_result": artifact_ref(path),
        "final_validation": artifact_ref(validation), "visual_review": artifact_ref(visual_path),
        "execution_summary": {"sensor_review_frames": 466, "sensor_sessions": 3,
                              "new_solver_invocations": 0, "products_structure": "4/4",
                              "products_quality": "0/4", "products_adopted": "0/4"},
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS_FOR_466_FRAME",
                         "display_quality": visual["quality"],
                         "sensor_visual_alignment": "INCONCLUSIVE_NO_INDEPENDENT_KEYPOINT_TRUTH",
                         "adoption": "NOT_ADOPTED"},
        "claim_limit": "Sensor display correction only; wrist alignment and products not adopted.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    })
    task.update(status=terminal, result=artifact_ref(result_path), updated_at=now_iso(),
                heartbeat_at=None, pid=None, proc_start_ticks=None, gpu_id=None)
    state["next_task"] = None
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": TASK, "attempt": 1, "status": terminal, "created_at": now_iso(),
        "message": "Sensor metric display correction sealed after fixed-sample visual review.",
    }])[-100:]
    successor = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": "HUMAN_TO_ROBOT_SENSOR_DISPLAY_TERMINAL",
        "plan_revision": REVISION, "execution_revision": REVISION,
        "status": "PASS_NO_ACTIVE_TASKS", "supersedes_index": artifact_ref(INDEX),
        "task_packets": [], "claim_limit": "Sensor display task terminal, no active algorithm task.",
    }
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
                               event_type="HUMAN_TO_ROBOT_SENSOR_DISPLAY_TERMINAL",
                               expected_revision=args.expected_revision,
                               generator_path=Path(__file__), task_packet_index_path=INDEX,
                               task_packet_index_value=successor)
    print(json.dumps({"status": terminal, "task_id": TASK,
                      "result": str(result_path),
                      "governance_revision": published["governance_revision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
