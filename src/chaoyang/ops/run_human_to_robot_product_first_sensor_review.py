"""Independent bounded review of the three retained Sensor full-session replays."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

OLD = ROOT / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001/lanes/sensor/review_v1/RESULT.json"
OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/sensor/review_v1"


def _save(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    old = load_json(OLD)
    if [row["frames"] for row in old["sessions"]] != [165, 179, 122]:
        raise ValueError("SENSOR_FROZEN_SESSION_DRIFT")
    OUT.mkdir(parents=True)
    rows = []
    for old_row in old["sessions"]:
        video = Path(old_row["video"]["path"])
        motion = Path(old_row["source_motion"]["path"])
        robot = Path(old_row["source_robot_fk"]["path"])
        capture = cv2.VideoCapture(str(video))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        if decoded != old_row["frames"]:
            raise ValueError(f"SENSOR_VIDEO_DECODE:{old_row['session_id']}:{decoded}")
        with np.load(motion, allow_pickle=False) as archive:
            motion_keys = list(archive.files)
        with np.load(robot, allow_pickle=False) as archive:
            robot_keys = list(archive.files)
        samples = [0, decoded // 2, decoded - 1]
        short = old_row["session_id"].split("_")[-1]
        contact_sheet = OUT.parent / f"SENSOR_SAMPLE_{short}.png"
        if not contact_sheet.is_file():
            raise FileNotFoundError(contact_sheet)
        rows.append({"session_id": old_row["session_id"], "timeline_frames": decoded,
                     "reviewed_frame_ids": samples, "reviewer": "AI_REVIEW_PROXY_NOT_HUMAN_GT",
                     "rgb_projection": "BONE_SEGMENTS_VISIBLE_AND_CLIPPED_BUT_PIXEL_ALIGNMENT_NOT_INDEPENDENTLY_VERIFIED",
                     "world_panel": "FIXED_ORIGIN_EQUAL_METRE_SCALE_VISIBLE_IN_REUSED_VIDEO",
                     "local_hand_and_grasp": "VISIBLE_IN_REUSED_VIDEO;NO_EXTERNAL_GT",
                     "common_robot_backend": "SAVED_Q_FK_REFERENCED_NOT_RESOLVED_AGAIN",
                     "source_invalid_out_of_frame_temporal_status": "AVAILABLE_IN_REUSED_CONSUMER;NOT_RELABELED_AS_SUCCESS",
                     "technical_visual_quality": "INCONCLUSIVE_NO_INDEPENDENT_PIXEL_KEYPOINT_TRUTH",
                     "user_review": "PENDING",
                     "video": artifact_ref(video), "sample_sheet": artifact_ref(contact_sheet),
                     "motion": artifact_ref(motion), "robot_fk": artifact_ref(robot),
                     "motion_fields": motion_keys, "robot_fields": robot_keys})
    result = {"schema_version": "HUMAN_TO_ROBOT_SENSOR_INDEPENDENT_REVIEW_V1",
              "task_id": TASK, "execution": "REUSED_ARRAYS_AND_VIDEO_REVIEWED", "structure": "PASS",
              "quality": "INCONCLUSIVE_VISUAL_ALIGNMENT", "adoption": "NOT_ADOPTED",
              "sample_scope": "3_EQUAL_SPACED_FRAMES_EACH_PLUS_COMPLETE_DECODE_NOT_FULL_HUMAN_FRAME_REVIEW",
              "sessions": rows, "source_result": artifact_ref(OLD),
              "claim_limit": "Three sample sheets and full decode support readable internal review only; no measured wrist alignment, absolute SLAM accuracy, calibration, or user visual adoption.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], "sessions": len(rows),
                      "frames": sum(row["timeline_frames"] for row in rows),
                      "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
