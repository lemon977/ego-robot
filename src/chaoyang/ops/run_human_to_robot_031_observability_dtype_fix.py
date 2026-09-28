"""Correct a uint16-mask decoding error in the historical 031 region ledger."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json

TASK = "human_to_robot_031_observability_dtype_fix_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/motion/observability_dtype_fix_v1"
V3 = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
MASK_ROOT = V3 / "lanes/exact78/masks031_right_epoch7_v1/play_cards_0915_031"
MANIFEST = MASK_ROOT / "MASK_MANIFEST.json"
SPEC = V3 / "lanes/exact78/MASK031_RIGHT_FULL_SPEC.json"
DOMAIN = V3 / "lanes/exact78/prepare_full_v1/play_cards_0915_031/DOMAIN_MANIFEST.json"
MOTION = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/"
                      "attempts/attempt_0001/lanes/lane2_motion/recovered_031_wave0/HAND_MOTION_V1.npz")
OLD = REPO_ROOT / ("_run/current/human_to_robot_root_cause_gated_r2_20260923/"
                   "attempts/attempt_0001/lanes/lane2_motion/observability_031_wave1/RESULT.json")


def read_region(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (960, 1280) or image.dtype != np.uint16:
        raise RuntimeError(f"MASK_DOMAIN_OR_DTYPE_INVALID:{path}")
    if image.max() > 1:
        raise RuntimeError(f"MASK_LABEL_UNEXPECTED:{path}")
    return image == 1


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    manifest, spec, domain, old = (load_json(p) for p in (MANIFEST, SPEC, DOMAIN, OLD))
    if (manifest.get("session_id") != "play_cards_0915_031" or manifest.get("frame_count") != 149
            or manifest.get("mask_root") != str(MASK_ROOT) or domain.get("frame_count") != 149
            or domain.get("source_index") != 1 or old.get("region_visible_frames") != 0):
        raise RuntimeError("FROZEN_INPUT_OR_OLD_RECEIPT_DRIFT")
    prompt = spec["sessions"][0]["prompts"][0]
    if (prompt.get("role") != "human" or prompt.get("text") != "hand"
            or prompt.get("anchor_frame") != 80 or prompt.get("boxes_xyxy") != [[610, 790, 845, 960]]):
        raise RuntimeError("SAM_PROMPT_DRIFT")
    with np.load(MOTION, allow_pickle=False) as arrays:
        predicted = np.asarray(arrays["predicted_valid"], dtype=bool)
        roi = np.asarray(arrays["roi_valid"], dtype=bool)
    if predicted.shape != (149, 2) or roi.shape != (149, 2):
        raise RuntimeError("MOTION_VALIDITY_DOMAIN_DRIFT")
    cv2.setNumThreads(2)
    DEST.mkdir(parents=True)
    video = DEST / "031_REGION_PROXY_FULL_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 960))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_NOT_OPEN")
    rows = []
    try:
        for frame in range(149):
            mask_path = MASK_ROOT / "human" / f"{frame:06d}.png"
            raw_path = Path(domain["frames"][frame]["rgb"])
            mask = read_region(mask_path)
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None or raw.shape != (960, 1280, 3):
                raise RuntimeError(f"RAW_DOMAIN_INVALID:{frame}")
            rendered = raw.copy()
            rendered[mask] = np.rint(raw[mask].astype(np.float32) * 0.55
                                      + np.array([0, 0, 255], np.float32) * 0.45).astype(np.uint8)
            cv2.putText(rendered, f"031 frame {frame:03d} | SAM region pixels {int(mask.sum())} | side UNKNOWN",
                        (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            writer.write(rendered)
            rows.append({"source_frame_id": frame, "region_visible": bool(mask.any()),
                         "region_area_px": int(mask.sum()),
                         "touches_boundary": bool(mask[0].any() or mask[-1].any()
                                                  or mask[:, 0].any() or mask[:, -1].any()),
                         "model_roi_left_right": roi[frame].astype(int).tolist(),
                         "model_prediction_left_right": predicted[frame].astype(int).tolist(),
                         "old_region_visible": bool(old["rows"][frame]["hand_region_visible"]),
                         "mask": artifact_ref(mask_path), "raw": artifact_ref(raw_path)})
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    while True:
        okay, image = cap.read()
        if not okay:
            break
        if image.shape != (960, 1280, 3):
            raise RuntimeError("VIDEO_FRAME_DOMAIN_INVALID")
        decoded += 1
    cap.release()
    if decoded != 149:
        raise RuntimeError(f"VIDEO_FRAME_COUNT_INVALID:{decoded}")
    visible = sum(row["region_visible"] for row in rows)
    old_visible = sum(row["old_region_visible"] for row in rows)
    result = {"schema_version": "HUMAN_TO_ROBOT_031_REGION_DTYPE_CORRECTION_V1",
              "task_id": TASK, "session_id": "play_cards_0915_031",
              "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "frame_count": 149, "mask_dtype": "uint16", "foreground_value": 1,
              "old_region_visible_frames": old_visible, "corrected_region_visible_frames": visible,
              "corrected_region_unknown_frames": 149 - visible,
              "region_visible_model_prediction_any_side": sum(r["region_visible"] and any(r["model_prediction_left_right"]) for r in rows),
              "region_visible_no_model_prediction_any_side": sum(r["region_visible"] and not any(r["model_prediction_left_right"]) for r in rows),
              "predicted_left_right": predicted.sum(axis=0).astype(int).tolist(),
              "roi_left_right": roi.sum(axis=0).astype(int).tolist(),
              "rows": rows,
              "inputs": {"manifest": artifact_ref(MANIFEST), "spec": artifact_ref(SPEC),
                         "domain": artifact_ref(DOMAIN), "motion": artifact_ref(MOTION),
                         "old_observability": artifact_ref(OLD)},
              "review_video": {**artifact_ref(video), "decoded_frames": decoded},
              "execution": "EXECUTED", "structure": "PASS", "quality": "REGION_ONLY_NOT_POSE_QUALITY",
              "adoption": "NOT_ADOPTED", "old_result_modified": False,
              "claim_limit": "SAM region proxy is HaWoR ROI upstream, not independent pose or side truth; matching 102 frames cannot validate HaWoR accuracy.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    with (DEST / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"task_id": TASK, "old_visible": old_visible, "corrected_visible": visible,
                      "decoded_video_frames": decoded}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
