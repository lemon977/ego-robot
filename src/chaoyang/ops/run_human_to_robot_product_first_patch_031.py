"""Call the observed-centre object patch sampler on frozen 031 frames 66–81."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.pipeline.object_patch_sample_v1 import sample_object_patch
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK

OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/object_patch_031/window_v1"
HAWOR = ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
DEPTH = ROOT / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/depth_full_v1/play_cards_0915_031/frames"
MASKS = ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_031_v1/task_object"
RAW = ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/play_cards_0915_031/raw"
TIPS = (4, 8, 12, 16, 20)


def _save(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    with np.load(HAWOR, allow_pickle=False) as source:
        joints = np.asarray(source["joints_2d"])
        predicted_valid = np.asarray(source["predicted_valid"], bool)
    rows = []
    reason_counts: dict[str, int] = {}
    for frame in range(66, 82):
        mask = cv2.imread(str(MASKS / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape != (960, 1280):
            raise ValueError(f"MASK_DOMAIN:{frame}")
        with np.load(DEPTH / f"{frame:06d}.npz", allow_pickle=False) as archive:
            depth = np.asarray(archive["depth_m"], np.float64)
            valid = np.asarray(archive["valid"], bool) & np.asarray(archive["lr_consistent"], bool)
            k = np.asarray(archive["physical_left_intrinsics"], np.float64)
        for finger, joint in enumerate(TIPS):
            row = {"source_frame_id": frame, "side": 1, "finger_index": finger,
                   "qualification_checked": True, "strict_contact_admissible": False}
            uv = joints[1, frame, joint]
            if not predicted_valid[1, frame] or not np.isfinite(uv).all():
                sampled = {"valid": False, "reason": "NO_MODEL_TIP"}
            elif not np.any(mask > 0):
                sampled = {"valid": False, "reason": "NO_OBJECT_INSTANCE"}
            else:
                pixels = np.argwhere(mask > 0)
                delta = pixels.astype(np.float64) - np.asarray([uv[1], uv[0]])
                index_nearest = int(np.argmin(np.sum(delta * delta, axis=1)))
                y, x = pixels[index_nearest]
                distance = float(np.linalg.norm(delta[index_nearest]))
                sampled = ({"valid": False, "reason": "OBJECT_TOO_FAR", "pixel_distance": distance}
                           if distance > 40.0 else
                           sample_object_patch(mask, depth, valid, k, (int(x), int(y))))
                sampled["model_tip_to_object_pixel_px"] = distance
            row.update(sampled)
            rows.append(row)
            reason_counts[row["reason"]] = reason_counts.get(row["reason"], 0) + 1
        raw = cv2.imread(str(RAW / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2] != mask.shape:
            raise ValueError(f"RAW_DOMAIN:{frame}")
        overlay = raw.copy()
        overlay[mask > 0] = (overlay[mask > 0].astype(float) * .6 +
                              np.array([0, 200, 200]) * .4).astype(np.uint8)
        for row in rows[-5:]:
            if row["valid"]:
                x, y = row["source_xy"]
                cv2.circle(overlay, (x, y), 6, (0, 0, 255), 2)
                cv2.putText(overlay, str(row["object_id"]), (x + 7, y),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        if not cv2.imwrite(str(OUT / f"overlay_{frame:06d}.png"), overlay):
            raise RuntimeError("OVERLAY_WRITE")
    checked = len(rows)
    screened = sum(bool(row["valid"]) for row in rows)
    result = {"schema_version": "HUMAN_TO_ROBOT_OBJECT_PATCH_031_V1", "task_id": TASK,
              "session_id": "play_cards_0915_031", "source_frames": list(range(66, 82)),
              "counts": {"unit": "finger_side_frame", "qualification_checked": checked,
                         "geometry_screened": screened, "strict_contact_admissible": 0,
                         "r1_executed_windows": 0, "reasons": reason_counts},
              "rows": rows, "inputs": {"hawor": artifact_ref(HAWOR),
                                        "depth_result": artifact_ref(DEPTH.parent / "RESULT.json"),
                                        "mask_manifest": artifact_ref(MASKS.parent / "MASK_MANIFEST.json")},
              "depth_semantics": "ACTUAL_CENTER_DEPTH;NEIGHBORHOOD_ONLY_SAME_INSTANCE_CONTINUITY",
              "quality": "MODEL_CONDITIONED_GEOMETRY_NOT_CONTACT_TRUTH",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False,
              "claim_limit": "Actual 031 fixed-window object patch sampling, not strict contact or R1."}
    _save(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["quality"], "counts": result["counts"], "result": str(OUT / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
