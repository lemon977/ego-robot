"""Check whether frozen 031 HaWoR joint0 is actually visible in its source image."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_source_coverage_diagnostic import TASK, OUT, V3

DEST = OUT / "lanes/motion_product/source_projection_031_v1"
HAWOR = V3 / "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
ROI = V3 / "lanes/ai2/full031_inputs_epoch7_c3/play_cards_0915_031_ROIS.npz"
HEIGHT, WIDTH = 960, 1280


def inside_image(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    if uv.shape[-1] != 2:
        raise ValueError("UV_LAST_DIM_MUST_BE_TWO")
    return np.isfinite(uv).all(axis=-1) & (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK or not index["task_packets"][0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    with np.load(HAWOR, allow_pickle=False) as hawor, np.load(ROI, allow_pickle=False) as roi:
        uv = np.asarray(hawor["joints_2d"][1], dtype=np.float64)
        valid = np.asarray(hawor["predicted_valid"][1], dtype=bool)
        ids = np.asarray(hawor["original_frame_indices"], dtype=np.int64)
        boxes = np.asarray(roi["boxes_xyxy"][:, 1], dtype=np.float64)
        intrinsics = np.asarray(hawor["intrinsics"], dtype=np.float64)
    if uv.shape != (149, 21, 2) or valid.sum() != 102 or boxes.shape != (149, 4) or intrinsics.shape != (149, 3, 3):
        raise ValueError("FROZEN_SOURCE_SHAPE_CHANGED")
    if not np.array_equal(ids, np.arange(149)):
        raise ValueError("FRAME_MAP_CHANGED")
    inside = inside_image(uv, WIDTH, HEIGHT)
    visible_wrist = valid & inside[:, 0]
    visible_any = valid & inside.any(axis=1)
    rows = []
    for frame in np.flatnonzero(valid):
        rows.append({
            "frame_id": int(ids[frame]), "roi_xyxy": boxes[frame].tolist(),
            "roi_width_px": float(boxes[frame, 2] - boxes[frame, 0]),
            "roi_height_px": float(boxes[frame, 3] - boxes[frame, 1]),
            "joint0_uv": uv[frame, 0].tolist(),
            "joint0_inside_image": bool(inside[frame, 0]),
            "projected_joints_inside_image": int(inside[frame].sum()),
        })
    result = {
        "schema_version": "HUMAN_TO_ROBOT_031_PROJECTED_WRIST_AUTHORITY_V1", "task_id": TASK,
        "session_id": "play_cards_0915_031", "execution": "EXECUTED", "structure": "PASS",
        "quality": "NO_PROJECTED_WRIST_IN_IMAGE", "adoption": "NOT_ADOPTED",
        "image_domain": "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY",
        "image_width": WIDTH, "image_height": HEIGHT,
        "timeline_frames": 149, "model_prediction_frames": 102,
        "projected_wrist_inside_image_frames": int(visible_wrist.sum()),
        "any_projected_joint_inside_image_frames": int(visible_any.sum()),
        "frame47_projected_joint_count_inside_image": int(inside[47].sum()),
        "frame47_roi_height_px": float(boxes[47, 3] - boxes[47, 1]),
        "frame47_joint0_y_px": float(uv[47, 0, 1]),
        "source_inference_preserved": True, "old_target_valid_preserved": True,
        "rows": rows, "inputs": {"hawor": artifact_ref(HAWOR), "roi": artifact_ref(ROI)},
        "interpretation": "HaWoR produced 102 model poses from non-direct ROIs, but projected wrist/joint0 is outside the 1280x960 image for all 102. The first pose at frame47 has no projected joint inside the image and a 12-pixel-high bottom-edge ROI. This does not prove the model's 3D result is false, but it cannot serve as an independently image-observed wrist target or establish IK failure against a trustworthy wrist center.",
        "claim_limit": "No replacement wrist labels, no new IK, no external metric truth, and no deletion of the 102-frame model-output denominator.",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    DEST.mkdir(parents=True)
    path = DEST / "RESULT.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SOURCE_PROJECTION_AUDITED", "result": str(path),
                      "wrist_inside": int(visible_wrist.sum()), "model_predictions": int(valid.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
