"""Frozen 007 mask-support and 031 target-lineage diagnostic; no new model call."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json

TASK = "human_to_robot_source_coverage_recovery_20260923"
OUT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PRIOR = REPO_ROOT / "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001"
V3 = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
R2 = REPO_ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
FRAMES = range(181, 197)
# A diagnostic screen region selected on frozen raw 181 and 184, not a hand segmentation GT.
LEFT_SCREEN_REGION_MODEL = (0, 250, 390, 720)  # x0, y0, x1, y1 in 960x720 model pixels


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def region_support(mask: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = rect
    if mask.ndim != 2 or not (0 <= x0 < x1 <= mask.shape[1] and 0 <= y0 < y1 <= mask.shape[0]):
        raise ValueError("REGION_OR_MASK_DOMAIN_INVALID")
    return int(np.count_nonzero(mask[y0:y1, x0:x1]))


def target_step_mm(position: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if position.ndim != 2 or position.shape[1] != 3 or valid.shape != position.shape[:1]:
        raise ValueError("TARGET_AXIS_MISMATCH")
    steps = np.full(len(position), np.nan)
    for frame in range(1, len(position)):
        if valid[frame] and valid[frame - 1]:
            steps[frame] = 1000.0 * np.linalg.norm(position[frame] - position[frame - 1])
    return steps


def scene() -> dict:
    root = PRIOR / "lanes/scene/cable_007"
    prepared = root / "window_v1"
    clean = root / "clean_window_v1"
    old_roles = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1"
    output = OUT / "lanes/scene/source_coverage_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = []
    masks = []
    selected = {181, 184, 190, 196}
    panels = []
    for local, frame in enumerate(FRAMES):
        mask_path = prepared / "model_masks" / f"{local:06d}.png"
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        human_path = old_roles / "human_forearm" / f"{frame:06d}.png"
        human = cv2.imread(str(human_path), cv2.IMREAD_UNCHANGED)
        if mask_image is None or human is None or mask_image.shape != (720, 960) or human.shape != (960, 1280):
            raise ValueError(f"MASK_DOMAIN_OR_INPUT_MISSING:{frame}")
        mask = mask_image > 0
        masks.append(mask)
        human_small = cv2.resize((human > 0).astype(np.uint8), (960, 720), interpolation=cv2.INTER_NEAREST) > 0
        x0, y0, x1, y1 = LEFT_SCREEN_REGION_MODEL
        model_left = region_support(mask, LEFT_SCREEN_REGION_MODEL)
        role_left = region_support(human_small, LEFT_SCREEN_REGION_MODEL)
        rows.append({
            "frame_id": frame, "model_mask_pixels": int(mask.sum()),
            "left_screen_model_mask_pixels": model_left,
            "left_screen_existing_human_role_pixels": role_left,
            "left_screen_region_pixels": (x1 - x0) * (y1 - y0),
            "model_mask": artifact_ref(mask_path), "human_role": artifact_ref(human_path),
            "left_screen_support_absent": model_left == 0,
        })
        if frame in selected:
            raw_path = prepared / "frames" / f"{local:06d}.png"
            new_path = clean / "clean" / f"{local:06d}.png"
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            new = cv2.imread(str(new_path), cv2.IMREAD_COLOR)
            if raw is None or new is None:
                raise FileNotFoundError(frame)
            new = cv2.resize(new, (960, 720), interpolation=cv2.INTER_AREA)
            overlay = raw.copy()
            overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
            for image, title in ((raw, "RAW"), (overlay, "MODEL MASK + FIXED SCREEN ROI"), (new, "REJECTED CLEAN")):
                cv2.putText(image, f"{frame} {title}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            panels.append(np.concatenate((raw, overlay, new), axis=1))
    stack = np.stack(masks)
    always_masked = np.all(stack, axis=0)
    for local, row in enumerate(rows):
        row["no_same_pixel_donor_fraction_within_mask"] = float(np.count_nonzero(always_masked & stack[local]) / np.count_nonzero(stack[local]))
    montage = output / "007_SOURCE_COVERAGE_181_184_190_196.png"
    if not cv2.imwrite(str(montage), np.concatenate(panels, axis=0)):
        raise RuntimeError("MONTAGE_WRITE_FAILED")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_SOURCE_COVERAGE_V1", "task_id": TASK,
        "session_id": "get_potato_chips_0915_007", "source_frames": list(FRAMES),
        "execution": "EXECUTED", "structure": "PASS", "quality": "DIAGNOSTIC_SUPPORT_GAP_CONFIRMED",
        "adoption": "NOT_ADOPTED", "left_screen_region_model_xyxy": list(LEFT_SCREEN_REGION_MODEL),
        "left_screen_region_authority": "FROZEN_RAW_VISUAL_PROXY_NOT_PIXEL_GT",
        "frames_left_screen_model_mask_absent": [row["frame_id"] for row in rows if row["left_screen_support_absent"]],
        "always_masked_pixels_across_window": int(always_masked.sum()),
        "any_masked_pixels_across_window": int(np.any(stack, axis=0).sum()),
        "montage": artifact_ref(montage), "rows": rows,
        "interpretation": "A nonempty human_forearm role and zero missing *role* pixels do not imply all visible hand pixels were masked. Frame 184 raw visibly contains a left hand while its fixed left screen region has zero model-mask pixels. This is input-support failure before any ProPainter quality claim; same-signature retry is invalid.",
        "claim_limit": "Frozen ROI is a screen proxy, not full hand segmentation or product acceptance; no new model call.",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    _save(output / "RESULT.json", result)
    return {"result": artifact_ref(output / "RESULT.json"), "montage": artifact_ref(montage),
            "frames_left_screen_model_mask_absent": result["frames_left_screen_model_mask_absent"]}


def motion() -> dict:
    output = OUT / "lanes/motion_product/target_source_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    hawor_path = V3 / "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
    robot_path = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
    roi_path = V3 / "lanes/ai2/full031_inputs_epoch7_c3/play_cards_0915_031_ROIS.npz"
    with np.load(hawor_path, allow_pickle=False) as h, np.load(robot_path, allow_pickle=False) as r, np.load(roi_path, allow_pickle=False) as roi:
        joints = np.asarray(h["joints_3d_camera"], dtype=np.float64)
        predicted = np.asarray(h["predicted_valid"], dtype=bool)
        observed = np.asarray(h["observed"], dtype=bool)
        inferred = np.asarray(h["inferred"], dtype=bool)
        target = np.asarray(r["T_target_root_cam"], dtype=np.float64)
        valid = np.asarray(r["target_valid"], dtype=bool)
        times = np.asarray(h["timestamp_ns"], dtype=np.int64)
        direct = np.asarray(roi["det_direct"], dtype=bool)
        roi_valid = np.asarray(roi["roi_valid"], dtype=bool)
    if joints.shape != (2, 149, 21, 3) or valid.shape != (149, 2):
        raise ValueError("SOURCE_SHAPE_INVALID")
    if predicted.sum(axis=1).tolist() != [0, 102] or observed.sum(axis=1).tolist() != [0, 0] or inferred.sum(axis=1).tolist() != [0, 102]:
        raise ValueError("FROZEN_SOURCE_COUNTS_CHANGED")
    if not np.array_equal(predicted.T, valid):
        raise ValueError("SOURCE_TARGET_VALIDITY_MISMATCH")
    if not np.array_equal(observed, predicted & direct.T) or not np.array_equal(inferred, predicted & ~direct.T):
        raise ValueError("SOURCE_FLAG_WRITER_SEMANTICS_CHANGED")
    if not np.array_equal(roi_valid.T & predicted, predicted):
        raise ValueError("PREDICTION_WITHOUT_ROI")
    if not np.array_equal(joints[1, valid[:, 1], 0], target[valid[:, 1], 1, :3, 3]):
        raise ValueError("JOINT0_TARGET_HANDOFF_CHANGED")
    xyz = target[:, 1, :3, 3]
    steps = target_step_mm(xyz, valid[:, 1])
    rows = []
    for frame in range(45, 52):
        dt = (times[frame] - times[frame - 1]) / 1e9 if frame else None
        rows.append({
            "frame_id": frame, "timestamp_ns": int(times[frame]), "target_valid": bool(valid[frame, 1]),
            "roi_valid": bool(roi_valid[frame, 1]), "roi_det_direct": bool(direct[frame, 1]),
            "source_observed": bool(observed[1, frame]), "source_inferred": bool(inferred[1, frame]),
            "target_xyz_m": xyz[frame].tolist() if valid[frame, 1] else None,
            "step_from_previous_mm": float(steps[frame]) if np.isfinite(steps[frame]) else None,
            "dt_s": float(dt) if dt is not None else None,
        })
    result = {
        "schema_version": "HUMAN_TO_ROBOT_031_TARGET_SOURCE_V1", "task_id": TASK,
        "session_id": "play_cards_0915_031", "execution": "EXECUTED", "structure": "PASS",
        "quality": "TARGET_TEMPORAL_DISCONTINUITY", "adoption": "NOT_ADOPTED",
        "timeline_frames": 149, "right_predicted_frames": 102, "left_predicted_frames": 0,
        "right_direct_roi_prediction_frames": 0, "right_nondirect_roi_prediction_frames": 102,
        "frame_47_to_48_step_mm": float(steps[48]),
        "source_flag_semantics": "inferred = predicted_valid AND NOT det_direct; still a HaWoR model inference on an ROI, not proof of motion-infiller generation or absent RGB support",
        "target_semantics": "HaWoR MANO joint0 passes unchanged into Robot wrist target position; not external 3D truth",
        "rows_45_51": rows,
        "inputs": {"hawor": artifact_ref(hawor_path), "robot": artifact_ref(robot_path), "roi": artifact_ref(roi_path)},
        "interpretation": "The first valid source target at 47 is a 3D position outlier inherited by Robot R0. Independent FK/render mismatch is not the cause. Preserve the old 102-frame input denominator; a new source-quality sidecar must distinguish model prediction from trustworthy motion target before any new IK or product acceptance.",
        "claim_limit": "No target repair, no new IK, and no inference about physical reachability or external wrist accuracy.",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    _save(output / "RESULT.json", result)
    return {"result": artifact_ref(output / "RESULT.json"), "frame_47_to_48_step_mm": float(steps[48])}


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK or not index["task_packets"][0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if (OUT / "DIAGNOSTIC_RESULT.json").exists():
        raise FileExistsError(OUT / "DIAGNOSTIC_RESULT.json")
    summary = {"task_id": TASK, "scene": scene(), "motion_product": motion(),
               "product_quality_promoted": False, "new_model_calls": 0, "new_ik_calls": 0}
    _save(OUT / "DIAGNOSTIC_RESULT.json", summary)
    print(json.dumps({"status": "DIAGNOSTIC_EXECUTED", "result": str(OUT / "DIAGNOSTIC_RESULT.json"),
                      "scene_absent_frames": summary["scene"]["frames_left_screen_model_mask_absent"],
                      "source_jump_mm": summary["motion_product"]["frame_47_to_48_step_mm"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
