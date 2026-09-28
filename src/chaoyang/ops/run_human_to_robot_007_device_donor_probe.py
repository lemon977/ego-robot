"""Bounded 007 device-mask and same-recording planar donor feasibility probe."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json

TASK = "human_to_robot_007_device_donor_probe_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/device_donor_probe_v1"
RAW = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw"
HUMAN = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/masks_007_recovered_join_v1/human"
ROLE = REPO_ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1/capture_device"
REBOUND = REPO_ROOT / "_run/current/human_to_robot_source_coverage_recovery_20260923/attempts/attempt_0001/lanes/scene/rebound_007_window_v2"
DONORS = (0, 100, 250, 377)
TARGET = 184
# Frozen on the raw frame before evaluating masks; AI visual proxy, not pixel GT.
COMPLAINT_POINTS = {
    "left_index_device": (209, 539),
    "left_middle_device": (276, 576),
    "left_thumb_device": (444, 766),
    "right_index_device": (657, 328),
    "right_middle_device": (703, 329),
    "right_ring_device": (749, 389),
    "left_white_line": (139, 466),
}


def table_mask(image: np.ndarray, exclude: np.ndarray, *, donor_frame: int | None = None) -> np.ndarray:
    if image.shape != (960, 1280, 3) or exclude.shape != (960, 1280):
        raise ValueError("TABLE_MASK_DOMAIN")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 100) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 160))
    if donor_frame == 0:
        mask[:530] = False  # upper room/chairs are not the task table
    elif donor_frame is not None:
        mask[:35] = False
    mask &= ~cv2.dilate(exclude.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
    return mask.astype(np.uint8) * 255


def fit_heldout_homography(
    donor: np.ndarray, target: np.ndarray, donor_table: np.ndarray, target_table: np.ndarray
) -> tuple[np.ndarray | None, dict]:
    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.008)
    da, xa = sift.detectAndCompute(cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY), donor_table)
    db, xb = sift.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), target_table)
    summary = {"donor_keypoints": len(da), "target_keypoints": len(db),
               "ratio_matches": 0, "fit_matches": 0, "holdout_matches": 0,
               "fit_inliers": 0, "holdout_inliers": 0, "holdout_median_px": None,
               "holdout_p95_px": None, "geometry_pass": False}
    if xa is None or xb is None:
        return None, summary
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(xa, xb, k=2)
    good = [a for a, b in pairs if a.distance < 0.7 * b.distance]
    good.sort(key=lambda m: (m.queryIdx, m.trainIdx))
    summary["ratio_matches"] = len(good)
    if len(good) < 40:
        return None, summary
    holdout = good[::4]
    fit = [m for i, m in enumerate(good) if i % 4 != 0]
    source = np.float32([da[m.queryIdx].pt for m in fit])
    dest = np.float32([db[m.trainIdx].pt for m in fit])
    h, inliers = cv2.findHomography(source, dest, cv2.RANSAC, 3.0)
    summary["fit_matches"] = len(fit)
    summary["holdout_matches"] = len(holdout)
    if h is None or not np.isfinite(h).all():
        return None, summary
    summary["fit_inliers"] = int(inliers.sum())
    val_source = np.float32([da[m.queryIdx].pt for m in holdout]).reshape(-1, 1, 2)
    val_dest = np.float32([db[m.trainIdx].pt for m in holdout])
    projected = cv2.perspectiveTransform(val_source, h).reshape(-1, 2)
    error = np.linalg.norm(projected - val_dest, axis=1)
    valid = error <= 5.0
    summary["holdout_inliers"] = int(valid.sum())
    if valid.any():
        summary["holdout_median_px"] = float(np.median(error[valid]))
        summary["holdout_p95_px"] = float(np.percentile(error[valid], 95))
    summary["geometry_pass"] = bool(
        summary["fit_inliers"] >= 25 and summary["holdout_inliers"] >= 8
        and summary["holdout_median_px"] is not None
        and summary["holdout_median_px"] <= 2.0
        and summary["holdout_p95_px"] <= 4.0
    )
    return h if summary["geometry_pass"] else None, summary


def _image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK or not index["task_packets"][0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    raw_path = RAW / f"{TARGET:06d}.png"
    raw = _image(raw_path)
    write_path = REBOUND / "write/000003.png"
    write = _image(write_path, cv2.IMREAD_GRAYSCALE) > 0
    role_path = ROLE / f"{TARGET:06d}.png"
    device_role = _image(role_path, cv2.IMREAD_GRAYSCALE) > 0
    if raw.shape != (960, 1280, 3) or write.shape != (960, 1280) or device_role.shape != write.shape:
        raise ValueError("TARGET_DOMAIN_MISMATCH")
    points = []
    for name, (x, y) in COMPLAINT_POINTS.items():
        points.append({"name": name, "x": x, "y": y,
                       "raw_bgr": raw[y, x].astype(int).tolist(),
                       "rebound_write": bool(write[y, x]),
                       "old_capture_device_role": bool(device_role[y, x])})
    target_table = table_mask(raw, write | device_role)
    rows = []
    winners = []
    for frame in DONORS:
        donor_path = RAW / f"{frame:06d}.png"
        donor = _image(donor_path)
        human_path = HUMAN / f"{frame:06d}.png"
        donor_human = _image(human_path, cv2.IMREAD_GRAYSCALE) > 0
        donor_device_path = ROLE / f"{frame:06d}.png"
        donor_device = _image(donor_device_path, cv2.IMREAD_GRAYSCALE) > 0
        donor_table = table_mask(donor, donor_human | donor_device, donor_frame=frame)
        h, metric = fit_heldout_homography(donor, raw, donor_table, target_table)
        row = {"donor_frame": frame, "raw": artifact_ref(donor_path),
               "human": artifact_ref(human_path), "device_role": artifact_ref(donor_device_path), **metric}
        if h is not None:
            warped_mask = cv2.warpPerspective(donor_table, h, (1280, 960), flags=cv2.INTER_NEAREST) > 0
            support = warped_mask & write
            row["masked_target_pixels_with_table_donor"] = int(support.sum())
            row["masked_target_pixels_total"] = int(write.sum())
            if support.any():
                winners.append((int(support.sum()), frame, h, donor, warped_mask))
        rows.append(row)
    DEST.mkdir(parents=True)
    overlay = raw.copy()
    overlay[write] = (overlay[write] * .55 + np.array([0, 0, 255]) * .45).astype(np.uint8)
    for item in points:
        cv2.circle(overlay, (item["x"], item["y"]), 7,
                   (0, 255, 0) if item["rebound_write"] else (0, 0, 255), 2)
        cv2.putText(overlay, item["name"], (item["x"] + 9, item["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    overlay_path = DEST / "007_FRAME184_DEVICE_COMPLAINT_AND_WRITE.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError("OVERLAY_WRITE_FAILED")
    candidate = None
    if winners:
        _, frame, h, donor, warped_mask = max(winners, key=lambda x: (x[0], -x[1]))
        warped = cv2.warpPerspective(donor, h, (1280, 960))
        support = warped_mask & write
        # Only a geometry-qualified table pixel may be replaced; plate/chips remain unknown.
        table_only = raw.copy()
        table_only[support] = warped[support]
        candidate_path = DEST / "007_FRAME184_TABLE_DONOR_DIAGNOSTIC.png"
        if not cv2.imwrite(str(candidate_path), table_only):
            raise RuntimeError("CANDIDATE_WRITE_FAILED")
        candidate = {"image": artifact_ref(candidate_path), "donor_frame": frame,
                     "replaced_pixels": int(support.sum()),
                     "claim_limit": "One-frame table-only diagnostic, not complete Clean; moving plate/chips and true background remain unverified."}
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_DEVICE_DONOR_PROBE_V1", "task_id": TASK,
        "session_id": "get_potato_chips_0915_007", "target_frame": TARGET,
        "fixed_window": [181, 196], "donor_frames": list(DONORS),
        "execution": "EXECUTED", "structure": "PASS", "quality": "PROBE_ONLY_NOT_CLEAN_QUALITY",
        "adoption": "NOT_ADOPTED", "complaint_points_authority": "AI_VISUAL_PROXY_NOT_PIXEL_GT",
        "complaint_points": points, "rows": rows, "raw": artifact_ref(raw_path),
        "rebound_write": artifact_ref(write_path), "device_role": artifact_ref(role_path),
        "overlay": artifact_ref(overlay_path), "table_only_candidate": candidate,
        "claim_limit": "Frozen same-recording donor and device-support probe only; no full Clean or product quality promotion.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PROBE_EXECUTED", "result": str(DEST / "RESULT.json"),
                      "geometry_pass_donors": [row["donor_frame"] for row in rows if row["geometry_pass"]],
                      "complaint_points_outside_write": [p["name"] for p in points if not p["rebound_write"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
