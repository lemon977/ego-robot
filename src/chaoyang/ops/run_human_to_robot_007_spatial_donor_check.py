"""Restrict frozen 007 table donors to interpolation inside matched fit-point hulls."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_007_device_donor_probe import fit_heldout_homography, table_mask
from chaoyang.ops.run_human_to_robot_007_late_donor_probe import (
    HUMAN, RAW, ROLE, TARGETS, source_masks, target_input,
)

TASK = "human_to_robot_007_spatial_donor_check_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/spatial_donor_check_v1"
PRIOR = REPO_ROOT / "_run/current/human_to_robot_007_late_donor_probe_20260923/attempts/attempt_0001/lanes/scene/late_donor_probe_v1/RESULT.json"
PAIRS = {192: (115, 135, 170, 175, 281, 282, 336, 363), 196: (48, 51, 197, 377)}


def matched_fit_inliers(donor: np.ndarray, target: np.ndarray, donor_table: np.ndarray,
                        target_table: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay the frozen matcher, returning only RANSAC fit inliers for spatial support."""
    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.008)
    source_points, source_desc = sift.detectAndCompute(cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY), donor_table)
    target_points, target_desc = sift.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), target_table)
    if source_desc is None or target_desc is None:
        raise RuntimeError("FROZEN_GEOMETRY_DESCRIPTORS_MISSING")
    good = [first for first, second in cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_desc, target_desc, k=2)
            if first.distance < 0.7 * second.distance]
    good.sort(key=lambda match: (match.queryIdx, match.trainIdx))
    fit = [match for i, match in enumerate(good) if i % 4 != 0]
    if len(good) < 40 or len(fit) < 30:
        raise RuntimeError("FROZEN_GEOMETRY_MATCHES_LOST")
    source_xy = np.float32([source_points[match.queryIdx].pt for match in fit])
    target_xy = np.float32([target_points[match.trainIdx].pt for match in fit])
    matrix, flags = cv2.findHomography(source_xy, target_xy, cv2.RANSAC, 3.0)
    if matrix is None or flags is None or not np.isfinite(matrix).all():
        raise RuntimeError("FROZEN_GEOMETRY_HOMOGRAPHY_LOST")
    inliers = flags.ravel().astype(bool)
    if int(inliers.sum()) < 25:
        raise RuntimeError("FROZEN_GEOMETRY_INLIERS_LOST")
    return matrix, source_xy[inliers], target_xy[inliers]


def hull_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("INVALID_SPATIAL_INLIERS")
    polygon = cv2.convexHull(np.rint(points).astype(np.int32)).reshape(-1, 2)
    mask = np.zeros(shape, dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(mask, polygon, 255)
    return mask > 0


def spatial_support(matrix: np.ndarray, source_points: np.ndarray, target_points: np.ndarray,
                    donor_table: np.ndarray, eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_hull = hull_mask(source_points, eligible.shape)
    target_hull = hull_mask(target_points, eligible.shape)
    warped_hull = cv2.warpPerspective(source_hull.astype(np.uint8), matrix,
                                       (eligible.shape[1], eligible.shape[0]),
                                       flags=cv2.INTER_NEAREST) > 0
    warped_table = cv2.warpPerspective(donor_table, matrix,
                                       (eligible.shape[1], eligible.shape[0]),
                                       flags=cv2.INTER_NEAREST) > 0
    supported = eligible & target_hull & warped_hull & warped_table
    return supported, target_hull, warped_hull


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    previous = load_json(PRIOR)
    if previous.get("full_cohort_evaluated") is not True or previous.get("targets") != list(TARGETS):
        raise RuntimeError("PRIOR_COHORT_NOT_COMPLETE")
    for frame, donor_frames in PAIRS.items():
        observed = tuple(row["donor_frame"] for row in previous["rows"]
                         if row["targets"][str(frame)]["eligible_table_pixels"] > 0)
        if observed != donor_frames:
            raise RuntimeError(f"FROZEN_DONOR_PAIR_DRIFT:{frame}")
    cv2.setNumThreads(2)
    inputs = {frame: target_input(frame) for frame in TARGETS}
    counts = {frame: np.zeros((960, 1280), dtype=np.uint8) for frame in TARGETS}
    DEST.mkdir(parents=True)
    pair_rows = []
    for frame, donor_frames in PAIRS.items():
        target = inputs[frame]
        for donor_frame in donor_frames:
            donor, human, role = source_masks(donor_frame)
            donor_table = table_mask(donor, (human > 0) | (role > 0), donor_frame=donor_frame)
            expected, metric = fit_heldout_homography(donor, target["raw"], donor_table, target["table"])
            if expected is None or not metric["geometry_pass"]:
                raise RuntimeError(f"PRIOR_GEOMETRY_NOT_REPRODUCIBLE:{frame}:{donor_frame}")
            matrix, source_points, target_points = matched_fit_inliers(donor, target["raw"], donor_table, target["table"])
            normalized = lambda value: value / value[2, 2]
            if not np.allclose(normalized(matrix), normalized(expected), rtol=1e-6, atol=1e-6):
                raise RuntimeError(f"FIT_MATRIX_NOT_REPRODUCIBLE:{frame}:{donor_frame}")
            support, target_hull, warped_source_hull = spatial_support(
                expected, source_points, target_points, donor_table, target["eligible"])
            counts[frame][support] += 1
            support_path = DEST / f"SUPPORT_{frame}_{donor_frame}.png"
            if not cv2.imwrite(str(support_path), support.astype(np.uint8) * 255):
                raise RuntimeError("SUPPORT_WRITE_FAILED")
            old = next(row["targets"][str(frame)] for row in previous["rows"] if row["donor_frame"] == donor_frame)
            pair_rows.append({"target_frame": frame, "donor_frame": donor_frame,
                              "fit_inliers": int(len(source_points)),
                              "target_hull_pixels": int(target_hull.sum()),
                              "warped_source_hull_pixels": int(warped_source_hull.sum()),
                              "old_unrestricted_eligible_pixels": old["eligible_table_pixels"],
                              "spatially_supported_eligible_pixels": int(support.sum()),
                              "support": artifact_ref(support_path),
                              "donor_raw": artifact_ref(RAW / f"{donor_frame:06d}.png"),
                              "donor_human": artifact_ref(HUMAN / f"{donor_frame:06d}.png"),
                              "donor_role": artifact_ref(ROLE / f"{donor_frame:06d}.png")})
    summaries = []
    for frame in TARGETS:
        count_path = DEST / f"SPATIAL_COUNT_{frame}.png"
        if not cv2.imwrite(str(count_path), counts[frame]):
            raise RuntimeError(f"COUNT_WRITE_FAILED:{frame}")
        eligible = inputs[frame]["eligible"]
        summaries.append({"target_frame": frame, "eligible_write_pixels": int(eligible.sum()),
                          "zero_spatial_donor_pixels": int(np.count_nonzero(eligible & (counts[frame] == 0))),
                          "one_or_more_spatial_donor_pixels": int(np.count_nonzero(eligible & (counts[frame] >= 1))),
                          "three_or_more_spatial_donor_pixels": int(np.count_nonzero(eligible & (counts[frame] >= 3))),
                          "previous_three_or_more_unrestricted_pixels": next(
                              row["three_or_more_donor_pixels"] for row in previous["summaries"]
                              if row["target_frame"] == frame),
                          "count": artifact_ref(count_path), "inputs": inputs[frame]["refs"]})
    output = {"schema_version": "HUMAN_TO_ROBOT_007_SPATIAL_DONOR_CHECK_V1", "task_id": TASK,
              "session_id": "get_potato_chips_0915_007", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "prior": artifact_ref(PRIOR), "pairs": pair_rows, "summaries": summaries,
              "execution": "EXECUTED", "structure": "PASS", "quality": "SOURCE_INTERPOLATION_DIAGNOSTIC_ONLY",
              "adoption": "NOT_ADOPTED", "claim_limit": "Fit-inlier convex hull interpolation support, not hidden background truth, color fidelity, Clean or product quality.",
              "training_eligible": False, "control_ground_truth": False,
              "physical_deployable": False, "external_metric_authority": False}
    with (DEST / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"task_id": TASK, "summaries": summaries}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
