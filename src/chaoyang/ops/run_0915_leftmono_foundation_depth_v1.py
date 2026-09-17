#!/usr/bin/env python3
"""Held-out-validated FoundationStereo depth for the 0915 physical-left replay.

The same-session rotation is estimated on ten frames and evaluated on ten
disjoint frames.  Lowe-ratio matches are further restricted by a fixed
Sampson-distance test before the unchanged median/P90 rectification gates are
applied.  The encoded pair has negative physical-left disparity, so both eyes
are horizontally flipped for inference and the result is flipped back.  This
preserves physical-left reference pixels without swapping the eyes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
PICO_PATH = PROJECT / "vendor/FoundationStereo/scripts/pico_stereo_depth.py"
WORKER_PATH = PROJECT / "src/chaoyang/ops/run_exact78_foundationstereo_corrected_depth_worker.py"
CHECKPOINT = PROJECT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
CHECKPOINT_SHA256 = "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def sampson_distance(fundamental: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_h = np.column_stack((left, np.ones(len(left))))
    right_h = np.column_stack((right, np.ones(len(right))))
    f_left = left_h @ fundamental.T
    ft_right = right_h @ fundamental
    numerator = np.sum(right_h * f_left, axis=1) ** 2
    denominator = f_left[:, 0] ** 2 + f_left[:, 1] ** 2 + ft_right[:, 0] ** 2 + ft_right[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def calibrate(
    stereo: Path,
    pico: Any,
    eyes: list[Any],
    eye_width: int,
    source_indices: tuple[int, int],
    frame_count: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, dict[str, Any]]:
    initial_k = pico.virtual_intrinsics(1280, 960, 90.0)
    base_maps = [pico.make_map(eye, np.eye(3), 1280, 960, initial_k) for eye in eyes]
    indices = np.linspace(10, frame_count - 11, 20, dtype=int)
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    capture = cv2.VideoCapture(str(stereo))
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, sbs = capture.read()
            if not ok:
                raise RuntimeError(f"stereo decode failed at calibration frame {frame_index}")
            left, right = pico.remap_pair(sbs, eye_width, base_maps, source_indices=source_indices)
            points_left, points_right = pico.feature_matches(left, right, max_features=8000)
            rows.append((int(frame_index), points_left, points_right))
    finally:
        capture.release()
    train_left = np.concatenate([left for index, (_, left, _) in enumerate(rows) if index % 2 == 0])
    train_right = np.concatenate([right for index, (_, _, right) in enumerate(rows) if index % 2 == 0])
    essential, inlier_mask = cv2.findEssentialMat(
        train_left,
        train_right,
        initial_k,
        method=cv2.USAC_MAGSAC,
        prob=0.9999,
        threshold=0.5,
    )
    if essential is None:
        raise RuntimeError("held-out rectification essential matrix failed")
    if essential.shape[0] > 3:
        essential = essential[:3]
    inliers, rotation, translation, _ = cv2.recoverPose(
        essential, train_left, train_right, initial_k, mask=inlier_mask
    )
    # The estimator uses only half of the 20 frozen frames; the disjoint ten
    # held-out frames below are the actual acceptance gate.  Fifty MAGSAC
    # inliers exceeds the legacy minimum of thirty without consuming holdout.
    if inliers < 50:
        raise RuntimeError(f"insufficient robust calibration inliers: {inliers}")
    # Translation sign is not used in map construction, but freeze the
    # conventional negative-x projection sign for positive disparity magnitude.
    translation = translation.reshape(3)
    if translation[0] > 0:
        projection_translation = -translation
    else:
        projection_translation = translation
    left_rotation, right_rotation, left_projection, right_projection, q, _, _ = cv2.stereoRectify(
        initial_k,
        None,
        initial_k,
        None,
        (1280, 960),
        rotation,
        projection_translation.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
        newImageSize=(1280, 960),
    )
    rectified_k = left_projection[:3, :3]
    inverse_k = np.linalg.inv(initial_k)
    fundamental = inverse_k.T @ essential @ inverse_k
    heldout_rows = []
    all_vertical: list[float] = []
    qualifying_frames = 0
    for index, (frame_index, points_left, points_right) in enumerate(rows):
        if index % 2 == 0:
            continue
        robust = sampson_distance(fundamental, points_left, points_right) <= 1.0
        rect_left = cv2.undistortPoints(points_left[:, None], initial_k, None, R=left_rotation, P=left_projection).reshape(-1, 2)
        rect_right = cv2.undistortPoints(points_right[:, None], initial_k, None, R=right_rotation, P=right_projection).reshape(-1, 2)
        vertical = np.abs(rect_left[:, 1] - rect_right[:, 1])[robust]
        all_vertical.extend(vertical.tolist())
        if len(vertical) >= 15:
            qualifying_frames += 1
        heldout_rows.append({
            "frame": frame_index,
            "ratio_matches": int(len(points_left)),
            "robust_matches": int(len(vertical)),
            "median_vertical_error_px": float(np.median(vertical)) if len(vertical) else None,
            "p90_vertical_error_px": float(np.quantile(vertical, 0.9)) if len(vertical) else None,
        })
    vertical_array = np.asarray(all_vertical, np.float64)
    quality = {
        "design": "10_ESTIMATION_FRAMES_PLUS_10_DISJOINT_HELDOUT_FRAMES",
        "estimation_indices": [row[0] for index, row in enumerate(rows) if index % 2 == 0],
        "heldout_indices": [row[0] for index, row in enumerate(rows) if index % 2 == 1],
        "train_ratio_matches": int(len(train_left)),
        "train_magsac_inliers": int(inliers),
        "heldout_robust_matches": int(len(vertical_array)),
        "heldout_frames_with_at_least_15_matches": qualifying_frames,
        "heldout_median_vertical_error_px": float(np.median(vertical_array)) if len(vertical_array) else None,
        "heldout_p90_vertical_error_px": float(np.quantile(vertical_array, 0.9)) if len(vertical_array) else None,
        "sampson_threshold_px": 1.0,
        "unchanged_gate": {"minimum_matches": 200, "minimum_qualifying_frames": 8, "median_max_px": 2.0, "p90_max_px": 5.0},
        "frames": heldout_rows,
    }
    quality["passed"] = bool(
        quality["heldout_robust_matches"] >= 200
        and qualifying_frames >= 8
        and quality["heldout_median_vertical_error_px"] <= 2.0
        and quality["heldout_p90_vertical_error_px"] <= 5.0
    )
    calibration = {
        "schema_version": "0915-leftmono-heldout-rectification-v1",
        "right_from_left_rotation": rotation.tolist(),
        "recover_pose_translation_unit": translation.tolist(),
        "projection_translation_unit": projection_translation.tolist(),
        "rectification_rotation_left": left_rotation.tolist(),
        "rectification_rotation_right": right_rotation.tolist(),
        "rectified_intrinsics": rectified_k.tolist(),
        "projection_left": left_projection.tolist(),
        "projection_right": right_projection.tolist(),
        "Q_unit_baseline": q.tolist(),
        "quality": quality,
    }
    maps = [
        pico.make_map(eyes[0], left_rotation, 1280, 960, rectified_k),
        pico.make_map(eyes[1], right_rotation, 1280, 960, rectified_k),
    ]
    return maps, rectified_k, calibration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--leftmono-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    session = args.session_root.resolve(strict=True)
    leftmono = args.leftmono_video.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output required: {output}")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("FoundationStereo checkpoint SHA drift")
    output.mkdir(parents=True)
    (output / "frames").mkdir()
    pico = load_module(PICO_PATH, "pico_0915_leftmono_depth")
    worker = load_module(WORKER_PATH, "worker_0915_leftmono_depth")
    camera = session / "camera_params.json"
    stereo_candidates = sorted((session / "source_stereo").glob("CameraRecord_*_stereo.mp4"))
    if len(stereo_candidates) != 1:
        raise RuntimeError("one same-session stereo input required")
    stereo = stereo_candidates[0]
    eyes, eye_width, _, baseline = pico.load_camera_params(camera)
    source_indices = pico.load_source_indices(camera)
    capture = cv2.VideoCapture(str(stereo))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count != 379:
        raise RuntimeError(f"unexpected stereo frame count {frame_count}")
    maps, rectified_k, calibration = calibrate(stereo, pico, eyes, eye_width, source_indices, frame_count)
    if not calibration["quality"]["passed"]:
        raise RuntimeError(f"held-out rectification gate failed: {calibration['quality']}")
    (output / "CALIBRATION.json").write_text(json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    left_rotation = np.asarray(calibration["rectification_rotation_left"], np.float64)
    primary_k = pico.virtual_intrinsics(1280, 960, 90.0)
    full_rectified_to_primary = primary_k @ left_rotation.T @ np.linalg.inv(rectified_k)
    full_rectified_to_primary /= full_rectified_to_primary[2, 2]
    half_to_full = np.asarray([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    depth_to_primary = full_rectified_to_primary @ half_to_full
    depth_to_primary /= depth_to_primary[2, 2]
    registration = {
        "schema_version": "0915-leftmono-depth-registration-v1",
        "H_depth_pixel_to_primary_leftmono_pixel": depth_to_primary.tolist(),
        "H_primary_leftmono_pixel_to_depth_pixel": np.linalg.inv(depth_to_primary).tolist(),
        "T_rectified_left_camera_to_primary_left_camera": np.block([[left_rotation.T, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]).tolist(),
        "primary_intrinsics": primary_k.tolist(),
        "depth_intrinsics": (rectified_k * np.asarray([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])).tolist(),
    }
    # Restore the homogeneous row exactly after the elementwise scale above.
    depth_k = rectified_k.copy()
    depth_k[:2] *= 0.5
    registration["depth_intrinsics"] = depth_k.tolist()
    (output / "REGISTRATION.json").write_text(json.dumps(registration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = worker.Model()
    stereo_capture = cv2.VideoCapture(str(stereo))
    rgb_capture = cv2.VideoCapture(str(leftmono))
    review = output / "FOUNDATIONSTEREO_PHYSICAL_LEFT_DEPTH_REVIEW.mp4"
    writer = cv2.VideoWriter(str(review), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    frame_rows = []
    started = time.monotonic()
    try:
        for frame_index in range(frame_count):
            ok, sbs = stereo_capture.read()
            ok_rgb, primary = rgb_capture.read()
            if not ok or not ok_rgb:
                raise RuntimeError(f"decode ended at frame {frame_index}")
            left, right = pico.remap_pair(sbs, eye_width, maps, source_indices=source_indices)
            # Both-eye flip converts the encoded negative left-reference
            # disparity to the positive convention expected by the model.
            disparity_flipped, _, _ = model.infer(cv2.flip(left, 1), cv2.flip(right, 1), baseline)
            disparity = cv2.flip(disparity_flipped, 1)
            x = np.arange(disparity.shape[1], dtype=np.float32)[None]
            # In the unflipped grid, the corresponding right pixel is x+d.
            valid = np.isfinite(disparity) & (disparity > 0.25) & ((x + disparity) < disparity.shape[1])
            depth = np.full(disparity.shape, np.nan, np.float32)
            depth[valid] = np.float32(depth_k[0, 0] * baseline) / disparity[valid]
            valid &= (depth >= 0.10) & (depth <= 3.0)
            depth[~valid] = np.nan
            np.savez_compressed(
                output / "frames" / f"{frame_index:06d}.npz",
                frame_id=np.asarray(frame_index, np.int32),
                disparity_magnitude_px=disparity.astype(np.float32),
                depth_m=depth,
                valid=valid,
                scaled_intrinsics=depth_k,
                depth_reference=np.asarray("PHYSICAL_LEFT_RECTIFIED_UNFLIPPED"),
            )
            values = depth[valid]
            frame_rows.append({
                "frame": frame_index,
                "valid_pixels": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "depth_p50_m": float(np.median(values)) if len(values) else None,
            })
            scalar = np.zeros(depth.shape, np.uint8)
            scalar[valid] = np.rint(255 * (1 - np.clip((depth[valid] - 0.1) / 2.9, 0, 1))).astype(np.uint8)
            color = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO)
            color[~valid] = 0
            panel = np.concatenate((cv2.resize(primary, (640, 480)), color), axis=1)
            cv2.rectangle(panel, (0, 0), (1279, 52), (0, 0, 0), -1)
            cv2.putText(panel, f"f{frame_index:03d} PHYSICAL LEFT RGB", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel, "FoundationStereo optical-Z | stereo only here", (650, 31), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(panel)
            if frame_index % 30 == 0:
                print(json.dumps({"stage": "depth", "frame": frame_index, "frames": frame_count, "valid_pixels": int(valid.sum())}), flush=True)
    finally:
        stereo_capture.release()
        rgb_capture.release()
        writer.release()
    valid_counts = np.asarray([row["valid_pixels"] for row in frame_rows], np.int64)
    passed = bool(np.median(valid_counts) >= 5_000 and np.count_nonzero(valid_counts >= 5_000) >= int(0.90 * frame_count))
    result = {
        "schema_version": "0915-leftmono-foundationstereo-depth-v1",
        "status": "PASS_DEVELOPMENT_DEPTH" if passed else "FAILED_QUALITY_C",
        "session_id": "get_potato_chips_0915_001",
        "frame_count": frame_count,
        "physical_eye_source_indices": {"left": source_indices[0], "right": source_indices[1]},
        "depth_reference": "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z",
        "inference_orientation": "FLIP_BOTH_EYES_THEN_FLIP_RESULT_BACK_NO_EYE_SWAP",
        "stereo_usage": "DEPTH_ONLY",
        "baseline_m": baseline,
        "depth_formula": "Z=scaled_rectified_fx*baseline/positive_disparity_magnitude",
        "rectification_quality": calibration["quality"],
        "depth_quality": {
            "median_valid_pixels": int(np.median(valid_counts)),
            "min_valid_pixels": int(valid_counts.min()),
            "frames_at_least_5000_valid_pixels": int(np.count_nonzero(valid_counts >= 5_000)),
        },
        "inputs": {"camera_params": ref(camera), "source_stereo": ref(stereo), "primary_leftmono": ref(leftmono), "checkpoint": ref(CHECKPOINT)},
        "artifacts": {"calibration": ref(output / "CALIBRATION.json"), "registration": ref(output / "REGISTRATION.json"), "review": ref(review)},
        "frames": frame_rows,
        "wall_seconds": time.monotonic() - started,
        "claim_limit": "Same-session stereo visible-surface optical-Z engineering estimate; no external metric-accuracy, occluded-surface, anatomical-depth, contact or deployment truth.",
    }
    (output / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "depth_quality": result["depth_quality"], "wall_seconds": result["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
