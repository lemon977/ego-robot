#!/usr/bin/env python3
"""CPU-only stereo-domain preflight for one processed 0915 session.

This operation compares the physical-eye SBS pixels in two domains without
loading FoundationStereo:

* sourceIndex-aware, resize-only pixels (diagnostic evidence only), and
* same-session equiDis62 plus held-out visual stereo rectification.

Only the calibrated/rectified candidate can be admitted to a later GPU depth
task.  A good resize-only epipolar score is not sufficient because its
distortion state and metric pinhole projection remain unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable
import uuid

import cv2
import numpy as np

from chaoyang.ops import run_0915_leftmono_foundation_depth_v1 as depth_v1


DEFAULT_SESSION = Path(
    "/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915/"
    "cleaned/playing_cards/play_cards_0915_001"
)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 960
DEFAULT_FOV_DEG = 90.0
DEFAULT_SAMPLE_COUNT = 12

MIN_ROBUST_MATCHES = 200
MIN_MATCHES_PER_FRAME = 15
MIN_QUALIFYING_FRAME_FRACTION = 0.75
MIN_MEAN_SPATIAL_COVERAGE = 0.08
MAX_MEDIAN_VERTICAL_PX = 2.0
MAX_P90_VERTICAL_PX = 5.0
MAX_P95_VERTICAL_PX = 8.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            payload, stream, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_pico_module() -> Any:
    """Load the pinned sourceIndex/camera parser used by the depth worker."""

    return depth_v1.load_module(depth_v1.PICO_PATH, "pico_0915_stereo_preflight_v1")


def split_physical_eyes(
    sbs: np.ndarray,
    *,
    eye_width: int,
    eye_height: int,
    source_indices: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return SBS halves in physical ``(left, right)`` order."""

    if tuple(sorted(source_indices)) != (0, 1):
        raise ValueError(
            "physical-eye sourceIndex must be an exact permutation of [0, 1]"
        )
    if sbs.shape[:2] != (eye_height, eye_width * 2):
        raise ValueError(
            f"decoded SBS geometry {sbs.shape[1]}x{sbs.shape[0]} does not match "
            f"camera geometry {eye_width * 2}x{eye_height}"
        )
    halves = (sbs[:, :eye_width], sbs[:, eye_width:])
    return halves[source_indices[0]], halves[source_indices[1]]


def sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("decoded frame count must be positive")
    if sample_count <= 0:
        raise ValueError("sample count must be positive")
    count = min(frame_count, sample_count)
    return [int(value) for value in np.unique(
        np.linspace(0, frame_count - 1, count, dtype=np.int64)
    )]


def inspect_video(video: Path) -> dict[str, Any]:
    """Fully decode the video and report its observed, not assumed, geometry."""

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open SBS video: {video}")
    reported_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_count = 0
    geometry: tuple[int, int] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current = (int(frame.shape[1]), int(frame.shape[0]))
            if geometry is None:
                geometry = current
            elif current != geometry:
                raise RuntimeError(
                    f"SBS geometry changed at decoded frame {decoded_count}: {current}"
                )
            decoded_count += 1
    finally:
        capture.release()
    if decoded_count <= 0 or geometry is None:
        raise RuntimeError("SBS video decoded zero frames")
    return {
        "reported_frame_count": reported_count,
        "decoded_frame_count": decoded_count,
        "reported_equals_decoded": reported_count == decoded_count,
        "fps": fps,
        "width": geometry[0],
        "height": geometry[1],
        "full_decode_passed": reported_count == decoded_count and fps > 0.0,
    }


def _matched_points(
    left: np.ndarray, right: np.ndarray, *, max_features: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Find ratio-tested matches and retain robust fundamental-matrix inliers."""

    detector = cv2.SIFT_create(max_features)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    key_left, desc_left = detector.detectAndCompute(left_gray, None)
    key_right, desc_right = detector.detectAndCompute(right_gray, None)
    if desc_left is None or desc_right is None:
        empty = np.empty((0, 2), np.float64)
        return empty, empty
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward_pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    reverse_pairs = matcher.knnMatch(desc_right, desc_left, k=2)
    forward = [first for first, second in forward_pairs
               if first.distance < 0.70 * second.distance]
    reverse = {
        first.queryIdx: first.trainIdx
        for first, second in reverse_pairs
        if first.distance < 0.70 * second.distance
    }
    good = [item for item in forward
            if reverse.get(item.trainIdx) == item.queryIdx]
    if len(good) < 8:
        empty = np.empty((0, 2), np.float64)
        return empty, empty
    points_left = np.asarray(
        [key_left[item.queryIdx].pt for item in good], dtype=np.float64,
    )
    points_right = np.asarray(
        [key_right[item.trainIdx].pt for item in good], dtype=np.float64,
    )
    fundamental, mask = cv2.findFundamentalMat(
        points_left, points_right, cv2.USAC_MAGSAC, 1.0, 0.999,
    )
    if fundamental is None or mask is None:
        empty = np.empty((0, 2), np.float64)
        return empty, empty
    robust = np.asarray(mask).reshape(-1).astype(bool)
    return points_left[robust], points_right[robust]


def vertical_residual_summary(
    points_left: np.ndarray,
    points_right: np.ndarray,
    *,
    width: int,
    height: int,
    grid_shape: tuple[int, int] = (8, 6),
) -> dict[str, Any]:
    """Summarize vertical residual and left-image spatial support."""

    left = np.asarray(points_left, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(points_right, dtype=np.float64).reshape(-1, 2)
    if left.shape != right.shape:
        raise ValueError("left/right correspondence shapes differ")
    if width <= 0 or height <= 0:
        raise ValueError("image geometry must be positive")
    rows, columns = int(grid_shape[1]), int(grid_shape[0])
    if rows <= 0 or columns <= 0:
        raise ValueError("grid geometry must be positive")
    if not len(left):
        return {
            "robust_matches": 0,
            "median_vertical_error_px": None,
            "p90_vertical_error_px": None,
            "p95_vertical_error_px": None,
            "spatial_grid_coverage": 0.0,
        }
    residual = np.abs(left[:, 1] - right[:, 1])
    cell_x = np.clip((left[:, 0] * columns / width).astype(int), 0, columns - 1)
    cell_y = np.clip((left[:, 1] * rows / height).astype(int), 0, rows - 1)
    occupied = len(set(zip(cell_x.tolist(), cell_y.tolist())))
    return {
        "robust_matches": int(len(residual)),
        "median_vertical_error_px": float(np.median(residual)),
        "p90_vertical_error_px": float(np.quantile(residual, 0.90)),
        "p95_vertical_error_px": float(np.quantile(residual, 0.95)),
        "spatial_grid_coverage": float(occupied / (rows * columns)),
    }


def _evaluate_frame_pair(
    left: np.ndarray, right: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    points_left, points_right = _matched_points(left, right)
    summary = vertical_residual_summary(
        points_left, points_right, width=left.shape[1], height=left.shape[0],
    )
    residual = (
        np.abs(points_left[:, 1] - points_right[:, 1])
        if len(points_left) else np.empty((0,), np.float64)
    )
    return summary, residual


def aggregate_candidate(
    frame_rows: list[dict[str, Any]], residuals: list[np.ndarray],
) -> dict[str, Any]:
    all_residuals = (
        np.concatenate([row for row in residuals if len(row)])
        if any(len(row) for row in residuals) else np.empty((0,), np.float64)
    )
    qualifying = sum(
        int(row["robust_matches"]) >= MIN_MATCHES_PER_FRAME for row in frame_rows
    )
    sampled = len(frame_rows)
    quality = {
        "sampled_frames": sampled,
        "robust_matches": int(len(all_residuals)),
        "qualifying_frames": int(qualifying),
        "qualifying_frame_fraction": float(qualifying / sampled) if sampled else 0.0,
        "median_vertical_error_px": (
            float(np.median(all_residuals)) if len(all_residuals) else None
        ),
        "p90_vertical_error_px": (
            float(np.quantile(all_residuals, 0.90)) if len(all_residuals) else None
        ),
        "p95_vertical_error_px": (
            float(np.quantile(all_residuals, 0.95)) if len(all_residuals) else None
        ),
        "mean_spatial_grid_coverage": (
            float(np.mean([row["spatial_grid_coverage"] for row in frame_rows]))
            if frame_rows else 0.0
        ),
        "frames": frame_rows,
    }
    quality["gate"] = {
        "minimum_robust_matches": MIN_ROBUST_MATCHES,
        "minimum_matches_per_qualifying_frame": MIN_MATCHES_PER_FRAME,
        "minimum_qualifying_frame_fraction": MIN_QUALIFYING_FRAME_FRACTION,
        "minimum_mean_spatial_grid_coverage": MIN_MEAN_SPATIAL_COVERAGE,
        "median_vertical_error_max_px": MAX_MEDIAN_VERTICAL_PX,
        "p90_vertical_error_max_px": MAX_P90_VERTICAL_PX,
        "p95_vertical_error_max_px": MAX_P95_VERTICAL_PX,
    }
    quality["passed"] = bool(
        quality["robust_matches"] >= MIN_ROBUST_MATCHES
        and quality["qualifying_frame_fraction"] >= MIN_QUALIFYING_FRAME_FRACTION
        and quality["mean_spatial_grid_coverage"] >= MIN_MEAN_SPATIAL_COVERAGE
        and quality["median_vertical_error_px"] is not None
        and quality["median_vertical_error_px"] <= MAX_MEDIAN_VERTICAL_PX
        and quality["p90_vertical_error_px"] <= MAX_P90_VERTICAL_PX
        and quality["p95_vertical_error_px"] <= MAX_P95_VERTICAL_PX
    )
    return quality


def evaluate_candidate(
    video: Path,
    *,
    indices: list[int],
    eye_width: int,
    eye_height: int,
    source_indices: tuple[int, int],
    transform: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open SBS video: {video}")
    frame_rows: list[dict[str, Any]] = []
    residuals: list[np.ndarray] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"SBS sample decode failed at frame {index}")
            # Always validate source geometry/routing even if transform remaps.
            split_physical_eyes(
                frame, eye_width=eye_width, eye_height=eye_height,
                source_indices=source_indices,
            )
            left, right = transform(frame)
            if left.shape != right.shape:
                raise RuntimeError("candidate left/right image shapes differ")
            summary, residual = _evaluate_frame_pair(left, right)
            frame_rows.append({"frame": int(index), **summary})
            residuals.append(residual)
    finally:
        capture.release()
    return aggregate_candidate(frame_rows, residuals)


def admission_decision(
    *,
    input_valid: bool,
    source_indices: tuple[int, int] | None,
    decode_valid: bool,
    rectification_calibration_passed: bool,
    calibrated_quality: dict[str, Any] | None,
    raw_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed; raw-resize quality alone never authorizes metric depth."""

    blockers: list[str] = []
    if not input_valid:
        blockers.append("INVALID_OR_INCOMPLETE_SAME_SESSION_CALIBRATION")
    if source_indices != (1, 0):
        blockers.append("PHYSICAL_EYE_SOURCE_INDEX_MISMATCH")
    if not decode_valid:
        blockers.append("SBS_FULL_DECODE_OR_GEOMETRY_FAILED")
    if not rectification_calibration_passed:
        blockers.append("CALIBRATED_RECTIFICATION_HELDOUT_GATE_FAILED")
    if calibrated_quality is None or calibrated_quality.get("passed") is not True:
        blockers.append("CALIBRATED_CANDIDATE_EPIPOLAR_GATE_FAILED")
    if raw_quality is not None and calibrated_quality is not None:
        comparative_values = (
            raw_quality.get("median_vertical_error_px"),
            raw_quality.get("p90_vertical_error_px"),
            calibrated_quality.get("median_vertical_error_px"),
            calibrated_quality.get("p90_vertical_error_px"),
        )
        if (
            any(value is None for value in comparative_values)
            or calibrated_quality["median_vertical_error_px"]
            > raw_quality["median_vertical_error_px"]
            or calibrated_quality["p90_vertical_error_px"]
            > raw_quality["p90_vertical_error_px"]
        ):
            blockers.append(
                "RECTIFIED_CANDIDATE_NOT_IMPROVED_OVER_RAW_AT_MEDIAN_AND_P90"
            )
    return {
        "gpu_depth_allowed": not blockers,
        "selected_candidate": (
            "SAME_SESSION_INTRINSICS_HELDOUT_ESTIMATED_RECTIFIED"
            if not blockers else None
        ),
        "first_blocker": blockers[0] if blockers else None,
        "blockers": blockers,
        "raw_resize_admission_policy": (
            "DIAGNOSTIC_ONLY_NOT_GPU_ELIGIBLE_WITHOUT_INDEPENDENT_ENCODED_"
            "DISTORTION_AND_METRIC_PINHOLE_AUTHORITY"
        ),
    }


def _camera_summary(
    params: dict[str, Any], *, baseline_m: float,
) -> dict[str, Any]:
    return {
        "version": params.get("version"),
        "per_eye_source_geometry": [params.get("width"), params.get("height")],
        "physical_eye_source_indices": {
            "left": params.get("left", {}).get("sourceIndex"),
            "right": params.get("right", {}).get("sourceIndex"),
        },
        "left": {
            "camera_id": params.get("left", {}).get("cameraId"),
            "intrinsics": params.get("left", {}).get("intrinsics"),
            "distortion": params.get("left", {}).get("distortion"),
        },
        "right": {
            "camera_id": params.get("right", {}).get("cameraId"),
            "intrinsics": params.get("right", {}).get("intrinsics"),
            "distortion": params.get("right", {}).get("distortion"),
        },
        "extrinsics": params.get("extrinsics"),
        "extrinsic_convention": params.get("extrinsic_convention"),
        "baseline_m": float(baseline_m),
        "encoded_domain_intrinsics_authority": params.get(
            "encoded_domain_intrinsics_authority"
        ),
        "encoded_domain_distortion_policy": params.get(
            "encoded_domain_distortion_policy"
        ),
    }


def run_preflight(
    session: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov_deg: float = DEFAULT_FOV_DEG,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict[str, Any]:
    if (width, height, float(fov_deg)) != (
        DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FOV_DEG,
    ):
        raise ValueError(
            "the v1 calibrated candidate is pinned to 1280x960 FOV90; "
            "a different domain requires a new contract version"
        )
    session = session.resolve(strict=True)
    camera = session / "camera_params.json"
    if not camera.is_file():
        raise RuntimeError(f"missing same-session camera_params: {camera}")
    stereo_candidates = sorted(
        (session / "source_stereo").glob("CameraRecord_*_stereo.mp4")
    )
    if len(stereo_candidates) != 1:
        raise RuntimeError(
            f"expected one same-session SBS video, found {len(stereo_candidates)}"
        )
    stereo = stereo_candidates[0]
    params = json.loads(camera.read_text(encoding="utf-8"))
    pico = load_pico_module()
    eyes, eye_width, eye_height, baseline_m = pico.load_camera_params(camera)
    source_indices = pico.load_source_indices(camera)
    video = inspect_video(stereo)
    geometry_valid = bool(
        video["full_decode_passed"]
        and video["width"] == eye_width * 2
        and video["height"] == eye_height
    )
    metadata_count = params.get("video_info", {}).get("frame_count")
    metadata_valid = (
        metadata_count is None or metadata_count == video["decoded_frame_count"]
    )
    indices = sample_indices(video["decoded_frame_count"], sample_count)

    def raw_transform(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left, right = split_physical_eyes(
            frame, eye_width=eye_width, eye_height=eye_height,
            source_indices=source_indices,
        )
        interpolation = cv2.INTER_AREA if width <= eye_width and height <= eye_height else cv2.INTER_LINEAR
        return (
            cv2.resize(left, (width, height), interpolation=interpolation),
            cv2.resize(right, (width, height), interpolation=interpolation),
        )

    raw_quality = evaluate_candidate(
        stereo, indices=indices, eye_width=eye_width, eye_height=eye_height,
        source_indices=source_indices, transform=raw_transform,
    )
    raw_candidate = {
        "domain": "PHYSICAL_EYES_SOURCEINDEX_AWARE_RESIZE_ONLY",
        "target_geometry": [width, height],
        "quality": raw_quality,
        "gpu_eligible": False,
        "authority": "DIAGNOSTIC_EPIPOLAR_EVIDENCE_ONLY",
    }

    calibration: dict[str, Any] | None = None
    calibrated_quality: dict[str, Any] | None = None
    calibrated_error: str | None = None
    maps: list[tuple[np.ndarray, np.ndarray]] | None = None
    try:
        maps, _rectified_k, calibration = depth_v1.calibrate(
            stereo, pico, eyes, eye_width, source_indices,
            int(video["decoded_frame_count"]),
        )

        def rectified_transform(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            assert maps is not None
            left, right = pico.remap_pair(
                frame, eye_width, maps, source_indices=source_indices,
            )
            return left, right

        calibrated_quality = evaluate_candidate(
            stereo, indices=indices, eye_width=eye_width, eye_height=eye_height,
            source_indices=source_indices, transform=rectified_transform,
        )
    except Exception as exc:  # fail-closed evidence, not a silent fallback
        calibrated_error = f"{type(exc).__name__}: {exc}"

    rectification_calibration_passed = bool(
        calibration is not None
        and calibration.get("quality", {}).get("passed") is True
    )
    distortion_coefficients_valid = all(
        isinstance(params.get(name, {}).get("distortion", {}).get("coeffs"), list)
        and len(params[name]["distortion"]["coeffs"]) == 8
        for name in ("left", "right")
    )
    input_valid = bool(
        eye_width > 0 and eye_height > 0 and baseline_m > 0.0
        and params.get("left", {}).get("distortion", {}).get("model") == "equiDis62"
        and params.get("right", {}).get("distortion", {}).get("model") == "equiDis62"
        and distortion_coefficients_valid
        and metadata_valid
    )
    decision = admission_decision(
        input_valid=input_valid,
        source_indices=source_indices,
        decode_valid=geometry_valid,
        rectification_calibration_passed=rectification_calibration_passed,
        calibrated_quality=calibrated_quality,
        raw_quality=raw_quality,
    )
    calibrated_candidate = {
        "domain": "SAME_SESSION_INTRINSICS_HELDOUT_ESTIMATED_RECTIFIED",
        "target_geometry": [width, height],
        "virtual_horizontal_fov_deg_before_rectification": fov_deg,
        "quality": calibrated_quality,
        "calibration": calibration,
        "error": calibrated_error,
        "gpu_eligible": decision["gpu_depth_allowed"],
        "authority": "SINGLE_SESSION_INTERNAL_STEREO_PREFLIGHT_ONLY",
        "rectification_rotation_source": (
            "IMAGE_MATCHED_ESSENTIAL_MATRIX_ESTIMATED_ON_10_FRAMES_"
            "ACCEPTED_ON_10_DISJOINT_HELDOUT_FRAMES"
        ),
        "metric_scale_source": "SAME_SESSION_FACTORY_BASELINE_M_TIMES_RECTIFIED_FX",
    }
    return {
        "schema_version": "0915-stereo-domain-preflight-v1",
        "status": (
            "PASS_GPU_DEPTH_ADMISSION" if decision["gpu_depth_allowed"]
            else "BLOCKED_GPU_DEPTH_ADMISSION"
        ),
        "session_id": session.name,
        "weights": "ABSENT",
        "gpu_used": False,
        "model_inference_run": False,
        "source_mutated": False,
        "inputs": {"camera_params": file_ref(camera), "source_stereo": file_ref(stereo)},
        "decode": video,
        "sample_indices": indices,
        "camera_calibration": _camera_summary(params, baseline_m=baseline_m),
        "input_validation": {
            "same_session_calibration_complete": input_valid,
            "equiDis62_coefficient_count_exact": distortion_coefficients_valid,
            "decoded_geometry_matches_camera": geometry_valid,
            "metadata_frame_count_matches_decode": metadata_valid,
            "physical_eye_source_indices": {
                "left": int(source_indices[0]), "right": int(source_indices[1]),
            },
        },
        "candidates": {
            "raw_resize": raw_candidate,
            "calibrated_rectified": calibrated_candidate,
        },
        "decision": decision,
        "claim_limit": (
            "CPU-only single-session stereo-domain evidence. Admission does not "
            "claim FoundationStereo quality, external metric accuracy, hidden "
            "geometry, contact truth, or deployment authority."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fov-deg", type=float, default=DEFAULT_FOV_DEG)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    args = parser.parse_args()
    output = args.output_json.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output path required: {output}")
    try:
        result = run_preflight(
            args.session_root, width=args.width, height=args.height,
            fov_deg=args.fov_deg, sample_count=args.sample_count,
        )
    except Exception as exc:  # preserve a fail-closed machine result
        result = {
            "schema_version": "0915-stereo-domain-preflight-v1",
            "status": "BLOCKED_GPU_DEPTH_ADMISSION",
            "weights": "ABSENT",
            "gpu_used": False,
            "model_inference_run": False,
            "source_mutated": False,
            "decision": {
                "gpu_depth_allowed": False,
                "selected_candidate": None,
                "first_blocker": "PREFLIGHT_RUNTIME_OR_INPUT_ERROR",
                "blockers": ["PREFLIGHT_RUNTIME_OR_INPUT_ERROR"],
            },
            "error": f"{type(exc).__name__}: {exc}",
            "claim_limit": "Failure receipt only; no stereo or depth claim.",
        }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "gpu_depth_allowed": result["decision"]["gpu_depth_allowed"],
        "output": str(output),
    }, sort_keys=True))
    return 0 if result["decision"]["gpu_depth_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
