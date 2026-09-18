#!/usr/bin/env python3
"""Governed FoundationStereo canary for ``play_cards_0915_001``.

The worker consumes the rectification accepted by the preceding CPU preflight;
it does not estimate a second calibration.  The only published geometry is
physical-left, rectified optical-Z in metres.  External metric accuracy remains
unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid

import cv2
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops import analyze_0915_stereo_domain_preflight_v1 as preflight_module
from chaoyang.ops import run_0915_leftmono_foundation_depth_v1 as depth_v1
from chaoyang.ops import run_exact78_foundationstereo_corrected_depth_worker as fs_worker


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_foundationstereo_single_session_canary_v1"
PHASE = "0915_FOUNDATIONSTEREO_SINGLE_SESSION_METRIC_CANARY"
SESSION_ID = "play_cards_0915_001"
SESSION = (
    Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915")
    / "cleaned/playing_cards" / SESSION_ID
)
T0_PREFLIGHT = (
    ROOT / "_run/current/0915_stereo_interaction_cpu_canary_v1/attempts/attempt_0001"
    / "lanes/stereo_preflight/STEREO_PREFLIGHT.json"
)
CHECKPOINT = ROOT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
ASSET_PIN = CHECKPOINT.parent / "ASSET_PIN.json"
CHECKPOINT_SHA256 = depth_v1.CHECKPOINT_SHA256
MODEL_WEIGHT = "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
OUTPUT_NAMESPACE = ROOT / "_run/current" / TASK_ID
VISUAL_NAMESPACE = ROOT / "docs/current/visuals/0915_FOUNDATIONSTEREO_CANARY_V1"
CENTRAL_GPU_LEASE = ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"
PINNED_PYTHON_LAUNCHER = ROOT / "src/chaoyang/ops/foundationstereo_gpu_python.sh"

DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
Z_NEAR_M = 0.10
Z_FAR_M = 3.0
MIN_DISPARITY_PX = 0.25
LR_MAX_RESIDUAL_PX = 2.0
QUALITY_THRESHOLDS = {
    "maximum_formula_recompute_error_m": 1.0e-6,
    "minimum_median_geometric_valid_fraction": 0.40,
    "minimum_median_lr_testable_fraction": 0.45,
    "minimum_median_lr_consistent_fraction": 0.60,
    "maximum_lr_residual_p90_px": 5.0,
    "minimum_median_final_valid_fraction": 0.25,
    "minimum_fraction_frames_final_valid_at_least_0_20": 0.90,
    "maximum_temporal_depth_median_step_p90_m": 0.35,
    "maximum_temporal_valid_fraction_step_p90": 0.20,
    "minimum_edge_evidence_frame_fraction": 0.80,
    "minimum_median_rgb_supported_disparity_edge_fraction": 0.15,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved), "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def published_ref(current: Path, published: Path) -> dict[str, Any]:
    current = current.resolve(strict=True)
    return {
        "path": str(published.resolve()), "bytes": current.stat().st_size,
        "sha256": sha256(current),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_output_namespace(output: Path, visual: Path) -> None:
    if not output.resolve().is_relative_to(OUTPUT_NAMESPACE.resolve()):
        raise RuntimeError(f"output must stay inside {OUTPUT_NAMESPACE}")
    if visual.resolve() != VISUAL_NAMESPACE.resolve():
        raise RuntimeError(f"visual root must equal {VISUAL_NAMESPACE}")


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from frozen task specification")
    if packet.get("weights") != [MODEL_WEIGHT]:
        raise RuntimeError("FoundationStereo canary must bind exactly one pinned weight")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("FoundationStereo canary is not the current next_task")
    task = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if task is None or task.get("status") not in {
        "PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE",
    }:
        raise RuntimeError("FoundationStereo canary is not executable")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize FoundationStereo canary")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    return packet


def heartbeat(status: str) -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()),
        "--status", status, "--phase", PHASE,
        *(["--gpu-id", "0"] if status == "RUNNING" else []),
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def validate_asset_pin() -> dict[str, Any]:
    pin = load_json(ASSET_PIN)
    pinned = pin.get("files", {}).get(CHECKPOINT.name, {})
    if (
        pin.get("status") != "PASS_LOCAL_CHECKPOINT_MATERIALIZED_NO_MODEL_LOAD"
        or pinned.get("bytes") != CHECKPOINT.stat().st_size
        or pinned.get("sha256") != CHECKPOINT_SHA256
        or sha256(CHECKPOINT) != CHECKPOINT_SHA256
    ):
        raise RuntimeError("FoundationStereo checkpoint asset pin drift")
    return pin


def validate_preflight(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    calibrated = payload.get("candidates", {}).get("calibrated_rectified", {})
    if payload.get("status") != "PASS_GPU_DEPTH_ADMISSION":
        raise RuntimeError("T0 stereo preflight did not pass GPU depth admission")
    if payload.get("session_id") != SESSION_ID:
        raise RuntimeError("T0 stereo preflight session identity mismatch")
    if payload.get("decision", {}).get("gpu_depth_allowed") is not True:
        raise RuntimeError("T0 preflight forbids GPU depth")
    if payload.get("decision", {}).get("selected_candidate") != (
        "SAME_SESSION_INTRINSICS_HELDOUT_ESTIMATED_RECTIFIED"
    ):
        raise RuntimeError("T0 selected stereo domain drift")
    if calibrated.get("gpu_eligible") is not True:
        raise RuntimeError("T0 rectified candidate is not GPU eligible")
    calibration = calibrated.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("quality", {}).get("passed") is not True:
        raise RuntimeError("T0 frozen rectification calibration is absent or rejected")
    expected = {
        "camera_params": SESSION / "camera_params.json",
        "source_stereo": next(iter(sorted((SESSION / "source_stereo").glob("CameraRecord_*_stereo.mp4"))), None),
    }
    for name, actual in expected.items():
        if actual is None or not actual.is_file():
            raise RuntimeError(f"fixed canary input missing: {name}")
        if payload.get("inputs", {}).get(name) != file_ref(actual):
            raise RuntimeError(f"T0 {name} closure drift")
    if payload.get("decode", {}).get("full_decode_passed") is not True:
        raise RuntimeError("T0 did not fully decode SBS")
    return payload


def physical_left_depth(
    disparity_flipped: np.ndarray, *, focal_px: float, baseline_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flipped-model disparity to unflipped physical-left optical-Z."""

    disparity = cv2.flip(np.asarray(disparity_flipped, np.float32), 1)
    x = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
    geometric = (
        np.isfinite(disparity) & (disparity > MIN_DISPARITY_PX)
        & ((x + disparity) < disparity.shape[1])
    )
    depth = np.full(disparity.shape, np.nan, np.float32)
    depth[geometric] = np.float32(focal_px * baseline_m) / disparity[geometric]
    geometric &= (depth >= Z_NEAR_M) & (depth <= Z_FAR_M)
    depth[~geometric] = np.nan
    return disparity, depth, geometric


def left_right_consistency(
    disparity_left: np.ndarray,
    disparity_right: np.ndarray,
    geometric_left: np.ndarray,
    *,
    max_residual_px: float = LR_MAX_RESIDUAL_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare left disparity with right-reference disparity at ``x_right=x+d``."""

    left = np.asarray(disparity_left, np.float32)
    right = np.asarray(disparity_right, np.float32)
    if left.shape != right.shape or left.shape != geometric_left.shape:
        raise ValueError("left/right consistency arrays must have identical shape")
    height, width = left.shape
    xx = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    xr = np.rint(xx + left).astype(np.int64)
    in_bounds = np.asarray(geometric_left, bool) & (xr >= 0) & (xr < width)
    sampled = np.full(left.shape, np.nan, np.float32)
    yy, _ = np.indices(left.shape)
    sampled[in_bounds] = right[yy[in_bounds], xr[in_bounds]]
    testable = in_bounds & np.isfinite(sampled) & (sampled > MIN_DISPARITY_PX)
    residual = np.full(left.shape, np.nan, np.float32)
    residual[testable] = np.abs(left[testable] - sampled[testable])
    consistent = testable & (residual <= max_residual_px)
    return residual, testable, consistent


def edge_support_metrics(left_bgr: np.ndarray, disparity: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    """Measure whether strong disparity boundaries have nearby RGB boundaries."""

    image = cv2.resize(left_bgr, (DEPTH_WIDTH, DEPTH_HEIGHT), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_edges = cv2.Canny(gray, 60, 140) > 0
    rgb_support = cv2.dilate(rgb_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    safe = np.asarray(valid, bool)
    filled = np.where(safe, disparity, 0.0).astype(np.float32)
    dx = np.abs(cv2.Sobel(filled, cv2.CV_32F, 1, 0, ksize=3))
    dy = np.abs(cv2.Sobel(filled, cv2.CV_32F, 0, 1, ksize=3))
    neighbourhood = cv2.erode(safe.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    disparity_edge = neighbourhood & (np.hypot(dx, dy) >= 4.0)
    count = int(disparity_edge.sum())
    supported = int((disparity_edge & rgb_support).sum())
    return {
        "strong_disparity_edge_pixels": count,
        "rgb_supported_disparity_edge_pixels": supported,
        "rgb_supported_disparity_edge_fraction": (
            float(supported / count) if count else None
        ),
    }


def build_registration_maps(
    full_left_map: tuple[np.ndarray, np.ndarray],
    *,
    source_width: int,
    source_height: int,
    sam_width: int = 1280,
    sam_height: int = 960,
) -> dict[str, np.ndarray]:
    """Map depth pixels through frozen rectification into resize-only SAM pixels."""

    map_x, map_y = (np.asarray(item, np.float32) for item in full_left_map)
    if map_x.shape != (960, 1280) or map_y.shape != (960, 1280):
        raise ValueError("frozen rectification maps must be 1280x960")
    yy, xx = np.indices((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.float32)
    full_x = 2.0 * xx + 0.5
    full_y = 2.0 * yy + 0.5
    source_x = cv2.remap(map_x, full_x, full_y, cv2.INTER_LINEAR)
    source_y = cv2.remap(map_y, full_x, full_y, cv2.INTER_LINEAR)
    sam_x = (source_x + 0.5) * (sam_width / source_width) - 0.5
    sam_y = (source_y + 0.5) * (sam_height / source_height) - 0.5
    source_xy = np.stack((source_x, source_y), axis=-1).astype(np.float32)
    sam_xy = np.stack((sam_x, sam_y), axis=-1).astype(np.float32)
    in_bounds = (
        np.isfinite(sam_xy).all(axis=2)
        & (sam_x >= 0.0) & (sam_x <= sam_width - 1)
        & (sam_y >= 0.0) & (sam_y <= sam_height - 1)
    )
    return {
        "depth_pixel_to_full_rectified_left_xy": np.stack((full_x, full_y), axis=-1),
        "depth_to_source_left_xy": source_xy,
        "depth_to_sam_resize_xy": sam_xy,
        "sam_resize_in_bounds": in_bounds,
    }


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def aggregate_quality(frame_rows: list[dict[str, Any]], expected_frames: int) -> dict[str, Any]:
    geometric = [float(row["geometric_valid_fraction"]) for row in frame_rows]
    testable = [float(row["lr_testable_fraction"]) for row in frame_rows]
    consistent = [float(row["lr_consistent_fraction_of_testable"]) for row in frame_rows]
    final = [float(row["final_valid_fraction"]) for row in frame_rows]
    residuals = [
        float(row["lr_residual_p90_px"])
        for row in frame_rows if row["lr_residual_p90_px"] is not None
    ]
    edge = [
        float(row["edge"]["rgb_supported_disparity_edge_fraction"])
        for row in frame_rows
        if row["edge"]["rgb_supported_disparity_edge_fraction"] is not None
    ]
    depth_medians = [row["depth_p50_m"] for row in frame_rows]
    depth_steps = [
        abs(float(b) - float(a)) for a, b in zip(depth_medians, depth_medians[1:])
        if a is not None and b is not None
    ]
    valid_steps = [abs(b - a) for a, b in zip(final, final[1:])]
    metrics = {
        "expected_frame_count": expected_frames,
        "decoded_frame_count": len(frame_rows),
        "median_geometric_valid_fraction": float(np.median(geometric)) if geometric else 0.0,
        "median_lr_testable_fraction": float(np.median(testable)) if testable else 0.0,
        "median_lr_consistent_fraction": float(np.median(consistent)) if consistent else 0.0,
        "lr_residual_p90_across_frame_p90_px": _percentile(residuals, 90),
        "median_final_valid_fraction": float(np.median(final)) if final else 0.0,
        "fraction_frames_final_valid_at_least_0_20": (
            float(np.mean(np.asarray(final) >= 0.20)) if final else 0.0
        ),
        "temporal_depth_median_step_p90_m": _percentile(depth_steps, 90),
        "temporal_valid_fraction_step_p90": _percentile(valid_steps, 90),
        "edge_evidence_frame_fraction": float(len(edge) / expected_frames),
        "median_rgb_supported_disparity_edge_fraction": (
            float(np.median(edge)) if edge else 0.0
        ),
        "maximum_formula_recompute_error_m": max(
            (
                float(row["formula_recompute_max_abs_error_m"])
                for row in frame_rows
                if row["formula_recompute_max_abs_error_m"] is not None
            ),
            default=None,
        ),
    }
    gates = {
        "full_decode": len(frame_rows) == expected_frames,
        "formula_recompute": (
            metrics["maximum_formula_recompute_error_m"] is not None
            and metrics["maximum_formula_recompute_error_m"]
            <= QUALITY_THRESHOLDS["maximum_formula_recompute_error_m"]
        ),
        "geometric_validity": metrics["median_geometric_valid_fraction"]
        >= QUALITY_THRESHOLDS["minimum_median_geometric_valid_fraction"],
        "lr_testable_coverage": metrics["median_lr_testable_fraction"]
        >= QUALITY_THRESHOLDS["minimum_median_lr_testable_fraction"],
        "lr_consistency": (
            metrics["median_lr_consistent_fraction"]
            >= QUALITY_THRESHOLDS["minimum_median_lr_consistent_fraction"]
            and metrics["lr_residual_p90_across_frame_p90_px"] is not None
            and metrics["lr_residual_p90_across_frame_p90_px"]
            <= QUALITY_THRESHOLDS["maximum_lr_residual_p90_px"]
        ),
        "final_validity": (
            metrics["median_final_valid_fraction"]
            >= QUALITY_THRESHOLDS["minimum_median_final_valid_fraction"]
            and metrics["fraction_frames_final_valid_at_least_0_20"]
            >= QUALITY_THRESHOLDS["minimum_fraction_frames_final_valid_at_least_0_20"]
        ),
        "temporal_distribution_stability": (
            metrics["temporal_depth_median_step_p90_m"] is not None
            and metrics["temporal_depth_median_step_p90_m"]
            <= QUALITY_THRESHOLDS["maximum_temporal_depth_median_step_p90_m"]
            and metrics["temporal_valid_fraction_step_p90"] is not None
            and metrics["temporal_valid_fraction_step_p90"]
            <= QUALITY_THRESHOLDS["maximum_temporal_valid_fraction_step_p90"]
        ),
        "rgb_edge_support": (
            metrics["edge_evidence_frame_fraction"]
            >= QUALITY_THRESHOLDS["minimum_edge_evidence_frame_fraction"]
            and metrics["median_rgb_supported_disparity_edge_fraction"]
            >= QUALITY_THRESHOLDS["minimum_median_rgb_supported_disparity_edge_fraction"]
        ),
    }
    return {
        "thresholds": QUALITY_THRESHOLDS,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "temporal_gate_scope": (
            "FRAME_DISTRIBUTION_STABILITY_ONLY_NOT_SCENE_FLOW_COMPENSATED_PER_PIXEL"
        ),
        "edge_gate_scope": "DISPARITY_BOUNDARY_RGB_SUPPORT_DIAGNOSTIC",
    }


def _review_decode(path: Path, expected_frames: int) -> dict[str, Any]:
    completed = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
        "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    capture = cv2.VideoCapture(str(path))
    frames = 0
    geometry = None
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frames += 1
            geometry = [int(image.shape[1]), int(image.shape[0])]
    finally:
        capture.release()
    if completed.returncode or frames != expected_frames or geometry != [1280, 480]:
        raise RuntimeError(
            f"depth review full-decode failed: rc={completed.returncode}, "
            f"frames={frames}, geometry={geometry}"
        )
    return {"full_decode": True, "frame_count": frames, "geometry": geometry}


def _review_frame(left: np.ndarray, depth: np.ndarray, valid: np.ndarray, index: int) -> np.ndarray:
    rgb = cv2.resize(left, (640, 480), interpolation=cv2.INTER_AREA)
    clipped = np.clip(depth, 0.20, 2.00)
    normalized = np.zeros(depth.shape, np.uint8)
    normalized[valid] = np.asarray(
        255.0 * (2.00 - clipped[valid]) / 1.80, np.uint8,
    )
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    canvas = np.hstack((rgb, color))
    cv2.rectangle(canvas, (0, 0), (1279, 58), (0, 0, 0), -1)
    cv2.putText(
        canvas, f"{SESSION_ID} | frame {index:03d} | physical-left rectified optical-Z",
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "left: rectified RGB | right: 0.2-2.0m Z | invalid/LR-rejected=black | external accuracy UNVERIFIED",
        (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (235, 235, 235), 1, cv2.LINE_AA,
    )
    return canvas


def run_worker(output: Path, visual: Path, preflight_path: Path) -> int:
    preflight = validate_preflight(preflight_path)
    validate_asset_pin()
    camera = Path(preflight["inputs"]["camera_params"]["path"])
    stereo = Path(preflight["inputs"]["source_stereo"]["path"])
    source_before = {"camera_params": file_ref(camera), "source_stereo": file_ref(stereo)}
    calibration = preflight["candidates"]["calibrated_rectified"]["calibration"]
    decoded = preflight["decode"]
    frame_count = int(decoded["decoded_frame_count"])
    fps = float(decoded["fps"])

    pico = preflight_module.load_pico_module()
    eyes, eye_width, eye_height, baseline_m = pico.load_camera_params(camera)
    source_indices = pico.load_source_indices(camera)
    if source_indices != (1, 0):
        raise RuntimeError("physical left/right sourceIndex must remain (1,0)")
    rectified_k = np.asarray(calibration["rectified_intrinsics"], np.float64)
    rotations = [
        np.asarray(calibration["rectification_rotation_left"], np.float64),
        np.asarray(calibration["rectification_rotation_right"], np.float64),
    ]
    maps = [
        pico.make_map(eyes[index], rotations[index], 1280, 960, rectified_k)
        for index in range(2)
    ]
    depth_k = rectified_k.copy()
    depth_k[:2] *= 0.5
    registration_maps = build_registration_maps(
        maps[0], source_width=eye_width, source_height=eye_height,
    )

    staging = output / f".depth-staging-{uuid.uuid4().hex}"
    if staging.exists():
        raise RuntimeError("fresh depth staging required")
    staging.mkdir()
    (staging / "frames").mkdir()
    staged_review = staging / "0915_FOUNDATIONSTEREO_DEPTH_REVIEW.mp4"
    writer = cv2.VideoWriter(
        str(staged_review), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot open depth review encoder")

    model = fs_worker.Model()
    if model.model_load_count != 1:
        raise RuntimeError("FoundationStereo must be loaded exactly once")
    capture = cv2.VideoCapture(str(stereo))
    if not capture.isOpened():
        raise RuntimeError("cannot open fixed SBS input")
    frame_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for frame_index in range(frame_count):
            ok, sbs = capture.read()
            if not ok:
                raise RuntimeError(f"SBS decode ended at frame {frame_index}")
            if sbs.shape[:2] != (eye_height, eye_width * 2):
                raise RuntimeError(f"SBS geometry drift at frame {frame_index}")
            left, right = pico.remap_pair(
                sbs, eye_width, maps, source_indices=source_indices,
            )
            disparity_flipped, _unused_depth, _unused_valid = model.infer(
                cv2.flip(left, 1), cv2.flip(right, 1), baseline_m,
            )
            disparity_left, depth, geometric = physical_left_depth(
                disparity_flipped, focal_px=float(depth_k[0, 0]),
                baseline_m=float(baseline_m),
            )
            disparity_right, _unused_depth, _unused_valid = model.infer(
                right, left, baseline_m,
            )
            disparity_right = np.asarray(disparity_right, np.float32)
            residual, testable, consistent = left_right_consistency(
                disparity_left, disparity_right, geometric,
            )
            valid = geometric & consistent
            depth[~valid] = np.nan
            edge = edge_support_metrics(left, disparity_left, valid)
            values = depth[valid]
            residual_values = residual[testable]
            recomputed = np.full(depth.shape, np.nan, np.float32)
            recomputed[valid] = np.float32(depth_k[0, 0] * baseline_m) / disparity_left[valid]
            formula_error = (
                float(np.max(np.abs(depth[valid] - recomputed[valid])))
                if valid.any() else None
            )
            geometric_count = int(geometric.sum())
            testable_count = int(testable.sum())
            row = {
                "frame": frame_index,
                "artifact": None,
                "geometric_valid_pixels": geometric_count,
                "geometric_valid_fraction": float(geometric.mean()),
                "lr_testable_pixels": testable_count,
                "lr_testable_fraction": float(testable.mean()),
                "lr_consistent_pixels": int(consistent.sum()),
                "lr_consistent_fraction_of_testable": (
                    float(consistent.sum() / testable_count) if testable_count else 0.0
                ),
                "lr_residual_p50_px": (
                    float(np.median(residual_values)) if residual_values.size else None
                ),
                "lr_residual_p90_px": (
                    float(np.percentile(residual_values, 90)) if residual_values.size else None
                ),
                "final_valid_pixels": int(valid.sum()),
                "final_valid_fraction": float(valid.mean()),
                "depth_p10_m": float(np.percentile(values, 10)) if values.size else None,
                "depth_p50_m": float(np.median(values)) if values.size else None,
                "depth_p90_m": float(np.percentile(values, 90)) if values.size else None,
                "formula_recompute_max_abs_error_m": formula_error,
                "edge": edge,
            }
            frame_path = staging / "frames" / f"{frame_index:06d}.npz"
            atomic_npz(
                frame_path,
                frame_id=np.asarray(frame_index, np.int32),
                disparity_left_px=disparity_left.astype(np.float32),
                depth_m=depth.astype(np.float32),
                valid=valid.astype(bool),
                lr_residual_px=residual.astype(np.float32),
                lr_consistent=consistent.astype(bool),
                depth_intrinsics=depth_k.astype(np.float64),
                depth_reference=np.asarray("PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z"),
            )
            row["artifact"] = published_ref(
                frame_path, output / "frames" / frame_path.name,
            )
            frame_rows.append(row)
            writer.write(_review_frame(left, depth, valid, frame_index))
            print(json.dumps({
                "frame": frame_index, "total": frame_count,
                "valid_fraction": row["final_valid_fraction"],
                "lr_consistent_fraction": row["lr_consistent_fraction_of_testable"],
            }, sort_keys=True), flush=True)
        extra, _image = capture.read()
        if extra:
            raise RuntimeError("SBS contains frames beyond T0 fully decoded frame count")
    finally:
        capture.release()
        writer.release()

    if model.inference_count != frame_count * 2:
        raise RuntimeError("FoundationStereo inference count is not exactly two per frame")
    review_decode = _review_decode(staged_review, frame_count)
    quality = aggregate_quality(frame_rows, frame_count)
    calibration_receipt = {
        "schema_version": "0915-foundationstereo-frozen-calibration-v1",
        "source": file_ref(preflight_path),
        "consumption": "EXACT_T0_ACCEPTED_CALIBRATION_NO_REESTIMATION",
        "selected_domain": preflight["decision"]["selected_candidate"],
        "calibration": calibration,
    }
    registration_map_path = staging / "REGISTRATION_MAPS.npz"
    atomic_npz(registration_map_path, **registration_maps)
    registration = {
        "schema_version": "0915-foundationstereo-registration-v1",
        "source_domain": "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS",
        "depth_reference": "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z",
        "depth_intrinsics": depth_k.tolist(),
        "full_rectified_intrinsics": rectified_k.tolist(),
        "H_depth_pixel_to_full_rectified_left_pixel": [
            [2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0],
        ],
        "H_full_rectified_left_pixel_to_depth_pixel": [
            [0.5, 0.0, -0.25], [0.0, 0.5, -0.25], [0.0, 0.0, 1.0],
        ],
        "nonlinear_registration_maps": published_ref(
            registration_map_path, output / "REGISTRATION_MAPS.npz",
        ),
        "depth_to_sam_resize_map": {
            "array": "depth_to_sam_resize_xy",
            "source": "PHYSICAL_LEFT_RECTIFIED_640x480_PIXEL_CENTERS",
            "target": "PHYSICAL_LEFT_SOURCEINDEX_1_RESIZE_ONLY_1280x960_PIXEL_CENTERS",
            "construction": (
                "T0_FROZEN_EQUIDIS62_RECTIFICATION_INVERSE_MAP_THEN_OPENCV_"
                "PIXEL_CENTER_RESIZE_2048x1536_TO_1280x960"
            ),
            "identity_assumed": False,
        },
        "mask_join_policy": (
            "ONLY_VIA_DEPTH_TO_SAM_RESIZE_XY_AND_SAM_RESIZE_IN_BOUNDS; "
            "IDENTITY_JOIN_FORBIDDEN"
        ),
    }
    depth_contract = {
        "schema_version": "0915-foundationstereo-depth-contract-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "frame_count": frame_count,
        "frame_geometry": [DEPTH_WIDTH, DEPTH_HEIGHT],
        "frame_pattern": "frames/%06d.npz",
        "frame_arrays": {
            "frame_id": "int32",
            "disparity_left_px": "float32",
            "depth_m": "float32; invalid=NaN",
            "valid": "bool; geometry+range+left-right consistency",
            "lr_residual_px": "float32; untestable=NaN",
            "lr_consistent": "bool",
            "depth_intrinsics": "float64[3,3]",
            "depth_reference": "PHYSICAL_LEFT_RECTIFIED_OPTICAL_Z",
        },
        "formula": "optical_z_m = rectified_fx_px * baseline_m / disparity_left_px",
        "baseline_m": float(baseline_m),
        "rectified_fx_px_at_depth_resolution": float(depth_k[0, 0]),
        "valid_range_m": [Z_NEAR_M, Z_FAR_M],
        "native_model_confidence": "ABSENT_NOT_FABRICATED",
        "external_accuracy": "UNVERIFIED",
        "occluded_or_hidden_geometry": "INVALID_NOT_COMPLETED",
        "registration": "REGISTRATION.json",
        "claim_limit": (
            "Internal same-session stereo optical-Z consistency only; no external "
            "millimetre accuracy, hidden geometry, contact truth, force, or deployment authority."
        ),
    }
    summary = {
        "schema_version": "0915-foundationstereo-depth-summary-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": "PASS" if quality["passed"] else "REJECTED_QUALITY",
        "depth_admission": "PASS" if quality["passed"] else "REJECTED_QUALITY",
        "external_accuracy": "UNVERIFIED",
        "frame_count": frame_count,
        "full_sbs_decode": True,
        "model_load_count": model.model_load_count,
        "model_inference_count": model.inference_count,
        "quality": quality,
        "frames": frame_rows,
        "review_decode": review_decode,
        "wall_seconds": time.monotonic() - started,
        "source_mutated": False,
    }
    atomic_json(staging / "CALIBRATION.json", calibration_receipt)
    atomic_json(staging / "REGISTRATION.json", registration)
    atomic_json(staging / "DEPTH_CONTRACT.json", depth_contract)
    atomic_json(staging / "DEPTH_SUMMARY.json", summary)
    source_after = {"camera_params": file_ref(camera), "source_stereo": file_ref(stereo)}
    if source_after != source_before:
        raise RuntimeError("source input changed during FoundationStereo canary")

    visual.mkdir(parents=True)
    final_review = visual / "0915_FOUNDATIONSTEREO_DEPTH_REVIEW.mp4"
    os.replace(staged_review, final_review)
    review_decode = _review_decode(final_review, frame_count)
    for name in (
        "frames", "CALIBRATION.json", "REGISTRATION_MAPS.npz", "REGISTRATION.json",
        "DEPTH_CONTRACT.json", "DEPTH_SUMMARY.json",
    ):
        os.replace(staging / name, output / name)
    staging.rmdir()
    worker_result = {
        "schema_version": "0915-foundationstereo-worker-result-v1",
        "status": "COMPLETED",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "inputs": {
            "t0_stereo_preflight": file_ref(preflight_path),
            "camera_params": source_before["camera_params"],
            "source_stereo": source_before["source_stereo"],
            "checkpoint": file_ref(CHECKPOINT),
            "asset_pin": file_ref(ASSET_PIN),
        },
        "access_contract": {
            "allowed_algorithm_inputs": [
                str(preflight_path), str(camera), str(stereo), str(CHECKPOINT), str(ASSET_PIN),
            ],
            "pico26": "NOT_CONSUMED",
            "controller_pose": "NOT_CONSUMED",
            "trackingData_hand": "NOT_CONSUMED",
            "sam_masks": "NOT_ALGORITHM_DEPENDENCY",
            "interaction_v0a": "NOT_ALGORITHM_DEPENDENCY",
            "source_mutated": False,
        },
        "calibration": file_ref(output / "CALIBRATION.json"),
        "registration": file_ref(output / "REGISTRATION.json"),
        "registration_maps": file_ref(output / "REGISTRATION_MAPS.npz"),
        "depth_contract": file_ref(output / "DEPTH_CONTRACT.json"),
        "depth_summary": file_ref(output / "DEPTH_SUMMARY.json"),
        "review": {"video": file_ref(final_review), **review_decode},
        "external_accuracy": "UNVERIFIED",
    }
    atomic_json(output / "DEPTH_WORKER_RESULT.json", worker_result)
    return 0


def write_terminal(
    output: Path,
    visual: Path,
    receipt: Path,
    packet: dict[str, Any],
    *,
    status: str,
    first_blocker: str | None,
) -> None:
    summary = load_json(output / "DEPTH_SUMMARY.json") if (output / "DEPTH_SUMMARY.json").is_file() else None
    result = {
        "schema_version": "0915-foundationstereo-single-session-canary-result-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": status,
        "depth_admission": summary.get("depth_admission") if summary else "NOT_PRODUCED",
        "external_accuracy": "UNVERIFIED",
        "weights": packet["weights"],
        "first_blocker": first_blocker,
        "calibration": file_ref(output / "CALIBRATION.json") if (output / "CALIBRATION.json").is_file() else None,
        "registration": file_ref(output / "REGISTRATION.json") if (output / "REGISTRATION.json").is_file() else None,
        "registration_maps": file_ref(output / "REGISTRATION_MAPS.npz") if (output / "REGISTRATION_MAPS.npz").is_file() else None,
        "depth_contract": file_ref(output / "DEPTH_CONTRACT.json") if (output / "DEPTH_CONTRACT.json").is_file() else None,
        "depth_summary": file_ref(output / "DEPTH_SUMMARY.json") if summary else None,
        "gpu_command_receipt": file_ref(output / "GPU_COMMAND_RECEIPT.json")
        if (output / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "visual": file_ref(visual / "0915_FOUNDATIONSTEREO_DEPTH_REVIEW.mp4")
        if (visual / "0915_FOUNDATIONSTEREO_DEPTH_REVIEW.mp4").is_file() else None,
        "source_mutated": False,
        "batch_started": False,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(receipt, {**result, "result": file_ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-foundationstereo-single-session-run-receipt-v1",
        "task_id": TASK_ID, "status": status,
        "result": file_ref(output / "RESULT.json"),
        "terminal_receipt": file_ref(receipt),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    packet = validate_route()
    output = args.output_root.resolve()
    visual = args.visual_root.resolve()
    receipt = args.receipt.resolve()
    validate_output_namespace(output, visual)
    for path in (output, visual, receipt):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    for path in (T0_PREFLIGHT, CHECKPOINT, ASSET_PIN, CENTRAL_GPU_LEASE, PINNED_PYTHON_LAUNCHER):
        if not path.is_file():
            raise RuntimeError(f"fixed canary input is missing: {path}")
    validate_preflight(T0_PREFLIGHT)
    output.mkdir(parents=True)
    heartbeat("WAIT_GPU_RESOURCE")
    gpu_receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_command = [
        str(PINNED_PYTHON_LAUNCHER), str(Path(__file__).resolve()), "--worker",
        "--output-root", str(output), "--visual-root", str(visual),
        "--preflight", str(T0_PREFLIGHT),
    ]
    lease_command = [
        sys.executable, str(CENTRAL_GPU_LEASE),
        "--task-id", TASK_ID, "--attempt-id", output.name,
        "--priority", "CANARY", "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds),
        "--wall-seconds", str(args.wall_seconds),
        "--receipt", str(gpu_receipt), "--claim-limit", packet["claim_limit"],
        "--", *worker_command,
    ]
    atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-foundationstereo-single-session-command-v1",
        "task_id": TASK_ID,
        "weights": packet["weights"],
        "t0_preflight": file_ref(T0_PREFLIGHT),
        "worker_command": worker_command,
        "lease_command": lease_command,
    })
    with (output / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        while process.poll() is None:
            time.sleep(30)
            # The wrapper releases its lease immediately after worker exit.
            # Poll before reading the lease so a completed worker is never
            # relabelled WAIT_GPU_RESOURCE in the final polling iteration.
            if process.poll() is not None:
                break
            lease_path = ROOT / "_run/current/GPU_LEASE.json"
            lease = load_json(lease_path) if lease_path.is_file() else {}
            acquired = (
                lease.get("status") == "ACQUIRED"
                and lease.get("task_id") == TASK_ID
                and lease.get("attempt_id") == output.name
            )
            heartbeat("RUNNING" if acquired else "WAIT_GPU_RESOURCE")
    gpu = load_json(gpu_receipt) if gpu_receipt.is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        blocked = gpu.get("status") == "BLOCKED_RESOURCE"
        status = "BLOCKED_RESOURCE" if blocked else "FAILED_RUNTIME_FINAL"
        write_terminal(
            output, visual, receipt, packet, status=status,
            first_blocker=str(
                gpu.get("reason") or gpu.get("error") or "FOUNDATIONSTEREO_RUNTIME_FAILED"
            ),
        )
        return 3 if blocked else 2
    summary = load_json(output / "DEPTH_SUMMARY.json")
    passed = summary.get("depth_admission") == "PASS"
    write_terminal(
        output, visual, receipt, packet,
        status="PASSED" if passed else "REJECTED_QUALITY",
        first_blocker=None if passed else "DEPTH_QUALITY_GATES_FAILED",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--preflight", type=Path, default=T0_PREFLIGHT)
    args = parser.parse_args()
    if args.worker:
        validate_output_namespace(args.output_root, args.visual_root)
        return run_worker(
            args.output_root.resolve(), args.visual_root.resolve(),
            args.preflight.resolve(strict=True),
        )
    if args.receipt is None:
        raise RuntimeError("orchestrator requires --receipt")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
