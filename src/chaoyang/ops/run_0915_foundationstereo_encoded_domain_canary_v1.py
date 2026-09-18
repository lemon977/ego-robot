#!/usr/bin/env python3
"""Run the fresh encoded-domain FoundationStereo canary for play_cards_0915_001."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import cv2
import numpy as np

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet
from chaoyang.ops import run_0915_foundationstereo_single_session_canary_v1 as common
from chaoyang.ops import run_exact78_foundationstereo_corrected_depth_worker as fs_worker
from chaoyang.pipeline.vst_encoded_video_domain import (
    ADMITTED_TRANSFORM,
    ENCODED_DOMAIN,
    split_resize_physical_eyes,
    validate_encoded_video_contract,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_foundationstereo_encoded_domain_canary_v1"
PHASE = "0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1"
SESSION_ID = "play_cards_0915_001"
SESSION = (
    Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915")
    / "cleaned/playing_cards" / SESSION_ID
)
SBS = SESSION / "source_stereo/CameraRecord_play_cards_0915_001_stereo.mp4"
CAMERA = SESSION / "camera_params.json"
PREFLIGHT_ROOT = (
    ROOT / "_run/current/0915_stereo_encoded_domain_preflight_v1"
    / "attempts/attempt_0001"
)
PREFLIGHT = PREFLIGHT_ROOT / "ENCODED_STEREO_PREFLIGHT.json"
PREFLIGHT_RESULT = PREFLIGHT_ROOT / "RESULT.json"
DOMAIN_CONFIRMATION = ROOT / "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json"
USER_CONFIRMATION = (
    ROOT / "tasks/receipts/0915_FOUNDATIONSTEREO_OBJECT6D_USER_CONFIRMATION_V1.json"
)
CONFIG = ROOT / "configs/systems/depth/foundationstereo_0915_encoded_domain_canary_v1.json"
CHECKPOINT = ROOT / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
CHECKPOINT_CONFIG = CHECKPOINT.parent / "cfg.yaml"
ASSET_PIN = CHECKPOINT.parent / "ASSET_PIN.json"
MODEL_WEIGHT = "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
MODEL_WORKER = ROOT / "src/chaoyang/ops/run_exact78_foundationstereo_corrected_depth_worker.py"
PIXEL_DOMAIN_CODE = ROOT / "src/chaoyang/pipeline/vst_encoded_video_domain.py"
GPU_LAUNCHER = ROOT / "src/chaoyang/ops/foundationstereo_gpu_python.sh"
GPU_LEASE_WRAPPER = ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_FOUNDATIONSTEREO_ENCODED_DOMAIN_CANARY_V1_RESULT.json"
DEPTH_SIZE = (640, 480)
EYE_SIZE = (1280, 960)
EXPECTED_FRAMES = 150
AUTHORIZED_SCOPE = "VISUAL_OBJECT6D_CANDIDATE_INPUT"
MIN_DISPARITY_PX = 0.25
Z_NEAR_M = 0.10
Z_FAR_M = 3.0


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def ref(path: Path) -> dict[str, Any]:
    return common.file_ref(path)


def atomic_json(path: Path, value: Any) -> None:
    common.atomic_json(path, value)


def atomic_json_new(path: Path, value: Any) -> None:
    common.atomic_json_new(path, value)


def atomic_npz(path: Path, **arrays: Any) -> None:
    common.atomic_npz(path, **arrays)


def canonical_sha256(value: Any) -> str:
    return common.canonical_sha256(value)


def camera_baseline_m(camera: dict[str, Any]) -> float:
    """Compute metric camera-centre separation from head-to-camera extrinsics."""

    centres = []
    for eye in ("left", "right"):
        matrix = np.asarray(camera["extrinsics"][eye], np.float64)
        if matrix.shape != (4, 4):
            raise RuntimeError(f"invalid {eye} extrinsic shape")
        centres.append(-matrix[:3, :3].T @ matrix[:3, 3])
    baseline = float(np.linalg.norm(centres[0] - centres[1]))
    if not 0.04 <= baseline <= 0.10:
        raise RuntimeError(f"implausible stereo baseline: {baseline}")
    return baseline


def scaled_intrinsics(
    camera: dict[str, Any], *, eye: str, width: int, height: int,
) -> np.ndarray:
    source_width = int(camera["width"])
    source_height = int(camera["height"])
    raw = camera[eye]["intrinsics"]
    sx = width / source_width
    sy = height / source_height
    return np.asarray([
        [float(raw["fx"]) * sx, 0.0, (float(raw["cx"]) + 0.5) * sx - 0.5],
        [0.0, float(raw["fy"]) * sy, (float(raw["cy"]) + 0.5) * sy - 0.5],
        [0.0, 0.0, 1.0],
    ], np.float64)


def mirrored_intrinsics(intrinsics: np.ndarray, width: int) -> np.ndarray:
    result = np.asarray(intrinsics, np.float64).copy()
    if result.shape != (3, 3):
        raise ValueError("intrinsics must be 3x3")
    result[0, 2] = float(width - 1) - result[0, 2]
    return result


def unflip_disparity_and_depth(
    disparity_model: np.ndarray, *, focal_px: float, baseline_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disparity = cv2.flip(np.asarray(disparity_model, np.float32), 1)
    x = np.arange(disparity.shape[1], dtype=np.float32)[None, :]
    valid = (
        np.isfinite(disparity)
        & (disparity > MIN_DISPARITY_PX)
        & ((x + disparity) < disparity.shape[1])
    )
    depth = np.full(disparity.shape, np.nan, np.float32)
    depth[valid] = np.float32(focal_px * baseline_m) / disparity[valid]
    valid &= (depth >= Z_NEAR_M) & (depth <= Z_FAR_M)
    depth[~valid] = np.nan
    return disparity, depth, valid


def rgb_roundtrip_metrics(left: np.ndarray) -> dict[str, Any]:
    physical = cv2.resize(left, DEPTH_SIZE, interpolation=cv2.INTER_LINEAR)
    model = cv2.resize(cv2.flip(left, 1), DEPTH_SIZE, interpolation=cv2.INTER_LINEAR)
    returned = cv2.flip(model, 1)
    difference = cv2.absdiff(physical, returned)
    mismatched = int(np.count_nonzero(np.any(difference != 0, axis=2)))
    return {
        "maximum_absolute_channel_error": int(difference.max()),
        "mismatched_pixels": mismatched,
        "pixel_count": int(physical.shape[0] * physical.shape[1]),
        "physical_left_depth_rgb_sha256": hashlib.sha256(
            np.ascontiguousarray(physical).tobytes()
        ).hexdigest(),
        "unflipped_model_left_sha256": hashlib.sha256(
            np.ascontiguousarray(returned).tobytes()
        ).hexdigest(),
    }


def coordinate_roundtrip_error(width: int) -> float:
    x = np.arange(width, dtype=np.float64)
    returned = (width - 1) - ((width - 1) - x)
    return float(np.max(np.abs(returned - x)))


def validate_static_contracts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG)
    preflight = load_json(PREFLIGHT)
    preflight_result = load_json(PREFLIGHT_RESULT)
    domain = load_json(DOMAIN_CONFIRMATION)
    user = load_json(USER_CONFIRMATION)
    if (
        config.get("task_id") != TASK_ID
        or config.get("session_id") != SESSION_ID
        or config.get("model_weight") != MODEL_WEIGHT
    ):
        raise RuntimeError("encoded-domain canary config identity drift")
    if (
        preflight.get("decision") != "PASS_DIRECT_FOUNDATION_INPUT"
        or preflight.get("gpu_successor_authorized") is not True
        or preflight.get("depth_produced") is not False
        or preflight.get("image_domain", {}).get("lens_undistortion_applied") is not False
        or preflight.get("metrics", {}).get("dominant_disparity_sign")
        != "NEGATIVE_LEFT_MINUS_RIGHT"
        or preflight_result.get("status") != "PASSED"
        or preflight_result.get("gpu_successor_authorized") is not True
    ):
        raise RuntimeError("encoded-domain preflight does not authorize this canary")
    if (
        domain.get("status") != "CONFIRMED_ENCODED_VIDEO_ALREADY_UNDISTORTED"
        or domain.get("admitted_visual_transform") != ADMITTED_TRANSFORM
    ):
        raise RuntimeError("VST encoded-domain confirmation drift")
    adapter = user.get("foundationstereo_adapter", {})
    if (
        user.get("status") != "CONFIRMED"
        or user.get("authorized_session") != SESSION_ID
        or user.get("authorized_tasks_in_order", [None])[0] != TASK_ID
        or adapter.get("physical_left_source_index") != 1
        or adapter.get("physical_right_source_index") != 0
        or adapter.get("swap_physical_cameras") is not False
        or adapter.get("horizontal_reflection_for_disparity_sign") is not True
        or adapter.get("reflect_both_eyes") is not True
        or adapter.get("output_spatial_unflip_to_physical_left") is not True
        or adapter.get("lens_undistortion_allowed") is not False
        or adapter.get("lens_remap_allowed") is not False
    ):
        raise RuntimeError("user-confirmed disparity-sign adapter drift")
    validate_encoded_video_contract({
        "encoded_video_domain": ENCODED_DOMAIN,
        "transform": ADMITTED_TRANSFORM,
        "lens_undistortion_applied": False,
        "distortion_coefficients_consumed": False,
        "operation": "SOURCE_INDEX_CROP_RESIZE_ONLY",
    })
    return config, preflight, user


def validate_route() -> dict[str, Any]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from frozen specification")
    if packet.get("weights") != [MODEL_WEIGHT]:
        raise RuntimeError("encoded FoundationStereo must bind exactly one checkpoint")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("encoded FoundationStereo is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next(
        (row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID),
        None,
    )
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize encoded FoundationStereo")
    if route.get("packet_sha256") != common.sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    return packet


def heartbeat(status: str, *, gpu_id: int | None = None) -> None:
    command = [
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()),
        "--status", status, "--phase", PHASE,
    ]
    if gpu_id is not None:
        command += ["--gpu-id", str(gpu_id)]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def validate_writer_claim(
    path: Path, *, output: Path, signature_sha: str, executor_epoch: int,
    require_descendant: bool,
) -> dict[str, Any]:
    claim = load_json(path)
    pid = claim.get("pid")
    if (
        claim.get("task_id") != TASK_ID
        or claim.get("session_id") != SESSION_ID
        or claim.get("attempt_id") != "attempt_0001"
        or claim.get("weights") != [MODEL_WEIGHT]
        or claim.get("status") != "CLAIMED"
        or claim.get("executor_epoch") != executor_epoch
        or claim.get("run_signature_sha256") != signature_sha
        or claim.get("unique_write_root") != str(output.resolve())
        or not isinstance(pid, int)
        or common.process_start_ticks(pid) != claim.get("proc_start_ticks")
    ):
        raise RuntimeError("encoded FoundationStereo writer fence mismatch")
    if require_descendant and not common.process_has_ancestor(os.getpid(), pid):
        raise RuntimeError("GPU worker is not a descendant of the fenced writer")
    return claim


def build_signature(packet_path: Path, executor_epoch: int) -> dict[str, Any]:
    config, preflight, _user = validate_static_contracts()
    runtime_paths = {
        "runner": Path(__file__),
        "model_worker": MODEL_WORKER,
        "pixel_domain_code": PIXEL_DOMAIN_CODE,
        "gpu_launcher": GPU_LAUNCHER,
        "gpu_lease_wrapper": GPU_LEASE_WRAPPER,
        "checkpoint": CHECKPOINT,
        "checkpoint_config": CHECKPOINT_CONFIG,
        "asset_pin": ASSET_PIN,
        "canary_config": CONFIG,
    }
    payload = {
        "schema_version": "0915-foundationstereo-encoded-run-signature-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "executor_epoch": executor_epoch,
        "weights": [MODEL_WEIGHT],
        "inputs": {
            "task_packet": ref(packet_path),
            "sbs": ref(SBS),
            "camera_params": ref(CAMERA),
            "preflight": ref(PREFLIGHT),
            "preflight_result": ref(PREFLIGHT_RESULT),
            "domain_confirmation": ref(DOMAIN_CONFIRMATION),
            "user_confirmation": ref(USER_CONFIRMATION),
        },
        "runtime": {name: ref(path) for name, path in runtime_paths.items()},
        "pixel_domain": {
            "encoded": ENCODED_DOMAIN,
            "source_transform": ADMITTED_TRANSFORM,
            "lens_undistortion_applied": False,
            "distortion_coefficients_consumed": False,
        },
        "adapter": config["input_domain"],
        "preflight_metrics_sha256": canonical_sha256(preflight["metrics"]),
        "schema_identity": config["output_schema_identities"],
    }
    return {**payload, "run_signature_sha256": canonical_sha256(payload)}


def _left_right_consistency(
    disparity_left: np.ndarray,
    disparity_right: np.ndarray,
    geometric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return common.left_right_consistency(
        disparity_left, disparity_right, geometric, max_residual_px=2.0,
    )


def _aggregate(rows: list[dict[str, Any]], alignment: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["quality_thresholds"]
    geometric = np.asarray([row["geometric_valid_fraction"] for row in rows])
    testable = np.asarray([row["lr_testable_fraction"] for row in rows])
    consistent = np.asarray([row["lr_consistent_fraction"] for row in rows])
    final = np.asarray([row["final_valid_fraction"] for row in rows])
    medians = [row["depth_p50_m"] for row in rows]
    steps = [
        abs(float(right) - float(left))
        for left, right in zip(medians, medians[1:])
        if left is not None and right is not None
    ]
    metrics = {
        "decoded_frame_count": len(rows),
        "median_geometric_valid_fraction": float(np.median(geometric)) if len(rows) else 0.0,
        "median_lr_testable_fraction": float(np.median(testable)) if len(rows) else 0.0,
        "median_lr_consistent_fraction": float(np.median(consistent)) if len(rows) else 0.0,
        "fraction_frames_final_valid_at_least_0_15": float(np.mean(final >= 0.15)) if len(rows) else 0.0,
        "temporal_depth_median_step_p90_m": float(np.percentile(steps, 90)) if steps else None,
        "rgb_roundtrip_max_abs_error": alignment["maximum_absolute_channel_error"],
        "rgb_roundtrip_max_mismatched_pixels": alignment["mismatched_pixels"],
        "coordinate_roundtrip_max_abs_error_px": alignment["coordinate_roundtrip_max_abs_error_px"],
    }
    gates = {
        "full_decode": len(rows) == EXPECTED_FRAMES,
        "geometric_validity": metrics["median_geometric_valid_fraction"] >= thresholds["minimum_median_geometric_valid_fraction"],
        "lr_testable_coverage": metrics["median_lr_testable_fraction"] >= thresholds["minimum_median_lr_testable_fraction"],
        "lr_consistency": metrics["median_lr_consistent_fraction"] >= thresholds["minimum_median_lr_consistent_fraction"],
        "final_validity": metrics["fraction_frames_final_valid_at_least_0_15"] >= thresholds["minimum_fraction_frames_final_valid_at_least_0_15"],
        "temporal_distribution_stability": (
            metrics["temporal_depth_median_step_p90_m"] is not None
            and metrics["temporal_depth_median_step_p90_m"] <= thresholds["maximum_temporal_depth_median_step_p90_m"]
        ),
        "pixelwise_rgb_alignment": (
            metrics["rgb_roundtrip_max_abs_error"] <= thresholds["rgb_roundtrip_max_abs_error"]
            and metrics["rgb_roundtrip_max_mismatched_pixels"] <= thresholds["rgb_roundtrip_max_mismatched_pixels"]
            and metrics["coordinate_roundtrip_max_abs_error_px"] <= thresholds["coordinate_roundtrip_max_abs_error_px"]
        ),
    }
    return {"thresholds": thresholds, "metrics": metrics, "gates": gates, "passed": all(gates.values())}


def _review_frame(left: np.ndarray, depth: np.ndarray, valid: np.ndarray, frame: int) -> np.ndarray:
    rgb = cv2.resize(left, DEPTH_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = np.zeros(depth.shape, np.uint8)
    if valid.any():
        clipped = np.clip(depth, 0.20, 2.00)
        normalized[valid] = np.asarray(255.0 * (2.00 - clipped[valid]) / 1.80, np.uint8)
    colour = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colour[~valid] = 0
    canvas = np.hstack((rgb, colour))
    cv2.rectangle(canvas, (0, 0), (1279, 58), (0, 0, 0), -1)
    cv2.putText(canvas, f"frame {frame:03d} | encoded resize-only | output physical-left", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "left: physical-left RGB | right: optical-Z 0.2-2.0m | no lens remap | external accuracy UNVERIFIED", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def run_worker(
    *, output: Path, visual: Path, claim_path: Path, signature_path: Path,
    executor_epoch: int,
) -> int:
    packet = validate_route()
    signature = load_json(signature_path)
    stored = signature.pop("run_signature_sha256", None)
    if stored != canonical_sha256(signature):
        raise RuntimeError("run signature digest mismatch")
    signature["run_signature_sha256"] = stored
    validate_writer_claim(
        claim_path, output=output, signature_sha=stored,
        executor_epoch=executor_epoch, require_descendant=True,
    )
    config, preflight, _user = validate_static_contracts()
    if packet["weights"] != [MODEL_WEIGHT]:
        raise RuntimeError("task packet weight drift")
    camera = load_json(CAMERA)
    if (int(camera["left"]["sourceIndex"]), int(camera["right"]["sourceIndex"])) != (1, 0):
        raise RuntimeError("physical eye sourceIndex drift")
    eye_width = int(camera["width"])
    eye_height = int(camera["height"])
    baseline_m = camera_baseline_m(camera)
    physical_k = scaled_intrinsics(camera, eye="left", width=DEPTH_SIZE[0], height=DEPTH_SIZE[1])
    model_k = mirrored_intrinsics(physical_k, DEPTH_SIZE[0])
    source_before = {"sbs": ref(SBS), "camera_params": ref(CAMERA)}

    output.mkdir(parents=True, exist_ok=True)
    staging = output / f".encoded-depth-staging-{uuid.uuid4().hex}"
    staging.mkdir()
    (staging / "frames").mkdir()
    review_staging = staging / "0915_FOUNDATIONSTEREO_ENCODED_DEPTH_REVIEW.mp4"
    capture = cv2.VideoCapture(str(SBS))
    if not capture.isOpened():
        raise RuntimeError("cannot open fixed SBS")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(str(review_staging), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    if not writer.isOpened():
        raise RuntimeError("cannot open review writer")
    model = fs_worker.Model()
    rows: list[dict[str, Any]] = []
    maximum_error = 0
    mismatched_pixels = 0
    first_rgb_sha = None
    first_returned_sha = None
    started = time.monotonic()
    try:
        for frame_index in range(EXPECTED_FRAMES):
            validate_writer_claim(
                claim_path, output=output, signature_sha=stored,
                executor_epoch=executor_epoch, require_descendant=True,
            )
            ok, sbs = capture.read()
            if not ok:
                raise RuntimeError(f"SBS decode ended at frame {frame_index}")
            if sbs.shape[:2] != (eye_height, eye_width * 2):
                raise RuntimeError(f"SBS geometry drift at frame {frame_index}")
            left, right = split_resize_physical_eyes(
                sbs, eye_width=eye_width, source_indices=(1, 0), output_size=EYE_SIZE,
            )
            alignment = rgb_roundtrip_metrics(left)
            maximum_error = max(maximum_error, alignment["maximum_absolute_channel_error"])
            mismatched_pixels += alignment["mismatched_pixels"]
            if first_rgb_sha is None:
                first_rgb_sha = alignment["physical_left_depth_rgb_sha256"]
                first_returned_sha = alignment["unflipped_model_left_sha256"]

            disparity_model, _unused_depth, _unused_valid = model.infer(
                cv2.flip(left, 1), cv2.flip(right, 1), baseline_m,
            )
            disparity, depth, geometric = unflip_disparity_and_depth(
                disparity_model, focal_px=float(physical_k[0, 0]), baseline_m=baseline_m,
            )
            disparity_right, _unused_depth, _unused_valid = model.infer(right, left, baseline_m)
            residual, testable, consistent = _left_right_consistency(
                disparity, np.asarray(disparity_right, np.float32), geometric,
            )
            valid = geometric & consistent
            depth[~valid] = np.nan
            values = depth[valid]
            testable_count = int(testable.sum())
            row = {
                "frame": frame_index,
                "geometric_valid_fraction": float(geometric.mean()),
                "lr_testable_fraction": float(testable.mean()),
                "lr_consistent_fraction": float(consistent.sum() / testable_count) if testable_count else 0.0,
                "final_valid_fraction": float(valid.mean()),
                "depth_p10_m": float(np.percentile(values, 10)) if values.size else None,
                "depth_p50_m": float(np.median(values)) if values.size else None,
                "depth_p90_m": float(np.percentile(values, 90)) if values.size else None,
            }
            frame_path = staging / "frames" / f"{frame_index:06d}.npz"
            atomic_npz(
                frame_path,
                frame_id=np.asarray(frame_index, np.int32),
                disparity_physical_left_px=disparity.astype(np.float32),
                depth_m=depth.astype(np.float32),
                valid=valid.astype(bool),
                lr_residual_px=residual.astype(np.float32),
                lr_consistent=consistent.astype(bool),
                physical_left_intrinsics=physical_k,
                model_mirrored_intrinsics=model_k,
                depth_reference=np.asarray("PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"),
            )
            row["artifact"] = common.published_ref(frame_path, output / "frames" / frame_path.name)
            rows.append(row)
            writer.write(_review_frame(left, depth, valid, frame_index))
            print(json.dumps({"frame": frame_index, "valid_fraction": row["final_valid_fraction"]}, sort_keys=True), flush=True)
        extra, _frame = capture.read()
        if extra:
            raise RuntimeError("SBS contains frames beyond frozen 150-frame denominator")
    finally:
        capture.release()
        writer.release()

    if model.model_load_count != 1 or model.inference_count != EXPECTED_FRAMES * 2:
        raise RuntimeError("FoundationStereo persistent-model accounting drift")
    alignment_summary = {
        "schema_version": "0915-foundationstereo-pixelwise-rgb-alignment-v1",
        "frame_count": len(rows),
        "comparison": "resize(physical_left)_640x480 == unflip(resize(flip(physical_left))_640x480)",
        "maximum_absolute_channel_error": maximum_error,
        "mismatched_pixels": mismatched_pixels,
        "coordinate_roundtrip_max_abs_error_px": coordinate_roundtrip_error(DEPTH_SIZE[0]),
        "first_frame_physical_left_depth_rgb_sha256": first_rgb_sha,
        "first_frame_unflipped_model_left_sha256": first_returned_sha,
        "depth_grid_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
    }
    quality = _aggregate(rows, alignment_summary, config)
    passed = bool(quality["passed"])
    adapter_contract = {
        "schema_version": "0915-foundationstereo-encoded-adapter-contract-v1",
        "physical_source_indices": {"left": 1, "right": 0},
        "camera_swap": False,
        "horizontal_reflection_for_disparity_sign": True,
        "both_eyes_reflected": True,
        "lens_undistortion_applied": False,
        "lens_remap_applied": False,
        "model_domain": "SIMULTANEOUSLY_MIRRORED_PHYSICAL_LEFT_RIGHT",
        "model_intrinsics": model_k.tolist(),
        "mirrored_principal_point_rule": "cx_mirrored_px = width_px - 1 - cx_physical_px",
        "output_spatial_unflip": True,
        "output_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
        "output_intrinsics": physical_k.tolist(),
        "baseline_m": baseline_m,
        "preflight": ref(PREFLIGHT),
        "preflight_disparity_sign": preflight["metrics"]["dominant_disparity_sign"],
    }
    depth_contract = {
        "schema_version": "0915-foundationstereo-encoded-depth-contract-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "frame_count": EXPECTED_FRAMES,
        "frame_geometry": [DEPTH_SIZE[0], DEPTH_SIZE[1]],
        "depth_reference": "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z",
        "formula": "optical_z_m = physical_left_fx_px * baseline_m / unflipped_disparity_px",
        "physical_left_intrinsics": physical_k.tolist(),
        "model_mirrored_intrinsics": model_k.tolist(),
        "baseline_m": baseline_m,
        "depth_to_physical_left_rgb": "IDENTITY_640x480_AFTER_OUTPUT_UNFLIP",
        "depth_to_sam_resize_homography": [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]],
        "native_model_confidence": "ABSENT_NOT_FABRICATED",
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": passed,
        "authorized_scopes": [AUTHORIZED_SCOPE] if passed else [],
        "occluded_or_hidden_geometry": "INVALID_NOT_COMPLETED",
    }
    summary = {
        "schema_version": "0915-foundationstereo-encoded-depth-summary-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": "PASS" if passed else "REJECTED_QUALITY",
        "depth_admission": "PASS" if passed else "REJECTED_QUALITY",
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": passed,
        "authorized_scopes": [AUTHORIZED_SCOPE] if passed else [],
        "frame_count": len(rows),
        "model_load_count": model.model_load_count,
        "model_inference_count": model.inference_count,
        "quality": quality,
        "frames": rows,
        "wall_seconds": time.monotonic() - started,
        "source_mutated": False,
    }
    atomic_json(staging / "ADAPTER_CONTRACT.json", adapter_contract)
    atomic_json(staging / "RGB_ALIGNMENT_QA.json", alignment_summary)
    atomic_json(staging / "DEPTH_CONTRACT.json", depth_contract)
    atomic_json(staging / "DEPTH_SUMMARY.json", summary)
    visual.mkdir(parents=True, exist_ok=False)
    final_review = visual / "0915_FOUNDATIONSTEREO_ENCODED_DEPTH_REVIEW.mp4"
    os.replace(review_staging, final_review)
    decode = common._review_decode(final_review, EXPECTED_FRAMES)
    summary["review_decode"] = decode
    atomic_json(staging / "DEPTH_SUMMARY.json", summary)
    for name in ("frames", "ADAPTER_CONTRACT.json", "RGB_ALIGNMENT_QA.json", "DEPTH_CONTRACT.json", "DEPTH_SUMMARY.json"):
        os.replace(staging / name, output / name)
    staging.rmdir()
    if {"sbs": ref(SBS), "camera_params": ref(CAMERA)} != source_before:
        raise RuntimeError("source inputs changed during encoded-domain canary")
    worker_result = {
        "schema_version": "0915-foundationstereo-encoded-worker-result-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": "COMPLETED",
        "adapter_contract": ref(output / "ADAPTER_CONTRACT.json"),
        "rgb_alignment_qa": ref(output / "RGB_ALIGNMENT_QA.json"),
        "depth_contract": ref(output / "DEPTH_CONTRACT.json"),
        "depth_summary": ref(output / "DEPTH_SUMMARY.json"),
        "review": {"video": ref(final_review), **decode},
        "run_signature": ref(signature_path),
        "writer_claim": ref(claim_path),
        "source_mutated": False,
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": passed,
        "authorized_scopes": [AUTHORIZED_SCOPE] if passed else [],
    }
    atomic_json(output / "DEPTH_WORKER_RESULT.json", worker_result)
    return 0


def write_terminal(
    *, output: Path, visual: Path, packet: dict[str, Any], status: str,
    first_blocker: str | None,
) -> None:
    summary = load_json(output / "DEPTH_SUMMARY.json") if (output / "DEPTH_SUMMARY.json").is_file() else None
    passed = bool(status == "PASSED" and summary and summary.get("depth_admission") == "PASS")
    result = {
        "schema_version": "0915-foundationstereo-encoded-canary-result-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": status,
        "depth_admission": summary.get("depth_admission") if summary else "NOT_PRODUCED",
        "first_blocker": first_blocker,
        "weights": packet["weights"],
        "horizontal_reflection_for_disparity_sign": True,
        "output_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
        "lens_undistortion_applied": False,
        "source_mutated": False,
        "external_accuracy": "UNVERIFIED",
        "consumption_authorized": passed,
        "authorized_scopes": [AUTHORIZED_SCOPE] if passed else [],
        "adapter_contract": ref(output / "ADAPTER_CONTRACT.json") if (output / "ADAPTER_CONTRACT.json").is_file() else None,
        "rgb_alignment_qa": ref(output / "RGB_ALIGNMENT_QA.json") if (output / "RGB_ALIGNMENT_QA.json").is_file() else None,
        "depth_contract": ref(output / "DEPTH_CONTRACT.json") if (output / "DEPTH_CONTRACT.json").is_file() else None,
        "depth_summary": ref(output / "DEPTH_SUMMARY.json") if summary else None,
        "depth_worker_result": ref(output / "DEPTH_WORKER_RESULT.json") if (output / "DEPTH_WORKER_RESULT.json").is_file() else None,
        "run_signature": ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": ref(output / "CLAIM.json"),
        "gpu_command_receipt": ref(output / "GPU_COMMAND_RECEIPT.json") if (output / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "visual": ref(visual / "0915_FOUNDATIONSTEREO_ENCODED_DEPTH_REVIEW.mp4") if (visual / "0915_FOUNDATIONSTEREO_ENCODED_DEPTH_REVIEW.mp4").is_file() else None,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(TERMINAL_RECEIPT, {**result, "result": ref(output / "RESULT.json")})
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-foundationstereo-encoded-run-receipt-v1",
        "task_id": TASK_ID,
        "status": status,
        "result": ref(output / "RESULT.json"),
        "terminal_receipt": ref(TERMINAL_RECEIPT),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    packet = validate_route()
    validate_static_contracts()
    if args.output_root.resolve() != OUTPUT.resolve() or args.visual_root.resolve() != VISUAL.resolve():
        raise RuntimeError("encoded canary output namespace drift")
    if args.receipt.resolve() != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("encoded canary terminal receipt path drift")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("executor epoch and fencing token are required")
    for path in (OUTPUT, VISUAL, TERMINAL_RECEIPT):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    signature = build_signature(packet_path, args.executor_epoch)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir()
    atomic_json_new(OUTPUT / "RUN_SIGNATURE.json", signature)
    claim = {
        "schema_version": "0915-foundationstereo-encoded-writer-claim-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "attempt_id": OUTPUT.name,
        "weights": packet["weights"],
        "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "CLAIMED",
        "pid": os.getpid(),
        "proc_start_ticks": common.process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"],
        "unique_write_root": str(OUTPUT),
        "task_packet": ref(packet_path),
    }
    atomic_json_new(OUTPUT / "CLAIM.json", claim)
    heartbeat("WAIT_GPU_RESOURCE")
    worker_command = [
        str(GPU_LAUNCHER), str(Path(__file__).resolve()), "--worker",
        "--output-root", str(OUTPUT), "--visual-root", str(VISUAL),
        "--executor-epoch", str(args.executor_epoch),
        "--claim", str(OUTPUT / "CLAIM.json"),
        "--run-signature", str(OUTPUT / "RUN_SIGNATURE.json"),
    ]
    gpu_receipt = OUTPUT / "GPU_COMMAND_RECEIPT.json"
    lease_command = [
        sys.executable, str(GPU_LEASE_WRAPPER),
        "--task-id", TASK_ID, "--attempt-id", OUTPUT.name,
        "--executor-epoch", str(args.executor_epoch), "--priority", "CANARY",
        "--gpu-id", str(args.gpu_id), "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds), "--wall-seconds", str(args.wall_seconds),
        "--receipt", str(gpu_receipt), "--claim-limit", packet["claim_limit"],
        "--", *worker_command,
    ]
    atomic_json(OUTPUT / "COMMAND.json", {
        "schema_version": "0915-foundationstereo-encoded-command-v1",
        "task_id": TASK_ID,
        "worker_command": worker_command,
        "lease_command": lease_command,
    })
    with (OUTPUT / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(10)
            if process.poll() is not None:
                break
            lease_path = ROOT / "_run/current/GPU_LEASE.json"
            lease = load_json(lease_path) if lease_path.is_file() else {}
            acquired = lease.get("status") == "ACQUIRED" and lease.get("task_id") == TASK_ID
            heartbeat("RUNNING" if acquired else "WAIT_GPU_RESOURCE", gpu_id=args.gpu_id if acquired else None)
    gpu = load_json(gpu_receipt) if gpu_receipt.is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        status = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        write_terminal(
            output=OUTPUT, visual=VISUAL, packet=packet, status=status,
            first_blocker=str(gpu.get("reason") or gpu.get("error") or "FOUNDATIONSTEREO_RUNTIME_FAILED"),
        )
        return 3 if status == "BLOCKED_RESOURCE" else 2
    summary = load_json(OUTPUT / "DEPTH_SUMMARY.json")
    passed = summary.get("depth_admission") == "PASS"
    write_terminal(
        output=OUTPUT, visual=VISUAL, packet=packet,
        status="PASSED" if passed else "REJECTED_QUALITY",
        first_blocker=None if passed else "DEPTH_QUALITY_GATES_FAILED",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=TERMINAL_RECEIPT)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--wall-seconds", type=int, default=7200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--run-signature", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.claim is None or args.run_signature is None:
            raise RuntimeError("worker requires claim and run signature")
        return run_worker(
            output=args.output_root.resolve(), visual=args.visual_root.resolve(),
            claim_path=args.claim.resolve(strict=True),
            signature_path=args.run_signature.resolve(strict=True),
            executor_epoch=args.executor_epoch,
        )
    if args.fencing_token is None:
        raise RuntimeError("orchestrator requires fencing token")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
