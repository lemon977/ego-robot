#!/usr/bin/env python3
"""FoundationStereo 12-frame canary for an admitted S1 encoded session."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops import run_0915_foundationstereo_encoded_domain_canary_v1 as encoded
from chaoyang.ops import run_exact78_foundationstereo_corrected_depth_worker as fs_worker
from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import (
    resize_only,
    split_source_index_eyes,
)

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_quality_closure_s1_20260923"
CHECKPOINT = REPO / "assets/models/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth"
EXPECTED_CHECKPOINT_SHA = "60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def review(left: np.ndarray, depth: np.ndarray, valid: np.ndarray, frame_id: int) -> np.ndarray:
    rgb = cv2.resize(left, (640, 480), interpolation=cv2.INTER_AREA)
    scalar = np.zeros(depth.shape, np.uint8)
    scalar[valid] = np.asarray(255 * (2.0 - np.clip(depth[valid], 0.2, 2.0)) / 1.8, np.uint8)
    colour = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO); colour[~valid] = 0
    panel = np.concatenate((rgb, colour), axis=1)
    cv2.rectangle(panel, (0, 0), (1280, 60), (0, 0, 0), -1)
    cv2.putText(panel, f"frame {frame_id} | physical-left encoded resize-only | no lens remap", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(panel, "right: internal optical-Z candidate | external metric authority=false", (14, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230,230,230), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--full-session", action="store_true")
    args = parser.parse_args()
    preflight_path = args.preflight.resolve(strict=True)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS_LOCAL_STEREO_METRIC_DEV_PREFLIGHT":
        raise RuntimeError("PREFLIGHT_NOT_ADMITTED")
    adapter = preflight["foundation_input_adapter"]
    if not (
        adapter.get("authorized") is True
        and adapter.get("horizontal_reflection_for_disparity_sign") is True
        and adapter.get("swap_physical_eyes") is False
        and adapter.get("output_spatial_unflip_required") is True
    ):
        raise RuntimeError("DISPARITY_ADAPTER_DRIFT")
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("CHECKPOINT_SHA_DRIFT")
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{output}")
    output.mkdir(parents=True); (output / "frames").mkdir()
    source = Path(preflight["inputs"]["source_stereo"]["path"]).resolve(strict=True)
    camera_path = Path(preflight["inputs"]["camera_params"]["path"]).resolve(strict=True)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    eye_width, eye_height = int(camera["width"]), int(camera["height"])
    baseline = float(preflight["metric_conversion_check"]["baseline_m"])
    physical_k = encoded.scaled_intrinsics(camera, eye="left", width=640, height=480)
    model_k = encoded.mirrored_intrinsics(physical_k, 640)
    sample_frames = (
        list(range(int(preflight["image_domain_check"]["decoded_frames"])))
        if args.full_session
        else list(map(int, preflight["image_domain_check"]["sample_frames"]))
    )
    sample_set = set(sample_frames)
    model = fs_worker.Model()
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(output / "DEPTH_CANARY_REVIEW.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (1280, 480))
    rows = []; frame_index = 0
    try:
        while True:
            ok, sbs = capture.read()
            if not ok: break
            if frame_index in sample_set:
                left, right = split_source_index_eyes(
                    sbs, eye_width=eye_width, eye_height=eye_height,
                    physical_left_source_index=1, physical_right_source_index=0,
                )
                left = resize_only(left, width=1280, height=960)
                right = resize_only(right, width=1280, height=960)
                disparity_model, _, _ = model.infer(cv2.flip(left, 1), cv2.flip(right, 1), baseline)
                disparity, depth, geometric = encoded.unflip_disparity_and_depth(
                    disparity_model, focal_px=float(physical_k[0, 0]), baseline_m=baseline,
                )
                disparity_right, _, _ = model.infer(right, left, baseline)
                residual, testable, consistent = encoded._left_right_consistency(
                    disparity, np.asarray(disparity_right, np.float32), geometric,
                )
                valid = geometric & consistent; depth[~valid] = np.nan
                values = depth[valid]; testable_count = int(testable.sum())
                row = {
                    "frame_id": frame_index,
                    "geometric_valid_fraction": float(geometric.mean()),
                    "lr_testable_fraction": float(testable.mean()),
                    "lr_consistent_fraction": float(consistent.sum() / testable_count) if testable_count else 0.0,
                    "final_valid_fraction": float(valid.mean()),
                    "depth_p10_m": float(np.percentile(values, 10)) if values.size else None,
                    "depth_p50_m": float(np.median(values)) if values.size else None,
                    "depth_p90_m": float(np.percentile(values, 90)) if values.size else None,
                }
                atomic_npz(
                    output / "frames" / f"{frame_index:06d}.npz",
                    frame_id=np.asarray(frame_index, np.int32), disparity_physical_left_px=disparity,
                    depth_m=depth, valid=valid, lr_residual_px=residual, lr_consistent=consistent,
                    physical_left_intrinsics=physical_k, model_mirrored_intrinsics=model_k,
                    depth_reference=np.asarray("PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"),
                )
                writer.write(review(left, depth, valid, frame_index)); rows.append(row)
            frame_index += 1
    finally:
        capture.release(); writer.release()
    if len(rows) != len(sample_frames) or model.model_load_count != 1 or model.inference_count != 2 * len(sample_frames):
        raise RuntimeError("MODEL_OR_SAMPLE_ACCOUNTING_DRIFT")
    geometric = np.asarray([row["geometric_valid_fraction"] for row in rows])
    testable = np.asarray([row["lr_testable_fraction"] for row in rows])
    consistent = np.asarray([row["lr_consistent_fraction"] for row in rows])
    final = np.asarray([row["final_valid_fraction"] for row in rows])
    medians = np.asarray([
        np.nan if row["depth_p50_m"] is None else row["depth_p50_m"]
        for row in rows
    ], np.float64)
    adjacent = np.abs(np.diff(medians)); adjacent = adjacent[np.isfinite(adjacent)]
    metrics = {
        "sample_count": len(rows),
        "median_geometric_valid_fraction": float(np.median(geometric)),
        "median_lr_testable_fraction": float(np.median(testable)),
        "median_lr_consistent_fraction": float(np.median(consistent)),
        "fraction_samples_final_valid_at_least_0_15": float(np.mean(final >= 0.15)),
        "temporal_depth_p50_step_p90_m": (
            float(np.percentile(adjacent, 90)) if args.full_session and adjacent.size else None
        ),
    }
    gates = {
        "sample_count": len(rows) == len(sample_frames),
        "geometric_validity": metrics["median_geometric_valid_fraction"] >= 0.25,
        "lr_testable_coverage": metrics["median_lr_testable_fraction"] >= 0.25,
        "lr_consistency": metrics["median_lr_consistent_fraction"] >= 0.50,
        "final_validity": metrics["fraction_samples_final_valid_at_least_0_15"] >= 0.80,
    }
    if args.full_session:
        gates["temporal_distribution_stability"] = (
            metrics["temporal_depth_p50_step_p90_m"] is not None
            and metrics["temporal_depth_p50_step_p90_m"] <= 0.35
        )
    passed = all(gates.values())
    video_path = output / "DEPTH_CANARY_REVIEW.mp4"
    verify = cv2.VideoCapture(str(video_path)); decoded = 0
    while True:
        ok, _ = verify.read()
        if not ok: break
        decoded += 1
    verify.release()
    if decoded != len(rows): raise RuntimeError("REVIEW_DECODE_MISMATCH")
    result = {
        "schema_version": "S1_FOUNDATIONSTEREO_ENCODED_DEPTH_CANARY_V1",
        "task_id": TASK, "session_id": preflight["session_id"], "created_at": now(),
        "status": (
            "PASS_OBJECT6D_SUCCESSOR_AUTHORIZED" if passed and args.full_session
            else "PASS_FULL_SESSION_SUCCESSOR_AUTHORIZED" if passed
            else "REJECTED_QUALITY"
        ),
        "execution_scope": "FULL_SESSION" if args.full_session else "SPARSE_CANARY",
        "input_domain": "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY",
        "lens_undistortion_applied": False,
        "adapter": adapter,
        "checkpoint": ref(CHECKPOINT), "preflight": ref(preflight_path),
        "model_load_count": model.model_load_count, "model_inference_count": model.inference_count,
        "metrics": metrics, "gates": gates, "rows": rows,
        "review_video": {**ref(video_path), "decoded_frames": decoded},
        "local_stereo_metric_dev": passed, "external_metric_authority": False,
        "contact_authority": False, "training_eligible": False,
        "claim_limit": "Sparse same-session visible-surface optical-Z canary only; no hidden geometry, Contact truth or external accuracy.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "metrics": metrics}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
