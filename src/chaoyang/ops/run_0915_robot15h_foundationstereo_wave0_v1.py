#!/usr/bin/env python3
"""Run one persistent FoundationStereo model across the four frozen 0915 W0 sessions."""

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
from typing import Any, Mapping
import uuid

import cv2
import numpy as np

from chaoyang.governance.robot15h_task_specs_v1 import FOUNDATION_WEIGHT, WINDOW_RUN_ID, build_packet
from chaoyang.ops import run_0915_foundationstereo_encoded_domain_canary_v1 as frozen
from chaoyang.ops import run_0915_foundationstereo_single_session_canary_v1 as common
from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import (
    EncodedStereoGateV1,
    aggregate_metrics as aggregate_preflight_metrics,
    frame_metrics as preflight_frame_metrics,
    robust_correspondences,
)
from chaoyang.pipeline.vst_encoded_video_domain import split_resize_physical_eyes


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot15h_foundationstereo_wave0_v1"
PHASE = "ROBOT15H_FOUNDATIONSTEREO_WAVE0_ENCODED_DOMAIN"
INVENTORY = ROOT / "_run/current/0915_robot15h_window_start_inventory_v1/attempts/attempt_0001/BATCH_MANIFEST.json"
AUTHORIZATION = ROOT / "tasks/receipts/0915_ROBOT15H_USER_AUTHORIZATION_V1.json"
DOMAIN = ROOT / "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json"
CONFIG = ROOT / "configs/systems/depth/foundationstereo_0915_encoded_domain_canary_v1.json"
SCALE_AUDIT = ROOT / "_run/current/0915_robot15h_scale_cause_audit_v1/attempts/attempt_0001/RESULT.json"
GPU_LAUNCHER = ROOT / "src/chaoyang/ops/foundationstereo_gpu_python.sh"
GPU_LEASE_WRAPPER = ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"
OUTPUT = ROOT / f"_run/current/{TASK_ID}/attempts/attempt_0001"
VISUAL = ROOT / "docs/current/visuals/0915_ROBOT15H_W0_FOUNDATIONSTEREO_V1"
TERMINAL_RECEIPT = ROOT / "tasks/receipts/0915_ROBOT15H_FOUNDATIONSTEREO_WAVE0_V1_RESULT.json"
DEPTH_SIZE = (640, 480)
EYE_SIZE = (1280, 960)
AUTHORIZED_SCOPE = "VISUAL_OBJECT6D_CANDIDATE_INPUT"


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


def canonical_sha(value: Any) -> str:
    return common.canonical_sha256(value)


def sha256(path: Path) -> str:
    return common.sha256(path)


def validate_route() -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = load_json(packet_path)
    if packet != build_packet(TASK_ID) or packet.get("weights") != [FOUNDATION_WEIGHT]:
        raise RuntimeError("current FoundationStereo W0 packet differs from frozen model spec")
    state = load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("FoundationStereo W0 is not current next_task")
    index = load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True or route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("FoundationStereo W0 is not the SHA-bound routable task")
    return packet, packet_path


def heartbeat(status: str, gpu_id: int | None = None) -> None:
    command = [
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(os.getpid()), "--status", status, "--phase", PHASE,
    ]
    if gpu_id is not None:
        command += ["--gpu-id", str(gpu_id)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError("governance heartbeat failed: " + (completed.stderr or completed.stdout)[-4000:])


def w0_rows() -> list[dict[str, Any]]:
    manifest = load_json(INVENTORY)
    rows = [row for row in manifest.get("sessions", []) if row.get("wave") == "W0"]
    if len(rows) != 4 or any(row.get("split") != "development" for row in rows):
        raise RuntimeError("frozen FoundationStereo W0 cohort drift")
    if sum(int(row["frame_count"]) for row in rows) != 1058:
        raise RuntimeError("frozen FoundationStereo W0 frame denominator drift")
    return rows


def camera_path(row: Mapping[str, Any]) -> Path:
    return Path(str(row["source_stereo"]["path"])).parent.parent / "camera_params.json"


def build_signature(
    packet_path: Path, executor_epoch: int, preflight: Mapping[str, Any],
) -> dict[str, Any]:
    rows = w0_rows()
    auth = load_json(AUTHORIZATION)
    domain = load_json(DOMAIN)
    scale = load_json(SCALE_AUDIT)
    if auth.get("status") != "AUTHORIZED" or auth.get("run_id") != WINDOW_RUN_ID:
        raise RuntimeError("Robot15h user authorization drift")
    if domain.get("status") != "CONFIRMED_ENCODED_VIDEO_ALREADY_UNDISTORTED":
        raise RuntimeError("VST encoded-domain confirmation drift")
    if scale.get("metric_translation_authorized") is not False:
        raise RuntimeError("scale audit authority unexpectedly changed")
    inputs = []
    for row in rows:
        video = Path(str(row["source_stereo"]["path"])).resolve(strict=True)
        if row["source_stereo"].get("sha256") != sha256(video):
            raise RuntimeError(f"W0 source SHA drift: {row['session_id']}")
        inputs.append({
            "session_id": row["session_id"], "task": row["task"],
            "source_group": row["source_group"], "frame_count": int(row["frame_count"]),
            "source_stereo": ref(video), "camera_params": ref(camera_path(row)),
        })
    payload = {
        "schema_version": "0915-robot15h-foundationstereo-wave0-run-signature-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID,
        "executor_epoch": executor_epoch, "weights": [FOUNDATION_WEIGHT],
        "sessions": inputs,
        "contracts": {"authorization": ref(AUTHORIZATION), "domain": ref(DOMAIN),
                      "config": ref(CONFIG), "scale_audit": ref(SCALE_AUDIT),
                      "task_packet": ref(packet_path)},
        "runtime": {
            "runner": ref(Path(__file__)),
            "frozen_canary_implementation": ref(Path(frozen.__file__)),
            "gpu_launcher": ref(GPU_LAUNCHER), "lease_wrapper": ref(GPU_LEASE_WRAPPER),
            "checkpoint": ref(ROOT / FOUNDATION_WEIGHT),
        },
        "pixel_domain": {
            "eyes": {"physical_left": 1, "physical_right": 0},
            "source_transform": "CROP_THEN_RESIZE_ONLY",
            "model_adapter": "SIMULTANEOUS_HORIZONTAL_REFLECTION_NO_CAMERA_SWAP",
            "output_transform": "HORIZONTAL_UNFLIP_TO_PHYSICAL_LEFT",
            "lens_undistortion_applied": False,
        },
        "metric_authority": "ENCODED_K_P_EXTERNAL_ACCURACY_UNVERIFIED",
        "preflight": {
            "policy": "24_FROZEN_UNIFORM_FRAMES_PER_SESSION_FULL_DECODE",
            "sha256": canonical_sha(preflight),
            "admitted_sessions": [
                row["session_id"] for row in preflight["sessions"] if row["gpu_admitted"]
            ],
        },
    }
    return {**payload, "run_signature_sha256": canonical_sha(payload)}


def validate_claim(path: Path, signature_sha: str, executor_epoch: int, require_descendant: bool) -> None:
    claim = load_json(path)
    pid = claim.get("pid")
    if (
        claim.get("task_id") != TASK_ID or claim.get("weights") != [FOUNDATION_WEIGHT]
        or claim.get("status") != "CLAIMED" or claim.get("executor_epoch") != executor_epoch
        or claim.get("run_signature_sha256") != signature_sha
        or claim.get("unique_write_root") != str(OUTPUT.resolve()) or not isinstance(pid, int)
        or common.process_start_ticks(pid) != claim.get("proc_start_ticks")
    ):
        raise RuntimeError("FoundationStereo W0 writer fence mismatch")
    if require_descendant and not common.process_has_ancestor(os.getpid(), pid):
        raise RuntimeError("FoundationStereo GPU worker is outside writer ancestry")


def aggregate(rows: list[dict[str, Any]], alignment: Mapping[str, Any], expected: int) -> dict[str, Any]:
    thresholds = load_json(CONFIG)["quality_thresholds"]
    geometric = np.asarray([row["geometric_valid_fraction"] for row in rows], np.float64)
    testable = np.asarray([row["lr_testable_fraction"] for row in rows], np.float64)
    consistent = np.asarray([row["lr_consistent_fraction"] for row in rows], np.float64)
    final = np.asarray([row["final_valid_fraction"] for row in rows], np.float64)
    medians = [row["depth_p50_m"] for row in rows]
    steps = [abs(float(b) - float(a)) for a, b in zip(medians, medians[1:]) if a is not None and b is not None]
    metrics = {
        "decoded_frame_count": len(rows),
        "median_geometric_valid_fraction": float(np.median(geometric)) if len(rows) else 0.0,
        "median_lr_testable_fraction": float(np.median(testable)) if len(rows) else 0.0,
        "median_lr_consistent_fraction": float(np.median(consistent)) if len(rows) else 0.0,
        "fraction_frames_final_valid_at_least_0_15": float(np.mean(final >= 0.15)) if len(rows) else 0.0,
        "temporal_depth_median_step_p90_m": float(np.percentile(steps, 90)) if steps else None,
        "rgb_roundtrip_max_abs_error": int(alignment["maximum_absolute_channel_error"]),
        "rgb_roundtrip_max_mismatched_pixels": int(alignment["mismatched_pixels"]),
        "coordinate_roundtrip_max_abs_error_px": float(alignment["coordinate_roundtrip_max_abs_error_px"]),
    }
    gates = {
        "full_decode": len(rows) == expected,
        "geometric_validity": metrics["median_geometric_valid_fraction"] >= thresholds["minimum_median_geometric_valid_fraction"],
        "lr_testable_coverage": metrics["median_lr_testable_fraction"] >= thresholds["minimum_median_lr_testable_fraction"],
        "lr_consistency": metrics["median_lr_consistent_fraction"] >= thresholds["minimum_median_lr_consistent_fraction"],
        "final_validity": metrics["fraction_frames_final_valid_at_least_0_15"] >= thresholds["minimum_fraction_frames_final_valid_at_least_0_15"],
        "temporal_distribution_stability": metrics["temporal_depth_median_step_p90_m"] is not None and metrics["temporal_depth_median_step_p90_m"] <= thresholds["maximum_temporal_depth_median_step_p90_m"],
        "pixelwise_rgb_alignment": (
            metrics["rgb_roundtrip_max_abs_error"] <= thresholds["rgb_roundtrip_max_abs_error"]
            and metrics["rgb_roundtrip_max_mismatched_pixels"] <= thresholds["rgb_roundtrip_max_mismatched_pixels"]
            and metrics["coordinate_roundtrip_max_abs_error_px"] <= thresholds["coordinate_roundtrip_max_abs_error_px"]
        ),
        "array_and_formula_contract": all(
            row.get("array_contract_passed") is True
            and float(row.get("depth_formula_max_abs_error_m", 1.0)) <= 1e-6
            for row in rows
        ),
    }
    return {"thresholds": thresholds, "metrics": metrics, "gates": gates, "passed": all(gates.values())}


def preflight_session(row: Mapping[str, Any], sample_count: int = 24) -> dict[str, Any]:
    """Fully decode one session and diagnose a frozen uniform subset without remapping."""

    expected = int(row["frame_count"])
    selected = set(np.linspace(0, expected - 1, min(sample_count, expected), dtype=int).tolist())
    source = Path(str(row["source_stereo"]["path"])).resolve(strict=True)
    camera = load_json(camera_path(row))
    eye_width, eye_height = int(camera["width"]), int(camera["height"])
    if (int(camera["left"]["sourceIndex"]), int(camera["right"]["sourceIndex"])) != (1, 0):
        raise RuntimeError("physical eye sourceIndex drift during preflight")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("cannot open W0 SBS during preflight")
    rows: list[dict[str, Any]] = []
    verticals: list[np.ndarray] = []
    disparities: list[np.ndarray] = []
    decoded = 0
    try:
        while True:
            ok, sbs = capture.read()
            if not ok:
                break
            if sbs.shape[:2] != (eye_height, eye_width * 2):
                raise RuntimeError(f"preflight SBS geometry drift at frame {decoded}")
            if decoded in selected:
                left, right = split_resize_physical_eyes(
                    sbs, eye_width=eye_width, source_indices=(1, 0), output_size=EYE_SIZE,
                )
                points_left, points_right = robust_correspondences(left, right)
                metrics = preflight_frame_metrics(
                    points_left, points_right, width=EYE_SIZE[0], height=EYE_SIZE[1],
                )
                rows.append({"frame_index": decoded, **metrics})
                verticals.append(
                    np.abs(points_left[:, 1] - points_right[:, 1])
                    if len(points_left) else np.empty(0, np.float64)
                )
                disparities.append(
                    points_left[:, 0] - points_right[:, 0]
                    if len(points_left) else np.empty(0, np.float64)
                )
            decoded += 1
    finally:
        capture.release()
    if decoded != expected:
        raise RuntimeError(f"preflight full decode mismatch: {decoded}!={expected}")
    aggregate = aggregate_preflight_metrics(rows, verticals, disparities, gate=EncodedStereoGateV1())
    sign_ok = aggregate["metrics"]["dominant_disparity_sign"] == "NEGATIVE_LEFT_MINUS_RIGHT"
    return {
        "session_id": row["session_id"], "task": row["task"],
        "source_group": row["source_group"], "decoded_frame_count": decoded,
        "sampled_frame_count": len(rows), "sampled_frame_ids": sorted(selected),
        "decision": aggregate["decision"], "dominant_sign_required": "NEGATIVE_LEFT_MINUS_RIGHT",
        "gpu_admitted": bool(aggregate["gpu_successor_authorized"] and sign_ok),
        "metrics": aggregate["metrics"], "gates": {**aggregate["gates"], "dominant_negative_sign": sign_ok},
        "thresholds": aggregate["thresholds"], "lens_undistortion_applied": False,
    }


def run_preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = [preflight_session(row) for row in rows]
    return {
        "schema_version": "0915-robot15h-foundationstereo-wave0-preflight-v1",
        "window_run_id": WINDOW_RUN_ID, "status": "COMPLETED",
        "sample_policy": "24_UNIFORM_FRAMES_PER_SESSION_WITH_FULL_VIDEO_DECODE",
        "sessions": results,
        "counts": {"total": len(results), "gpu_admitted": sum(row["gpu_admitted"] for row in results),
                   "rejected_quality": sum(not row["gpu_admitted"] for row in results)},
        "lens_undistortion_applied": False,
    }


def process_session(model: Any, row: Mapping[str, Any], visual: Path) -> dict[str, Any]:
    session_id = str(row["session_id"])
    task = str(row["task"])
    expected = int(row["frame_count"])
    source = Path(str(row["source_stereo"]["path"])).resolve(strict=True)
    camera_file = camera_path(row).resolve(strict=True)
    camera = load_json(camera_file)
    if (int(camera["left"]["sourceIndex"]), int(camera["right"]["sourceIndex"])) != (1, 0):
        raise RuntimeError("physical eye sourceIndex drift")
    eye_width, eye_height = int(camera["width"]), int(camera["height"])
    baseline_m = frozen.camera_baseline_m(camera)
    physical_k = frozen.scaled_intrinsics(camera, eye="left", width=DEPTH_SIZE[0], height=DEPTH_SIZE[1])
    model_k = frozen.mirrored_intrinsics(physical_k, DEPTH_SIZE[0])
    source_before = {"video": ref(source), "camera": ref(camera_file)}
    target = OUTPUT / "sessions" / task / session_id
    staging = target.with_name(f".{session_id}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    (staging / "frames").mkdir()
    review_tmp = staging / f"{session_id}_FOUNDATIONSTEREO_REVIEW.mp4"
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("cannot open W0 SBS")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(str(review_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 480))
    if not writer.isOpened():
        raise RuntimeError("cannot open W0 Depth review writer")
    rows: list[dict[str, Any]] = []
    maximum_error, mismatched = 0, 0
    first_physical = first_returned = None
    inference_before = int(model.inference_count)
    started = time.monotonic()
    try:
        for frame_index in range(expected):
            ok, sbs = capture.read()
            if not ok:
                raise RuntimeError(f"SBS decode ended at {frame_index}/{expected}")
            if sbs.shape[:2] != (eye_height, eye_width * 2):
                raise RuntimeError(f"SBS geometry drift at frame {frame_index}")
            left, right = split_resize_physical_eyes(
                sbs, eye_width=eye_width, source_indices=(1, 0), output_size=EYE_SIZE,
            )
            alignment = frozen.rgb_roundtrip_metrics(left)
            maximum_error = max(maximum_error, int(alignment["maximum_absolute_channel_error"]))
            mismatched += int(alignment["mismatched_pixels"])
            if first_physical is None:
                first_physical = alignment["physical_left_depth_rgb_sha256"]
                first_returned = alignment["unflipped_model_left_sha256"]
            disparity_model, _unused_depth, _unused_valid = model.infer(
                cv2.flip(left, 1), cv2.flip(right, 1), baseline_m,
            )
            disparity, depth, geometric = frozen.unflip_disparity_and_depth(
                disparity_model, focal_px=float(physical_k[0, 0]), baseline_m=baseline_m,
            )
            disparity_right, _unused_depth, _unused_valid = model.infer(right, left, baseline_m)
            residual, testable, consistent = frozen._left_right_consistency(
                disparity, np.asarray(disparity_right, np.float32), geometric,
            )
            valid = geometric & consistent
            depth[~valid] = np.nan
            recomputed = np.full_like(depth, np.nan)
            positive = valid & np.isfinite(disparity) & (disparity > 0)
            recomputed[positive] = float(physical_k[0, 0]) * baseline_m / disparity[positive]
            formula_error = (
                float(np.max(np.abs(recomputed[positive] - depth[positive])))
                if np.any(positive) else 0.0
            )
            array_contract = bool(
                depth.shape == (DEPTH_SIZE[1], DEPTH_SIZE[0])
                and disparity.shape == depth.shape and valid.shape == depth.shape
                and np.isnan(depth[~valid]).all() and np.isfinite(depth[valid]).all()
            )
            values = depth[valid]
            testable_count = int(testable.sum())
            item = {
                "frame": frame_index,
                "geometric_valid_fraction": float(geometric.mean()),
                "lr_testable_fraction": float(testable.mean()),
                "lr_consistent_fraction": float(consistent.sum() / testable_count) if testable_count else 0.0,
                "final_valid_fraction": float(valid.mean()),
                "depth_p10_m": float(np.percentile(values, 10)) if values.size else None,
                "depth_p50_m": float(np.median(values)) if values.size else None,
                "depth_p90_m": float(np.percentile(values, 90)) if values.size else None,
                "depth_formula_max_abs_error_m": formula_error,
                "array_contract_passed": array_contract,
            }
            frame_path = staging / "frames" / f"{frame_index:06d}.npz"
            atomic_npz(
                frame_path, frame_id=np.asarray(frame_index, np.int32),
                disparity_physical_left_px=disparity.astype(np.float32), depth_m=depth.astype(np.float32),
                correspondence_magnitude_px=disparity.astype(np.float32),
                valid=valid.astype(bool), lr_residual_px=residual.astype(np.float32),
                lr_consistent=consistent.astype(bool), physical_left_intrinsics=physical_k,
                model_mirrored_intrinsics=model_k,
                depth_reference=np.asarray("PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"),
            )
            item["artifact"] = common.published_ref(frame_path, target / "frames" / frame_path.name)
            rows.append(item)
            writer.write(frozen._review_frame(left, depth, valid, frame_index))
            if frame_index % 30 == 0:
                print(json.dumps({"session_id": session_id, "frame": frame_index, "expected": expected}), flush=True)
        extra, _ = capture.read()
        if extra:
            raise RuntimeError("SBS contains frames beyond frozen denominator")
    finally:
        capture.release()
        writer.release()
    if int(model.inference_count) - inference_before != expected * 2:
        raise RuntimeError("persistent-model inference accounting drift")
    alignment = {
        "schema_version": "0915-foundationstereo-pixelwise-rgb-alignment-v1",
        "frame_count": expected,
        "comparison": "resize(physical_left)==unflip(resize(flip(physical_left)))",
        "maximum_absolute_channel_error": maximum_error, "mismatched_pixels": mismatched,
        "coordinate_roundtrip_max_abs_error_px": frozen.coordinate_roundtrip_error(DEPTH_SIZE[0]),
        "first_frame_physical_left_depth_rgb_sha256": first_physical,
        "first_frame_unflipped_model_left_sha256": first_returned,
        "depth_grid_domain": "ORIGINAL_PHYSICAL_LEFT_640x480",
    }
    quality = aggregate(rows, alignment, expected)
    passed = bool(quality["passed"])
    atomic_json(staging / "ADAPTER_CONTRACT.json", {
        "schema_version": "0915-robot15h-foundationstereo-adapter-contract-v1",
        "physical_source_indices": {"left": 1, "right": 0}, "camera_swap": False,
        "horizontal_reflection_for_disparity_sign": True, "both_eyes_reflected": True,
        "lens_undistortion_applied": False, "lens_remap_applied": False,
        "model_domain": "SIMULTANEOUSLY_MIRRORED_PHYSICAL_LEFT_RIGHT",
        "model_intrinsics": model_k.tolist(), "output_spatial_unflip": True,
        "output_domain": "ORIGINAL_PHYSICAL_LEFT_640x480", "output_intrinsics": physical_k.tolist(),
        "baseline_m": baseline_m,
        "encoded_projection_matrices": "ABSENT_EXTERNAL_ACCURACY_UNVERIFIED",
    })
    atomic_json(staging / "RGB_ALIGNMENT_QA.json", alignment)
    atomic_json(staging / "DEPTH_CONTRACT.json", {
        "schema_version": "0915-robot15h-foundationstereo-depth-contract-v1",
        "task_id": TASK_ID, "session_id": session_id, "source_group": row["source_group"],
        "frame_count": expected, "frame_geometry": list(DEPTH_SIZE),
        "depth_reference": "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z",
        "formula": "optical_z_m = scaled_factory_left_fx_px * camera_centre_baseline_m / unflipped_disparity_px",
        "disparity_semantics": "POSITIVE_RIGHTWARD_CORRESPONDENCE_MAGNITUDE_IN_PHYSICAL_LEFT_PIXEL_GRID",
        "physical_left_intrinsics": physical_k.tolist(), "model_mirrored_intrinsics": model_k.tolist(),
        "baseline_m": baseline_m, "external_accuracy": "UNVERIFIED",
        "encoded_projection_matrices": "ABSENT", "strict_metric_contact_authorized": False,
        "consumption_authorized": passed,
        "authorized_scopes": [AUTHORIZED_SCOPE] if passed else [],
        "occluded_or_hidden_geometry": "INVALID_NOT_COMPLETED",
    })
    summary = {
        "schema_version": "0915-robot15h-foundationstereo-session-summary-v1",
        "task_id": TASK_ID, "session_id": session_id, "source_group": row["source_group"],
        "status": "PASS" if passed else "REJECTED_QUALITY", "frame_count": expected,
        "external_accuracy": "UNVERIFIED", "strict_metric_contact_authorized": False,
        "consumption_authorized": passed, "authorized_scopes": [AUTHORIZED_SCOPE] if passed else [],
        "model_load_count_batch": int(model.model_load_count),
        "session_inference_count": int(model.inference_count) - inference_before,
        "quality": quality, "frames": rows, "wall_seconds": time.monotonic() - started,
        "source_mutated": False,
    }
    atomic_json(staging / "DEPTH_SUMMARY.json", summary)
    visual.mkdir(parents=True, exist_ok=True)
    final_review = visual / f"{session_id}_FOUNDATIONSTEREO_REVIEW.mp4"
    os.replace(review_tmp, final_review)
    decode = common._review_decode(final_review, expected)
    summary["review_decode"] = decode
    atomic_json(staging / "DEPTH_SUMMARY.json", summary)
    atomic_json(staging / "RESULT.json", {
        "schema_version": "0915-robot15h-foundationstereo-session-result-v1",
        "session_id": session_id, "task": task, "source_group": row["source_group"],
        "status": "PASSED" if passed else "REJECTED_QUALITY",
        "frame_count": expected, "external_accuracy": "UNVERIFIED",
        "strict_metric_contact_authorized": False, "consumption_authorized": passed,
        "authorized_scopes": [AUTHORIZED_SCOPE] if passed else [], "source_mutated": False,
        "review": {"video": ref(final_review), **decode},
    })
    if {"video": ref(source), "camera": ref(camera_file)} != source_before:
        raise RuntimeError("source input changed during FoundationStereo W0")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, target)
    result = load_json(target / "RESULT.json")
    result["result"] = ref(target / "RESULT.json")
    return result


def run_worker(claim_path: Path, signature_path: Path, executor_epoch: int) -> int:
    packet, _ = validate_route()
    signature = load_json(signature_path)
    stored = signature.pop("run_signature_sha256", None)
    if stored != canonical_sha(signature):
        raise RuntimeError("FoundationStereo W0 signature digest mismatch")
    signature["run_signature_sha256"] = stored
    validate_claim(claim_path, stored, executor_epoch, require_descendant=True)
    if packet["weights"] != [FOUNDATION_WEIGHT]:
        raise RuntimeError("FoundationStereo W0 weight identity drift")
    preflight = load_json(OUTPUT / "PREFLIGHT_BATCH.json")
    admission = {row["session_id"]: bool(row["gpu_admitted"]) for row in preflight["sessions"]}
    admitted_rows = [row for row in w0_rows() if admission.get(row["session_id"]) is True]
    if not admitted_rows:
        raise RuntimeError("worker invoked with no preflight-admitted sessions")
    model = frozen.fs_worker.Model()
    results: list[dict[str, Any]] = []
    for row in w0_rows():
        if not admission.get(row["session_id"], False):
            target = OUTPUT / "sessions" / str(row["task"]) / str(row["session_id"])
            target.mkdir(parents=True, exist_ok=True)
            rejection = {
                "schema_version": "0915-robot15h-foundationstereo-session-result-v1",
                "session_id": row["session_id"], "task": row["task"], "source_group": row["source_group"],
                "status": "REJECTED_QUALITY", "frame_count": int(row["frame_count"]),
                "first_blocker": "ENCODED_DOMAIN_PREFLIGHT_REJECTED", "consumption_authorized": False,
                "strict_metric_contact_authorized": False, "external_accuracy": "UNVERIFIED",
            }
            atomic_json(target / "RESULT.json", rejection)
            results.append({**rejection, "result": ref(target / "RESULT.json")})
            continue
        try:
            results.append(process_session(model, row, VISUAL))
        except Exception as error:  # preserve one session terminal and continue the frozen cohort
            target = OUTPUT / "sessions" / str(row["task"]) / str(row["session_id"])
            target.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": "0915-robot15h-foundationstereo-session-result-v1",
                "session_id": row["session_id"], "task": row["task"], "source_group": row["source_group"],
                "status": "FAILED_RUNTIME", "frame_count": int(row["frame_count"]),
                "first_blocker": f"{type(error).__name__}:{error}", "consumption_authorized": False,
                "strict_metric_contact_authorized": False, "external_accuracy": "UNVERIFIED",
            }
            atomic_json(target / "RESULT.json", failure)
            results.append({**failure, "result": ref(target / "RESULT.json")})
    if int(model.model_load_count) != 1:
        raise RuntimeError("FoundationStereo was not loaded exactly once")
    expected_inferences = 2 * sum(int(row["frame_count"]) for row in admitted_rows)
    if int(model.inference_count) != expected_inferences:
        raise RuntimeError("FoundationStereo W0 total inference accounting drift")
    passed = sum(row["status"] == "PASSED" for row in results)
    rejected = sum(row["status"] == "REJECTED_QUALITY" for row in results)
    failed = sum(row["status"] == "FAILED_RUNTIME" for row in results)
    batch = {
        "schema_version": "0915-robot15h-foundationstereo-wave0-batch-v1",
        "window_run_id": WINDOW_RUN_ID, "status": "COMPLETED_ALL_TERMINAL",
        "counts": {"attempted": 4, "passed": passed, "rejected_quality": rejected, "failed_runtime": failed},
        "model_load_count": int(model.model_load_count), "model_inference_count": int(model.inference_count),
        "expected_inference_count": expected_inferences,
        "sessions": results, "external_accuracy": "UNVERIFIED",
        "strict_metric_contact_authorized": False, "source_mutated": False,
    }
    atomic_json(OUTPUT / "BATCH_RESULT.json", batch)
    return 0 if failed == 0 else 2


def write_terminal(packet: Mapping[str, Any], status: str, blocker: str | None) -> None:
    batch = load_json(OUTPUT / "BATCH_RESULT.json") if (OUTPUT / "BATCH_RESULT.json").is_file() else None
    counts = batch.get("counts") if batch else {"attempted": 0, "passed": 0, "rejected_quality": 0, "failed_runtime": 4}
    result = {
        "schema_version": "0915-robot15h-foundationstereo-wave0-result-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": status,
        "first_blocker": blocker, "weights": packet["weights"], "counts": counts,
        "horizontal_reflection_for_disparity_sign": True,
        "output_domain": "ORIGINAL_PHYSICAL_LEFT_640x480", "lens_undistortion_applied": False,
        "external_accuracy": "UNVERIFIED", "strict_metric_contact_authorized": False,
        "source_mutated": False, "training_eligible": False, "physical_deployment_authorized": False,
        "batch_result": ref(OUTPUT / "BATCH_RESULT.json") if batch else None,
        "gpu_command_receipt": ref(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else None,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(OUTPUT / "RESULT.json", result)
    atomic_json(TERMINAL_RECEIPT, {**result, "result": ref(OUTPUT / "RESULT.json")})
    atomic_json(OUTPUT / "RUN_RECEIPT.json", {
        "schema_version": "0915-robot15h-foundationstereo-wave0-run-receipt-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": status,
        "result": ref(OUTPUT / "RESULT.json"), "terminal_receipt": ref(TERMINAL_RECEIPT),
    })


def run_orchestrator(args: argparse.Namespace) -> int:
    packet, packet_path = validate_route()
    if args.output_root.resolve() != OUTPUT.resolve() or args.visual_root.resolve() != VISUAL.resolve() or args.receipt.resolve() != TERMINAL_RECEIPT.resolve():
        raise RuntimeError("FoundationStereo W0 namespace drift")
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        raise RuntimeError("positive epoch and fencing token >=16 characters required")
    for path in (OUTPUT, VISUAL, TERMINAL_RECEIPT):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh output required: {path}")
    rows = w0_rows()
    preflight = run_preflight(rows)
    signature = build_signature(packet_path, args.executor_epoch, preflight)
    OUTPUT.mkdir(parents=True)
    atomic_json_new(OUTPUT / "RUN_SIGNATURE.json", signature)
    atomic_json_new(OUTPUT / "PREFLIGHT_BATCH.json", preflight)
    claim = {
        "schema_version": "0915-robot15h-foundationstereo-wave0-writer-claim-v1",
        "task_id": TASK_ID, "window_run_id": WINDOW_RUN_ID, "status": "CLAIMED",
        "weights": packet["weights"], "claimed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(), "proc_start_ticks": common.process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(args.fencing_token.encode()).hexdigest(),
        "run_signature_sha256": signature["run_signature_sha256"], "unique_write_root": str(OUTPUT.resolve()),
    }
    atomic_json_new(OUTPUT / "CLAIM.json", claim)
    if preflight["counts"]["gpu_admitted"] == 0:
        results = []
        for row in rows:
            target = OUTPUT / "sessions" / str(row["task"]) / str(row["session_id"])
            target.mkdir(parents=True, exist_ok=True)
            rejection = {
                "schema_version": "0915-robot15h-foundationstereo-session-result-v1",
                "session_id": row["session_id"], "task": row["task"], "source_group": row["source_group"],
                "status": "REJECTED_QUALITY", "frame_count": int(row["frame_count"]),
                "first_blocker": "ENCODED_DOMAIN_PREFLIGHT_REJECTED", "consumption_authorized": False,
                "strict_metric_contact_authorized": False, "external_accuracy": "UNVERIFIED",
            }
            atomic_json(target / "RESULT.json", rejection)
            results.append({**rejection, "result": ref(target / "RESULT.json")})
        atomic_json(OUTPUT / "BATCH_RESULT.json", {
            "schema_version": "0915-robot15h-foundationstereo-wave0-batch-v1",
            "window_run_id": WINDOW_RUN_ID, "status": "COMPLETED_ALL_TERMINAL",
            "counts": {"attempted": 4, "passed": 0, "rejected_quality": 4, "failed_runtime": 0},
            "model_load_count": 0, "model_inference_count": 0, "expected_inference_count": 0,
            "sessions": results, "external_accuracy": "UNVERIFIED",
            "strict_metric_contact_authorized": False, "source_mutated": False,
        })
        atomic_json(OUTPUT / "GPU_COMMAND_RECEIPT.json", {
            "schema_version": "gpu-command-receipt-v71",
            "task_id": TASK_ID, "attempt_id": OUTPUT.name,
            "status": "SKIPPED_NO_PREFLIGHT_ADMITTED_SESSIONS",
            "gpu_acquired": False, "worker_executed": False,
        })
        write_terminal(packet, "PASSED", None)
        return 0
    heartbeat("WAIT_GPU_RESOURCE")
    worker_command = [
        str(GPU_LAUNCHER), str(Path(__file__).resolve()), "--worker",
        "--output-root", str(OUTPUT), "--visual-root", str(VISUAL),
        "--executor-epoch", str(args.executor_epoch), "--claim", str(OUTPUT / "CLAIM.json"),
        "--run-signature", str(OUTPUT / "RUN_SIGNATURE.json"),
    ]
    lease_command = [
        sys.executable, str(GPU_LEASE_WRAPPER), "--task-id", TASK_ID,
        "--attempt-id", OUTPUT.name, "--executor-epoch", str(args.executor_epoch),
        "--priority", "CANARY", "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib), "--wait-seconds", str(args.gpu_wait_seconds),
        "--wall-seconds", str(args.wall_seconds), "--receipt", str(OUTPUT / "GPU_COMMAND_RECEIPT.json"),
        "--claim-limit", packet["claim_limit"], "--", *worker_command,
    ]
    atomic_json(OUTPUT / "COMMAND.json", {"worker_command": worker_command, "lease_command": lease_command})
    with (OUTPUT / "GPU_WRAPPER.log").open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(lease_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        while process.poll() is None:
            time.sleep(30)
            if process.poll() is not None:
                break
            lease_path = ROOT / "_run/current/GPU_LEASE.json"
            lease = load_json(lease_path) if lease_path.is_file() else {}
            acquired = lease.get("status") == "ACQUIRED" and lease.get("task_id") == TASK_ID
            heartbeat("RUNNING" if acquired else "WAIT_GPU_RESOURCE", args.gpu_id if acquired else None)
    gpu = load_json(OUTPUT / "GPU_COMMAND_RECEIPT.json") if (OUTPUT / "GPU_COMMAND_RECEIPT.json").is_file() else {}
    if process.returncode != 0 or gpu.get("status") != "PASSED":
        status = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        write_terminal(packet, status, str(gpu.get("reason") or gpu.get("error") or "FOUNDATIONSTEREO_W0_RUNTIME_FAILED"))
        return 3 if status == "BLOCKED_RESOURCE" else 2
    write_terminal(packet, "PASSED", None)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--visual-root", required=True, type=Path)
    parser.add_argument("--receipt", type=Path, default=TERMINAL_RECEIPT)
    parser.add_argument("--executor-epoch", required=True, type=int)
    parser.add_argument("--fencing-token")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--wall-seconds", type=int, default=10_800)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--claim", type=Path)
    parser.add_argument("--run-signature", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.claim is None or args.run_signature is None:
            raise RuntimeError("worker requires claim and signature")
        return run_worker(args.claim.resolve(strict=True), args.run_signature.resolve(strict=True), args.executor_epoch)
    if args.fencing_token is None:
        raise RuntimeError("orchestrator requires fencing token")
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
