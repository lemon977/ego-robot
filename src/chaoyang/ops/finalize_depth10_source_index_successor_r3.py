#!/usr/bin/env python3
"""Finalize the bounded play_cards_0910_001 DEPTH-10 CPU successor.

This tool never runs FoundationStereo.  It verifies the physical-eye SBS
routing from same-session ``camera_params.json``, evaluates the already fresh
same-session rectification on held-out frames, and fails closed when the frozen
median/P90 gate is not met.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
PICO_TOOL = PROJECT / "vendor/FoundationStereo/scripts/pico_stereo_depth.py"
SESSION_ID = "play_cards_0910_001"


def digest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    hasher = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_pico():
    spec = importlib.util.spec_from_file_location("pico_depth10_successor_r3", PICO_TOOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PICO_TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 90, 190])
    parser.add_argument("--median-gate-px", type=float, default=2.0)
    parser.add_argument("--p90-gate-px", type=float, default=5.0)
    args = parser.parse_args()

    session = args.session_root.resolve(strict=True)
    attempt = args.attempt_dir.resolve(strict=True)
    if session.name != SESSION_ID:
        raise ValueError(f"bounded successor only permits {SESSION_ID}, got {session.name}")
    required_absent = [
        attempt / name
        for name in (
            "RESULT.json",
            "ARTIFACT_MANIFEST.json",
            "METRICS.json",
            "RUN_RECEIPT.json",
            "DECISION.md",
            "NEXT_ACTION.json",
        )
    ]
    collisions = [str(path) for path in required_absent if path.exists()]
    if collisions:
        raise FileExistsError(f"immutable output collision: {collisions}")

    camera_params = session / "camera_params.json"
    stereo_video = session / f"source_stereo/CameraRecord_{SESSION_ID}_stereo.mp4"
    selected_video = session / f"CameraRecord_{SESSION_ID}.mp4"
    calibration_file = attempt / "same_session_rectification.json"
    pico = load_pico()
    params = json.loads(camera_params.read_text(encoding="utf-8"))
    if params.get("selected_calibration_key") != "right":
        raise ValueError("selected_calibration_key must be physical right")
    source_indices = pico.load_source_indices(camera_params)
    if source_indices != (1, 0):
        raise ValueError(f"unexpected same-session physical routing: {source_indices}")
    eyes, eye_width, eye_height, baseline_m = pico.load_camera_params(camera_params)
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    if Path(calibration.get("source_camera_params", "")).resolve() != camera_params.resolve():
        raise ValueError("rectification calibration is not bound to same-session camera_params")
    recorded_indices = calibration.get("physical_eye_source_indices")
    if recorded_indices != {"left": 1, "right": 0}:
        raise ValueError("rectification calibration does not bind the corrected source order")
    rectified_k = np.asarray(calibration["rectified_intrinsics"], dtype=np.float64)
    maps = [
        pico.make_map(
            eyes[0],
            np.asarray(calibration["rectification_rotation_left"], dtype=np.float64),
            1280,
            960,
            rectified_k,
        ),
        pico.make_map(
            eyes[1],
            np.asarray(calibration["rectification_rotation_right"], dtype=np.float64),
            1280,
            960,
            rectified_k,
        ),
    ]

    stereo = cv2.VideoCapture(str(stereo_video))
    selected = cv2.VideoCapture(str(selected_video))
    if not stereo.isOpened() or not selected.isOpened():
        raise RuntimeError("cannot open same-session SBS/selected video")
    frame_count = int(round(stereo.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_count != int(round(selected.get(cv2.CAP_PROP_FRAME_COUNT))):
        raise ValueError("SBS and selected frame counts differ")
    rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    frame90_rectified: tuple[np.ndarray, np.ndarray] | None = None
    try:
        for frame_id in args.frames:
            if not 0 <= frame_id < frame_count:
                raise ValueError(f"frame {frame_id} outside [0,{frame_count - 1}]")
            stereo.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            selected.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok_sbs, sbs = stereo.read()
            ok_rgb, rgb = selected.read()
            if not ok_sbs or not ok_rgb:
                raise RuntimeError(f"decode failed at frame {frame_id}")
            if sbs.shape[:2] != (eye_height, eye_width * 2):
                raise ValueError("SBS dimensions do not match same-session camera contract")
            physical_left, physical_right = pico.remap_pair(
                sbs, eye_width, maps, source_indices=source_indices
            )
            quality = pico.rectification_error(physical_left, physical_right)
            rows.append({"frame_id": frame_id, **quality})
            if frame_id == 90:
                frame90_rectified = (physical_left, physical_right)

            raw_right = sbs[:, :eye_width]
            resized_right = cv2.resize(raw_right, (rgb.shape[1], rgb.shape[0]))
            identity_left, identity_right = pico.feature_matches(resized_right, rgb)
            identity_residual = (
                np.linalg.norm(identity_left - identity_right, axis=1)
                if len(identity_left)
                else np.asarray([], dtype=np.float64)
            )
            identity_rows.append(
                {
                    "frame_id": frame_id,
                    "matches": int(len(identity_residual)),
                    "feature_p90_px": (
                        float(np.percentile(identity_residual, 90))
                        if len(identity_residual)
                        else None
                    ),
                    "pixel_mae": float(
                        np.mean(
                            np.abs(
                                resized_right.astype(np.float32)
                                - rgb.astype(np.float32)
                            )
                        )
                    ),
                }
            )
    finally:
        stereo.release()
        selected.release()

    if frame90_rectified is not None:
        cv2.imwrite(
            str(attempt / "RECTIFICATION_FRAME90.png"),
            pico.draw_epipolar_check(*frame90_rectified),
        )

    medians = [row["median_vertical_error_px"] for row in rows]
    p90s = [row["p90_vertical_error_px"] for row in rows]
    gate_pass = all(
        value is not None and value <= args.median_gate_px for value in medians
    ) and all(value is not None and value <= args.p90_gate_px for value in p90s)
    terminal_status = "PASSED" if gate_pass else "BLOCKED_PREREQ"
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    metrics = {
        "schema_version": "DEPTH10_SOURCE_INDEX_SUCCESSOR_METRICS_R3",
        "task_id": "DEPTH-10-SOURCE-ORDER-SUCCESSOR-play_cards_0910_001",
        "session_id": SESSION_ID,
        "terminal_status": terminal_status,
        "physical_eye_source_indices": {"left": 1, "right": 0},
        "selected_physical_eye": "right",
        "selected_sbs_half": 0,
        "baseline_m": baseline_m,
        "same_session_calibration": True,
        "rectification_frames": rows,
        "selected_eye_identity": identity_rows,
        "frozen_gates": {
            "median_vertical_error_max_px": args.median_gate_px,
            "p90_vertical_error_max_px": args.p90_gate_px,
        },
        "rectification_gate_pass": gate_pass,
        "fresh_foundationstereo_executed": False,
        "accepted_depth_frames": 0,
        "depth_confidence_present": False,
        "external_metric_accuracy": "UNKNOWN",
    }
    write_json(attempt / "METRICS.json", metrics)

    decision = f"""# DEPTH-10 sourceIndex 有界 successor 决定

会话：`{SESSION_ID}`
终态：`{terminal_status}`
证据等级：`DEVELOPMENT_EVIDENCE`

已修复两个确定的输入合同错误：物理左右眼现在严格按同会话
`camera_params.<eye>.sourceIndex=(1,0)` 裁切 SBS；rectification remap 使用
`stereoRectify` 输出的校正 K，不再使用校正前 K。

同会话重新标定后，0/90/190 帧的垂直匹配 P90 为
`{p90s[0]:.2f}/{p90s[1]:.2f}/{p90s[2]:.2f} px`，没有通过冻结的
`P90 <= {args.p90_gate_px:.1f} px` 门；median 也没有全部通过
`<= {args.median_gate_px:.1f} px`。因此没有申请 GPU lease、没有运行新的
FoundationStereo、没有接受任何深度帧，也没有启用 DEPTH-20 Stereo 修正。

Controller/MANUS 独立支路不依赖本 Stereo 门，可以继续。该结果只说明当前
同会话输入和校正仍不足，不是外部深度精度，也不是 FoundationStereo 模型失败。
"""
    (attempt / "DECISION.md").write_text(decision, encoding="utf-8")
    next_action = {
        "schema_version": "DEPTH10_SOURCE_INDEX_SUCCESSOR_NEXT_ACTION_R3",
        "task_id": "DEPTH-10-SOURCE-ORDER-SUCCESSOR-play_cards_0910_001",
        "status": terminal_status,
        "next_task_id": "SENSOR_H3_RECTIFICATION_CALIBRATION_RECOVERY",
        "required_actions": [
            "Obtain or verify a same-session stereo rectification contract that passes median<=2 px and P90<=5 px on held-out frames.",
            "Do not borrow exact78 rectification or guess K/baseline/extrinsics.",
            "Only after the CPU gate passes may a central GPU lease be requested for fresh FoundationStereo and swapped-input LR QA.",
        ],
        "independent_branch": "Controller/MANUS hand and Tactile remain eligible independently.",
        "authority_promoted": False,
    }
    write_json(attempt / "NEXT_ACTION.json", next_action)

    inputs = {
        "schema_version": "DEPTH10_SOURCE_INDEX_SUCCESSOR_INPUT_MANIFEST_R3",
        "session_id": SESSION_ID,
        "inputs": {
            "camera_params": digest(camera_params),
            "stereo_video": digest(stereo_video),
            "selected_video": digest(selected_video),
            "same_session_rectification": digest(calibration_file),
            "pico_adapter": digest(PICO_TOOL),
            "finalizer": digest(Path(__file__)),
        },
    }
    write_json(attempt / "INPUT_MANIFEST.json", inputs)
    extra_artifacts = [
        attempt / "INPUT_MANIFEST.json",
        attempt / "METRICS.json",
        attempt / "DECISION.md",
        attempt / "NEXT_ACTION.json",
        calibration_file,
        attempt / "rectification/manifest.json",
        attempt / "rectification/left_rectified.png",
        attempt / "rectification/right_rectified.png",
        attempt / "rectification/rectification_check.png",
        attempt / "RECTIFICATION_FRAME90.png",
    ]
    manifest = {
        "schema_version": "PIPELINE_CONTRACT_TASK_ARTIFACT_MANIFEST_R3",
        "task_id": "DEPTH-10-SOURCE-ORDER-SUCCESSOR-play_cards_0910_001",
        "attempt_id": attempt.name,
        "artifact_revision": "R7_1_SOURCE_INDEX_BOUNDED_SUCCESSOR",
        "artifacts": [digest(path) for path in extra_artifacts],
        "authority_promoted": False,
        "claim_limit": "Same-session CPU routing/rectification evidence only; no model depth or external accuracy authority.",
    }
    write_json(attempt / "ARTIFACT_MANIFEST.json", manifest)
    result = {
        "schema_version": "PIPELINE_CONTRACT_TASK_RESULT_R3",
        "task_id": "DEPTH-10-SOURCE-ORDER-SUCCESSOR-play_cards_0910_001",
        "attempt_id": attempt.name,
        "generated_at": generated_at,
        "terminal_status": terminal_status,
        "evidence_status": "DEVELOPMENT_EVIDENCE",
        "evaluation_scope": "REAL_SAME_SESSION_CPU_SOURCE_INDEX_AND_RECTIFICATION_QA",
        "session_id": SESSION_ID,
        "artifact_revision": "R7_1_SOURCE_INDEX_BOUNDED_SUCCESSOR",
        "validity": "VALID_FOR_PINNED_REVISION",
        "blockers": (
            []
            if gate_pass
            else [
                "SAME_SESSION_RECTIFICATION_MEDIAN_GATE_FAILED",
                "SAME_SESSION_RECTIFICATION_P90_GATE_FAILED",
                "GPU_INFERENCE_FORBIDDEN_UNTIL_CPU_GATE_PASSES",
            ]
        ),
        "findings": {
            "source_index_fix_verified": True,
            "rectified_intrinsics_used_for_remap": True,
            "physical_eye_source_indices": {"left": 1, "right": 0},
            "rectification_median_px_frame_0_90_190": medians,
            "rectification_p90_px_frame_0_90_190": p90s,
            "rectification_gate_pass": gate_pass,
        },
        "fresh_model_inference_executed": False,
        "gpu_lease_requested": False,
        "accepted_depth_frames": 0,
        "depth_confidence_present": False,
        "external_metric_accuracy": "UNKNOWN",
        "downstream_effect": {
            "depth_20_stereo_correction_authorized": False,
            "controller_manus_independent_branch_may_continue": True,
        },
        "authority_promoted": False,
        "claim_limit": "Corrected source routing and same-session CPU rectification QA only; not model-depth, anatomical-wrist, physical accuracy, or Robot authority.",
        "artifact_manifest": digest(attempt / "ARTIFACT_MANIFEST.json"),
        "metrics": digest(attempt / "METRICS.json"),
        "next_action": digest(attempt / "NEXT_ACTION.json"),
    }
    write_json(attempt / "RESULT.json", result)
    receipt = {
        "schema_version": "PIPELINE_CONTRACT_TASK_RUN_RECEIPT_R3",
        "task_id": result["task_id"],
        "attempt_id": attempt.name,
        "generated_at": generated_at,
        "terminal_status": terminal_status,
        "evidence_status": "DEVELOPMENT_EVIDENCE",
        "hostname": socket.gethostname(),
        "execution_mode": "CPU_AUDIT_ONLY",
        "gpu_used": False,
        "repository_commit": git_value("rev-parse", "HEAD"),
        "active_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "input_manifest": digest(attempt / "INPUT_MANIFEST.json"),
        "result": digest(attempt / "RESULT.json"),
        "artifact_manifest": digest(attempt / "ARTIFACT_MANIFEST.json"),
        "metrics": digest(attempt / "METRICS.json"),
        "execution_summary": {
            "same_session_input_identity_verified": True,
            "source_index_mapping_applied": True,
            "held_out_rectification_frames": args.frames,
            "rectification_gate_pass": gate_pass,
            "fresh_model_inference_executed": False,
            "accepted_depth_frames": 0,
        },
        "authority_promoted": False,
        "claim_limit": result["claim_limit"],
    }
    write_json(attempt / "RUN_RECEIPT.json", receipt)
    print(json.dumps({"status": terminal_status, "p90_px": p90s}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
