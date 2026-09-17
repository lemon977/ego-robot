#!/usr/bin/env python3
"""Publish a reproducible fail-closed DEPTH-10 diagnostic and wrist review."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def digest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    sha = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha.hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def video_probe(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    value = {
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    cap.release()
    return value


def hawor_points(
    comparison: dict[str, Any], controller: np.ndarray
) -> np.ndarray:
    output = np.full_like(controller, np.nan, dtype=np.float64)
    for row in comparison["per_frame_delta_xyz_mm"]:
        frame = int(row["frame"])
        for side, side_index in (("left", 0), ("right", 1)):
            delta = row[side]["hawor_minus_controller"]
            if delta is not None and all(value is not None for value in delta):
                output[frame, side_index] = controller[frame, side_index] + np.asarray(delta) / 1000.0
    return output


def build_wrist_video(
    *, raw_video: Path, hand_sidecar: Path, comparison_json: Path, output: Path
) -> int:
    with np.load(hand_sidecar, allow_pickle=False) as archive:
        controller = np.asarray(archive["T_wrist_to_camera"], dtype=np.float64)[:, :, :3, 3]
        manus = np.asarray(archive["joint_xyz_camera_m"], dtype=np.float64)[:, :, 0]
        valid = np.asarray(archive["hand_valid"], dtype=np.bool_)
    comparison = json.loads(comparison_json.read_text(encoding="utf-8"))
    hawor = hawor_points(comparison, controller)
    finite_points = np.concatenate([controller[valid], manus[valid], hawor[np.isfinite(hawor).all(axis=2)]])
    x_min, z_min = np.nanpercentile(finite_points[:, [0, 2]], 1, axis=0) - 0.08
    x_max, z_max = np.nanpercentile(finite_points[:, [0, 2]], 99, axis=0) + 0.08
    if x_max <= x_min or z_max <= z_min:
        raise RuntimeError("invalid wrist plot range")

    cap = cv2.VideoCapture(str(raw_video))
    if not cap.isOpened():
        raise RuntimeError("cannot open wrist source video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_count != len(controller):
        raise RuntimeError("wrist/video frame count mismatch")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 600))
    if not writer.isOpened():
        raise RuntimeError("cannot create wrist review")

    def point_xy(point: np.ndarray) -> tuple[int, int]:
        x = 675 + int((point[0] - x_min) / (x_max - x_min) * 565)
        y = 555 - int((point[2] - z_min) / (z_max - z_min) * 465)
        return x, y

    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id = decoded
        left = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((600, 1280, 3), dtype=np.uint8)
        canvas[70:550, :640] = left
        cv2.rectangle(canvas, (660, 70), (1260, 570), (100, 100, 100), 1)
        cv2.putText(canvas, "Raw 1280x960", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(canvas, "Camera X-Z absolute trajectory", (670, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(canvas, f"frame {frame_id}/{frame_count-1}  t={frame_id/fps:.2f}s", (20, 585), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(canvas, "Stereo: UNAVAILABLE (rectification P90 > 5 px)", (675, 595), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,255), 2)
        for side_index, side_label in enumerate(("L", "R")):
            if valid[frame_id, side_index]:
                pc = point_xy(controller[frame_id, side_index])
                pm = point_xy(manus[frame_id, side_index])
                cv2.circle(canvas, pc, 8, (255, 255, 0), -1)
                cv2.rectangle(canvas, (pm[0]-6, pm[1]-6), (pm[0]+6, pm[1]+6), (0, 255, 255), -1)
                cv2.putText(canvas, side_label, (pc[0]+8, pc[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if np.isfinite(hawor[frame_id, side_index]).all():
                ph = point_xy(hawor[frame_id, side_index])
                cv2.drawMarker(canvas, ph, (255, 0, 255), cv2.MARKER_CROSS, 18, 3)
        cv2.putText(canvas, "Controller=circle  MANUS-root=square  HaWoR=cross", (675, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230,230,230), 1)
        writer.write(canvas)
        decoded += 1
    cap.release()
    writer.release()
    if decoded != frame_count:
        raise RuntimeError(f"decoded {decoded}/{frame_count}")
    check = video_probe(output)
    if check["frame_count"] != frame_count:
        raise RuntimeError("published wrist review does not fully decode")
    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-attempt", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--hand-sidecar", type=Path, required=True)
    parser.add_argument("--comparison-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    prior = args.prior_attempt.resolve(strict=True)
    metrics_prior_path = prior / "METRICS.json"
    metrics_prior = json.loads(metrics_prior_path.read_text(encoding="utf-8"))
    session = args.session_root.resolve(strict=True)
    camera = session / "camera_params.json"
    selected = session / f"CameraRecord_{session.name}.mp4"
    sbs = session / f"source_stereo/CameraRecord_{session.name}_stereo.mp4"
    calibration = prior / "same_session_rectification.json"
    selected_probe = video_probe(selected)
    sbs_probe = video_probe(sbs)
    params = json.loads(camera.read_text(encoding="utf-8"))
    rows = metrics_prior["rectification_frames"]
    sampled = len(rows)
    frame_count = selected_probe["frame_count"]
    p90_values = [float(row["p90_vertical_error_px"]) for row in rows]
    match_counts = [int(row["matches"]) for row in rows]
    gate = bool(metrics_prior["rectification_gate_pass"])
    if gate:
        raise RuntimeError("this bounded finalizer is only for the frozen failed diagnostic")

    summary = {
        "schema_version": "DEPTH10_REPRODUCIBLE_DIAGNOSTIC_METRICS_R22",
        "session_id": session.name,
        "terminal_status": "BLOCKED_PREREQ",
        "image_domains": {
            "sbs_encoded": [sbs_probe["width"], sbs_probe["height"]],
            "physical_eye_source": [int(params["width"]), int(params["height"])],
            "selected_mp4": [selected_probe["width"], selected_probe["height"]],
            "rectified_output": [1280, 960],
            "selected_physical_eye": metrics_prior["selected_physical_eye"],
            "physical_eye_source_indices": metrics_prior["physical_eye_source_indices"],
        },
        "sampling": {
            "frame_ids": [int(row["frame_id"]) for row in rows],
            "frame_range_inclusive": [0, frame_count - 1],
            "sampled_frames": sampled,
            "total_frames": frame_count,
            "sampled_video_frame_coverage": sampled / frame_count,
            "valid_match_frames": sum(count > 0 for count in match_counts),
            "valid_match_frame_coverage_within_sample": sum(count > 0 for count in match_counts) / sampled,
            "valid_match_count_total": sum(match_counts),
        },
        "vertical_epipolar_residual": {
            "per_frame": rows,
            "p90_px_by_frame": p90_values,
            "p90_px_max": max(p90_values),
            "median_gate_px": metrics_prior["frozen_gates"]["median_vertical_error_max_px"],
            "p90_gate_px": metrics_prior["frozen_gates"]["p90_vertical_error_max_px"],
            "gate_pass": False,
        },
        "calibration": digest(calibration),
        "camera_contract": digest(camera),
        "foundationstereo_executed": False,
        "foundationstereo_forbidden_reason": "SAME_SESSION_RECTIFICATION_P90_GATE_FAILED",
        "stereo_wrist_status": "UNAVAILABLE",
        "depth_confidence_present": False,
        "external_metric_accuracy": "UNKNOWN",
    }
    metrics_path = output / "METRICS.json"
    write_json(metrics_path, summary)
    wrist_video = output / f"{session.name}_Controller_MANUS_HaWoR_Stereo不可用_绝对3D全片.mp4"
    build_wrist_video(
        raw_video=selected,
        hand_sidecar=args.hand_sidecar.resolve(strict=True),
        comparison_json=args.comparison_json.resolve(strict=True),
        output=wrist_video,
    )
    (output / "DECISION.md").write_text(
        "# DEPTH-10可复现诊断\n\n"
        "同会话rectification在冻结帧上的vertical epipolar P90均超过5 px，故终态为"
        "`BLOCKED_PREREQ`。未申请GPU、未运行FoundationStereo。可视化继续展示"
        "Controller、MANUS root与稀疏HaWoR；Stereo明确为`UNAVAILABLE`。旧Stereo"
        "比较只能作为历史开发证据，不得用于本revision的手腕修正。\n",
        encoding="utf-8",
    )
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "DEPTH10_REPRODUCIBLE_NEXT_ACTION_R22",
        "status": "BLOCKED_PREREQ",
        "next_task_id": "SENSOR_H3_RECTIFICATION_CALIBRATION_RECOVERY",
        "required_gate": "vertical epipolar P90 <= 5 px with reproducible match coverage",
        "foundationstereo_may_run": False,
    })
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "DEPTH10_REPRODUCIBLE_RUN_RECEIPT_R22",
        "task_id": "DEPTH10-RECTIFICATION-DIAGNOSTIC-R22",
        "artifact_revision": "R7_2_DEPTH10_REPRODUCIBLE_BLOCKED",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "inputs": [
            digest(metrics_prior_path), digest(calibration), digest(camera), digest(selected), digest(sbs),
            digest(args.hand_sidecar.resolve(strict=True)), digest(args.comparison_json.resolve(strict=True)),
        ],
        "producer": digest(Path(__file__)),
        "gpu_calls": 0,
        "authority_promoted": False,
    })
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "DEPTH10_REPRODUCIBLE_ARTIFACT_MANIFEST_R22",
        "artifacts": [
            digest(metrics_path), digest(output / "DECISION.md"), digest(output / "NEXT_ACTION.json"),
            digest(output / "RUN_RECEIPT.json"), digest(wrist_video),
        ],
        "authority_promoted": False,
    })
    result = {
        "schema_version": "DEPTH10_REPRODUCIBLE_RESULT_R22",
        "task_id": "DEPTH10-RECTIFICATION-DIAGNOSTIC-R22",
        "session_id": session.name,
        "terminal_status": "BLOCKED_PREREQ",
        "blocker": "SAME_SESSION_RECTIFICATION_P90_GATE_FAILED",
        "foundationstereo_executed": False,
        "stereo_wrist_status": "UNAVAILABLE",
        "metrics": digest(metrics_path),
        "wrist_review": digest(wrist_video),
        "authority_promoted": False,
        "claim_limit": "Same-session rectification QA and three-source wrist visualization only; no Stereo or external metric authority.",
    }
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
