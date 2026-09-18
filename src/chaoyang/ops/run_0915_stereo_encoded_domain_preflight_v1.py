#!/usr/bin/env python3
"""Run the CPU-only encoded VST stereo preflight on one 150-frame session."""

from __future__ import annotations

import argparse
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
from chaoyang.pipeline.stereo_encoded_domain_preflight_v1 import (
    EncodedStereoGateV1,
    aggregate_metrics,
    frame_metrics,
    robust_correspondences,
)
from chaoyang.pipeline.vst_encoded_video_domain import (
    ADMITTED_TRANSFORM,
    ENCODED_DOMAIN,
    split_resize_physical_eyes,
    validate_encoded_video_contract,
)


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_stereo_encoded_domain_preflight_v1"
PHASE = "0915_STEREO_ENCODED_DOMAIN_PREFLIGHT_V1"
SESSION_ID = "play_cards_0915_001"
SESSION = (
    Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915")
    / "cleaned/playing_cards" / SESSION_ID
)
SBS = SESSION / "source_stereo/CameraRecord_play_cards_0915_001_stereo.mp4"
CAMERA = SESSION / "camera_params.json"
OUTPUT_SIZE = (1280, 960)
EXPECTED_FRAMES = 150
NOTABLE_FRAMES = (81, 94)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ref(path: Path) -> dict[str, Any]:
    candidate = path.resolve(strict=True)
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": _sha256(candidate),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _process_start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split()[21])


def _validate_route(output: Path) -> tuple[dict[str, Any], Path]:
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    packet = _load_json(packet_path)
    if packet != build_packet(TASK_ID):
        raise RuntimeError("current task packet differs from frozen specification")
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("encoded-domain preflight must have weights ABSENT")
    expected = (ROOT / str(packet["write_set"][0])).resolve()
    if output.resolve() != expected:
        raise RuntimeError(f"output root must equal packet writer root: {expected}")
    state = _load_json(ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task is not current next_task")
    index = _load_json(ROOT / "tasks/current/INDEX.json")
    route = next((row for row in index.get("task_packets", [])
                  if row.get("task_id") == TASK_ID), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize this preflight")
    if route.get("packet_sha256") != _sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")
    visual = (ROOT / str(packet["write_set"][1])).resolve()
    return packet, visual


def _heartbeat(pid: int) -> None:
    completed = subprocess.run(
        [
            sys.executable, "-m", "chaoyang.governance.heartbeat_task",
            "--task-id", TASK_ID, "--pid", str(pid), "--status", "RUNNING",
            "--phase", PHASE,
        ],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )


def _review_frame(
    left: np.ndarray,
    right: np.ndarray,
    points_left: np.ndarray,
    points_right: np.ndarray,
    row: dict[str, Any],
) -> np.ndarray:
    scale = 0.5
    left_small = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.hstack((left_small, right_small))
    width = left_small.shape[1]
    if len(points_left):
        residual = np.abs(points_left[:, 1] - points_right[:, 1])
        order = np.argsort(residual)[: min(50, len(residual))]
        for index in order:
            first = tuple(np.rint(points_left[index] * scale).astype(int))
            second_xy = np.rint(points_right[index] * scale).astype(int)
            second = (int(second_xy[0] + width), int(second_xy[1]))
            color = (60, 210, 60) if residual[index] <= 4.0 else (40, 80, 230)
            cv2.line(canvas, first, second, color, 1, cv2.LINE_AA)
    overlay = np.zeros((70, canvas.shape[1], 3), np.uint8)
    median = row["median_abs_vertical_px"]
    p95 = row["p95_abs_vertical_px"]
    text = (
        f"frame {row['frame_index']:03d} | matches {row['robust_matches']} | "
        f"|dy| med {median if median is not None else -1:.2f} "
        f"p95 {p95 if p95 is not None else -1:.2f}px"
    )
    cv2.putText(overlay, text, (18, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        "physical LEFT sourceIndex=1 | physical RIGHT sourceIndex=0 | crop+resize only",
        (18, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 220, 255), 1,
        cv2.LINE_AA,
    )
    return np.vstack((overlay, canvas))


def _diagnosis(row: dict[str, Any], gate: EncodedStereoGateV1) -> list[str]:
    reasons: list[str] = []
    if row["robust_matches"] < gate.minimum_matches_per_frame:
        reasons.append("LOW_ROBUST_MATCH_COUNT")
    if row["p95_abs_vertical_px"] is not None and (
        row["p95_abs_vertical_px"] > gate.maximum_p95_vertical_px
    ):
        reasons.append("HIGH_VERTICAL_RESIDUAL")
    fraction = row["positive_disparity_fraction"]
    if fraction is not None and max(fraction, 1.0 - fraction) < gate.minimum_disparity_sign_consistency:
        reasons.append("MIXED_DISPARITY_SIGN")
    return reasons or ["NO_FRAME_LOCAL_ANOMALY_UNDER_GLOBAL_GATES"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output root required: {output}")
    packet, visual = _validate_route(output)
    if visual.exists() or visual.is_symlink():
        raise RuntimeError(f"fresh visual root required: {visual}")
    output.mkdir(parents=True)
    visual.mkdir(parents=True)

    pid = os.getpid()
    start_ticks = _process_start_ticks(pid)
    epoch = time.time_ns()
    fencing_token = uuid.uuid4().hex
    packet_path = ROOT / f"tasks/current/{TASK_ID}/TASK_PACKET.json"
    signature = {
        "schema_version": "0915-stereo-encoded-domain-run-signature-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "executor_epoch": epoch,
        "input_manifest": {
            "task_packet": _ref(packet_path),
            "sbs": _ref(SBS),
            "camera_params": _ref(CAMERA),
            "vst_domain_confirmation": _ref(
                ROOT / "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json"
            ),
            "user_authorization": _ref(
                ROOT / "tasks/receipts/0915_CPU_NEXT_TASKS_USER_AUTHORIZATION.json"
            ),
        },
        "code": {
            "runner": _ref(Path(__file__)),
            "algorithm": _ref(
                ROOT / "src/chaoyang/pipeline/stereo_encoded_domain_preflight_v1.py"
            ),
            "pixel_domain": _ref(
                ROOT / "src/chaoyang/pipeline/vst_encoded_video_domain.py"
            ),
        },
        "weights": "ABSENT",
        "calibration": "ENCODED_DOMAIN_EPIPOLAR_STATE_UNDER_TEST",
        "pixel_transform": ADMITTED_TRANSFORM,
        "lens_undistortion_applied": False,
        "gpu_used": False,
    }
    signature["run_signature_sha256"] = _canonical_sha(signature)
    _atomic_json(output / "RUN_SIGNATURE.json", signature)
    claim = {
        "schema_version": "0915-stereo-encoded-domain-writer-claim-v1",
        "task_id": TASK_ID,
        "attempt_id": output.name,
        "pid": pid,
        "proc_start_ticks": start_ticks,
        "executor_epoch": epoch,
        "run_signature_sha256": signature["run_signature_sha256"],
        "fencing_token_sha256": hashlib.sha256(fencing_token.encode()).hexdigest(),
        "unique_write_root": str(output),
        "status": "CLAIMED",
    }
    _atomic_json(output / "WRITER_CLAIM.json", claim)
    _heartbeat(pid)

    domain_contract = {
        "encoded_video_domain": ENCODED_DOMAIN,
        "transform": ADMITTED_TRANSFORM,
        "lens_undistortion_applied": False,
        "distortion_coefficients_consumed": False,
        "operation": "SOURCE_INDEX_CROP_RESIZE_ONLY",
    }
    validate_encoded_video_contract(domain_contract)
    camera = _load_json(CAMERA)
    eye_width = int(camera["width"])
    eye_height = int(camera["height"])
    source_indices = (
        int(camera["left"]["sourceIndex"]), int(camera["right"]["sourceIndex"]),
    )
    if source_indices != (1, 0):
        raise RuntimeError(f"unexpected physical sourceIndex mapping: {source_indices}")

    capture = cv2.VideoCapture(str(SBS))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open SBS video: {SBS}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    review_path = visual / "0915_STEREO_ENCODED_DOMAIN_PREFLIGHT_REVIEW.mp4"
    writer = cv2.VideoWriter(
        str(review_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (OUTPUT_SIZE[0], OUTPUT_SIZE[1] // 2 + 70),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("cannot open stereo review writer")

    rows: list[dict[str, Any]] = []
    verticals: list[np.ndarray] = []
    disparities: list[np.ndarray] = []
    contact_frames: dict[int, np.ndarray] = {}
    contact_indices = {0, 30, 60, 81, 94, 149}
    gate = EncodedStereoGateV1()
    try:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            left, right = split_resize_physical_eyes(
                frame, eye_width=eye_width, source_indices=source_indices,
                output_size=OUTPUT_SIZE,
            )
            points_left, points_right = robust_correspondences(left, right)
            row = {
                "frame_index": frame_index,
                **frame_metrics(
                    points_left, points_right,
                    width=OUTPUT_SIZE[0], height=OUTPUT_SIZE[1],
                ),
            }
            vertical = np.abs(points_left[:, 1] - points_right[:, 1]) \
                if len(points_left) else np.empty(0, np.float64)
            disparity = points_left[:, 0] - points_right[:, 0] \
                if len(points_left) else np.empty(0, np.float64)
            review = _review_frame(left, right, points_left, points_right, row)
            writer.write(review)
            if frame_index in contact_indices:
                contact_frames[frame_index] = review.copy()
            rows.append(row)
            verticals.append(vertical)
            disparities.append(disparity)
            frame_index += 1
            if frame_index % 20 == 0:
                _heartbeat(pid)
    finally:
        capture.release()
        writer.release()

    if len(rows) != EXPECTED_FRAMES or reported_frames != EXPECTED_FRAMES:
        raise RuntimeError(
            f"expected {EXPECTED_FRAMES} frames, reported={reported_frames}, decoded={len(rows)}"
        )
    if set(NOTABLE_FRAMES) - {row["frame_index"] for row in rows}:
        raise RuntimeError("mandatory diagnostic frames 81/94 were not decoded")
    aggregate = aggregate_metrics(rows, verticals, disparities, gate=gate)
    notable = {
        str(index): {
            "metrics": rows[index],
            "diagnosis": _diagnosis(rows[index], gate),
        }
        for index in NOTABLE_FRAMES
    }
    per_frame = {
        "schema_version": "0915-stereo-encoded-domain-per-frame-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "frame_count": len(rows),
        "mandatory_diagnostic_frames": list(NOTABLE_FRAMES),
        "frames": rows,
    }
    _atomic_json(output / "PER_FRAME_METRICS.json", per_frame)
    preflight = {
        "schema_version": "0915-stereo-encoded-domain-preflight-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "image_domain": domain_contract,
        "physical_eye_mapping": {
            "left_source_index": source_indices[0],
            "right_source_index": source_indices[1],
            "decoded_eye_geometry": [eye_width, eye_height],
            "analysis_geometry": list(OUTPUT_SIZE),
        },
        "decode": {
            "reported_frame_count": reported_frames,
            "decoded_frame_count": len(rows),
            "full_decode": len(rows) == reported_frames == EXPECTED_FRAMES,
            "fps": fps,
        },
        **aggregate,
        "notable_frame_diagnosis": notable,
        "external_metric_accuracy": "NOT_EVALUATED",
        "depth_produced": False,
        "gpu_used": False,
    }
    _atomic_json(output / "ENCODED_STEREO_PREFLIGHT.json", preflight)

    ordered = [contact_frames[index] for index in sorted(contact_frames)]
    sheet = np.vstack(ordered)
    sheet_path = visual / "0915_STEREO_ENCODED_DOMAIN_PREFLIGHT_CONTACT_SHEET.jpg"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError("cannot write stereo contact sheet")
    (visual / "README_ZH.md").write_text(
        "# 0915 Encoded-domain Stereo Preflight V1\n\n"
        "本目录只显示物理左/右目按 `sourceIndex` 裁切后 resize 的原始编码像素；"
        "未使用镜头去畸变、相机模型 remap、Depth warm-start 或 GPU。\n\n"
        f"- 决策：`{aggregate['decision']}`\n"
        f"- 150 帧完整解码：是\n"
        f"- frame 81/94：已单独诊断\n"
        "- 本结果只决定后续 stereo 输入路径，不构成深度或毫米精度证明。\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": "0915-stereo-encoded-domain-preflight-result-v1",
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "status": "PASSED",
        "weights": "ABSENT",
        "gpu_used": False,
        "preflight_decision": aggregate["decision"],
        "gpu_successor_authorized": aggregate["gpu_successor_authorized"],
        "first_blocker": (
            None if aggregate["gpu_successor_authorized"]
            else aggregate["decision"]
        ),
        "source_mutated": False,
        "preflight": _ref(output / "ENCODED_STEREO_PREFLIGHT.json"),
        "per_frame_metrics": _ref(output / "PER_FRAME_METRICS.json"),
        "review_video": _ref(review_path),
        "contact_sheet": _ref(sheet_path),
        "claim_limit": packet["claim_limit"],
    }
    _atomic_json(output / "RESULT.json", result)
    receipt = {
        "schema_version": "0915-stereo-encoded-domain-run-receipt-v1",
        "task_id": TASK_ID,
        "status": "PASSED",
        "run_signature": _ref(output / "RUN_SIGNATURE.json"),
        "writer_claim": _ref(output / "WRITER_CLAIM.json"),
        "result": _ref(output / "RESULT.json"),
        "full_decode": True,
        "gpu_used": False,
        "source_mutated": False,
    }
    _atomic_json(output / "RUN_RECEIPT.json", receipt)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

