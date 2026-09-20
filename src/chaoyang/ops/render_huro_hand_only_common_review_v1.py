#!/usr/bin/env python3
"""Render a same-frame local/HuRo-derived root-relative KaiHand comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, load_json, sha256_file
from chaoyang.pipeline.huro_hand_only_retarget_v1 import keypoints_from_q, load_hand_model


EDGES = tuple((0, start) for start in (1, 5, 9, 13, 17)) + tuple(
    (start + offset, start + offset + 1)
    for start in (1, 5, 9, 13, 17)
    for offset in range(3)
)


def _exact(item: dict[str, Any]) -> Path:
    path = Path(str(item["path"])).resolve(strict=True)
    if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != item["sha256"]:
        raise RuntimeError(f"artifact mismatch: {path}")
    return path


def _label(canvas: np.ndarray, text: str, xy: tuple[int, int], color=(235, 235, 235), scale=0.52) -> None:
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _project(points: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = box
    relative = points - points[:1]
    u = (relative[:, 0] + 0.12) / 0.24
    v = (-relative[:, 2] + 0.01) / 0.20
    return np.column_stack((x + u * width, y + v * height)).round().astype(int)


def _draw(canvas: np.ndarray, points: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    uv = _project(points, box)
    for a, b in EDGES:
        cv2.line(canvas, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA)
    for point in uv:
        cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _local_field_names(result: dict[str, Any], local: dict[str, np.ndarray]) -> tuple[str, str]:
    comparison = result.get("comparison", {})
    q_field = str(comparison.get("local_q22_field", "q22"))
    valid_field = str(comparison.get("local_valid_field", "q22_computed"))
    if q_field not in local or valid_field not in local:
        raise RuntimeError(f"local comparison fields unavailable: {q_field}/{valid_field}")
    return q_field, valid_field


def render(*, result_path: Path, rgb_video: Path, output_root: Path) -> dict[str, Any]:
    result_path = result_path.resolve(strict=True)
    result = load_json(result_path)
    if (
        result.get("schema_version") != "HURO_DERIVED_HAND_ONLY_RESULT_V1"
        or result.get("status") != "PASSED_DEVELOPMENT_HAND_ONLY_PRODUCTION"
    ):
        raise RuntimeError("HuRo-derived result is not an admitted development input")
    session = str(result["session_id"])
    frame_count = int(result["frame_count"])
    local_path = _exact(result["comparison"]["local_r0"])
    huro_path = _exact(result["outputs"]["states"])
    left_urdf = _exact(result["inputs"]["kaihand_left_urdf"])
    right_urdf = _exact(result["inputs"]["kaihand_right_urdf"])
    rgb_video = rgb_video.resolve(strict=True)
    local = _load_npz(local_path)
    huro = _load_npz(huro_path)
    q_field, valid_field = _local_field_names(result, local)
    local_q = np.asarray(local[q_field], dtype=np.float64)
    local_valid = np.asarray(local[valid_field], dtype=bool)
    huro_q = np.asarray(huro["q22"], dtype=np.float64)
    huro_valid = np.asarray(huro["valid"], dtype=bool)
    expected = (frame_count, 2, 22)
    if local_q.shape != expected or huro_q.shape != expected:
        raise RuntimeError("q22 shape mismatch")
    if local_valid.shape != (frame_count, 2) or huro_valid.shape != (frame_count, 2):
        raise RuntimeError("validity shape mismatch")

    output_root = output_root.resolve()
    video_path = output_root / f"{session}_LOCAL_VS_HURO_DERIVED_COMMON_REVIEW.mp4"
    receipt_path = output_root / "COMMON_RENDER_RECEIPT_V2.json"
    if video_path.exists() or receipt_path.exists():
        raise RuntimeError("immutable render output already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(rgb_video))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    temporary = video_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("video open failed")
    models = (load_hand_model(left_urdf, "left"), load_hand_model(right_urdf, "right"))
    common_count = 0
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"RGB decode stopped at {frame_index}")
            canvas = np.full((720, 1280, 3), 18, np.uint8)
            canvas[:480, :640] = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            _label(canvas, "Physical-left RGB (resize-only; no lens remap)", (14, 26), scale=0.62)
            _label(canvas, "Frozen local retarget", (660, 27), (80, 230, 100), 0.58)
            _label(canvas, "HuRo-derived hand-only", (976, 27), (60, 175, 255), 0.56)
            local_points: list[np.ndarray | None] = [None, None]
            huro_points: list[np.ndarray | None] = [None, None]
            common_sides = []
            for side in range(2):
                if local_valid[frame_index, side] and np.isfinite(local_q[frame_index, side]).all():
                    local_points[side] = keypoints_from_q(models[side], local_q[frame_index, side])
                    _draw(canvas, local_points[side], (650 + 150 * side, 55, 140, 270), (80, 230, 100))
                if huro_valid[frame_index, side] and np.isfinite(huro_q[frame_index, side]).all():
                    huro_points[side] = keypoints_from_q(models[side], huro_q[frame_index, side])
                    _draw(canvas, huro_points[side], (970 + 150 * side, 55, 140, 270), (60, 175, 255))
                if local_points[side] is not None and huro_points[side] is not None:
                    common_sides.append(side)
                    common_count += 1
                    _draw(canvas, local_points[side], (660 + 300 * side, 405, 270, 255), (80, 230, 100))
                    _draw(canvas, huro_points[side], (660 + 300 * side, 405, 270, 255), (60, 175, 255))
            _label(canvas, "Root-relative difference overlay (not accuracy)", (658, 388), scale=0.60)
            if common_sides:
                differences = [
                    np.degrees(local_q[frame_index, side] - huro_q[frame_index, side])
                    for side in common_sides
                ]
                rms = float(np.sqrt(np.mean(np.square(np.concatenate(differences)))))
                metric = f"common sides={common_sides}  q RMS diff={rms:.1f} deg"
            else:
                metric = "common sides=NONE; no fill"
            _label(canvas, f"frame={frame_index:03d}  {metric}", (14, 525), scale=0.58)
            _label(canvas, "Same frozen MANO21 target / KaiHand assets / frame id", (14, 560))
            _label(canvas, "DEVELOPMENT_ONLY; no wrist/world/contact/control/deployment authority", (14, 594), (80, 190, 255))
            _label(canvas, "Green=local; Orange=HuRo-derived; invalid remains absent", (14, 650), scale=0.56)
            writer.write(canvas)
        if capture.read()[0]:
            raise RuntimeError("RGB has more frames than the frozen result")
    finally:
        capture.release()
        writer.release()
    temporary.replace(video_path)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"], check=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    receipt = {
        "schema_version": "HURO_DERIVED_HAND_ONLY_COMMON_RENDER_V2",
        "status": "PASS_DIAGNOSTIC_RENDER",
        "session_id": session,
        "frame_count": frame_count,
        "common_valid_side_frames_rendered": common_count,
        "method_comparison": "COMMON_FRAME_DIFFERENCE_ONLY_NOT_ACCURACY",
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY_NO_LENS_REMAP",
        "inputs": {
            "result": artifact_ref(result_path),
            "rgb": artifact_ref(rgb_video),
            "local": artifact_ref(local_path),
            "huro_derived": artifact_ref(huro_path),
            "renderer": artifact_ref(Path(__file__)),
        },
        "output": artifact_ref(video_path),
        "decode": json.loads(probe.stdout)["streams"][0],
        "claims": {
            "accuracy": False,
            "control_ground_truth": False,
            "physical_deployable": False,
            "external_metric_authority": False,
        },
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--rgb-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    value = render(result_path=args.result, rgb_video=args.rgb_video, output_root=args.output_root)
    print(json.dumps({"status": value["status"], "output": value["output"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
