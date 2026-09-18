#!/usr/bin/env python3
"""Render a shallow HaWoR batch diagnostic without changing model evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any
import uuid

import cv2
import numpy as np


CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
COLORS = ((255, 210, 30), (20, 80, 255))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        rendered_path = str(resolved.relative_to(relative_to.resolve()))
    except ValueError:
        rendered_path = str(resolved)
    return {
        "path": rendered_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def overlay_frame(
    image: np.ndarray,
    joints: np.ndarray,
    observed: np.ndarray,
    boxes: np.ndarray,
    frame: int,
    session_id: str,
) -> np.ndarray:
    output = image.copy()
    for side in (0, 1):
        color = COLORS[side]
        if np.isfinite(boxes[side]).all():
            x1, y1, x2, y2 = np.rint(boxes[side]).astype(int)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        if observed[side]:
            points = np.rint(joints[side]).astype(np.int32)
            for chain in CHAINS:
                cv2.polylines(output, [points[np.asarray(chain)]], False,
                              color, 3, cv2.LINE_AA)
            cv2.circle(output, tuple(points[0]), 7, color, -1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, 82), (0, 0, 0), -1)
    cv2.putText(output, f"{session_id} | physical-left | frame {frame:04d}",
                (14, 30), cv2.FONT_HERSHEY_SIMPLEX, .68,
                (245, 245, 245), 2, cv2.LINE_AA)
    states = [f"L={'OBS' if observed[0] else 'MISS'}",
              f"R={'OBS' if observed[1] else 'MISS'}"]
    cv2.putText(output, "  ".join(states) + " | CYAN=L ORANGE=R",
                (14, 64), cv2.FONT_HERSHEY_SIMPLEX, .62,
                (245, 245, 245), 2, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--session-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    video = args.video.resolve(strict=True)
    npz_path = args.hawor_npz.resolve(strict=True)
    session_result_path = args.session_result.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh diagnostic output required: {output}")
    output.mkdir(parents=True)
    session_result = json.loads(session_result_path.read_text(encoding="utf-8"))
    session_id = str(session_result["session_id"])
    with np.load(npz_path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float32)
        observed = np.asarray(archive["observed"], bool)
        boxes = np.asarray(archive["detector_boxes_xyxy"], np.float32)
        fps = float(np.asarray(archive["fps"]).item())
    frame_count = int(observed.shape[1])
    if joints.shape[:3] != (2, frame_count, 21) or boxes.shape != (2, frame_count, 4):
        raise RuntimeError("HaWoR diagnostic array shape drift")
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    review = output / f"{session_id}_HAWOR_PHYSICAL_LEFT_REVIEW.mp4"
    writer = cv2.VideoWriter(str(review), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not capture.isOpened() or not writer.isOpened():
        raise RuntimeError("HaWoR diagnostic video reader/writer failed")
    sample_indices = set(np.linspace(0, frame_count - 1, 12, dtype=int).tolist())
    samples: list[np.ndarray] = []
    try:
        for frame in range(frame_count):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"physical-left video ended at frame {frame}")
            rendered = overlay_frame(
                image, joints[:, frame], observed[:, frame], boxes[:, frame],
                frame, session_id,
            )
            writer.write(rendered)
            if frame in sample_indices:
                samples.append(cv2.resize(rendered, (640, 480),
                                          interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
        writer.release()
    sheet = output / f"{session_id}_HAWOR_CONTACT_SHEET.jpg"
    canvas = np.zeros((3 * 480, 4 * 640, 3), np.uint8)
    for index, image in enumerate(samples):
        row, column = divmod(index, 4)
        canvas[row * 480:(row + 1) * 480,
               column * 640:(column + 1) * 640] = image
    if not cv2.imwrite(str(sheet), canvas):
        raise RuntimeError("failed to write HaWoR diagnostic contact sheet")
    decoded = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(review),
        "-f", "null", "-",
    ], capture_output=True, text=True, check=False)
    if decoded.returncode:
        raise RuntimeError(f"HaWoR diagnostic full decode failed: {decoded.stderr[-1000:]}")
    result = {
        "schema_version": "0915-hawor-batch-diagnostic-v1",
        "status": "PASS_REVIEW_ONLY",
        "session_id": session_id,
        "frame_count": frame_count,
        "observed_frames": {
            "left": int(observed[0].sum()),
            "right": int(observed[1].sum()),
            "bilateral": int(np.all(observed, axis=0).sum()),
        },
        "observed_fraction": {
            "left": float(observed[0].mean()),
            "right": float(observed[1].mean()),
        },
        "inputs": {
            "video": ref(video, output),
            "hawor_npz": ref(npz_path, output),
            "session_result": ref(session_result_path, output),
        },
        "artifacts": {
            "review": ref(review, output),
            "contact_sheet": ref(sheet, output),
        },
        "full_decode": "PASS_FFMPEG_XERROR",
        "claim_limit": (
            "Review-only rendering of frozen HaWoR observations; it does not alter "
            "detections, fill missing hands, or grant anatomical ground-truth authority."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
