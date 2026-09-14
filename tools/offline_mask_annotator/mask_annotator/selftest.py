"""Synthetic end-to-end smoke test; no project data is read or packaged."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image

from .config import load_config
from .project import export_project, load_annotation, prepare_project, save_annotation
from .schema import CLASSES
from .validation import validate_project


def _make_video(path: Path) -> None:
    command = [
        shutil.which("ffmpeg") or "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", "testsrc2=size=320x240:rate=10:duration=3",
        "-c:v", "mpeg4",
        "-q:v", "3",
        str(path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def _operations(width: int, height: int) -> list[dict[str, Any]]:
    operations = []
    cell_w = max(12, width // 10)
    cell_h = max(12, height // 6)
    for index, item in enumerate(CLASSES[1:], start=0):
        col, row = index % 4, index // 4
        x0 = 8 + col * (cell_w + 6)
        y0 = 8 + row * (cell_h + 6)
        x1 = min(width - 2, x0 + cell_w)
        y1 = min(height - 2, y0 + cell_h)
        operations.append({
            "op_id": f"synthetic-{item['id']}",
            "kind": "polygon",
            "class_id": item["id"],
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        })
    operations.append({
        "op_id": "synthetic-brush-overwrite",
        "kind": "brush",
        "class_id": 1,
        "radius": 4,
        "points": [[width * 0.6, height * 0.7], [width * 0.8, height * 0.8]],
    })
    return operations


def _execute(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    video = root / "synthetic.mp4"
    output = root / "annotations"
    config_path = root / "config.json"
    _make_video(video)
    config_payload = {
        "input_mp4": str(video),
        "output_dir": str(output),
        "session_id": "synthetic_selftest",
        "sampling": {"frame_count": 6, "timestamps_csv": ""},
        "server": {"host": "127.0.0.1", "port": 18765, "open_browser": False},
        "validation": {
            "require_complete": True,
            "required_nonempty_classes": [item["symbol"] for item in CLASSES],
        },
    }
    config_path.write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    config = load_config(config_path)
    manifest = prepare_project(config)
    pts = [float(frame["source_pts_time"]) for frame in manifest["frames"]]
    if len(manifest["frames"]) != 6 or any(b <= a for a, b in zip(pts, pts[1:])):
        raise RuntimeError("synthetic sampling count/PTS order failed")
    for frame in manifest["frames"]:
        annotation = load_annotation(config, frame)
        saved = save_annotation(config, frame, {
            "revision": annotation["revision"],
            "complete": True,
            "operations": _operations(frame["image_width"], frame["image_height"]),
        })
        if saved["revision"] != 1:
            raise RuntimeError("annotation revision did not advance")
    export = export_project(config)
    validation = validate_project(config)
    first_class = output / export["frames"][0]["class_id"]["relpath"]
    with Image.open(first_class) as image:
        values = set(image.getdata())
    if validation["status"] != "PASS" or values != set(range(9)):
        raise RuntimeError({"validation": validation, "class_ids": sorted(values)})

    timestamps = root / "timestamps.csv"
    timestamps.write_text(
        "timestamp_seconds\n0.05\n0.55\n1.05\n1.55\n2.05\n2.75\n",
        encoding="utf-8",
    )
    timestamp_config_path = root / "config_timestamps.json"
    timestamp_payload = dict(config_payload)
    timestamp_payload["output_dir"] = str(root / "annotations_timestamps")
    timestamp_payload["session_id"] = "synthetic_timestamp_selftest"
    timestamp_payload["sampling"] = {
        "frame_count": 6,
        "timestamps_csv": str(timestamps),
    }
    timestamp_config_path.write_text(
        json.dumps(timestamp_payload, indent=2) + "\n", encoding="utf-8"
    )
    timestamp_manifest = prepare_project(load_config(timestamp_config_path))
    timestamp_indices = [frame["source_decoded_index"] for frame in timestamp_manifest["frames"]]
    if len(timestamp_indices) != 6 or len(set(timestamp_indices)) != 6:
        raise RuntimeError("explicit timestamp integration sampling failed")
    return {
        "schema_version": "offline-mask-synthetic-self-test-v1",
        "status": "PASS",
        "synthetic_only": True,
        "sampled_frames": len(manifest["frames"]),
        "decoded_indices": [frame["source_decoded_index"] for frame in manifest["frames"]],
        "pts_time": [frame["source_pts_time"] for frame in manifest["frames"]],
        "exported_frames": export["frame_count"],
        "validation_status": validation["status"],
        "class_ids_seen": sorted(values),
        "explicit_timestamp_decoded_indices": timestamp_indices,
        "explicit_timestamp_pts_time": [
            frame["source_pts_time"] for frame in timestamp_manifest["frames"]
        ],
        "work_dir": str(root),
    }


def run_self_test(work_dir: Path | None = None) -> dict[str, Any]:
    if work_dir is not None:
        if work_dir.exists() and any(work_dir.iterdir()):
            raise RuntimeError(f"self-test work dir must be empty: {work_dir}")
        return _execute(work_dir)
    with tempfile.TemporaryDirectory(prefix="offline-mask-selftest-") as value:
        result = _execute(Path(value))
        result["work_dir"] = "TEMPORARY_REMOVED_AFTER_TEST"
        return result
