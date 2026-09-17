#!/usr/bin/env python3
"""Audit play_cards_0910_001 selected-eye/SBS mapping without inventing calibration."""

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


def digest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_frame(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    info = {
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame zero: {path}")
    return frame, info


def label(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 45), (20, 20, 20), -1)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--previous-depth-result", type=Path, required=True)
    parser.add_argument("--rectification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    root = args.session_root.resolve(strict=True)
    session = root.name
    if session != "play_cards_0910_001":
        raise ValueError("this bounded audit is pinned to play_cards_0910_001")
    camera_path = root / "camera_params.json"
    selected_video = root / f"CameraRecord_{session}.mp4"
    stereo_video = root / "source_stereo" / f"CameraRecord_{session}_stereo.mp4"
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    rectification_path = args.rectification.resolve(strict=True)
    rectification = json.loads(rectification_path.read_text(encoding="utf-8"))
    previous_path = args.previous_depth_result.resolve(strict=True)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_metrics_path = Path(previous["metrics"]["path"]).resolve(strict=True)
    previous_metrics = json.loads(previous_metrics_path.read_text(encoding="utf-8"))

    selected_frame, selected_info = first_frame(selected_video)
    stereo_frame, stereo_info = first_frame(stereo_video)
    if stereo_info["width"] % 2:
        raise ValueError("SBS width is not even")
    eye_width = stereo_info["width"] // 2
    source0 = stereo_frame[:, :eye_width]
    source1 = stereo_frame[:, eye_width:]
    source_indices = {
        "left": int(camera["left"]["sourceIndex"]),
        "right": int(camera["right"]["sourceIndex"]),
    }
    mapping_consistent = (
        source_indices == {"left": 1, "right": 0}
        and rectification.get("physical_eye_source_indices") == source_indices
        and camera.get("selected_calibration_key") == "right"
        and selected_info["width"] == int(camera["video_info"]["width"])
        and selected_info["height"] == int(camera["video_info"]["height"])
    )
    declared_source = root / camera["source_calibration"]["path"]
    original_source_present = declared_source.is_file()
    p90 = previous_metrics["vertical_epipolar_residual"]
    gate_pass = bool(p90["gate_pass"])
    if gate_pass or previous.get("foundationstereo_executed") is not False:
        raise ValueError("previous fail-closed Depth result contract changed unexpectedly")

    panel_size = (480, 360)
    panels = [
        label(cv2.resize(source0, panel_size), "SBS sourceIndex 0 = physical RIGHT", (0, 230, 255)),
        label(cv2.resize(source1, panel_size), "SBS sourceIndex 1 = physical LEFT", (255, 180, 0)),
        label(cv2.resize(selected_frame, panel_size), "selected MP4 = RIGHT / 1280x960", (80, 255, 80)),
    ]
    canvas = np.hstack(panels)
    footer = np.full((120, canvas.shape[1], 3), 24, np.uint8)
    lines = [
        "Same-session mapping evidence: consistent" if mapping_consistent else "Mapping evidence: CONFLICT",
        "Rectification P90(px): " + ", ".join(f"{value:.2f}" for value in p90["p90_px_by_frame"]),
        "Gate <=5px: FAIL; FoundationStereo NOT RUN; original factory calibration payload: "
        + ("present" if original_source_present else "missing (SHA declared only)"),
    ]
    for index, text in enumerate(lines):
        cv2.putText(footer, text, (12, 30 + 36 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (235, 235, 235) if index == 0 else (0, 180, 255), 2, cv2.LINE_AA)
    visual = output / f"{session}_selected_eye与SBS证据审计.png"
    if not cv2.imwrite(str(visual), np.vstack([canvas, footer])):
        raise RuntimeError("cannot write audit visualization")

    metrics = {
        "schema_version": "R22_SELECTED_EYE_SOURCE_INDEX_AUDIT_METRICS_V1",
        "session_id": session,
        "selected_eye": "right",
        "selected_calibration_key": camera.get("selected_calibration_key"),
        "physical_eye_source_indices": source_indices,
        "rectification_source_indices": rectification.get("physical_eye_source_indices"),
        "mapping_consistent": mapping_consistent,
        "selected_video": selected_info,
        "stereo_sbs_video": stereo_info,
        "stereo_eye_source_resolution": [eye_width, stereo_info["height"]],
        "camera_params_source_domain": [camera["width"], camera["height"]],
        "same_session_camera_params": digest(camera_path),
        "declared_original_factory_calibration": {
            "path": str(declared_source),
            "present": original_source_present,
            "declared_sha256": camera["source_calibration"].get("sha256"),
        },
        "rectification": digest(rectification_path),
        "vertical_epipolar_residual": p90,
        "foundationstereo_executed": False,
        "stereo_wrist_status": "UNAVAILABLE",
        "external_metric_accuracy": "UNKNOWN",
    }
    write_json(output / "METRICS.json", metrics)
    (output / "DECISION.md").write_text(
        "# selected-eye / SBS证据审计决定\n\n"
        "同会话`camera_params.json`明确记录：physical right=`sourceIndex 0`、physical left=`sourceIndex 1`，"
        "selected MP4选择right；SBS为4096×1536，两半各2048×1536。该映射与现有rectification文件一致。\n\n"
        "但rectification抽样P90仍为47.98/35.09/14.31 px，远高于5 px门；且camera_params中声明的"
        "原始factory calibration payload当前未随session materialize。不得猜测K、baseline或外参，"
        "因此继续BLOCKED_PREREQ，不运行FoundationStereo。\n",
        encoding="utf-8",
    )
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "R22_SELECTED_EYE_SOURCE_INDEX_NEXT_V1",
        "status": "BLOCKED_PREREQ",
        "next_task_id": "RECOVER_DECLARED_FACTORY_CALIBRATION_OR_FIX_RECTIFICATION_WITH_EVIDENCE",
    })
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "R22_SELECTED_EYE_SOURCE_INDEX_RECEIPT_V1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_params": digest(camera_path), "selected_video": digest(selected_video),
        "stereo_video": digest(stereo_video), "rectification": digest(rectification_path),
        "previous_depth_result": digest(previous_path), "previous_metrics": digest(previous_metrics_path),
        "producer": digest(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu_calls": 0, "authority_promoted": False,
    })
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "R22_SELECTED_EYE_SOURCE_INDEX_ARTIFACTS_V1",
        "artifacts": [digest(visual), digest(output / "METRICS.json"), digest(output / "DECISION.md"), digest(output / "NEXT_ACTION.json")],
    })
    write_json(output / "RESULT.json", {
        "schema_version": "R22_SELECTED_EYE_SOURCE_INDEX_RESULT_V1",
        "terminal_status": "BLOCKED_PREREQ",
        "blocker": "RECTIFICATION_P90_GATE_FAILED_AND_ORIGINAL_FACTORY_PAYLOAD_NOT_MATERIALIZED",
        "session_id": session, "mapping_consistent": mapping_consistent,
        "foundationstereo_executed": False, "stereo_wrist_status": "UNAVAILABLE",
        "visualization": digest(visual), "authority_promoted": False,
        "claim_limit": "Source-index and same-session evidence audit only; no Stereo, calibration or physical metric authority.",
    })
    print(json.dumps({"status": "BLOCKED_PREREQ", "mapping_consistent": mapping_consistent}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
