"""FFmpeg 4.4-compatible wrapper for the frozen fresh Robot diagnostic stage.

The v1 IK, base search, rasterization and quality gates remain untouched.  Only
its encoder factory is replaced inside one synchronous process-local call; the
original factory is restored even when rendering fails.  This module is not a
standalone operation and cannot confer visual or training qualification.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from typing import Any, Iterator

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as legacy


ROOT = Path(__file__).resolve().parents[3]
_ENCODER_OVERRIDE_LOCK = threading.Lock()
VIDEO_NAMES = ("ego_robot", "external_full_robot", "raw_motion_source")


def ffmpeg44_encoder(path: Path, fps: float, width: int, height: int,
                     temp_dir: Path) -> subprocess.Popen:
    """Preserve input frame cadence without FFmpeg 5's ``-fps_mode`` option."""
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise legacy.FreshRobotError("ENCODER_ARGUMENT_INVALID")
    if temp_dir.resolve().parent != path.resolve().parent:
        raise legacy.FreshRobotError("ENCODER_TEMP_DIR_DRIFT")
    return subprocess.Popen(
        ["/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
         "-framerate", f"{fps:.9f}", "-i", "pipe:0", "-an", "-vsync", "0",
         "-c:v", "libx264", "-threads", "2", "-crf", "18",
         "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, cwd=ROOT,
        env={**os.environ, "TMPDIR": str(temp_dir), "TMP": str(temp_dir),
             "TEMP": str(temp_dir)},
    )


@contextmanager
def _ffmpeg44_override() -> Iterator[None]:
    """A synchronous, process-local compatibility shim; not thread-safe outside it."""
    with _ENCODER_OVERRIDE_LOCK:
        original = legacy._encoder
        legacy._encoder = ffmpeg44_encoder
        try:
            yield
        finally:
            legacy._encoder = original


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")


def run_robot_stage_ffmpeg44(*, hawor_npz: Path, frame_provenance: Path,
                             input_audit: Path, raw_video: Path,
                             source_kind: str, session_id: str,
                             output_root: Path) -> dict[str, Any]:
    """Write a new successor RESULT only after all three videos decode and time-check."""
    if output_root.exists():
        raise legacy.FreshRobotError("FRESH_OUTPUT_REQUIRED")
    with np.load(hawor_npz, allow_pickle=False) as archive:
        expected = len(archive["original_frame_indices"])
        fps = float(archive["fps_nominal"])
    if expected <= 0 or not math.isfinite(fps) or fps <= 0:
        raise legacy.FreshRobotError("FRESH_HAWOR_TIMELINE_INVALID")

    output_root.mkdir(parents=True)
    stage_root = output_root / "v1_stage_with_ffmpeg44_encoder"
    with _ffmpeg44_override():
        stage = legacy.run_robot_stage(
            hawor_npz=hawor_npz, frame_provenance=frame_provenance,
            input_audit=input_audit, raw_video=raw_video, source_kind=source_kind,
            session_id=session_id, output_root=stage_root)
    if (stage.get("status") != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS"
            or stage.get("frame_count") != expected
            or stage.get("session_id") != session_id):
        raise legacy.FreshRobotError("V1_STAGE_RESULT_IDENTITY_DRIFT")
    stage_result = stage_root / "RESULT.json"
    if not stage_result.is_file():
        raise legacy.FreshRobotError("V1_STAGE_RESULT_ABSENT")
    stored = json.loads(stage_result.read_text(encoding="utf-8"))
    if stored != stage:
        raise legacy.FreshRobotError("V1_STAGE_RESULT_DRIFT")
    videos = (stage.get("outputs") or {}).get("videos")
    if not isinstance(videos, dict) or set(videos) != set(VIDEO_NAMES):
        raise legacy.FreshRobotError("V1_STAGE_VIDEO_SET_DRIFT")
    verified: dict[str, Any] = {}
    for name in VIDEO_NAMES:
        prior = videos[name]
        if not isinstance(prior, dict) or not isinstance(prior.get("video"), dict):
            raise legacy.FreshRobotError(f"V1_STAGE_VIDEO_REF_ABSENT:{name}")
        path = Path(prior["video"]["path"])
        if path.resolve() != stage_root / "render" / {
                "ego_robot": "ego_robot_diagnostic.mp4",
                "external_full_robot": "fixed_external_full_robot.mp4",
                "raw_motion_source": "raw_motion_source_review.mp4"}[name]:
            raise legacy.FreshRobotError(f"V1_STAGE_VIDEO_PATH_DRIFT:{name}")
        if file_ref(path) != prior["video"]:
            raise legacy.FreshRobotError(f"V1_STAGE_VIDEO_SHA_DRIFT:{name}")
        check = legacy._verify_video(path, expected, (legacy.WIDTH, legacy.HEIGHT), fps)
        if check != prior:
            raise legacy.FreshRobotError(f"V1_STAGE_VIDEO_VERIFICATION_DRIFT:{name}")
        verified[name] = check
    result = {
        "schema_version": "exact78-fresh-robot-ffmpeg44-successor-result-v1",
        "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": session_id, "frame_count": expected,
        "nominal_duration_seconds": expected / fps,
        "duration_basis": "N_OVER_NOMINAL_FPS_NOT_CAPTURE_SYNCHRONIZATION",
        "encoder_change": "FFMPEG_4_4_VSYNC_0_REPLACES_FPS_MODE_PASSTHROUGH_ONLY",
        "stage_result": file_ref(stage_result), "verified_videos": verified,
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
        "claim_limit": "Camera-frame diagnostic only. The v1 IK, base search, raster and numerical gates are unchanged; no world/timebase or occlusion proof.",
    }
    _write_new(output_root / "RESULT.json", result)
    return result
