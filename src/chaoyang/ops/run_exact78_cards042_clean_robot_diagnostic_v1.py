"""Unregistered fenced CPU operation for Cards042 Clean + Robot diagnostics.

No route currently imports this module. A new finite packet and maintained
candidate route must authorize it before any video output is created.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json, process_identity


TASK = "exact78_raw_to_robot_t2_cards042_clean_robot_diagnostic_v1_20261001"
OPERATION = "run_exact78_cards042_clean_robot_diagnostic_v1"
ATTEMPT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
SPEC = ATTEMPT / "COMPOSITE_SPEC.json"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
VIDEO_NAMES = {
    "ego_clean_robot": "ego_clean_robot_diagnostic.mp4",
    "raw_mask_clean_composite": "raw_mask_clean_composite_review.mp4",
}


class CompositeOperationError(RuntimeError):
    pass


def authorize(spec_path: Path) -> dict[str, Any]:
    """Packet/index/writer check before importing raster or making output."""
    if spec_path.resolve() != SPEC:
        raise CompositeOperationError("COMPOSITE_SPEC_PATH_DRIFT")
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    routes = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if len(routes) != 1 or routes[0]["task_id"] != TASK:
        raise CompositeOperationError("COMPOSITE_NOT_ROUTABLE")
    packet_path = REPO_ROOT / routes[0]["packet_path"]
    if packet_path != PACKET or artifact_ref(packet_path)["sha256"] != routes[0]["packet_sha256"]:
        raise CompositeOperationError("COMPOSITE_PACKET_SHA_DRIFT")
    packet = load_json(PACKET)
    if (packet.get("task_id") != TASK
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or packet.get("inputs", {}).get("composite_spec") != artifact_ref(spec_path)
            or packet.get("commands") != [
                f"PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli run {OPERATION} --spec {SPEC}"]):
        raise CompositeOperationError("COMPOSITE_SPEC_NOT_PACKET_BOUND")
    writer = packet["writer"]
    identity = process_identity(writer["pid"])
    if (writer.get("scope") != TASK or not identity["alive"]
            or writer.get("proc_start_ticks") != identity["start_ticks"]
            or load_json(ATTEMPT / "WRITER.json") != writer
            or not isinstance(writer.get("fencing_token"), str)
            or len(writer["fencing_token"]) < 32):
        raise CompositeOperationError("COMPOSITE_WRITER_FENCE_DRIFT")
    if packet["write_set"][0] != str(ATTEMPT.relative_to(REPO_ROOT)):
        raise CompositeOperationError("COMPOSITE_WRITE_SET_DRIFT")
    from chaoyang.governance.exact78_operation_gate import authorize_exact78_operation

    authorize_exact78_operation(OPERATION, ["--spec", str(spec_path)])
    return packet


def _image(ref: dict[str, Any], flag: int, shape: tuple[int, ...], role: str) -> np.ndarray:
    from chaoyang.pipeline import exact78_clean_robot_diagnostic_cards042_v1 as candidate

    image = cv2.imread(str(candidate._file_ref(ref, role)), flag)
    if image is None or image.shape != shape or image.dtype != np.uint8:
        raise CompositeOperationError(f"COMPOSITE_IMAGE_DOMAIN_DRIFT:{role}")
    return image


def review_panel(binding: dict[str, Any], frame: int,
                 composite: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Four-panel visual audit; masks are display-only and never sort depth."""
    raw_ref = binding["raw_rows"][frame]["model_input"]
    clean_ref = binding["clean_rows"][frame]["clean"]
    mask_refs = binding["mask_rows"][frame]["masks"]
    raw = _image(raw_ref, cv2.IMREAD_COLOR, (960, 1280, 3), f"raw_{frame}")
    clean = _image(clean_ref, cv2.IMREAD_COLOR, (960, 1280, 3), f"clean_{frame}")
    removal = _image(mask_refs["removal"], cv2.IMREAD_GRAYSCALE,
                     (960, 1280), f"removal_{frame}")
    protected = _image(mask_refs["protected"], cv2.IMREAD_GRAYSCALE,
                       (960, 1280), f"protected_{frame}")
    if composite.shape != (480, 640, 3) or composite.dtype != np.uint8:
        raise CompositeOperationError(f"COMPOSITE_FRAME_DOMAIN_DRIFT:{frame}")
    small_raw = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
    small_clean = cv2.resize(clean, (640, 480), interpolation=cv2.INTER_AREA)
    red = cv2.resize(removal, (640, 480), interpolation=cv2.INTER_NEAREST) > 0
    green = cv2.resize(protected, (640, 480), interpolation=cv2.INTER_NEAREST) > 0
    mask_panel = small_raw.copy()
    mask_panel[red] = ((mask_panel[red].astype(np.uint16) +
                        np.array((0, 0, 255), np.uint16)) // 2).astype(np.uint8)
    mask_panel[green] = ((mask_panel[green].astype(np.uint16) +
                          np.array((0, 255, 0), np.uint16)) // 2).astype(np.uint8)
    panels = [small_raw, mask_panel, small_clean, composite.copy()]
    titles = ["RAW F%03d" % frame, "MASK DISPLAY ONLY - UNVERIFIED",
              "CLEAN SYNTHETIC - VISUAL ONLY", "ROBOT OVER CLEAN - NO OCCLUSION PROOF"]
    for panel, title in zip(panels, titles, strict=True):
        cv2.rectangle(panel, (0, 0), (640, 27), (0, 0, 0), -1)
        cv2.putText(panel, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
    return grid, {"raw": raw_ref, "clean": clean_ref,
                  "display_masks": {key: mask_refs[key] for key in ("removal", "protected")},
                  "removal_display_pixels": int(np.count_nonzero(removal)),
                  "protected_display_pixels": int(np.count_nonzero(protected)),
                  "mask_geometry_authority": False,
                  "review_panel_array_sha256": hashlib.sha256(grid.tobytes()).hexdigest()}


def verify_video(path: Path, frames_expected: int, size: tuple[int, int], fps: int) -> dict[str, Any]:
    """Complete decode, exact CFR frame PTS, stream duration, and file SHA."""
    if not path.is_file() or path.is_symlink():
        raise CompositeOperationError("COMPOSITE_VIDEO_MISSING_OR_LINK")
    probe = subprocess.run(["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries",
                            "stream=width,height,avg_frame_rate,r_frame_rate,time_base,duration_ts,nb_frames:frame=best_effort_timestamp,pkt_duration",
                            "-of", "json", str(path)], capture_output=True, text=True, check=True)
    payload = json.loads(probe.stdout)
    streams, frames = payload.get("streams", []), payload.get("frames", [])
    if len(streams) != 1 or len(frames) != frames_expected:
        raise CompositeOperationError("COMPOSITE_VIDEO_STREAM_OR_FRAME_COUNT")
    stream = streams[0]
    try:
        period = Fraction(1, fps) / Fraction(stream["time_base"])
        if (int(stream["width"]), int(stream["height"])) != size or period.denominator != 1:
            raise CompositeOperationError("COMPOSITE_VIDEO_DIMENSION_OR_TIMEBASE")
        if (Fraction(stream["avg_frame_rate"]) != fps
                or Fraction(stream["r_frame_rate"]) != fps
                or int(stream["duration_ts"]) != frames_expected * int(period)
                or stream.get("nb_frames") not in (None, "N/A", str(frames_expected))):
            raise CompositeOperationError("COMPOSITE_VIDEO_FPS_OR_DURATION")
        for index, frame in enumerate(frames):
            if (int(frame["best_effort_timestamp"]) != index * int(period)
                    or int(frame["pkt_duration"]) != int(period)):
                raise CompositeOperationError(f"COMPOSITE_VIDEO_PTS:{index}")
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        raise CompositeOperationError(f"COMPOSITE_VIDEO_PROBE_INVALID:{error}") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise CompositeOperationError("COMPOSITE_VIDEO_DECODE_OPEN")
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[1::-1] != size:
                raise CompositeOperationError(f"COMPOSITE_VIDEO_DECODE_SHAPE:{decoded}")
            decoded += 1
    finally:
        capture.release()
    if decoded != frames_expected:
        raise CompositeOperationError(f"COMPOSITE_VIDEO_DECODE_COUNT:{decoded}")
    return {"video": artifact_ref(path), "decoded_frames": decoded,
            "width": size[0], "height": size[1], "fps": fps,
            "duration_seconds": frames_expected / fps,
            "timing_claim": "ENCODED_CFR_INDEX_ONLY_NOT_CAPTURE_SYNC"}


def _single_thread_encoder(path: Path, size: tuple[int, int], temp: Path) -> subprocess.Popen:
    """FFmpeg 4.4-compatible CFR encoder; only one is live at any moment."""
    if temp.resolve().parent != path.resolve().parent or path.exists():
        raise CompositeOperationError("COMPOSITE_ENCODER_OUTPUT_DOMAIN")
    return subprocess.Popen(
        ["/usr/bin/ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-framerate", "30",
         "-i", "pipe:0", "-an", "-vsync", "0", "-c:v", "libx264",
         "-threads", "1", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, cwd=REPO_ROOT,
        env={**os.environ, "TMPDIR": str(temp), "TMP": str(temp), "TEMP": str(temp)},
    )


def _finish_encoder(worker: subprocess.Popen) -> None:
    assert worker.stdin is not None
    worker.stdin.close()
    if worker.wait(timeout=120):
        raise CompositeOperationError("COMPOSITE_ENCODER_FAILED")


def run_diagnostic(spec: dict[str, Any]) -> dict[str, Any]:
    """Called only after authorize(); no model, geometry, or active-task reads."""
    from chaoyang.pipeline import exact78_clean_robot_diagnostic_cards042_v1 as candidate

    if (set(spec) != {"task_id", "session_id", "inputs", "output_root"}
            or spec["task_id"] != TASK or spec["session_id"] != candidate.SESSION
            or Path(spec["output_root"]).resolve() != ATTEMPT / "candidate"
            or (ATTEMPT / "candidate").exists()):
        raise CompositeOperationError("COMPOSITE_SPEC_OR_OUTPUT_DRIFT")
    binding = candidate.prepare_binding(spec["inputs"])
    cv2.setNumThreads(1)
    root = ATTEMPT / "candidate"
    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=False)
    temp = videos_dir / "tmp"
    temp.mkdir()
    frames_dir = root / "composite_frames"
    frames_dir.mkdir()
    video_paths = {key: videos_dir / name for key, name in VIDEO_NAMES.items()}
    worker = _single_thread_encoder(video_paths["ego_clean_robot"], (640, 480), temp)
    rows = []
    try:
        for frame, (composite, row) in enumerate(candidate.iter_frames(binding)):
            if row["output_index"] != frame:
                raise CompositeOperationError(f"COMPOSITE_FRAME_INDEX_DRIFT:{frame}")
            row["geometry_decision"] = "UNKNOWN"
            row["display_policy"] = candidate.DISPLAY_POLICY
            image_path = frames_dir / f"{frame:06d}.png"
            if not cv2.imwrite(str(image_path), composite):
                raise CompositeOperationError(f"COMPOSITE_PNG_WRITE_FAILED:{frame}")
            row["composite_frame_png"] = artifact_ref(image_path)
            assert worker.stdin is not None
            worker.stdin.write(composite.tobytes())
            rows.append(row)
        if len(rows) != 171:
            raise CompositeOperationError("COMPOSITE_OUTPUT_FRAME_COUNT")
        _finish_encoder(worker)
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)
    ego_video = verify_video(video_paths["ego_clean_robot"], 171, (640, 480), 30)
    worker = _single_thread_encoder(video_paths["raw_mask_clean_composite"],
                                    (1280, 960), temp)
    try:
        for frame, row in enumerate(rows):
            image_path = frames_dir / f"{frame:06d}.png"
            if artifact_ref(image_path) != row["composite_frame_png"]:
                raise CompositeOperationError(f"COMPOSITE_STAGED_FRAME_SHA_DRIFT:{frame}")
            composite_frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if (composite_frame is None or composite_frame.shape != (480, 640, 3)
                    or hashlib.sha256(composite_frame.tobytes()).hexdigest()
                    != row["composite_array_sha256"]):
                raise CompositeOperationError(f"COMPOSITE_STAGED_FRAME_PIXEL_DRIFT:{frame}")
            grid, review = review_panel(binding, frame, composite_frame)
            row.update(review)
            assert worker.stdin is not None
            worker.stdin.write(grid.tobytes())
        _finish_encoder(worker)
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)
    videos = {
        "ego_clean_robot": ego_video,
        "raw_mask_clean_composite": verify_video(
            video_paths["raw_mask_clean_composite"], 171, (1280, 960), 30),
    }
    status_path = root / "FRAME_STATUS.json"
    with status_path.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "exact78-clean-robot-frame-status-v1", "rows": rows},
                  stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    # The sealed direction result supplies existing first/third-person and
    # Raw+motion reviews. These are links, never inputs to this composite.
    old_review = load_json(Path(binding["render"]["candidate_result"]["path"]))["videos"]
    result = {
        "schema_version": "exact78-cards042-clean-robot-diagnostic-v1",
        "status": "COMPLETE_DIAGNOSTIC_VIDEO_NOT_VISUAL_PASS",
        "session_id": candidate.SESSION, "frame_count": 171, "fps_nominal": 30,
        "diagnostic_video_complete": True,
        "inputs": spec["inputs"], "frame_status": artifact_ref(status_path),
        "videos": videos,
        "review_classes": {
            "raw_plus_motion_source": old_review["raw_motion_source"],
            "ego_robot_plus_fixed_external": {
                "new_ego_clean_robot": videos["ego_clean_robot"],
                "sealed_external": old_review["external_full_robot"]},
            "raw_mask_clean_composite": videos["raw_mask_clean_composite"],
        },
        "time_mode": "CLIP_INDEX_ONLY_TIMEBASE_UNVERIFIED",
        "geometry_decision": "UNKNOWN", "display_policy": candidate.DISPLAY_POLICY,
        "scene_depth_consumed": False, "object_mask_consumed_for_occlusion": False,
        "clean_role": "OFFLINE_SYNTHETIC_VISUAL_BACKGROUND_ONLY",
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
        "human_review": "PENDING", "ai_review": "PENDING",
        "claim_limit": "Review-only Clean+Robot overlay; no occlusion, object depth, world/timebase, visual or training authority",
    }
    result_path = root / "RESULT.json"
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    authorize(args.spec)
    result = run_diagnostic(load_json(args.spec))
    if result["status"] != "COMPLETE_DIAGNOSTIC_VIDEO_NOT_VISUAL_PASS":
        raise CompositeOperationError("COMPOSITE_RESULT_STATUS_DRIFT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
