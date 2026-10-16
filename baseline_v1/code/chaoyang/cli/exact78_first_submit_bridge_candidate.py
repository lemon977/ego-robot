"""CPU-only first-submit descriptor for a new, known-format Exact78 raw session.

This module is deliberately not routed by ``chaoyang submit``.  It neither
registers a Task Packet nor launches the T0-only model runner.  A successor
must freeze its result and pin a generalized stage runner before dispatch.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

from chaoyang.cli.exact78_submit import build_request
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    DATA_ROOT,
    ColdStartInputError,
    file_ref,
    validate_producer_reads,
)


_SESSION_ID = re.compile(r"^[A-Za-z0-9_]+$")
_TRACKER_SIZE = (1280, 960)
_EXPECTED_KINDS = {"TRACKER_CLIP_RGB", "0915_RAW_VST"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ColdStartInputError(f"SOURCE_JSON_INVALID:{path}") from exc
    if not isinstance(value, dict):
        raise ColdStartInputError("SOURCE_JSON_OBJECT_REQUIRED")
    return value


def _verified_file(source: dict[str, Any], role: str) -> Path:
    ref = source.get("files", {}).get(role)
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ColdStartInputError(f"SOURCE_ROLE_ABSENT:{role}")
    path = Path(ref["path"]).resolve(strict=True)
    session = Path(source["session_root"]).resolve(strict=True)
    if session not in path.parents or file_ref(path) != ref:
        raise ColdStartInputError(f"SOURCE_ROLE_SHA_OR_SCOPE_DRIFT:{role}")
    return path


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ColdStartInputError(f"{label}_INVALID")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ColdStartInputError(f"{label}_INVALID")
    return number


def _admit_known_domain(source: dict[str, Any]) -> dict[str, Any]:
    """Admit only the two already documented RGB layouts, without T0 IDs."""
    kind = source.get("kind")
    if kind not in _EXPECTED_KINDS:
        raise ColdStartInputError("UNSUPPORTED_SOURCE_DOMAIN")
    expected_roles = ({"video", "source_stereo_video", "camera_params", "slam", "clip_manifest"}
                      if kind == "TRACKER_CLIP_RGB" else
                      {"raw/vst.h264", "raw/vst.qpc.ts.jsonl", "raw/vst.ts.jsonl",
                       "raw/camera_params.json", "raw/camera_params.meta.json"})
    if set(source.get("files", {})) != expected_roles:
        raise ColdStartInputError("SOURCE_ROLE_SET_DRIFT")
    session_id = source.get("session_id")
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ColdStartInputError("UNSAFE_SESSION_ID")
    if kind == "TRACKER_CLIP_RGB":
        if (source.get("physical_eye") != "tracker_session_left_RGB"
                or source.get("crop_xywh") is not None
                or source.get("time_mode") != "TRACKER_CLIP_METADATA_ONLY"):
            raise ColdStartInputError("TRACKER_SOURCE_CLAIM_DRIFT")
        manifest = _object(_verified_file(source, "clip_manifest"))
        video = manifest.get("video")
        rect = manifest.get("video_rectification")
        if not isinstance(video, dict) or not isinstance(rect, dict):
            raise ColdStartInputError("TRACKER_RECTIFIED_LEFT_RGB_UNVERIFIED")
        if (rect.get("eye") != "left" or rect.get("rectified") is not True
                or rect.get("resolution") != list(_TRACKER_SIZE)
                or (video.get("width"), video.get("height")) != _TRACKER_SIZE):
            raise ColdStartInputError("TRACKER_RECTIFIED_LEFT_RGB_UNVERIFIED")
        intrinsic = rect.get("intrinsics")
        if (not isinstance(intrinsic, list) or len(intrinsic) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in intrinsic)):
            raise ColdStartInputError("TRACKER_INTRINSICS_UNVERIFIED")
        _positive_number(intrinsic[0][0], "TRACKER_FX")
        _positive_number(intrinsic[1][1], "TRACKER_FY")
        declared_frames = video.get("frame_count")
        if isinstance(declared_frames, bool) or not isinstance(declared_frames, int) or declared_frames <= 0:
            raise ColdStartInputError("TRACKER_FRAME_COUNT_INVALID")
        fps = _positive_number(video.get("fps"), "TRACKER_FPS")
        return {
            "raw_video_role": "video", "expected_dimensions": list(_TRACKER_SIZE),
            "declared_frame_count": declared_frames, "fps_nominal": fps,
            "physical_eye": "tracker_session_left_RGB", "crop_xywh": None,
            "time_mode": "TRACKER_CLIP_METADATA_ONLY",
            "capture_to_decode_alignment": "CLIP_INDEX_ONLY",
            "calibration_claim": "CLIP_RECTIFIED_K_ONLY_WORLD_POSE_NOT_SOLVED",
            "qpc_ticks": None, "capture_timestamps": None,
        }

    if (source.get("physical_eye") != "left_sourceIndex_1"
            or source.get("time_mode") != "TIMEBASE_UNVERIFIED"):
        raise ColdStartInputError("VST0915_SOURCE_CLAIM_DRIFT")
    camera = _object(_verified_file(source, "raw/camera_params.json"))
    width, height = camera.get("width"), camera.get("height")
    left, right = camera.get("left"), camera.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ColdStartInputError("VST0915_PHYSICAL_EYE_OR_SBS_UNVERIFIED")
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0
            or camera.get("outputWidth") != 2 * width
            or camera.get("outputHeight") != height
            or left.get("sourceIndex") != 1
            or right.get("sourceIndex") != 0
            or source.get("crop_xywh") != [width, 0, width, height]):
        raise ColdStartInputError("VST0915_PHYSICAL_EYE_OR_SBS_UNVERIFIED")
    for field in ("fx", "fy"):
        _positive_number(left.get(field), f"VST0915_{field.upper()}")
    fps = _positive_number(camera.get("fps"), "VST0915_FPS")
    ticks = _verified_file(source, "raw/vst.qpc.ts.jsonl").read_text(encoding="utf-8").splitlines()
    timestamps = _verified_file(source, "raw/vst.ts.jsonl").read_text(encoding="utf-8").splitlines()
    if (not ticks or len(ticks) != len(timestamps)
            or any(not value.isdecimal() for value in ticks + timestamps)):
        raise ColdStartInputError("VST0915_TIMESTAMP_SIDECAR_INVALID")
    return {
        "raw_video_role": "raw/vst.h264", "expected_dimensions": [2 * width, height],
        "declared_frame_count": len(ticks), "fps_nominal": fps,
        "physical_eye": "left_sourceIndex_1", "crop_xywh": [width, 0, width, height],
        "time_mode": "TIMEBASE_UNVERIFIED",
        "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
        "calibration_claim": "SCALED_FACTORY_FISHEYE_K_MODEL_INPUT_ONLY_NOT_METRIC",
        "qpc_ticks": ticks, "capture_timestamps": timestamps,
    }


def _decode_frame_rows(video: Path, *, timeout_seconds: int = 1800) -> list[dict[str, Any]]:
    """Ask ffprobe to decode video frames; no image/model output is written."""
    command = ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_frames", "-show_entries",
               "frame=pts,best_effort_timestamp,pict_type,width,height", "-of", "json", str(video)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ColdStartInputError("CPU_FRAME_DECODE_PROBE_FAILED") from exc
    if completed.returncode != 0:
        raise ColdStartInputError("CPU_FRAME_DECODE_PROBE_FAILED")
    try:
        frames = json.loads(completed.stdout).get("frames")
    except (ValueError, AttributeError) as exc:
        raise ColdStartInputError("CPU_FRAME_DECODE_PROBE_INVALID") from exc
    if not isinstance(frames, list) or not frames:
        raise ColdStartInputError("CPU_FRAME_DECODE_PROBE_EMPTY")
    return frames


def plan_single(session: Path, *, allowed_root: Path = DATA_ROOT) -> dict[str, Any]:
    """Describe a first submit; deliberately never register or run it."""
    request = build_request(session=session, allowed_root=allowed_root)
    sources = request.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise ColdStartInputError("ONE_RAW_SESSION_REQUIRED")
    source = sources[0]
    domain = _admit_known_domain(source)
    raw_video = _verified_file(source, domain["raw_video_role"])
    frames = _decode_frame_rows(raw_video)
    if len(frames) != domain["declared_frame_count"]:
        raise ColdStartInputError("RAW_DECODE_FRAME_COUNT_DRIFT")
    if any([row.get("width"), row.get("height")] != domain["expected_dimensions"]
           for row in frames):
        raise ColdStartInputError("RAW_DECODE_DIMENSION_DRIFT")
    # Re-read every declared raw role after the decode probe. This closes a
    # source mutation window; it is not a native-open audit of model execution.
    validate_producer_reads(source, [Path(ref["path"]) for ref in source["files"].values()])
    frame_rows = []
    for index, row in enumerate(frames):
        frame_rows.append({
            "decoded_display_index": index,
            "pts": row.get("pts"),
            "best_effort_timestamp": row.get("best_effort_timestamp"),
            "pict_type": row.get("pict_type"),
            "qpc_raw_tick": (int(domain["qpc_ticks"][index]) if domain["qpc_ticks"] is not None else None),
            "capture_timestamp_raw": (int(domain["capture_timestamps"][index])
                                      if domain["capture_timestamps"] is not None else None),
            "capture_to_decode_alignment": domain["capture_to_decode_alignment"],
            "sbs_physical_eye": domain["physical_eye"],
            "crop_xywh": domain["crop_xywh"],
        })
    descriptor = {
        "schema_version": "exact78-first-submit-descriptor-candidate-v1",
        "session_id": source["session_id"], "session_root": source["session_root"],
        "source_kind": source["kind"], "source_files": source["files"],
        "raw_video": source["files"][domain["raw_video_role"]],
        "submission_input_sha256": request["submission_input_sha256"],
        "decoded_frames": len(frame_rows), "fps_nominal": domain["fps_nominal"],
        "physical_eye": domain["physical_eye"], "crop_xywh": domain["crop_xywh"],
        "time_mode": domain["time_mode"],
        "capture_to_decode_alignment": domain["capture_to_decode_alignment"],
        "calibration_claim": domain["calibration_claim"],
        "independent_source_group": "UNVERIFIED",
        "world_pose_authority": False, "metric_authority": False,
        "frame_rows": frame_rows,
    }
    descriptor_sha = hashlib.sha256(_canonical(descriptor)).hexdigest()
    task_id = f"exact78_newraw_{source['session_id']}_{descriptor_sha[:16]}_v1"
    target_draft = {
        "task_id": task_id, "session_id": source["session_id"],
        "session": source["session_root"], "frame_count": len(frame_rows),
        "fps_nominal": domain["fps_nominal"], "time_mode": domain["time_mode"],
        "physical_eye": domain["physical_eye"],
        "attempt_root": f"_run/current/{task_id}/attempts/attempt_0001",
        "packet_path": f"tasks/current/{task_id}/TASK_PACKET.json",
        "receipt_path": f"tasks/receipts/{task_id.upper()}.json",
    }
    return {
        "status": "BLOCKED_RUNNER_NOT_GENERALIZED",
        "schema_version": "exact78-first-submit-bridge-candidate-v1",
        "descriptor": descriptor, "descriptor_sha256": descriptor_sha,
        "target_draft": target_draft,
        "task_packet_draft": {
            "task_id": task_id, "execution_mode": "EXACT78_CANDIDATE",
            "execution_allowed": False, "commands": [], "writer": None,
            "inputs": {"submission_input_sha256": request["submission_input_sha256"],
                       "descriptor_sha256": descriptor_sha},
            "runtime_code": "MUST_PIN_GENERALIZED_RUNNER_AND_RECIPE",
            "deadline_at": "MUST_SET_NEW_FINITE_DEADLINE",
            "run_signature": "UNAVAILABLE_UNTIL_RECIPE_AND_RUNNER_PINNED",
        },
        "execution_started_here": False, "registered": False,
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False,
        "blocking_gaps": ["GENERALIZED_HAWOR_STAGE_DESCRIPTOR_INPUT",
                          "GENERALIZED_ROBOT_STAGE_ENTRY",
                          "NEW_FINITE_PACKET_REGISTER_RUN_LEASE_WRAPPER"],
    }


__all__ = ["plan_single"]
