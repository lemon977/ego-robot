"""Candidate-only RGB frontend shared by the 0901, 0902 and 0915 sources.

The input spec and every producer are pinned by the enclosing Task Packet.  The
0915 branch reads capture ``raw/vst.h264`` directly: physical left is SBS half
one, and encoded pixels are cropped and resized without lens remapping.  Clean
is a synthetic offline visual layer, never geometry or training evidence.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, process_identity
from chaoyang.ops.run_exact78_fresh_frontend_v1 import checked, normalize, save
from chaoyang.pipeline.vst_encoded_video_domain import (
    ADMITTED_TRANSFORM, ENCODED_DOMAIN, split_resize_physical_eyes,
    validate_encoded_video_contract,
)


TRACKER_SOURCES = frozenset({"TRACKER_0901", "TRACKER_0902"})
VST_SOURCE = "VST_0915"
SOURCE_FAMILIES = TRACKER_SOURCES | {VST_SOURCE}
IMAGE_SIZE = (1280, 960)
MASK_MISSING_SLEEVE_CABLE = "MASK_SLEEVE_CABLE_UNVERIFIED"
MAP_GENERATOR = REPO_ROOT / "vendor/FoundationStereo/scripts/pico_stereo_depth.py"
TRACKER_MAP_RULE = "GENERATE_FROM_CAPTURE_CAMERA_PARAMS"
VST_MAP_RULE = "NO_MAP_VST_ENCODED"


def validate_spec(spec: dict[str, Any]) -> None:
    required = {
        "task_id", "session_id", "task", "source_family", "source_root",
        "source_video", "camera_params", "map_rule", "map_generator_code", "source_index",
        "source_eye_size", "frame_count", "source_fps", "image_domain",
        "transform", "frame_timing", "sam_checkpoint", "propainter_entry",
        "propainter_weights", "output_root",
    }
    if not required <= spec.keys() or spec["source_family"] not in SOURCE_FAMILIES:
        raise ValueError("FRONTEND_SPEC_INCOMPLETE")
    if spec["task"] not in {"poker", "chips"}:
        raise ValueError("UNKNOWN_TASK")
    if type(spec["frame_count"]) is not int or spec["frame_count"] <= 0:
        raise ValueError("INVALID_FRAME_COUNT")
    if type(spec["source_fps"]) is not int or spec["source_fps"] not in (25, 30):
        raise ValueError("SOURCE_FPS_MUST_BE_NOMINAL_25_OR_30")
    if type(spec["source_index"]) is not int or spec["source_index"] not in (0, 1):
        raise ValueError("INVALID_SOURCE_INDEX")
    eye_size = spec["source_eye_size"]
    if (not isinstance(eye_size, list) or len(eye_size) != 2
            or any(type(value) is not int or value <= 0 for value in eye_size)):
        raise ValueError("INVALID_SOURCE_EYE_SIZE")
    if not isinstance(spec["propainter_weights"], dict) or set(spec["propainter_weights"]) != {
        "raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"
    }:
        raise ValueError("PROPAINTER_WEIGHTS_NOT_PINNED")
    root = Path(spec["source_root"]).resolve(strict=True)
    video = Path(spec["source_video"]["path"]).resolve(strict=True)
    camera_path = Path(spec["camera_params"]["path"]).resolve(strict=True)
    if not video.is_relative_to(root):
        raise ValueError("VIDEO_OUTSIDE_ORIGINAL_SESSION")
    if spec["source_family"] in TRACKER_SOURCES:
        if video.parent.name != "source_stereo" or video.suffix.lower() != ".mp4":
            raise ValueError("TRACKER_SOURCE_MUST_BE_ORIGINAL_STEREO")
        if camera_path != root / "camera_params.json":
            raise ValueError("TRACKER_CAMERA_MUST_BE_SAME_SESSION")
        if (spec["map_rule"] != TRACKER_MAP_RULE or spec.get("camera_maps") not in (None, "ABSENT")
                or not isinstance(spec["map_generator_code"], dict)):
            raise ValueError("TRACKER_MAP_MUST_BE_GENERATED_IN_STAGE")
        if spec["transform"] != "SAME_SESSION_EQUIDIS62_TO_PINHOLE":
            raise ValueError("TRACKER_IMAGE_TRANSFORM_UNVERIFIED")
    else:
        if video != root / "raw" / "vst.h264":
            raise ValueError("VST_SOURCE_MUST_BE_CAPTURE_RAW")
        if camera_path != root / "raw" / "camera_params.json":
            raise ValueError("VST_CAMERA_MUST_BE_CAPTURE_RAW")
        if (spec["source_index"] != 1 or spec["map_rule"] != VST_MAP_RULE
                or spec["map_generator_code"] != "ABSENT"
                or spec.get("camera_maps") not in (None, "ABSENT")):
            raise ValueError("VST_PHYSICAL_LEFT_OR_MAP_VIOLATION")
        validate_encoded_video_contract({
            "encoded_video_domain": spec["image_domain"],
            "transform": spec["transform"],
            "lens_undistortion_applied": False,
            "distortion_coefficients_consumed": False,
        })
    if spec["frame_timing"] != "ABSENT" and not isinstance(spec["frame_timing"], dict):
        raise ValueError("FRAME_TIMING_MUST_BE_REF_OR_ABSENT")


def map_stage(spec: dict[str, Any], output: Path) -> None:
    """Seal a same-session visual map before RAW; no stereo extrinsics needed."""
    validate_spec(spec)
    if (output / "MAP.json").exists():
        raise RuntimeError("MAP_RESULT_ALREADY_EXISTS")
    camera = load_json(checked(spec["camera_params"]))
    resolution = camera.get("reference_resolution", {})
    if spec["source_family"] in TRACKER_SOURCES:
        if (resolution.get("w"), resolution.get("h")) != tuple(spec["source_eye_size"]):
            raise RuntimeError("CALIBRATION_EYE_SIZE_DRIFT")
        left = camera.get("left", {})
        if left.get("sourceIndex", spec["source_index"]) != spec["source_index"]:
            raise RuntimeError("CALIBRATION_SOURCE_INDEX_DRIFT")
        distortion = left.get("distortion", {})
        intrinsic = left.get("intrinsics", {})
        names = ("fx", "fy", "cx", "cy")
        coeffs = distortion.get("coeffs")
        if (distortion.get("model") != "equiDis62" or not isinstance(coeffs, list)
                or len(coeffs) != 8 or any(name not in intrinsic for name in names)):
            raise RuntimeError("TRACKER_LEFT_CALIBRATION_DOMAIN")
        values = np.asarray([intrinsic[name] for name in names] + coeffs, np.float64)
        if not np.isfinite(values).all() or values[0] <= 0 or values[1] <= 0:
            raise RuntimeError("TRACKER_LEFT_CALIBRATION_NONFINITE")
        generator = checked(spec["map_generator_code"])
        if generator.resolve(strict=True) != MAP_GENERATOR.resolve(strict=True):
            raise RuntimeError("MAP_GENERATOR_CODE_DRIFT")
        sys.path.insert(0, str(MAP_GENERATOR.parent))
        from pico_stereo_depth import Eye, make_map, virtual_intrinsics

        width, height = IMAGE_SIZE
        k = virtual_intrinsics(width, height, 90.0)
        maps = make_map(Eye(values[:4], values[4:]), np.eye(3), width, height, k)
        map_path = output / "CAMERA_MAPS.npz"
        with map_path.open("xb") as stream:
            np.savez(stream, rectify_to_native_x=maps[0], rectify_to_native_y=maps[1])
        map_ref: dict[str, Any] | str = artifact_ref(map_path)
        generator_ref: dict[str, Any] | str = spec["map_generator_code"]
        output_intrinsics: list[list[float]] | None = k.tolist()
    else:
        if (camera.get("left", {}).get("sourceIndex"), camera.get("right", {}).get("sourceIndex")) != (1, 0):
            raise RuntimeError("VST_CALIBRATION_PHYSICAL_EYE_DRIFT")
        map_ref = "ABSENT"
        generator_ref = "ABSENT"
        output_intrinsics = None
    atomic_json(output / "MAP.json", {
        "schema_version": "exact78-fresh-frontend-map-v2",
        "session_id": spec["session_id"], "source_family": spec["source_family"],
        "camera_params": spec["camera_params"], "source_video": spec["source_video"],
        "source_index": spec["source_index"], "map_rule": spec["map_rule"],
        "generator_code": generator_ref, "output_intrinsics": output_intrinsics,
        "camera_maps": map_ref, "metric_authority": False,
    })


def select_primary_frame(
    sbs: np.ndarray, spec: dict[str, Any], maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    eye_width, eye_height = spec["source_eye_size"]
    if sbs.shape[:2] != (eye_height, eye_width * 2) or sbs.ndim != 3 or sbs.shape[2] != 3:
        raise ValueError("RAW_SBS_IMAGE_DOMAIN")
    half = int(spec["source_index"])
    crop = (eye_width * half, 0, eye_width, eye_height)
    if spec["source_family"] == VST_SOURCE:
        if maps is not None:
            raise ValueError("VST_REMAP_FORBIDDEN")
        primary, _ = split_resize_physical_eyes(
            sbs, eye_width=eye_width, source_indices=(1, 0), output_size=IMAGE_SIZE,
        )
        return primary, crop
    if maps is None or maps[0].shape != IMAGE_SIZE[::-1] or maps[1].shape != IMAGE_SIZE[::-1]:
        raise ValueError("TRACKER_CAMERA_MAP_DOMAIN")
    eye = sbs[:, crop[0]:crop[0] + crop[2]]
    return cv2.remap(eye, maps[0], maps[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT), crop


def _timing_rows(spec: dict[str, Any]) -> tuple[list[dict[str, Any]] | None, str]:
    ref = spec["frame_timing"]
    if ref == "ABSENT":
        return None, "TIMEBASE_UNVERIFIED"
    record = load_json(checked(ref))
    frozen = record.get("sessions", {}).get(spec["session_id"])
    if frozen is not None:
        if frozen.get("decoded_frames") != spec["frame_count"]:
            raise ValueError("FROZEN_DECODE_COUNT_DRIFT")
        nominal_fps = frozen.get("nominal_fps_from_capture_metadata")
        if nominal_fps is None or abs(float(nominal_fps) - spec["source_fps"]) > 0.01:
            raise ValueError("SOURCE_FPS_CAPTURE_METADATA_DRIFT")
        rows = frozen.get("frame_rows")
        mode = "TIMEBASE_UNVERIFIED"
        if spec["source_family"] == VST_SOURCE and frozen.get("time_mode") != "TIMEBASE_UNVERIFIED":
            raise ValueError("VST_FROZEN_TIMEBASE_DRIFT")
    else:
        rows = record.get("frames")
        mode = record.get("time_mode")
    if not isinstance(rows, list) or len(rows) != spec["frame_count"]:
        raise ValueError("FRAME_TIMING_COUNT")
    ids = [row.get("decoded_display_index", row.get("decode_display_index")) for row in rows]
    if ids != list(range(spec["frame_count"])):
        raise ValueError("FRAME_TIMING_DUPLICATE_MISSING_OR_REORDERED")
    capture_ids = [row.get("capture_frame_id") for row in rows]
    if any(value is not None for value in capture_ids) and (
        any(type(value) is not int for value in capture_ids)
        or capture_ids != list(range(capture_ids[0], capture_ids[0] + len(capture_ids)))
    ):
        raise ValueError("CAPTURE_FRAME_ID_DUPLICATE_OR_MISSING")
    if mode not in {"SOURCE_TIMEBASE_VERIFIED", "TIMEBASE_UNVERIFIED"}:
        raise ValueError("FRAME_TIME_MODE")
    if spec["source_family"] == VST_SOURCE and mode != "TIMEBASE_UNVERIFIED":
        raise ValueError("VST_QPC_TIMEBASE_NOT_PROVEN")
    normalized = []
    for row in rows:
        normalized.append({
            "decode_display_index": row.get("decoded_display_index", row.get("decode_display_index")),
            "capture_frame_id": row.get("capture_frame_id"),
            "capture_to_decode_alignment": row.get("capture_to_decode_alignment", "UNVERIFIED"),
            "qpc_ticks": row.get("qpc_ticks", row.get("qpc_raw_tick")),
            "pts": row.get("pts"),
            "capture_timestamp_raw": row.get("capture_timestamp_raw"),
        })
    for field in ("pts", "qpc_ticks"):
        values = [row.get(field) for row in normalized]
        if any(value is not None for value in values):
            if (any(type(value) not in (int, float) or not np.isfinite(value) for value in values)
                    or any(right <= left for left, right in zip(values, values[1:]))):
                raise ValueError(f"FRAME_{field.upper()}_MISSING_OR_NONMONOTONIC")
    return normalized, mode


def raw(spec: dict[str, Any], output: Path) -> None:
    validate_spec(spec)
    video = checked(spec["source_video"])
    map_result_path = output / "MAP.json"
    if not map_result_path.is_file():
        raise RuntimeError("MAP_STAGE_NOT_SEALED")
    map_record = load_json(map_result_path)
    if (map_record.get("session_id") != spec["session_id"]
            or map_record.get("source_family") != spec["source_family"]
            or map_record.get("camera_params") != spec["camera_params"]
            or map_record.get("source_video") != spec["source_video"]
            or map_record.get("source_index") != spec["source_index"]
            or map_record.get("map_rule") != spec["map_rule"]):
        raise RuntimeError("MAP_STAGE_IDENTITY_DRIFT")
    camera = load_json(checked(spec["camera_params"]))
    left_index = camera.get("left", {}).get("sourceIndex")
    right_index = camera.get("right", {}).get("sourceIndex")
    if spec["source_family"] == VST_SOURCE and (left_index, right_index) != (1, 0):
        raise RuntimeError("VST_CALIBRATION_PHYSICAL_EYE_DRIFT")
    if left_index is not None and left_index != spec["source_index"]:
        raise RuntimeError("CALIBRATION_SOURCE_INDEX_DRIFT")
    resolution = camera.get("reference_resolution", {})
    if resolution and (resolution.get("w"), resolution.get("h")) != tuple(spec["source_eye_size"]):
        raise RuntimeError("CALIBRATION_EYE_SIZE_DRIFT")
    maps = None
    if spec["source_family"] in TRACKER_SOURCES:
        if map_record.get("generator_code") != spec["map_generator_code"]:
            raise RuntimeError("MAP_STAGE_GENERATOR_DRIFT")
        with np.load(checked(map_record["camera_maps"]), allow_pickle=False) as archive:
            maps = (np.asarray(archive["rectify_to_native_x"], np.float32),
                    np.asarray(archive["rectify_to_native_y"], np.float32))
    elif map_record.get("camera_maps") != "ABSENT":
        raise RuntimeError("VST_MAP_FORBIDDEN")
    timing, time_mode = _timing_rows(spec)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("RAW_VIDEO_OPEN")
    encoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if spec["source_family"] in TRACKER_SOURCES and (
        not np.isfinite(encoded_fps) or abs(encoded_fps - spec["source_fps"]) > 0.01
    ):
        capture.release()
        raise RuntimeError("TRACKER_SOURCE_FPS_DRIFT")
    fps_status = ("TRACKER_ENCODED_AND_CAPTURE_METADATA_MATCH" if spec["source_family"] in TRACKER_SOURCES
                  else "VST_CAPTURE_NOMINAL_ONLY_TIMEBASE_UNVERIFIED")
    rows = []
    try:
        for index in range(spec["frame_count"]):
            ok, sbs = capture.read()
            if not ok:
                raise RuntimeError(f"RAW_DECODE_MISSING:{index}")
            image, crop = select_primary_frame(sbs, spec, maps)
            stamp = timing[index] if timing is not None else {}
            rows.append({
                "decode_display_index": index,
                "capture_frame_id": stamp.get("capture_frame_id"),
                "capture_to_decode_alignment": stamp.get("capture_to_decode_alignment", "UNVERIFIED"),
                "sbs_source_index": spec["source_index"],
                "physical_eye": "LEFT",
                "crop_xywh": list(crop),
                "transform": spec["transform"],
                "qpc_ticks": stamp.get("qpc_ticks"),
                "pts": stamp.get("pts"),
                "capture_timestamp_raw": stamp.get("capture_timestamp_raw"),
                "time_mode": time_mode,
                "model_input_index": index,
                "model_input": save(output / "raw" / f"{index:06d}.png", image),
                "array_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            })
        if capture.read()[0]:
            raise RuntimeError("RAW_DECODE_EXTRA_FRAME")
    finally:
        capture.release()
    atomic_json(output / "RAW.json", {
        "schema_version": "exact78-fresh-frontend-raw-v2",
        "session_id": spec["session_id"], "source_family": spec["source_family"],
        "source_video": spec["source_video"], "camera_params": spec["camera_params"],
        "map_result": artifact_ref(map_result_path),
        "camera_maps": map_record["camera_maps"], "frame_timing": spec["frame_timing"],
        "source_eye_size": spec["source_eye_size"],
        "source_fps": spec["source_fps"], "encoded_fps_reported": encoded_fps,
        "fps_status": fps_status, "time_mode": time_mode,
        "rows": rows, "metric_authority": False,
    })


def _prompts(spec: dict[str, Any]) -> dict[str, str]:
    result = {
        "human": "a person's hand and forearm",
        "object": "a playing card" if spec["task"] == "poker" else "a potato chip",
    }
    if spec["source_family"] == VST_SOURCE:
        result.update({"equipment": "a tactile glove", "finger_sleeve": "a finger sleeve",
                       "yellow_cable": "a yellow cable"})
    else:
        result["equipment"] = "a wrist worn tracking device"
    return result


def role_status(source_family: str) -> str:
    return MASK_MISSING_SLEEVE_CABLE if source_family == VST_SOURCE else "MODEL_OUTPUT_UNREVIEWED"


def masks(spec: dict[str, Any], output: Path) -> None:
    import torch
    sys.path.insert(0, str(REPO_ROOT / "vendor/SAM3"))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter

    raw_record = load_json(output / "RAW.json")
    if raw_record["session_id"] != spec["session_id"] or len(raw_record["rows"]) != spec["frame_count"]:
        raise RuntimeError("RAW_STAGE_NOT_SEALED")
    torch.set_num_threads(2)
    adapter, evidence = build_pinned_adapter(
        official_code_root=REPO_ROOT / "vendor/SAM3",
        checkpoint_path=checked(spec["sam_checkpoint"]),
    )
    prompts = _prompts(spec)
    rows = []
    started = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for raw_row in raw_record["rows"]:
                index = raw_row["decode_display_index"]
                image = cv2.imread(str(checked(raw_row["model_input"])))
                if image is None or image.shape != (960, 1280, 3):
                    raise RuntimeError(f"MODEL_INPUT_DOMAIN:{index}")
                frame_dir = output / "sam_inputs" / f"{index:06d}"
                sam_input = save(frame_dir / "000000.jpg", image)
                selected = {}
                counts = {}
                for role, prompt in prompts.items():
                    state = adapter.model.init_state(resource_path=str(frame_dir),
                        offload_video_to_cpu=True, async_loading_frames=False)
                    try:
                        _, prediction = adapter.model.add_prompt(
                            inference_state=state, frame_idx=0, text_str=prompt,
                            output_prob_thresh=0.5,
                        )
                        instances, scores = normalize(prediction, image.shape[:2])
                        chosen = instances[scores >= 0.5]
                        selected[role] = np.any(chosen, axis=0) if len(chosen) else np.zeros(image.shape[:2], bool)
                        counts[role] = int(len(chosen))
                    finally:
                        state.clear()
                equipment = selected["equipment"].copy()
                if spec["source_family"] == VST_SOURCE:
                    equipment |= selected["finger_sleeve"] | selected["yellow_cable"]
                human_equipment = selected["human"] | equipment
                unknown = human_equipment & selected["object"]
                protected = selected["object"] & ~human_equipment
                removal = human_equipment & ~selected["object"]
                images = {**selected, "human_equipment": human_equipment,
                          "unknown": unknown, "protected": protected, "removal": removal}
                refs = {key: save(output / "masks" / key / f"{index:06d}.png",
                                  value.astype(np.uint8) * 255)
                        for key, value in images.items()}
                rows.append({
                    "decode_display_index": index, "model_input": raw_row["model_input"],
                    "sam_input": sam_input, "sam_output_index": 0,
                    "masks": refs, "instance_counts": counts,
                    "removal_pixels": int(removal.sum()),
                    "unknown_overlap_pixels": int(unknown.sum()),
                    "role_status": role_status(spec["source_family"]),
                    "mask_quality": "MODEL_OUTPUT_NOT_GROUND_TRUTH",
                })
                if index % 10 == 0:
                    print(json.dumps({"stage": "SAM", "frame": index,
                                      "seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        adapter.predictor.shutdown()
    if not rows or not any(row["removal_pixels"] for row in rows):
        raise RuntimeError("EMPTY_HUMAN_EQUIPMENT_MASK_ALL_FRAMES")
    atomic_json(output / "MASK.json", {
        "schema_version": "exact78-fresh-frontend-mask-v2",
        "session_id": spec["session_id"], "source_family": spec["source_family"],
        "raw_result": artifact_ref(output / "RAW.json"),
        "rows": rows, "prompts": prompts, "probability_threshold": 0.5,
        "build_evidence": evidence, "fresh_model_inference": True,
        "physical_instance_tracking": "NOT_CLAIMED",
        "visible_object_identity": "UNVERIFIED_TEXT_UNION",
        "sleeve_cable_status": MASK_MISSING_SLEEVE_CABLE if spec["source_family"] == VST_SOURCE
                               else "NOT_APPLICABLE",
        "training_eligible": False,
    })


def prepare(spec: dict[str, Any], output: Path) -> None:
    record = load_json(output / "MASK.json")
    if record["session_id"] != spec["session_id"] or len(record["rows"]) != spec["frame_count"]:
        raise RuntimeError("MASK_STAGE_NOT_SEALED")
    rows = []
    for row in record["rows"]:
        index = row["decode_display_index"]
        image = cv2.imread(str(checked(row["model_input"])))
        removal_image = cv2.imread(str(checked(row["masks"]["removal"])), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (960, 1280, 3) or removal_image is None or removal_image.shape != (960, 1280):
            raise RuntimeError(f"PREPARE_IMAGE_DOMAIN:{index}")
        removal = removal_image > 0
        prepared_image = save(output / "propainter_input/frames" / f"{index:06d}.png",
                              cv2.resize(image, (960, 720), interpolation=cv2.INTER_AREA))
        prepared_mask = save(output / "propainter_input/masks" / f"{index:06d}.png",
                             cv2.resize(removal.astype(np.uint8) * 255, (960, 720), interpolation=cv2.INTER_NEAREST))
        rows.append({"decode_display_index": index, "model_input": row["model_input"],
                     "masks": row["masks"], "role_status": row["role_status"],
                     "propainter_input": prepared_image, "propainter_mask": prepared_mask,
                     "propainter_input_index": index})
    atomic_json(output / "PREPARE.json", {
        "schema_version": "exact78-fresh-frontend-prepare-v2",
        "session_id": spec["session_id"], "mask_result": artifact_ref(output / "MASK.json"),
        "rows": rows,
        "qualified_real_donors": 0, "old_depth_or_donor_maps_consumed": False,
        "fallback": "SYNTHETIC_PROPAINTER_NOT_OBSERVED_BACKGROUND",
    })


def propainter_command(spec: dict[str, Any], output: Path, entry: Path) -> list[str]:
    """Build an unchanged ProPainter recipe with native source FPS metadata."""
    return [sys.executable, "-B", str(entry),
        "--video", str(output / "propainter_input/frames"),
        "--mask", str(output / "propainter_input/masks"),
        "--output", str(output / "propainter_model"), "--width", "960", "--height", "720",
        "--mask_dilation", "4", "--ref_stride", "10", "--neighbor_length", "10",
        "--subvideo_length", "80", "--raft_iter", "20",
        "--save_fps", str(spec["source_fps"]), "--save_frames", "--fp16"]


def verify_internal_mp4(path: Path, frame_count: int, fps: int) -> dict[str, Any]:
    """Fully decode and time-check an internal ProPainter video before sealing it."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"PROPAINTER_VIDEO_MISSING_OR_LINK:{path}")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,duration,width,height:format=duration:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ], capture_output=True, text=True, check=False)
    if probe.returncode:
        raise RuntimeError(f"PROPAINTER_VIDEO_PROBE:{path}:{probe.stderr[-500:]}")
    try:
        record = json.loads(probe.stdout)
        streams = record["streams"]
        frames = record["frames"]
        if len(streams) != 1 or len(frames) != frame_count:
            raise ValueError("stream or frame count")
        stream = streams[0]
        observed_fps = Fraction(stream["avg_frame_rate"])
        if observed_fps != fps or (stream["width"], stream["height"]) != (960, 720):
            raise ValueError("frame rate or dimensions")
        pts = [Fraction(row["best_effort_timestamp_time"]) for row in frames]
        frame_period = Fraction(1, fps)
        if any(abs(stamp - index * frame_period) > Fraction(1, 10000)
               for index, stamp in enumerate(pts)):
            raise ValueError("frame PTS sequence")
        expected_duration = Fraction(frame_count, fps)
        durations = [Fraction(stream["duration"]), Fraction(record["format"]["duration"])]
        if any(abs(duration - expected_duration) > Fraction(1, 10000) for duration in durations):
            raise ValueError("stream or container duration")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"PROPAINTER_VIDEO_TIME_OR_DOMAIN:{path}:{error}") from error
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"PROPAINTER_VIDEO_DECODE_OPEN:{path}")
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.shape != (720, 960, 3):
                raise RuntimeError(f"PROPAINTER_VIDEO_DECODE_DOMAIN:{path}:{decoded}")
            decoded += 1
            if decoded > frame_count:
                raise RuntimeError(f"PROPAINTER_VIDEO_EXTRA_FRAME:{path}")
    finally:
        capture.release()
    if decoded != frame_count:
        raise RuntimeError(f"PROPAINTER_VIDEO_DECODE_COUNT:{path}:{decoded}")
    return {"artifact": artifact_ref(path), "decoded_frames": decoded,
            "fps": fps, "duration_seconds": float(expected_duration),
            "dimensions_wh": [960, 720], "pts_mode": "CFR_INDEX_ALIGNED"}


def clean(spec: dict[str, Any], output: Path) -> None:
    prepared = load_json(output / "PREPARE.json")
    if prepared["session_id"] != spec["session_id"] or len(prepared["rows"]) != spec["frame_count"]:
        raise RuntimeError("PREPARE_STAGE_NOT_SEALED")
    entry = checked(spec["propainter_entry"])
    for name, ref in spec["propainter_weights"].items():
        weight = checked(ref)
        expected = (REPO_ROOT / "vendor/ProPainter/weights" / name).resolve(strict=True)
        if weight.resolve(strict=True) != expected:
            raise RuntimeError(f"PROPAINTER_WEIGHT_PATH_DRIFT:{name}")
    command = propainter_command(spec, output, entry)
    with (output / "propainter.log").open("x") as log:
        result = subprocess.run(command, cwd=REPO_ROOT / "vendor/ProPainter",
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"PROPAINTER_RETURN_CODE:{result.returncode}")
    model_root = output / "propainter_model/frames/frames"
    if sorted(path.name for path in model_root.glob("*.png")) != [
        f"{index:04d}.png" for index in range(spec["frame_count"])
    ]:
        raise RuntimeError("PROPAINTER_FRAME_SET")
    video_root = model_root.parent
    internal_videos = {name: verify_internal_mp4(
        video_root / name, spec["frame_count"], spec["source_fps"])
        for name in ("masked_in.mp4", "inpaint_out.mp4")}
    rows = []
    for row in prepared["rows"]:
        index = row["decode_display_index"]
        source = cv2.imread(str(checked(row["model_input"])))
        removal_image = cv2.imread(str(checked(row["masks"]["removal"])), cv2.IMREAD_GRAYSCALE)
        protected_image = cv2.imread(str(checked(row["masks"]["protected"])), cv2.IMREAD_GRAYSCALE)
        unknown_image = cv2.imread(str(checked(row["masks"]["unknown"])), cv2.IMREAD_GRAYSCALE)
        generated = cv2.imread(str(model_root / f"{index:04d}.png"))
        if (source is None or generated is None or source.shape != (960, 1280, 3)
                or any(value is None or value.shape != (960, 1280)
                       for value in (removal_image, protected_image, unknown_image))):
            raise RuntimeError(f"CLEAN_FRAME_DOMAIN:{index}")
        removal, protected, unknown = (value > 0 for value in
                                       (removal_image, protected_image, unknown_image))
        generated = cv2.resize(generated, (1280, 960), interpolation=cv2.INTER_CUBIC)
        visual = source.copy()
        visual[removal] = generated[removal]
        if not np.array_equal(visual[~removal], source[~removal]) or not np.array_equal(visual[protected], source[protected]):
            raise RuntimeError("CLEAN_PROTECTED_OR_OUTSIDE_CHANGED")
        provenance = np.zeros(source.shape[:2], np.uint8)
        provenance[removal] = 3
        provenance[protected] = 2
        provenance[unknown] = 4
        rows.append({"decode_display_index": index,
                     "propainter_input_index": row["propainter_input_index"],
                     "propainter_input": row["propainter_input"],
                     "propainter_mask": row["propainter_mask"],
                     "model_output_index": index,
                     "model_output": artifact_ref(model_root / f"{index:04d}.png"),
                     "clean": save(output / "clean" / f"{index:06d}.png", visual),
                     "pixel_source": save(output / "pixel_source" / f"{index:06d}.png", provenance),
                     "model_input": row["model_input"], "masks": row["masks"],
                     "role_status": row["role_status"], "synthetic_pixels": int(removal.sum())})
    atomic_json(output / "CLEAN.json", {
        "schema_version": "exact78-fresh-frontend-clean-v2",
        "session_id": spec["session_id"], "prepare_result": artifact_ref(output / "PREPARE.json"),
        "rows": rows,
        "internal_videos": internal_videos,
        "source_fps": spec["source_fps"], "time_mode": load_json(output / "RAW.json")["time_mode"],
        "source_enum": {"0": "TARGET_RAW", "2": "PROTECTED_VISIBLE_RAW",
                        "3": "SYNTHETIC_PROPAINTER", "4": "UNKNOWN_RAW_RETAINED"},
        "fresh_mask_and_clean_inference": True, "real_donor_pixels": 0,
        "use_scope": "OFFLINE_BIDIRECTIONAL_VISUAL_ONLY",
        "depth_object6d_contact_truth": False, "training_eligible": False,
        "human_review": "PENDING", "quality": "NOT_YET_ADOPTED",
    })


def authorize_frontend(task_id: str) -> tuple[dict[str, Any], Path]:
    """Verify the active Task Packet and fenced writer without legacy runner imports."""
    routes = [row for row in load_json(REPO_ROOT / "tasks/current/INDEX.json")["task_packets"]
              if row["task_id"] == task_id and row.get("execution_allowed") is True]
    if len(routes) != 1:
        raise RuntimeError("FRONTEND_NOT_ROUTABLE")
    route = routes[0]
    packet_path = REPO_ROOT / route["packet_path"]
    if artifact_ref(packet_path)["sha256"] != route["packet_sha256"]:
        raise RuntimeError("FRONTEND_PACKET_SHA_DRIFT")
    packet = load_json(packet_path)
    if packet.get("task_id") != task_id:
        raise RuntimeError("FRONTEND_PACKET_IDENTITY_DRIFT")
    writer = packet["writer"]
    if writer.get("scope") != task_id:
        raise RuntimeError("FRONTEND_WRITER_SCOPE_DRIFT")
    identity = process_identity(writer["pid"])
    if not identity["alive"] or identity["start_ticks"] != writer["proc_start_ticks"]:
        raise RuntimeError("FRONTEND_WRITER_FENCE_DRIFT")
    task_root = (REPO_ROOT / packet["write_set"][0]).resolve()
    if task_root.parent != REPO_ROOT / "_run/current" / task_id / "attempts":
        raise RuntimeError("FRONTEND_WRITE_SET_DRIFT")
    if load_json(task_root / "WRITER.json") != writer:
        raise RuntimeError("FRONTEND_WRITER_TOKEN_DRIFT")
    return packet, task_root


def seal_stage(spec: dict[str, Any], output: Path, stage: str, spec_ref: dict[str, Any]) -> None:
    """Hash-check a complete stage before its downstream phase is allowed."""
    names = {"map": "MAP", "raw": "RAW", "masks": "MASK", "prepare": "PREPARE", "clean": "CLEAN"}
    name = names[stage]
    previous = {"raw": "MAP", "masks": "RAW", "prepare": "MASK", "clean": "PREPARE"}.get(stage)
    previous_ref = "ABSENT"
    if previous is not None:
        previous_path = output / f"SEAL_{previous}.json"
        if not previous_path.is_file():
            raise RuntimeError(f"UPSTREAM_STAGE_NOT_SEALED:{previous}")
        previous_ref = artifact_ref(previous_path)
    result_path = output / f"{name}.json"
    seal_path = output / f"SEAL_{name}.json"
    if seal_path.exists() or not result_path.is_file():
        raise RuntimeError(f"STAGE_SEAL_TARGET_INVALID:{name}")
    result = load_json(result_path)
    if result.get("session_id") != spec["session_id"]:
        raise RuntimeError(f"STAGE_SESSION_DRIFT:{name}")
    rows = result.get("rows")
    if stage != "map" and (
        not isinstance(rows, list) or len(rows) != spec["frame_count"]
        or [row.get("decode_display_index") for row in rows] != list(range(spec["frame_count"]))
    ):
        raise RuntimeError(f"STAGE_FRAME_MAPPING_DRIFT:{name}")

    def verify(value: Any) -> None:
        if isinstance(value, dict):
            if {"path", "bytes", "sha256"} <= value.keys():
                checked(value)
            else:
                for child in value.values():
                    verify(child)
        elif isinstance(value, list):
            for child in value:
                verify(child)

    verify(result)
    atomic_json(seal_path, {
        "schema_version": "exact78-fresh-frontend-stage-seal-v2",
        "session_id": spec["session_id"], "stage": name,
        "frame_count": spec["frame_count"] if stage != "map" else 0,
        "result": artifact_ref(result_path), "upstream_seal": previous_ref,
        "frontend_spec": spec_ref, "quality_adopted": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=("map", "raw", "masks", "prepare", "clean", "all"), required=True)
    args = parser.parse_args()
    spec = load_json(args.spec)
    validate_spec(spec)
    packet, task_root = authorize_frontend(spec["task_id"])
    if packet.get("inputs", {}).get("frontend_spec") != artifact_ref(args.spec):
        raise RuntimeError("FRONTEND_SPEC_NOT_PACKET_BOUND")
    output = Path(spec["output_root"]).resolve()
    output.relative_to(task_root)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    phases = {"map": map_stage, "raw": raw, "masks": masks, "prepare": prepare, "clean": clean}
    sequence = tuple(phases) if args.phase == "all" else (args.phase,)
    for phase in sequence:
        if phase != "map":
            upstream = {"raw": "MAP", "masks": "RAW", "prepare": "MASK", "clean": "PREPARE"}[phase]
            if not (output / f"SEAL_{upstream}.json").is_file():
                raise RuntimeError(f"UPSTREAM_STAGE_NOT_SEALED:{upstream}")
        phases[phase](spec, output)
        seal_stage(spec, output, phase, artifact_ref(args.spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
