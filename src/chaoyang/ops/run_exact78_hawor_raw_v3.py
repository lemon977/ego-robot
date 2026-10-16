#!/usr/bin/env python3
"""Pure-RGB Exact78 HaWoR candidate; execution requires a separate Task Packet.

Only the frozen clip RGB or raw 0915 VST is decoded.  T0 is an identity/time
contract, not a source of hand motion.  This operation does not run the HaWoR
infiller and never manufactures world-space poses or missing predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


PROJECT = Path(__file__).resolve().parents[3]
HAWOR = PROJECT / "vendor/HaWoR"
TARGET_SIZE = (1280, 960)
SIDES = ("left", "right")
MANO_NAMES = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


class InputRejected(RuntimeError):
    """A cold-start input or frozen identity failed closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise InputRejected(f"not a file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise InputRejected(f"JSON object required: {path}")
    return value


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def require_child(path: Path, directory: Path) -> Path:
    child = path.resolve(strict=True)
    if not child.is_relative_to(directory.resolve(strict=True)):
        raise InputRejected(f"source escaped session: {path}")
    return child


def checked_frozen_ref(path: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    actual = ref(path)
    if (actual["path"] != frozen.get("path")
            or actual["bytes"] != frozen.get("bytes")
            or actual["sha256"] != frozen.get("sha256")):
        raise InputRejected(f"T0 source SHA drift: {path}")
    return actual


def tracker_inputs(session: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    files = frozen["files"]
    manifest_path = require_child(session / "clip_manifest.json", session)
    manifest_ref = checked_frozen_ref(manifest_path, files["clip_manifest"])
    manifest = load_object(manifest_path)
    # Whitelist only RGB, camera and SLAM members. Never open tracking/HumanEgo.
    allowed = ("video", "camera_params", "slam")
    paths = {key: require_child(session / manifest["files"][key], session)
             for key in allowed}
    refs = {key: checked_frozen_ref(paths[key], files["camera" if key == "camera_params" else key])
            for key in allowed}
    video = manifest["video"]
    rect = manifest["video_rectification"]
    if (rect.get("eye") != "left" or rect.get("rectified") is not True
            or [int(x) for x in rect.get("resolution", [])] != [1280, 960]
            or [int(video.get("width", 0)), int(video.get("height", 0))] != [1280, 960]):
        raise InputRejected("tracker left RGB calibration/resolution drift")
    fps = float(video["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise InputRejected("invalid native clip fps")
    intrinsic = rect["intrinsics"]
    if (len(intrinsic) != 3 or any(len(row) != 3 for row in intrinsic)
            or float(intrinsic[0][0]) <= 0 or float(intrinsic[1][1]) <= 0):
        raise InputRejected("invalid clip intrinsics")
    return {
        "source_video": paths["video"], "allowed_refs": {"clip_manifest": manifest_ref, **refs},
        "intrinsics": intrinsic, "fps_nominal": fps,
        "frame_count": int(video["frame_count"]), "crop_xywh": None,
        "time_mode": "CLIP_FRAME_INDEX_ONLY", "physical_eye": "tracker_session_left_RGB",
        "calibration_claim": "CLIP_RECTIFIED_K_ONLY_WORLD_POSE_NOT_SOLVED",
    }


def vst0915_inputs(session: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    raw = require_child(session / "raw", session)
    files = frozen["files"]
    names = ("vst.h264", "vst.qpc.ts.jsonl", "vst.ts.jsonl",
             "camera_params.json", "camera_params.meta.json")
    paths = {name: require_child(raw / name, raw) for name in names}
    refs = {name: checked_frozen_ref(paths[name], files[name]) for name in names}
    camera = load_object(paths["camera_params.json"])
    left = camera["left"]
    width, height = int(camera["width"]), int(camera["height"])
    if (left.get("sourceIndex") != 1 or camera["right"].get("sourceIndex") != 0
            or int(camera["outputWidth"]) != 2 * width
            or int(camera["outputHeight"]) != height):
        raise InputRejected("0915 physical-eye/SBS identity drift")
    qpc = paths["vst.qpc.ts.jsonl"].read_text(encoding="utf-8").splitlines()
    timestamp = paths["vst.ts.jsonl"].read_text(encoding="utf-8").splitlines()
    if (not qpc or len(qpc) != len(timestamp)
            or any(not row.isdecimal() for row in qpc + timestamp)):
        raise InputRejected("0915 raw timestamp sidecar mismatch")
    sx, sy = TARGET_SIZE[0] / width, TARGET_SIZE[1] / height
    intrinsic = [[float(left["fx"]) * sx, 0.0, float(left["cx"]) * sx],
                 [0.0, float(left["fy"]) * sy, float(left["cy"]) * sy],
                 [0.0, 0.0, 1.0]]
    return {
        "source_video": paths["vst.h264"], "allowed_refs": refs,
        "intrinsics": intrinsic, "fps_nominal": float(camera["fps"]),
        "frame_count": len(qpc), "crop_xywh": [width, 0, width, height],
        "source_dimensions": [2 * width, height],
        "time_mode": "TIMEBASE_UNVERIFIED", "physical_eye": "left_sourceIndex_1",
        "calibration_claim": "SCALED_FACTORY_FISHEYE_K_MODEL_INPUT_ONLY_NOT_METRIC",
    }


def build_spec(*, session: Path, session_id: str, source_kind: str,
               input_freeze: dict[str, Any], frame_time_map: dict[str, Any]) -> dict[str, Any]:
    session = session.resolve(strict=True)
    if session.name == "raw" and source_kind == "vst0915":
        session = session.parent
    frozen = input_freeze.get("selected", input_freeze.get("sessions", {})).get(session_id)
    mapping = frame_time_map.get("sessions", {}).get(session_id)
    if not isinstance(frozen, dict) or not isinstance(mapping, dict):
        raise InputRejected("session is absent from frozen T0 input/time map")
    if frozen.get("source_path") != str(session):
        raise InputRejected("T0 session path differs from requested session")
    if source_kind == "tracker":
        if "chips_cards_tracker_" not in str(session):
            raise InputRejected("tracker source identity mismatch")
        spec = tracker_inputs(session, frozen)
    elif source_kind == "vst0915":
        if "chips_cards_hands_0915" not in str(session):
            raise InputRejected("0915 source identity mismatch")
        spec = vst0915_inputs(session, frozen)
    else:
        raise InputRejected("unknown source kind")
    rows = mapping.get("frame_rows")
    if (not isinstance(rows, list) or len(rows) != spec["frame_count"]
            or mapping.get("decoded_frames") != spec["frame_count"]):
        raise InputRejected("T0 raw/decode frame-count disagreement")
    if any(row.get("decoded_display_index") != index for index, row in enumerate(rows)):
        raise InputRejected("T0 decode display order is not contiguous")
    if any(row.get("sbs_physical_eye") != spec["physical_eye"]
           or row.get("crop_xywh") != spec["crop_xywh"] for row in rows):
        raise InputRejected("T0 physical-eye/crop disagreement")
    if source_kind == "vst0915" and (
        mapping.get("time_mode") != "TIMEBASE_UNVERIFIED"
        or mapping.get("capture_to_decode_alignment") != "INDEX_ALIGNMENT_UNVERIFIED"
    ):
        raise InputRejected("0915 T0 diagnostic timebase has changed")
    return {**spec, "session_id": session_id, "session": session,
            "frame_rows_t0": rows, "source_kind": source_kind,
            "input_role": frozen.get("role"), "split_claim": frozen.get("split_claim")}


def frame_status(row: dict[str, Any], index: int, spec: dict[str, Any],
                 valid_left: bool, valid_right: bool) -> dict[str, Any]:
    preserved = {key: row.get(key) for key in (
        "capture_frame_id", "capture_timestamp_raw", "pts", "best_effort_timestamp",
        "qpc_raw_tick", "pict_type", "capture_to_decode_alignment")}
    hands = {}
    for side, valid in zip(SIDES, (valid_left, valid_right)):
        hands[side] = {
            "roi_source": "HAWOR_DETECTION_OR_TRACKED_RGB" if valid else "NONE",
            "motion_source": "HAWOR_MODEL_PREDICTION" if valid else "MISSING",
            "visibility": "UNKNOWN", "wrist_position_valid": valid,
            "wrist_orientation_valid": valid, "local_hand_shape_valid": valid,
        }
    return {
        **preserved, "decoded_display_index": index, "model_input_index": index,
        "output_index": index, "sbs_physical_eye": spec["physical_eye"],
        "crop_xywh": spec["crop_xywh"], "time_mode": spec["time_mode"],
        "hands": hands,
    }


def model_asset_refs() -> dict[str, dict[str, Any]]:
    """Freeze every directly loaded upstream weight and MANO asset."""
    return {
        "hawor_checkpoint": ref(HAWOR / "weights/hawor/checkpoints/hawor.ckpt"),
        "hawor_model_config": ref(HAWOR / "weights/hawor/model_config.yaml"),
        "hand_detector": ref(HAWOR / "weights/external/detector.pt"),
        "mano_left": ref(HAWOR / "_DATA/data_left/mano_left/MANO_LEFT.pkl"),
        "mano_right": ref(HAWOR / "_DATA/data/mano/MANO_RIGHT.pkl"),
        "hawor_environment_manifest": ref(
            PROJECT / "_run/current/environments/hawor-py310-v1/"
            ".chaoyang-hardlink-snapshot/manifest.json.gz"),
    }


def extract_model_frames(spec: dict[str, Any], image_dir: Path) -> tuple[list[dict[str, Any]], str]:
    import cv2  # imported only when the governed model operation runs

    image_dir.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(spec["source_video"]))
    if not cap.isOpened():
        raise InputRejected("source RGB video cannot be opened")
    rows = []
    aggregate = hashlib.sha256()
    try:
        for index in range(spec["frame_count"]):
            ok, image = cap.read()
            if not ok:
                raise InputRejected(f"RGB decode ended early at {index}")
            if spec["source_kind"] == "vst0915":
                width, height = spec["source_dimensions"]
                if [image.shape[1], image.shape[0]] != [width, height]:
                    raise InputRejected(f"0915 raw SBS frame dimensions drift at {index}")
                x, y, w, h = spec["crop_xywh"]
                image = image[y:y + h, x:x + w]
                image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            elif [image.shape[1], image.shape[0]] != list(TARGET_SIZE):
                raise InputRejected(f"tracker RGB dimensions drift at {index}")
            path = image_dir / f"{index:06d}.jpg"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise InputRejected(f"cannot stage model RGB frame {index}")
            entry = ref(path)
            rows.append({"decoded_display_index": index, "model_input_index": index,
                         "input_jpeg": entry})
            aggregate.update(bytes.fromhex(entry["sha256"]))
        ok, _ = cap.read()
        if ok:
            raise InputRejected("RGB decoder has frames beyond T0 frozen count")
    finally:
        cap.release()
    return rows, aggregate.hexdigest()


def run_hawor(spec: dict[str, Any], image_dir: Path) -> dict[str, Any]:
    """Use upstream detector/model only, never hawor_infiller or old all_data."""
    import cv2
    import numpy as np
    import torch

    sys.path.insert(0, str(HAWOR))
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import rotation_matrix_to_angle_axis

    class Args:
        pass

    args = Args()
    # Vendor uses this as a sequence namespace. JPEGs were decoded/cropped
    # from the audited source once; the virtual path is never opened as video.
    args.video_path = str(image_dir.parent.parent / "model_input.mp4")
    args.input_type = "file"
    args.img_focal = (float(spec["intrinsics"][0][0]) +
                      float(spec["intrinsics"][1][1])) / 2.0
    args.checkpoint = str(HAWOR / "weights/hawor/checkpoints/hawor.ckpt")
    if not Path(args.checkpoint).is_file():
        raise InputRejected("frozen HaWoR checkpoint absent")
    previous_cwd = Path.cwd()
    try:
        os.chdir(HAWOR)
        start, end, seq_folder_s, image_files = detect_track_video(args)
        if len(image_files) != spec["frame_count"]:
            raise InputRejected("HaWoR detector input-frame count drift")
        chunks, _ = hawor_motion_estimation(args, start, end, seq_folder_s)
    finally:
        os.chdir(previous_cwd)
    seq_folder = Path(seq_folder_s)
    count = spec["frame_count"]
    joints = np.full((2, count, 21, 3), np.nan, np.float32)
    uv = np.full((2, count, 21, 2), np.nan, np.float32)
    betas = np.full((2, count, 10), np.nan, np.float32)
    root = np.full((2, count, 3, 3), np.nan, np.float32)
    pose = np.full((2, count, 15, 3, 3), np.nan, np.float32)
    valid = np.zeros((2, count), bool)
    k = np.asarray(spec["intrinsics"], np.float32)
    if k.shape != (3, 3):
        raise InputRejected("model input intrinsics shape drift")
    previous_cwd = Path.cwd()
    try:
        os.chdir(HAWOR)
        for side, mano in ((0, run_mano_left), (1, run_mano)):
            for chunk in chunks.get(side, []):
                frames = np.asarray(chunk, dtype=np.int64)
                if (frames.size == 0 or np.any(frames < 0) or np.any(frames >= count)
                        or np.any(valid[side, frames])):
                    raise InputRejected("HaWoR chunk frame/index overlap")
                payload = load_object(seq_folder / "cam_space" / str(side) /
                                      f"{frames[0]}_{frames[-1]}.json")
                rr = np.asarray(payload["init_root_orient"], np.float32)[0]
                hr = np.asarray(payload["init_hand_pose"], np.float32)[0]
                tr = np.asarray(payload["init_trans"], np.float32)[0]
                be = np.asarray(payload["init_betas"], np.float32)[0]
                if (rr.shape != (len(frames), 3, 3)
                        or hr.shape != (len(frames), 15, 3, 3)
                        or tr.shape != (len(frames), 3)
                        or be.shape != (len(frames), 10)):
                    raise InputRejected("HaWoR camera prediction tensor shape drift")
                with torch.inference_mode():
                    result = mano(
                        torch.from_numpy(tr[None]),
                        rotation_matrix_to_angle_axis(torch.from_numpy(rr[None])),
                        rotation_matrix_to_angle_axis(torch.from_numpy(hr[None])),
                        betas=torch.from_numpy(be[None]), use_cuda=False,
                    )["joints"][0].detach().cpu().numpy().astype(np.float32)
                if result.shape != (len(frames), 21, 3) or not np.isfinite(result).all():
                    raise InputRejected("invalid HaWoR/MANO camera joints")
                joints[side, frames], root[side, frames] = result, rr
                pose[side, frames], betas[side, frames] = hr, be
                q = result @ k.T
                uv[side, frames] = q[:, :, :2] / q[:, :, 2:3]
                valid[side, frames] = True
    finally:
        os.chdir(previous_cwd)
    return {"joints_3d_camera": joints, "joints_2d": uv,
            "root_orient_camera": root, "hand_pose_rotmat": pose,
            "betas": betas, "prediction_valid": valid,
            "intrinsics": np.repeat(k[None], count, axis=0),
            "original_frame_indices": np.arange(count, dtype=np.int32),
            "fps_nominal": np.asarray(spec["fps_nominal"], np.float32),
            "mano_joint_names": np.asarray(MANO_NAMES),
            "anatomical_side_names": np.asarray(SIDES)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-kind", choices=("tracker", "vst0915"), required=True)
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--frame-time-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    if not output.is_relative_to(PROJECT / "_run") or output.exists():
        raise InputRejected("fresh project-local _run output required")
    frozen_ref = ref(args.input_freeze)
    time_ref = ref(args.frame_time_map)
    spec = build_spec(session=args.session_root, session_id=args.session_id,
                      source_kind=args.source_kind,
                      input_freeze=load_object(args.input_freeze),
                      frame_time_map=load_object(args.frame_time_map))
    assets = model_asset_refs()
    started = time.monotonic()
    output.mkdir(parents=True)
    image_dir = output / "model_input" / "extracted_images"
    jpeg_rows, jpeg_digest = extract_model_frames(spec, image_dir)
    arrays = run_hawor(spec, image_dir)
    import numpy as np

    prediction_valid = arrays["prediction_valid"]
    npz = output / "HAWOR_CAMERA_PREDICTIONS.npz"
    np.savez_compressed(npz, **arrays)
    frame_rows = [frame_status(row, index, spec,
                               bool(prediction_valid[0, index]),
                               bool(prediction_valid[1, index]))
                  for index, row in enumerate(spec["frame_rows_t0"])]
    frame_path = output / "FRAME_PROVENANCE.json"
    write_new_json(frame_path, {
        "schema_version": "exact78-hawor-frame-provenance-v3",
        "session_id": spec["session_id"], "frame_count": spec["frame_count"],
        "fps_nominal": spec["fps_nominal"], "fps_is_timestamp_truth": False,
        "model_input_width": TARGET_SIZE[0], "model_input_height": TARGET_SIZE[1],
        "intrinsics_image_size": list(TARGET_SIZE),
        "physical_eye": spec["physical_eye"], "time_mode": spec["time_mode"],
        "frame_rows": frame_rows,
    })
    audit_path = output / "INPUT_AUDIT.json"
    write_new_json(audit_path, {
        "schema_version": "exact78-hawor-input-audit-v3",
        "session_id": spec["session_id"], "source_kind": spec["source_kind"],
        "source_role": spec["input_role"], "source_split_claim": spec["split_claim"],
        "raw_source_ref": spec["allowed_refs"]["video" if spec["source_kind"] == "tracker" else "vst.h264"],
        "read_refs": {**spec["allowed_refs"], "t0_input_freeze": frozen_ref,
                      "t0_frame_time_map": time_ref},
        "model_input_jpeg_refs": jpeg_rows,
        "model_input_jpeg_sha256_aggregate": jpeg_digest,
        "runtime_code": ref(Path(__file__)),
        "model_assets": assets,
        "vendor_model_code": {
            "detect_track_video": ref(HAWOR / "scripts/scripts_test_video/detect_track_video.py"),
            "hawor_video": ref(HAWOR / "scripts/scripts_test_video/hawor_video.py"),
            "pipeline_tools": ref(HAWOR / "lib/pipeline/tools.py"),
            "mano_process": ref(HAWOR / "hawor/utils/process.py"),
        },
        "calibration_claim": spec["calibration_claim"],
        "same_session_slam_usage": "SHA_ONLY_CAMERA_WORLD_ALIGNMENT_NOT_ESTABLISHED"
                                   if spec["source_kind"] == "tracker" else "NOT_APPLICABLE",
        "forbidden_reads": "NO_HARDWARE_HAND_NO_ALL_DATA_NO_OLD_HAWOR_NO_MASK_CLEAN_ROBOT",
        "infiller_invoked": False, "world_pose_emitted": False,
    })
    result_path = output / "RESULT.json"
    write_new_json(result_path, {
        "schema_version": "exact78-hawor-pure-rgb-result-v3",
        "status": "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS",
        "session_id": spec["session_id"], "source_kind": spec["source_kind"],
        "source_frames_decoded": spec["frame_count"],
        "model_input_frames": len(jpeg_rows),
        "model_prediction_frames": {side: int(prediction_valid[index].sum())
                                    for index, side in enumerate(SIDES)},
        "model_infill_frames": 0, "deterministic_fill_frames": 0,
        "npz": ref(npz), "frame_provenance": ref(frame_path),
        "input_audit": ref(audit_path),
        "time_mode": spec["time_mode"], "world_pose_valid": False,
        "control_ground_truth": False, "training_eligible": False,
        "visual_qualified": False, "product_qualified": False,
        "wall_seconds": round(time.monotonic() - started, 3),
        "claim_limit": "RGB-derived HaWoR camera predictions only, not direct 3D observations or synchronized world motion.",
    })
    print(json.dumps({"status": "MODEL_PREDICTIONS_CAMERA_ONLY_NO_QUALITY_PASS",
                      "result": ref(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
