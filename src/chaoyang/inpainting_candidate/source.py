"""Read-only, task-independent original VST/controller/MANUS source preflight.

No processed video, aligned HDF5 datasets, historical seed or fitted session
configuration is read. Camera metadata assertions remain distinct from verified
encoded geometry. QPC sidecars alone do not prove exposure time or clock sync.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from .contracts import canonical_sha, file_ref

RAW_FILES = {
    "manifest": "manifest.json", "video": "raw/vst.h264",
    "video_qpc": "raw/vst.qpc.ts.jsonl", "video_wall": "raw/vst.ts.jsonl",
    "controller_head": "raw/pico.jsonl", "manus": "raw/manus.jsonl",
    "manus_meta": "raw/manus.meta.json", "camera": "raw/camera_params.json",
    "camera_meta": "raw/camera_params.meta.json", "installation_container": "dataset.hdf5",
}
SOURCE_REQUIRED = frozenset({"manifest", "video", "video_qpc", "video_wall"})


class SourceError(ValueError):
    pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise SourceError(f"RAW_JSON_INVALID:{path.name}") from exc


def _rows(path: Path) -> list[Any]:
    result = []
    with path.open(encoding="utf-8-sig") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                try:
                    result.append(json.loads(text))
                except ValueError as exc:
                    raise SourceError(f"RAW_JSONL_INVALID:{path.name}:{line}") from exc
    if not result:
        raise SourceError(f"RAW_STREAM_EMPTY:{path.name}")
    return result


def _integer_timeline(values: list[Any], label: str) -> list[int]:
    if not values or any(type(value) is not int for value in values):
        raise SourceError(f"RAW_INTEGER_TIMELINE_REQUIRED:{label}")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise SourceError(f"RAW_DUPLICATE_OR_BACKWARD_TIME:{label}")
    return values


def probe_vst(path: str | Path) -> dict[str, Any]:
    """Count actual displayed pictures read-only; do not re-encode or set -r."""
    try:
        completed = subprocess.run([
            "ffprobe", "-v", "error", "-threads", "2", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_read_frames,codec_name", "-of", "json", str(path),
        ], check=True, capture_output=True, text=True, timeout=180)
        streams = json.loads(completed.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError("one stream required")
        row = streams[0]
        width, height, count = (int(row[key]) for key in ("width", "height", "nb_read_frames"))
        if width <= 0 or height <= 0 or count <= 0 or width % 2:
            raise ValueError("invalid dimensions or frame count")
    except (OSError, subprocess.SubprocessError, KeyError, ValueError, TypeError) as exc:
        raise SourceError("RAW_VIDEO_PROBE_FAILED") from exc
    return {"width": width, "height": height, "decoded_frames": count,
            "codec": row.get("codec_name"), "frame_order": "DECODER_DISPLAY_ORDER",
            "source_video": file_ref(path)}


def camera_route(params: dict, meta: dict, camera_reference: dict,
                 video_probe: dict | None = None) -> dict:
    """Same physical-right selection rule for all sources, only when explicit.

    Selection is not a claim that factory fisheye intrinsics describe encoded
    upright VST. No inference by task, folder date, device name or image content.
    """
    if meta.get("sha256") != camera_reference["sha256"]:
        raise SourceError("RAW_CAMERA_SNAPSHOT_SHA_MISMATCH")
    eyes = {eye: params.get(eye, {}) for eye in ("left", "right")}
    routes = {eye: value.get("sourceIndex") for eye, value in eyes.items()}
    route_valid = all(type(value) is int and value in (0, 1) for value in routes.values()) and len(set(routes.values())) == 2
    selected_eye = "right" if route_valid else None
    if "width" in params:
        reference_size = [params.get("width"), params.get("height")]
    else:
        resolution = params.get("reference_resolution", {})
        reference_size = [resolution.get("w"), resolution.get("h")]
    encoded_size = [video_probe["width"] // 2, video_probe["height"]] if video_probe else None
    domain_match = encoded_size == reference_size if video_probe else None
    intrinsic = eyes["right"].get("intrinsics", eyes["right"])
    intrinsic_present = all(isinstance(intrinsic.get(key), (int, float)) for key in ("fx", "fy", "cx", "cy"))
    return {"status": "SOURCE_INDEX_DECLARATION" if route_valid else "BLOCKED_PHYSICAL_EYE_UNRESOLVED",
            "physical_eye": selected_eye, "sbs_index": routes.get(selected_eye) if selected_eye else None,
            "eye_routes": routes, "camera_device_serial": meta.get("device_serial"),
            "factory_reference_size": reference_size, "actual_encoded_eye_size": encoded_size,
            "encoded_dimension_match": domain_match, "intrinsics_present": intrinsic_present,
            "calibration_domain_status": "UNVERIFIED_FACTORY_TO_ENCODED_DOMAIN" if domain_match is not False else "ENCODED_CALIBRATION_DOMAIN_MISMATCH",
            "upright_axes": "UNKNOWN", "lens_remap": False,
            "projection_qualified": False,
            "next_action": "Bind encoded-domain axes/calibration evidence; do not use guessed remap or historical session fits."}


def read_installation_attribute(path: str | Path) -> dict:
    """Only the legal installation attribute; HDF5 motion datasets never read."""
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("controller_to_wrist_calibration")
        if value is None:
            return {"status": "ABSENT", "hdf5_datasets_read": False}
        if isinstance(value, bytes):
            value = value.decode("utf-8-sig")
        prior = json.loads(value)
        for side in ("left", "right"):
            pose = prior[side]
            if len(pose["pos"]) != 3 or len(pose["quat"]) != 4:
                raise ValueError("installation pose shape")
            import math
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose["pos"] + pose["quat"]):
                raise ValueError("installation nonfinite")
            if sum(v * v for v in pose["quat"]) <= 0:
                raise ValueError("installation zero quaternion")
    except ImportError:
        return {"status": "BLOCKED_HDF5_READER_UNAVAILABLE", "hdf5_datasets_read": False}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SourceError("RAW_INSTALLATION_ATTRIBUTE_INVALID") from exc
    return {"status": "PRESENT_INSTALLATION_PRIOR", "attribute": "controller_to_wrist_calibration",
            "value": prior, "sha256": canonical_sha(prior), "hdf5_datasets_read": False,
            "authority": "INSTALLATION_PRIOR_NOT_MEASURED_ANATOMICAL_WRIST"}


def inspect_raw_session(session: str | Path, *, decode_probe: bool = False) -> dict:
    """Automatically bind original files and diagnose capabilities, not quality.

    The G0 default is metadata/bytes only. A later finite packet can request the
    CPU decoder probe. This does not call motion/segmentation/inpainting models.
    """
    root = Path(session).resolve(strict=True)
    if not root.is_dir():
        raise SourceError("RAW_SESSION_NOT_DIRECTORY")
    missing = [relative for relative in RAW_FILES.values() if not (root / relative).is_file()]
    required_missing = [RAW_FILES[name] for name in SOURCE_REQUIRED if not (root / RAW_FILES[name]).is_file()]
    refs = {name: file_ref(root / relative) for name, relative in RAW_FILES.items()
            if (root / relative).is_file()}
    if any(not Path(reference["path"]).is_relative_to(root) for reference in refs.values()):
        raise SourceError("RAW_FILE_ESCAPES_SESSION_ROOT")
    if required_missing:
        return {"schema": "inpainting-candidate-raw-source-v1", "status": "BLOCKED_INPUT_INCOMPLETE",
                "session_dir": str(root), "missing": missing, "required_missing": required_missing,
                "actual_inputs": refs,
                "control_ground_truth": False}
    manifest = _read_json(root / RAW_FILES["manifest"])
    identity = manifest.get("session")
    if not isinstance(identity, str) or not identity:
        raise SourceError("RAW_SESSION_IDENTITY_MISSING")
    # The original metadata owns identity; no "pick_place" prefix or ID patches.
    qpc = _integer_timeline(_rows(root / RAW_FILES["video_qpc"]), "VST_QPC")
    wall = _rows(root / RAW_FILES["video_wall"])
    if len(wall) != len(qpc) or any(type(value) is not int for value in wall):
        raise SourceError("RAW_VIDEO_WALL_QPC_COUNT_OR_TYPE_MISMATCH")
    probe = probe_vst(root / RAW_FILES["video"]) if decode_probe else None
    if probe and probe["decoded_frames"] != len(qpc):
        raise SourceError("RAW_VIDEO_DECODE_QPC_FRAME_COUNT_MISMATCH")
    if "camera" in refs and "camera_meta" in refs:
        params = _read_json(root / RAW_FILES["camera"])
        meta = _read_json(root / RAW_FILES["camera_meta"])
        route = camera_route(params, meta, refs["camera"], probe)
    else:
        route = {"status": "BLOCKED_CAMERA_METADATA_ABSENT", "physical_eye": None,
                 "sbs_index": None, "projection_qualified": False,
                 "calibration_domain_status": "ABSENT", "upright_axes": "UNKNOWN"}
    if "manus_meta" in refs:
        manus_meta = _read_json(root / RAW_FILES["manus_meta"])
        if manus_meta.get("session", identity) != identity:
            raise SourceError("RAW_MANUS_META_SESSION_IDENTITY_MISMATCH")
    installation = (read_installation_attribute(root / RAW_FILES["installation_container"])
                    if "installation_container" in refs else {"status": "ABSENT", "hdf5_datasets_read": False})
    pico_rows = _rows(root / RAW_FILES["controller_head"]) if "controller_head" in refs else []
    manus_rows = _rows(root / RAW_FILES["manus"]) if "manus" in refs else []
    if any(not isinstance(row, dict) for row in pico_rows + manus_rows):
        raise SourceError("RAW_MOTION_RECORD_MUST_BE_OBJECT")
    if any(not isinstance(row.get("data", {}), dict) or not isinstance(row.get("data", {}).get("Controller", {}), dict)
           for row in pico_rows):
        raise SourceError("RAW_CONTROLLER_HEAD_SCHEMA_UNSUPPORTED")
    motion_presence = {
        "controller_head": {"records": len(pico_rows),
                            "head_records": sum(isinstance(row.get("data", {}).get("Head"), dict) for row in pico_rows),
                            "controller_records": {side: sum(isinstance(row.get("data", {}).get("Controller", {}).get(side), dict)
                                                              for row in pico_rows) for side in ("left", "right")},
                            "pico_hand_bones_consumed": False, "retarget_arrays_consumed": False,
                            "status": "PRESENT_NOT_YET_MOTION_VALIDATED" if pico_rows else "BLOCKED_CONTROLLER_HEAD_ABSENT"},
        "manus": {"records": len(manus_rows),
                  "side_records": {side: sum(row.get("side") == side for row in manus_rows) for side in ("left", "right")},
                  "status": "PRESENT_NOT_YET_MOTION_VALIDATED" if manus_rows else "BLOCKED_MANUS_ABSENT"},
    }
    frame_map = [{"native_frame_index": index, "output_frame": None, "decoded_display_index": index if probe else None,
                  "acquisition_frame_id": "UNKNOWN", "sbs_source_frame": index if probe else None,
                  "physical_eye": route["physical_eye"], "sbs_index": route["sbs_index"],
                  "qpc_declared_ns": value, "wall_ns": wall[index],
                  "model_input_frame": None, "output_pts": None}
                 for index, value in enumerate(qpc)]
    return {"schema": "inpainting-candidate-raw-source-v1", "status": "SOURCE_PREFLIGHT_CANDIDATE",
            "session_id": identity, "session_dir": str(root), "actual_inputs": refs,
            "source_signature": canonical_sha({"present": refs, "absent": missing}), "native_sidecar_frames": len(qpc),
            "decoded_frames": probe["decoded_frames"] if probe else None,
            "frame_count_status": "DECODE_COUNT_MATCHED_MAPPING_UNVERIFIED" if probe else "NOT_DECODED",
            "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS",
            "qpc_frequency": "ABSENT", "qpc_sampling_position": "UNKNOWN",
            "cross_clock_sync_verified": False, "camera": route, "installation": installation,
            "motion": motion_presence, "frame_mapping": frame_map,
            "missing": missing,
            "capabilities": {"raw_rgb_and_time": "SOURCE_AVAILABLE_TIMEBASE_UNVERIFIED",
                             "physical_eye_crop": "SOURCE_ROUTE_DECLARED" if route.get("physical_eye") else "BLOCKED_EYE_ROUTE_UNRESOLVED",
                             "mask_from_rgb": "CANDIDATE_RGB_AVAILABLE_GEOMETRY_NOT_REQUIRED",
                             "native_motion": "PRESENT_NOT_YET_VALIDATED" if pico_rows and manus_rows else "BLOCKED_NATIVE_MOTION_INPUT_ABSENT",
                             "installed_wrist_motion": "INSTALLATION_PRIOR_PRESENT_NOT_YET_VALIDATED" if installation["status"] == "PRESENT_INSTALLATION_PRIOR" else "BLOCKED_INSTALLATION_ABSENT",
                             "robot_projection": "UNVERIFIED" if route.get("physical_eye") else "BLOCKED_CALIBRATION_ABSENT"},
            "aligned_rows_read": False, "processed_video_read": False,
            "hdf5_datasets_read": False, "control_ground_truth": False,
            "training_eligible": False,
            "next_action": "Validate native motion and encoded route under the next finite candidate packet."}
