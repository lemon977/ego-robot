"""CPU-only raw VST image evidence for explicitly visible seed annotation.

Only frozen video/QPC/camera/source-manifest references are consumed. No raw
pose streams or HDF5 data are loaded. Native display-order decoding stops after
the greatest requested index; this stage does not claim a new full-video audit.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np

from . import audit, source
from .camera import normalize_camera

CONSUMED_ROLES = ("raw_vst", "video_qpc", "camera_params", "camera_meta", "source_manifest")
SOURCE_FILENAMES = {
    "raw_vst": "raw/vst.h264", "video_qpc": "raw/vst.qpc.ts.jsonl",
    "camera_params": "raw/camera_params.json", "camera_meta": "raw/camera_params.meta.json",
    "source_manifest": "manifest.json",
}
BASE_SIZE = (1280, 960)
IMAGE_SIZE = (640, 480)


class SeedEvidenceError(ValueError):
    pass


def _indices_for_sessions(sids, frame_indices):
    if frame_indices is None:
        selected = {sid: [0] for sid in sids}
    elif isinstance(frame_indices, dict):
        if set(frame_indices) != set(sids):
            raise SeedEvidenceError("EXPLICIT_ANCHOR_SESSION_SET_MISMATCH")
        selected = {sid: frame_indices[sid] for sid in sids}
    elif isinstance(frame_indices, (list, tuple)) or type(frame_indices) is int:
        selected = {sid: frame_indices for sid in sids}
    else:
        raise SeedEvidenceError("FIXED_NATIVE_FRAME_SELECTION_REQUIRED")
    result = {}
    for sid, values in selected.items():
        values = [values] if type(values) is int else values
        if (not isinstance(values, (list, tuple)) or not values
                or any(type(value) is not int or value < 0 for value in values)
                or len(set(values)) != len(values)):
            raise SeedEvidenceError("UNIQUE_NONNEGATIVE_NATIVE_FRAME_INDICES_REQUIRED")
        result[sid] = sorted(values)
    return result


def _project_ref(path, repo):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(repo):
        raise SeedEvidenceError("FROZEN_MANIFEST_OR_CAMERA_CONFIG_OUTSIDE_PROJECT")
    return source.ref(resolved)


def run_seed_evidence(repo, out, manifest_path, config_path, sids, frame_indices=None):
    """Freeze lossless right-SBS1 raw images without reading poses or HDF5.

    frame_indices may be a shared integer/list or an exact per-session mapping.
    Defaults to each native frame zero. K_override is interpreted at 1280x960 by
    the existing normalizer, then its first two rows scale by 1/2 for 640x480.
    """
    started = time.monotonic()
    repo = Path(repo).resolve(strict=True)
    if repo != audit.PROJECT_ROOT.resolve():
        raise SeedEvidenceError("SEED_EVIDENCE_REPOSITORY_IDENTITY_MISMATCH")
    if (not isinstance(sids, (list, tuple)) or not sids
            or any(not isinstance(sid, str) or not sid or Path(sid).name != sid
                   or sid in (".", "..") or "\\" in sid for sid in sids)
            or len(set(sids)) != len(sids)):
        raise SeedEvidenceError("UNIQUE_EXPLICIT_RAW_SESSION_IDS_REQUIRED")
    manifest_ref, config_ref = _project_ref(manifest_path, repo), _project_ref(config_path, repo)
    manifest, config = source.read_json(Path(manifest_ref["path"])), \
                       source.read_json(Path(config_ref["path"]))
    rows = {}
    for row in manifest["sessions"]:
        if row["session_id"] in rows:
            raise SeedEvidenceError("DUPLICATE_FROZEN_MANIFEST_SESSION_ID")
        rows[row["session_id"]] = row
    if not set(sids) <= set(rows):
        raise SeedEvidenceError("SEED_SESSION_MISSING_FROM_FROZEN_MANIFEST")
    selected_indices = _indices_for_sessions(sids, frame_indices)
    output = audit._output_path(out)
    if output == repo or Path(out).is_symlink():
        raise SeedEvidenceError("SEED_OUTPUT_MUST_BE_ORDINARY_PROJECT_SUBDIRECTORY")
    output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    session_records, all_images = [], []
    for sid in sids:
        row = rows[sid]
        if row.get("input_status") not in ("READY_RAW_AUDIT", "READY_RAW_WITH_INVALID_SHORT_GAP"):
            raise SeedEvidenceError(f"RAW_PIXEL_INPUT_NOT_ADMITTED:{sid}")
        count = row.get("raw_frame_count")
        if type(count) is not int or count <= 0 or selected_indices[sid][-1] >= count:
            raise SeedEvidenceError(f"SEED_FRAME_OUTSIDE_FROZEN_NATIVE_COUNT:{sid}")
        source_dir = (Path(manifest["source_root"]) / sid).resolve(strict=True)
        verified_refs = {}
        for role in CONSUMED_ROLES:
            reference = row["source_refs"][role]
            expected_path = (source_dir / SOURCE_FILENAMES[role]).resolve(strict=True)
            if Path(reference["path"]).resolve(strict=True) != expected_path:
                raise SeedEvidenceError(f"RAW_SOURCE_REFERENCE_SESSION_ROUTE_MISMATCH:{sid}:{role}")
            verified_refs[role] = audit._verify_exact_ref(reference)
        if row["camera_snapshot_sha256"] != verified_refs["camera_params"]["sha256"]:
            raise SeedEvidenceError(f"CAMERA_SNAPSHOT_REFERENCE_MISMATCH:{sid}")
        serial = row["device_serial"]
        selected = config.get("devices", {}).get(serial)
        if (not isinstance(selected, dict) or selected.get("eye") != "right"
                or selected.get("sbs_index") != 1):
            raise SeedEvidenceError(f"EXPLICIT_RIGHT_SBS_INDEX_ONE_REQUIRED:{serial}")
        if tuple(selected.get("output_size", BASE_SIZE)) != BASE_SIZE:
            raise SeedEvidenceError("K_OVERRIDE_MUST_REMAIN_IN_1280X960_BASE_DOMAIN")
        if selected.get("correction_evidence"):
            audit._verify_exact_ref(selected["correction_evidence"], project_only=True)
        camera = normalize_camera(
            source.read_json(Path(verified_refs["camera_params"]["path"])),
            source.read_json(Path(verified_refs["camera_meta"]["path"])),
            device_serial=serial, camera_sha256=row["camera_snapshot_sha256"],
            config=selected, output_size=BASE_SIZE)
        image_k = np.diag([.5, .5, 1.]) @ camera.K
        evidence_camera = replace(camera, width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], K=image_k)
        qpc = source.timeline(audit.read_integer_jsonl(
            verified_refs["video_qpc"]["path"]), "seed_evidence_native_video")
        if len(qpc) != count:
            raise SeedEvidenceError(f"SEED_NATIVE_QPC_COUNT_MISMATCH:{sid}")
        wanted = set(selected_indices[sid])
        maximum = max(wanted)
        images, decoded_prefix = [], 0
        decoder = source.crop_frames(source_dir, evidence_camera)
        try:
            for frame, bgr in enumerate(decoder):
                decoded_prefix = frame + 1
                if bgr.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3) or bgr.dtype != np.uint8:
                    raise SeedEvidenceError("RAW_SEED_IMAGE_PIXEL_DOMAIN_MISMATCH")
                if frame in wanted:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    image_path = audit._output_path(
                        output / "images" / sid / f"raw_{frame:06d}.png")
                    if not image_path.is_relative_to(output):
                        raise SeedEvidenceError("SEED_IMAGE_ESCAPES_CURRENT_OUTPUT_ROOT")
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    ok, encoded = cv2.imencode(".png", bgr)
                    if not ok:
                        raise SeedEvidenceError("SEED_IMAGE_PNG_ENCODING_FAILED")
                    with image_path.open("xb") as handle:
                        handle.write(encoded.tobytes())
                    image_record = {
                        "session_id": sid, "device_serial": serial, "raw_frame_index": frame,
                        "qpc_ns": int(qpc[frame]), "relative_qpc_ns": int(qpc[frame] - qpc[0]),
                        "image": source.ref(image_path), "image_size": list(IMAGE_SIZE),
                        "rgb_shape": list(rgb.shape), "rgb_dtype": "uint8", "rgb_order": "RGB",
                        "rgb_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
                        "camera_snapshot_sha256": row["camera_snapshot_sha256"],
                        "camera_config": config_ref, "verified_source_refs": verified_refs,
                        "K_base_used": camera.K.tolist(), "K_image_used": image_k.tolist(),
                        "T_head_camera_used": camera.T_head_camera.tolist(),
                        "T_head_camera_factory": camera.T_head_camera_factory.tolist(),
                        "eye": camera.eye, "sbs_index": camera.sbs_index,
                        "upright_axes": camera.calibration["upright_axes"],
                        "eye_identity_qualification": camera.calibration["encoded_eye_identity_qualification"],
                        "physical_eye_verified": camera.calibration["physical_eye_identity_verified"],
                        "frame_semantics": "NATIVE_DECODE_DISPLAY_ORDER_TO_RECORDED_VIDEO_QPC_ROW",
                        "timestamp_semantics": "SOFTWARE_HOST_ARRIVAL_NOT_EXPOSURE_SYNCHRONIZATION",
                        "pixels_overlaid": False, "pixels_rotated": False, "lens_remap": False,
                    }
                    images.append(image_record)
                if frame == maximum:
                    break
        finally:
            decoder.close()
        if decoded_prefix != maximum + 1 or len(images) != len(wanted):
            raise SeedEvidenceError(f"RAW_VIDEO_ENDED_BEFORE_REQUESTED_SEED_FRAME:{sid}")
        session_records.append({
            "session_id": sid, "device_serial": serial, "raw_frame_count": count,
            "requested_raw_frame_indices": selected_indices[sid],
            "decoded_native_prefix_frames": decoded_prefix,
            "full_video_decode_performed": decoded_prefix == count,
            "verified_consumed_source_refs": verified_refs,
            "source_refs_from_frozen_manifest": copy.deepcopy(row["source_refs"]),
            "consumed_source_roles": list(CONSUMED_ROLES),
            "camera_config": config_ref, "camera": camera.calibration,
            "camera_snapshot_sha256": row["camera_snapshot_sha256"],
            "images": images,
        })
        all_images.extend(images)
    evidence = {
        "schema": "pick_place_visible_seed_source_images_v1", "status": "PASSED",
        "input_manifest": manifest_ref, "camera_config": config_ref,
        "source_root": manifest["source_root"], "session_ids": list(sids),
        "sessions": session_records, "images": all_images,
        "image_count": len(all_images), "image_size": list(IMAGE_SIZE),
        "native_decoder_threads": 2, "opencv_threads": 1,
        "image_operation": "ORIGINAL_ENCODED_RIGHT_SBS_INDEX1_CROP_UNIFORM_RESIZE_ONLY",
        "K_resize_policy": "USE_1280X960_CONFIG_K_THEN_SCALE_FIRST_TWO_ROWS_BY_ONE_HALF",
        "full_video_decode_performed": all(session["full_video_decode_performed"] for session in session_records), "new_full_source_audit_claimed": False, "raw_pose_loaded": False,
        "hdf5_data_consumed": False, "GPU_used": False, "fitting_performed": False,
        "qpc_modified": False, "lens_remap": False, "no_overlays": True,
        "training_eligible": False, "control_ground_truth": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    audit._write_json(output / "SOURCE_SEED_IMAGES.json", evidence)
    result = {
        "schema": "pick_place_visible_seed_images_result_v1", "status": "PASSED",
        "source_seed_images": source.ref(output / "SOURCE_SEED_IMAGES.json"),
        "input_manifest": manifest_ref, "camera_config": config_ref,
        "session_ids": list(sids), "sessions_requested": len(sids), "sessions_complete": len(sids),
        "image_count": len(all_images), "native_prefix_frames_decoded": sum(
            session["decoded_native_prefix_frames"] for session in session_records),
        "full_video_decode_performed": all(session["full_video_decode_performed"] for session in session_records), "new_full_source_audit_claimed": False, "raw_pose_loaded": False,
        "hdf5_data_consumed": False, "GPU_used": False, "fitting_performed": False,
        "training_eligible": False, "control_ground_truth": False,
    }
    audit._write_json(output / "RESULT.json", result)
    return result
