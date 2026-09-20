#!/usr/bin/env python3
"""Build the Chips023 Exact78 stereo-preflight input without decoding video.

The source-session metadata contains raw per-eye intrinsics and head-to-camera
extrinsics, but that is not equivalent to projection matrices for an encoded,
resize-only stereo domain.  This builder therefore publishes those values as
candidate evidence only.  It creates an executable preflight input only when a
separate encoded-domain calibration authority is present and byte-bound to the
source camera parameters and SBS video.  Otherwise it emits a fail-closed
blocker receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from chaoyang.pipeline.exact78_stereo_preflight_v31 import check_metric_conversion


SESSION_ID = "get_potato_chips_0902_023"
INPUT_SCHEMA = "EXACT78_STEREO_PREFLIGHT_INPUT_V31"
AUTHORITY_SCHEMA = "EXACT78_ENCODED_STEREO_CALIBRATION_AUTHORITY_V31"
PROVENANCE_SCHEMA = "EXACT78_STEREO_PREFLIGHT_INPUT_PROVENANCE_V31"
RESULT_SCHEMA = "EXACT78_STEREO_PREFLIGHT_INPUT_BUILD_RESULT_V31"
FRAME_SAMPLE_COUNT = 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def path_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=False)
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    return {"exists": True, **artifact(resolved)}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def is_archive_path(path: Path) -> bool:
    return "archive" in path.resolve(strict=False).parts


def deterministic_frame_indices(frame_count: int) -> list[int]:
    """Return twelve deterministic, endpoint-inclusive frame indices."""

    if frame_count < FRAME_SAMPLE_COUNT:
        raise ValueError(f"at least {FRAME_SAMPLE_COUNT} frames required")
    last = frame_count - 1
    frames = [index * last // (FRAME_SAMPLE_COUNT - 1) for index in range(FRAME_SAMPLE_COUNT)]
    if len(set(frames)) != FRAME_SAMPLE_COUNT:
        raise AssertionError("deterministic frame selector produced duplicates")
    return frames


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must be homogeneous")
    return matrix


def _raw_intrinsics(camera: Mapping[str, Any], side: str) -> np.ndarray:
    values = camera[side]["intrinsics"]
    matrix = np.asarray(
        [
            [values["fx"], 0.0, values["cx"]],
            [0.0, values["fy"], values["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0.0:
        raise ValueError(f"invalid {side} raw intrinsics")
    return matrix


def _baseline_candidate(camera: Mapping[str, Any]) -> float:
    # camera_params declares both matrices as head_to_camera.  Camera centres
    # are therefore computed in the common head frame via inverse(T_camera_head).
    left = _matrix4(camera["extrinsics"]["left"], "extrinsics.left")
    right = _matrix4(camera["extrinsics"]["right"], "extrinsics.right")
    left_center_head = np.linalg.inv(left)[:3, 3]
    right_center_head = np.linalg.inv(right)[:3, 3]
    return float(np.linalg.norm(right_center_head - left_center_head))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _source_paths(session_root: Path) -> dict[str, Path]:
    manifest_path = session_root / "clip_manifest.json"
    camera_path = session_root / "camera_params.json"
    if not manifest_path.is_file():
        return {
            "manifest": manifest_path,
            "camera_params": camera_path,
            "source_stereo": session_root / "source_stereo" / "MISSING.mp4",
        }
    manifest = _load_json(manifest_path)
    relative_video = str(
        manifest.get("files", {}).get("source_stereo_video", "source_stereo/MISSING.mp4")
    )
    return {
        "manifest": manifest_path,
        "camera_params": camera_path,
        "source_stereo": session_root / relative_video,
    }


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def inspect_current_assets(session_root: Path) -> tuple[dict[str, Any], list[str]]:
    root = session_root.resolve(strict=False)
    paths = _source_paths(root)
    blockers: list[str] = []
    if root.name != SESSION_ID:
        blockers.append("SESSION_ID_MISMATCH")
    if is_archive_path(root):
        blockers.append("ARCHIVE_SOURCE_FORBIDDEN")
    for name, path in paths.items():
        if is_archive_path(path):
            blockers.append(f"ARCHIVE_{name.upper()}_FORBIDDEN")
        if not _inside(path, root):
            blockers.append(f"{name.upper()}_ESCAPES_SESSION_ROOT")
        if not path.is_file():
            blockers.append(f"MISSING_{name.upper()}")

    refs = {name: path_ref(path) for name, path in paths.items()}
    provenance: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA,
        "session_id": SESSION_ID,
        "session_root": str(root),
        "source_assets": refs,
        "non_archive_sources_only": not any("ARCHIVE" in item for item in blockers),
        "source_video_decoded": False,
        "lens_transform_applied": False,
    }
    if blockers:
        return provenance, sorted(set(blockers))

    manifest = _load_json(paths["manifest"])
    camera = _load_json(paths["camera_params"])
    if manifest.get("clip_name") != SESSION_ID:
        blockers.append("CLIP_MANIFEST_SESSION_MISMATCH")
    source_video = manifest.get("source_stereo_video", {})
    frame_count = int(source_video.get("frame_count", 0))
    sbs_width = int(source_video.get("width", 0))
    sbs_height = int(source_video.get("height", 0))
    reference = camera.get("reference_resolution", {})
    eye_width = int(reference.get("w", 0))
    eye_height = int(reference.get("h", 0))
    if sbs_width != 2 * eye_width or sbs_height != eye_height:
        blockers.append("SBS_AND_CAMERA_REFERENCE_GEOMETRY_MISMATCH")
    try:
        frame_indices = deterministic_frame_indices(frame_count)
        left_k = _raw_intrinsics(camera, "left")
        right_k = _raw_intrinsics(camera, "right")
        baseline = _baseline_candidate(camera)
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        blockers.append(f"INVALID_CAMERA_OR_FRAME_METADATA:{type(error).__name__}")
        frame_indices, left_k, right_k, baseline = [], None, None, None

    notes = [str(item) for item in manifest.get("notes", [])]
    provenance.update(
        {
            "manifest_geometry": {
                "sbs_width": sbs_width,
                "sbs_height": sbs_height,
                "frame_count": frame_count,
                "fps": source_video.get("fps"),
                "physical_eye_width": eye_width,
                "physical_eye_height": eye_height,
            },
            "deterministic_frame_indices": frame_indices,
            "raw_camera_candidate": {
                "pixel_domain": f"RAW_CAMERA_PARAMS_{eye_width}x{eye_height}",
                "left_k": left_k.tolist() if left_k is not None else None,
                "right_k": right_k.tolist() if right_k is not None else None,
                "baseline_candidate_m": baseline,
                "baseline_derivation": (
                    "distance between left/right camera centres obtained by "
                    "inverting declared head_to_camera extrinsics"
                ),
                "encoded_projection_matrices_present": False,
                "authorized_for_metric_conversion": False,
            },
            "metadata_declarations": {
                "camera_params_video_info_is_distorted": camera.get("video_info", {}).get(
                    "is_distorted"
                ),
                "clip_notes_label_source_stereo_distorted": any(
                    "distorted stereo source" in note.lower() for note in notes
                ),
                "interpretation": (
                    "Metadata declarations are recorded as provenance, not accepted as "
                    "proof of the actual encoded pixel domain."
                ),
            },
        }
    )
    return provenance, sorted(set(blockers))


def _validate_authority(
    authority_path: Path,
    provenance: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    blockers: list[str] = []
    if is_archive_path(authority_path):
        return None, ["ARCHIVE_ENCODED_CALIBRATION_AUTHORITY_FORBIDDEN"], path_ref(authority_path)
    if not authority_path.is_file():
        return None, ["MISSING_ENCODED_DOMAIN_CALIBRATION_AUTHORITY"], path_ref(authority_path)
    authority_ref = artifact(authority_path)
    try:
        authority = _load_json(authority_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, ["INVALID_ENCODED_DOMAIN_CALIBRATION_AUTHORITY_JSON"], authority_ref
    if authority.get("schema_version") != AUTHORITY_SCHEMA:
        blockers.append("ENCODED_CALIBRATION_AUTHORITY_SCHEMA_MISMATCH")
    if authority.get("session_id") != SESSION_ID:
        blockers.append("ENCODED_CALIBRATION_AUTHORITY_SESSION_MISMATCH")

    sources = provenance.get("source_assets", {})
    bindings = authority.get("source_bindings", {})
    for key in ("camera_params", "source_stereo"):
        expected = sources.get(key, {}).get("sha256")
        actual = bindings.get(f"{key}_sha256")
        if not isinstance(expected, str) or actual != expected:
            blockers.append(f"ENCODED_CALIBRATION_{key.upper()}_SHA_MISMATCH")

    image = authority.get("image_domain")
    metric = authority.get("metric_conversion")
    if not isinstance(image, dict) or not isinstance(metric, dict):
        blockers.append("ENCODED_CALIBRATION_DOMAIN_OBJECTS_MISSING")
        return None, sorted(set(blockers)), authority_ref

    geometry = provenance.get("manifest_geometry", {})
    required_image = {
        "eye_width": geometry.get("physical_eye_width"),
        "eye_height": geometry.get("physical_eye_height"),
        "physical_left_source_index": 1,
        "physical_right_source_index": 0,
        "decoded_vst_already_undistorted": True,
        "lens_undistortion_applied": False,
        "horizontal_reflection_for_disparity_sign": True,
        "outputs_unflipped_to_physical_left": True,
    }
    for key, expected in required_image.items():
        if image.get(key) != expected:
            blockers.append(f"ENCODED_CALIBRATION_IMAGE_DOMAIN_{key.upper()}_INVALID")

    try:
        model_width = int(image["model_width"])
        model_height = int(image["model_height"])
        expected_domain = f"PHYSICAL_LEFT_{model_width}x{model_height}"
        metric_result = check_metric_conversion(
            physical_left_k=np.asarray(metric["physical_left_k"], np.float64),
            physical_right_k=np.asarray(metric["physical_right_k"], np.float64),
            model_left_k=np.asarray(metric["model_left_k"], np.float64),
            model_right_k=np.asarray(metric["model_right_k"], np.float64),
            projection_left=np.asarray(metric["projection_left"], np.float64),
            projection_right=np.asarray(metric["projection_right"], np.float64),
            baseline_m=float(metric["baseline_m"]),
            width=model_width,
            height=model_height,
            reflected_for_model=True,
            outputs_unflipped_to_physical_left=True,
            calibration_pixel_domain=str(metric["calibration_pixel_domain"]),
            expected_pixel_domain=expected_domain,
        )
    except (KeyError, TypeError, ValueError):
        blockers.append("ENCODED_CALIBRATION_METRIC_FIELDS_INVALID")
    else:
        if not metric_result["passed"]:
            blockers.append("ENCODED_CALIBRATION_METRIC_CONVERSION_NOT_CLOSED")
    return authority if not blockers else None, sorted(set(blockers)), authority_ref


def build(
    session_root: Path,
    output_root: Path,
    *,
    encoded_calibration: Path | None,
) -> dict[str, Any]:
    output = output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh/no-clobber output root required: {output}")
    output.mkdir(parents=True)

    provenance, blockers = inspect_current_assets(session_root)
    authority: dict[str, Any] | None = None
    authority_ref: dict[str, Any] | None = None
    if not blockers:
        if encoded_calibration is None:
            blockers.append("MISSING_ENCODED_DOMAIN_K_P_BASELINE_AUTHORITY")
        else:
            authority, authority_blockers, authority_ref = _validate_authority(
                encoded_calibration.resolve(strict=False), provenance
            )
            blockers.extend(authority_blockers)
    elif encoded_calibration is not None:
        authority_ref = path_ref(encoded_calibration)

    provenance = {
        **provenance,
        "encoded_calibration_authority": authority_ref,
        "archive_consumed": False,
    }
    atomic_json(output / "PROVENANCE.json", provenance)

    config_ref: dict[str, Any] | None = None
    if not blockers and authority is not None:
        config = {
            "schema_version": INPUT_SCHEMA,
            "session_id": SESSION_ID,
            "frame_indices": provenance["deterministic_frame_indices"],
            "image_domain": dict(authority["image_domain"]),
            "metric_conversion": {
                key: value
                for key, value in authority["metric_conversion"].items()
                if key not in {"model_left_k", "model_right_k"}
            },
            "provenance": {
                "source_assets": provenance["source_assets"],
                "encoded_calibration_authority": authority_ref,
                "frame_selection": "ENDPOINT_INCLUSIVE_INTEGER_LINSPACE_12",
                "source_video_decoded_by_builder": False,
                "lens_transform_applied_by_builder": False,
            },
        }
        config_path = output / "EXACT78_STEREO_PREFLIGHT_INPUT_V31.json"
        atomic_json(config_path, config)
        config_ref = artifact(config_path)

    blockers = sorted(set(blockers))
    status = "READY_CPU_STEREO_PREFLIGHT" if not blockers else "BLOCKED_EXTERNAL_ASSET"
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "session_id": SESSION_ID,
        "execution_allowed": not blockers,
        "gpu_required": False,
        "model_inference_performed": False,
        "source_video_decoded": False,
        "lens_transform_applied": False,
        "blockers": blockers,
        "outputs": {
            "config": config_ref,
            "provenance": artifact(output / "PROVENANCE.json"),
        },
        "claim_limit": (
            "READY authorizes only the separate CPU correspondence preflight. It does "
            "not establish horizontal epipolar geometry, metric depth, or external accuracy."
        ),
    }
    if blockers:
        blocker_path = output / "BLOCKER.json"
        atomic_json(
            blocker_path,
            {
                "schema_version": "EXACT78_STEREO_PREFLIGHT_INPUT_BLOCKER_V31",
                "status": status,
                "session_id": SESSION_ID,
                "first_blocker": blockers[0],
                "blockers": blockers,
                "metric_conversion_authorized": False,
                "projection_matrices_synthesized": False,
            },
        )
        result["outputs"]["blocker"] = artifact(blocker_path)
    atomic_json(output / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--encoded-calibration", type=Path)
    args = parser.parse_args()
    result = build(
        args.session_root,
        args.output_root,
        encoded_calibration=args.encoded_calibration,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["execution_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
