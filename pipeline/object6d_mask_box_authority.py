"""Validate Object6D inputs and project independent per-frame mask boxes.

This is a CPU-only admission adapter.  It consumes an acquisition manifest, an
instance of the existing functional Object6D drop-in schema, a task role map,
and one byte-pinned NPZ observation bundle.  It never estimates a missing pose,
uses hand/PICO points as an object substitute, propagates boxes, or runs SAM.

Only rows reported as ``KNOWN_OBJECT6D_DIRECT`` may be used as box prompts by a
later SAM refinement stage.  Every other row is explicitly ``UNKNOWN`` and is
not a prompt authority.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import jsonschema
import numpy as np


CONTRACT_SCHEMA = "object6d-mask-box-adapter-v1"
ROLE_SCHEMA = "object6d-mask-role-mapping-v1"
OUTPUT_SCHEMA = "object6d-independent-mask-box-authority-v1"
SHA256_LENGTH = 64
EXPECTED_BUNDLE_KEYS = frozenset(
    {
        "K",
        "T_world_camera",
        "frame_id",
        "camera_timestamp_ns",
        "timestamp_ns",
        "timestamp_s",
        "object_ids",
        "source_instance_ids",
        "source_frame_index",
        "geometry_sha256",
        "T_world_object",
        "T_object_to_camera",
        "valid",
        "visible",
        "confidence",
        "provenance",
        "direct_observation",
        "identity_valid",
        "identity_score",
        "reprojection_rmse_px",
    }
)


class Object6DBoxContractError(RuntimeError):
    """The Object6D box-authority contract is invalid or incomplete."""


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_dict(value: object, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise Object6DBoxContractError(f"{field} must be a JSON object")
    if not all(type(key) is str for key in value):
        raise Object6DBoxContractError(f"{field} keys must be strings")
    return value


def _require_list(value: object, *, field: str) -> list[Any]:
    if type(value) is not list:
        raise Object6DBoxContractError(f"{field} must be a JSON array")
    return value


def _require_exact_keys(
    value: dict[str, Any], *, required: set[str], field: str
) -> None:
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise Object6DBoxContractError(
            f"{field} key mismatch; missing={missing}, extra={extra}"
        )


def _require_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value == "replace_me":
        raise Object6DBoxContractError(f"{field} must be a non-placeholder string")
    return value


def _require_number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise Object6DBoxContractError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise Object6DBoxContractError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise Object6DBoxContractError(f"{field} is below {minimum}")
    if maximum is not None and result > maximum:
        raise Object6DBoxContractError(f"{field} exceeds {maximum}")
    return result


def _load_json_bytes(path: Path, *, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Object6DBoxContractError(f"cannot load {field}: {error}") from error
    return _require_dict(value, field=field), payload


def _resolve_project_file(project_root: Path, relative: object, *, field: str) -> Path:
    text = _require_string(relative, field=field)
    candidate = Path(text)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise Object6DBoxContractError(f"{field} must be a project-relative path")
    path = project_root.joinpath(candidate)
    try:
        resolved = path.resolve(strict=True)
        root = project_root.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise Object6DBoxContractError(f"{field} escapes or is absent") from error
    if path.is_symlink() or not resolved.is_file():
        raise Object6DBoxContractError(f"{field} must be a regular non-symlink file")
    return resolved


def _load_artifact(
    project_root: Path, value: object, *, field: str
) -> tuple[Path, bytes, str]:
    ref = _require_dict(value, field=field)
    _require_exact_keys(ref, required={"path", "bytes", "sha256"}, field=field)
    path = _resolve_project_file(project_root, ref["path"], field=f"{field}.path")
    expected_bytes = ref["bytes"]
    expected_sha = ref["sha256"]
    if type(expected_bytes) is not int or isinstance(expected_bytes, bool):
        raise Object6DBoxContractError(f"{field}.bytes must be an integer")
    if (
        type(expected_sha) is not str
        or len(expected_sha) != SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in expected_sha)
    ):
        raise Object6DBoxContractError(f"{field}.sha256 is invalid")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise Object6DBoxContractError(f"cannot read {field}") from error
    digest = _sha256(payload)
    if len(payload) != expected_bytes or digest != expected_sha:
        raise Object6DBoxContractError(f"{field} byte identity mismatch")
    return path, payload, digest


def _artifact_identity(value: object, *, field: str) -> tuple[str, int, str]:
    ref = _require_dict(value, field=field)
    _require_exact_keys(ref, required={"path", "bytes", "sha256"}, field=field)
    path = _require_string(ref["path"], field=f"{field}.path")
    size = ref["bytes"]
    digest = ref["sha256"]
    if type(size) is not int or isinstance(size, bool) or size <= 0:
        raise Object6DBoxContractError(f"{field}.bytes must be positive")
    if type(digest) is not str or len(digest) != SHA256_LENGTH:
        raise Object6DBoxContractError(f"{field}.sha256 is invalid")
    return path, size, digest


def _load_npz(payload: bytes) -> dict[str, np.ndarray]:
    import io

    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != EXPECTED_BUNDLE_KEYS:
                raise Object6DBoxContractError(
                    "observation bundle keys do not match the exact contract"
                )
            arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except Object6DBoxContractError:
        raise
    except (OSError, ValueError, EOFError) as error:
        raise Object6DBoxContractError(
            f"observation bundle is not a safe NPZ: {error}"
        ) from error
    if any(value.dtype.hasobject for value in arrays.values()):
        raise Object6DBoxContractError("object arrays are forbidden")
    return arrays


def _require_array(
    arrays: dict[str, np.ndarray],
    key: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    value = arrays[key]
    if value.shape != shape or value.dtype != dtype:
        raise Object6DBoxContractError(
            f"{key} must have shape={shape} dtype={dtype}, got {value.shape}/{value.dtype}"
        )
    return value


def _require_unicode_array(
    arrays: dict[str, np.ndarray], key: str, *, shape: tuple[int, ...]
) -> np.ndarray:
    value = arrays[key]
    if value.shape != shape or value.dtype.kind != "U":
        raise Object6DBoxContractError(f"{key} must be a Unicode array with shape={shape}")
    return value


def _require_se3(value: np.ndarray, *, field: str, atol: float) -> None:
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise Object6DBoxContractError(f"{field} must be finite SE(3)")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=atol):
        raise Object6DBoxContractError(f"{field} bottom row is invalid")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=atol):
        raise Object6DBoxContractError(f"{field} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=atol):
        raise Object6DBoxContractError(f"{field} rotation is not right handed")


def _corners(center: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
    signs = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3)))
    return center[None, :] + signs * dimensions[None, :]


def _project_box(
    corners_object: np.ndarray,
    transform_camera_object: np.ndarray,
    intrinsic: np.ndarray,
    *,
    min_depth_m: float,
) -> tuple[np.ndarray, float] | None:
    homogeneous = np.concatenate(
        [corners_object, np.ones((len(corners_object), 1), dtype=np.float64)], axis=1
    )
    camera = (transform_camera_object @ homogeneous.T).T[:, :3]
    if not np.isfinite(camera).all() or np.any(camera[:, 2] <= min_depth_m):
        return None
    pixels_h = (intrinsic @ camera.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    box = np.asarray(
        [
            np.min(pixels[:, 0]),
            np.min(pixels[:, 1]),
            np.max(pixels[:, 0]),
            np.max(pixels[:, 1]),
        ],
        dtype=np.float64,
    )
    depth = float(np.mean(camera[:, 2]))
    return box, depth


def _clip_box(box: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    raw_width = max(0.0, float(box[2] - box[0]))
    raw_height = max(0.0, float(box[3] - box[1]))
    raw_area = raw_width * raw_height
    clipped = np.asarray(
        [
            np.clip(box[0], 0.0, float(width - 1)),
            np.clip(box[1], 0.0, float(height - 1)),
            np.clip(box[2], 0.0, float(width - 1)),
            np.clip(box[3], 0.0, float(height - 1)),
        ]
    )
    clipped_area = max(0.0, float(clipped[2] - clipped[0])) * max(
        0.0, float(clipped[3] - clipped[1])
    )
    retained = clipped_area / raw_area if raw_area > 0.0 else 0.0
    return clipped, 1.0 - retained


def _load_and_validate_sources(
    contract_path: Path,
) -> tuple[
    dict[str, Any],
    Path,
    bytes,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, str],
]:
    contract, contract_payload = _load_json_bytes(contract_path, field="contract")
    required_contract_keys = {
        "schema_version",
        "project_root",
        "task_id",
        "session_id",
        "frame_count",
        "image_size",
        "source_contracts",
        "role_mapping",
        "observation_bundle",
        "functional_array_mapping",
        "gates",
        "required_frame_indices",
        "fallback_policy",
        "provider",
    }
    _require_exact_keys(contract, required=required_contract_keys, field="contract")
    if contract["schema_version"] != CONTRACT_SCHEMA:
        raise Object6DBoxContractError("unsupported contract schema_version")
    project_root_text = _require_string(contract["project_root"], field="project_root")
    project_root = Path(project_root_text)
    if not project_root.is_absolute() or not project_root.is_dir():
        raise Object6DBoxContractError("project_root must be an existing absolute directory")
    if contract["fallback_policy"] != "UNKNOWN_NO_STATIC_NO_PICO_NO_PROPAGATION":
        raise Object6DBoxContractError("fallback policy must fail closed")
    provider = _require_dict(contract["provider"], field="provider")
    _require_exact_keys(
        provider,
        required={
            "producer",
            "version",
            "provider_family",
            "code_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "license_sha256",
            "independent_from_sam3",
            "temporal_policy",
        },
        field="provider",
    )
    for key in (
        "producer",
        "version",
        "provider_family",
        "code_sha256",
        "config_sha256",
        "checkpoint_sha256",
        "license_sha256",
    ):
        _require_string(provider[key], field=f"provider.{key}")
    for key in ("code_sha256", "config_sha256", "checkpoint_sha256", "license_sha256"):
        if len(provider[key]) != SHA256_LENGTH or any(
            char not in "0123456789abcdef" for char in provider[key]
        ):
            raise Object6DBoxContractError(f"provider.{key} must be SHA256")
    if (
        provider["independent_from_sam3"] is not True
        or "sam" in provider["provider_family"].casefold()
    ):
        raise Object6DBoxContractError("Object6D provider must be independent from SAM")
    if provider["temporal_policy"] != "PER_FRAME_DIRECT_OBSERVATION_CURRENT_FRAME_ONLY":
        raise Object6DBoxContractError("Object6D provider temporal policy is not direct")

    sources = _require_dict(contract["source_contracts"], field="source_contracts")
    _require_exact_keys(
        sources,
        required={
            "functional_dropin_schema",
            "functional_dropin_manifest",
            "acquisition_manifest",
        },
        field="source_contracts",
    )
    schema_path, schema_payload, schema_digest = _load_artifact(
        project_root,
        sources["functional_dropin_schema"],
        field="functional_dropin_schema",
    )
    functional_path, functional_payload, functional_digest = _load_artifact(
        project_root,
        sources["functional_dropin_manifest"],
        field="functional_dropin_manifest",
    )
    acquisition_path, acquisition_payload, acquisition_digest = _load_artifact(
        project_root,
        sources["acquisition_manifest"],
        field="acquisition_manifest",
    )
    role_path, role_payload, role_digest = _load_artifact(
        project_root, contract["role_mapping"], field="role_mapping"
    )
    _, bundle_payload, bundle_digest = _load_artifact(
        project_root, contract["observation_bundle"], field="observation_bundle"
    )

    try:
        schema = _require_dict(json.loads(schema_payload), field="functional schema")
        functional = _require_dict(
            json.loads(functional_payload), field="functional manifest"
        )
        acquisition = _require_dict(
            json.loads(acquisition_payload), field="acquisition manifest"
        )
        role_mapping = _require_dict(json.loads(role_payload), field="role mapping")
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(functional)
    except (UnicodeDecodeError, json.JSONDecodeError, jsonschema.SchemaError) as error:
        raise Object6DBoxContractError(f"invalid functional schema: {error}") from error
    except jsonschema.ValidationError as error:
        raise Object6DBoxContractError(
            f"functional manifest does not satisfy its pinned schema: {error.message}"
        ) from error

    arrays = _load_npz(bundle_payload)
    pins = {
        "contract": _sha256(contract_payload),
        "functional_schema": schema_digest,
        "functional_manifest": functional_digest,
        "acquisition_manifest": acquisition_digest,
        "role_mapping": role_digest,
        "observation_bundle": bundle_digest,
        "functional_schema_path": str(schema_path),
        "functional_manifest_path": str(functional_path),
        "acquisition_manifest_path": str(acquisition_path),
        "role_mapping_path": str(role_path),
    }
    return (
        contract,
        project_root,
        contract_payload,
        functional,
        acquisition,
        role_mapping,
        arrays,
        pins,
    )


def build_box_authority(contract_path: Path) -> dict[str, Any]:
    """Validate a drop-in contract and return deterministic per-frame box rows."""

    (
        contract,
        project_root,
        _contract_payload,
        functional,
        acquisition,
        role_mapping,
        arrays,
        pins,
    ) = _load_and_validate_sources(contract_path)
    task_id = _require_string(contract["task_id"], field="task_id")
    session_id = _require_string(contract["session_id"], field="session_id")
    if task_id not in ("chips", "poker"):
        raise Object6DBoxContractError("task_id must be chips or poker")
    frame_count = contract["frame_count"]
    if type(frame_count) is not int or isinstance(frame_count, bool) or frame_count <= 0:
        raise Object6DBoxContractError("frame_count must be positive")
    if (
        functional.get("task_id") != task_id
        or functional.get("session_id") != session_id
        or functional.get("frame_count") != frame_count
    ):
        raise Object6DBoxContractError("functional manifest session identity mismatch")
    if (
        acquisition.get("task_id") != task_id
        or acquisition.get("session_id") != session_id
        or acquisition.get("metric_unit") != "meter"
    ):
        raise Object6DBoxContractError("acquisition manifest session/unit mismatch")
    convention = _require_dict(
        acquisition.get("coordinate_convention"), field="coordinate_convention"
    )
    if (
        convention.get("pose_field") != "T_object_to_camera"
        or convention.get("camera_axes") != "+X right, +Y down, +Z forward"
        or convention.get("matrix_convention")
        != "column vectors, p_camera = T_object_to_camera @ p_object"
    ):
        raise Object6DBoxContractError("acquisition camera convention is incompatible")
    functional_frames = _require_dict(
        functional.get("coordinate_frames"), field="functional coordinate_frames"
    )
    if (
        functional_frames.get("T_world_camera_convention")
        != "column_vector_left_multiply"
        or functional_frames.get("camera_axis_convention")
        != "+X right, +Y down, +Z forward"
    ):
        raise Object6DBoxContractError("functional camera convention is incompatible")

    if role_mapping.get("schema_version") != ROLE_SCHEMA:
        raise Object6DBoxContractError("role mapping schema_version mismatch")
    if role_mapping.get("task_id") != task_id:
        raise Object6DBoxContractError("role mapping task mismatch")
    if role_mapping.get("template_only") is not False:
        raise Object6DBoxContractError("role mapping template must be instantiated/frozen")
    instances = _require_list(role_mapping.get("instances"), field="role instances")
    if not instances:
        raise Object6DBoxContractError("role mapping must contain instances")
    object_count = len(instances)

    image_size = _require_dict(contract["image_size"], field="image_size")
    _require_exact_keys(image_size, required={"width", "height"}, field="image_size")
    width = image_size["width"]
    height = image_size["height"]
    if (
        type(width) is not int
        or isinstance(width, bool)
        or width < 2
        or type(height) is not int
        or isinstance(height, bool)
        or height < 2
    ):
        raise Object6DBoxContractError("image dimensions must be integers >=2")

    _require_array(arrays, "K", shape=(3, 3), dtype=np.dtype(np.float64))
    _require_array(
        arrays,
        "T_world_camera",
        shape=(frame_count, 4, 4),
        dtype=np.dtype(np.float64),
    )
    _require_array(
        arrays,
        "frame_id",
        shape=(frame_count,),
        dtype=np.dtype(np.int64),
    )
    _require_array(
        arrays,
        "camera_timestamp_ns",
        shape=(frame_count,),
        dtype=np.dtype(np.int64),
    )
    _require_array(
        arrays,
        "timestamp_ns",
        shape=(frame_count,),
        dtype=np.dtype(np.int64),
    )
    _require_array(
        arrays,
        "timestamp_s",
        shape=(frame_count,),
        dtype=np.dtype(np.float64),
    )
    _require_unicode_array(arrays, "object_ids", shape=(object_count,))
    _require_unicode_array(
        arrays, "source_instance_ids", shape=(frame_count, object_count)
    )
    _require_array(
        arrays,
        "source_frame_index",
        shape=(frame_count, object_count),
        dtype=np.dtype(np.int64),
    )
    _require_unicode_array(arrays, "geometry_sha256", shape=(object_count,))
    for key in ("T_world_object", "T_object_to_camera"):
        _require_array(
            arrays,
            key,
            shape=(frame_count, object_count, 4, 4),
            dtype=np.dtype(np.float64),
        )
    for key in ("valid", "visible", "direct_observation", "identity_valid"):
        _require_array(
            arrays,
            key,
            shape=(frame_count, object_count),
            dtype=np.dtype(np.bool_),
        )
    for key in (
        "confidence",
        "identity_score",
        "reprojection_rmse_px",
    ):
        _require_array(
            arrays,
            key,
            shape=(frame_count, object_count),
            dtype=np.dtype(np.float64),
        )
    _require_unicode_array(arrays, "provenance", shape=(frame_count, object_count))

    intrinsic = arrays["K"]
    if (
        not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
        or not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-12)
        or not 0.0 <= intrinsic[0, 2] < width
        or not 0.0 <= intrinsic[1, 2] < height
    ):
        raise Object6DBoxContractError("K is not a valid rectified pinhole intrinsic")

    gates = _require_dict(contract["gates"], field="gates")
    required_gate_keys = {
        "max_timestamp_delta_ns",
        "min_confidence",
        "min_identity_score",
        "max_reprojection_rmse_px",
        "max_edge_clip_fraction",
        "min_depth_m",
        "se3_atol",
        "pose_composition_atol",
        "max_center_speed_px_per_s",
        "adjacent_area_ratio_min",
        "adjacent_area_ratio_max",
    }
    _require_exact_keys(gates, required=required_gate_keys, field="gates")
    if (
        type(gates["max_timestamp_delta_ns"]) is not int
        or isinstance(gates["max_timestamp_delta_ns"], bool)
        or gates["max_timestamp_delta_ns"] < 0
    ):
        raise Object6DBoxContractError("max_timestamp_delta_ns must be an integer")
    max_timestamp_delta = gates["max_timestamp_delta_ns"]
    min_confidence = _require_number(
        gates["min_confidence"], field="min_confidence", minimum=0.0, maximum=1.0
    )
    min_identity_score = _require_number(
        gates["min_identity_score"],
        field="min_identity_score",
        minimum=0.0,
        maximum=1.0,
    )
    max_reprojection = _require_number(
        gates["max_reprojection_rmse_px"],
        field="max_reprojection_rmse_px",
        minimum=0.0,
    )
    max_edge_clip = _require_number(
        gates["max_edge_clip_fraction"],
        field="max_edge_clip_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    min_depth = _require_number(
        gates["min_depth_m"], field="min_depth_m", minimum=0.0
    )
    se3_atol = _require_number(gates["se3_atol"], field="se3_atol", minimum=0.0)
    composition_atol = _require_number(
        gates["pose_composition_atol"],
        field="pose_composition_atol",
        minimum=0.0,
    )
    max_speed = _require_number(
        gates["max_center_speed_px_per_s"],
        field="max_center_speed_px_per_s",
        minimum=0.0,
    )
    area_ratio_min = _require_number(
        gates["adjacent_area_ratio_min"],
        field="adjacent_area_ratio_min",
        minimum=0.0,
    )
    area_ratio_max = _require_number(
        gates["adjacent_area_ratio_max"],
        field="adjacent_area_ratio_max",
        minimum=area_ratio_min,
    )

    timestamp_ns = arrays["timestamp_ns"]
    camera_timestamp_ns = arrays["camera_timestamp_ns"]
    timestamp_s = arrays["timestamp_s"]
    if not np.array_equal(arrays["frame_id"], np.arange(frame_count, dtype=np.int64)):
        raise Object6DBoxContractError("frame_id must be exact zero-based media order")
    if np.any(timestamp_ns <= 0) or np.any(np.diff(timestamp_ns) <= 0):
        raise Object6DBoxContractError("pose timestamps must be positive and monotonic")
    if np.any(camera_timestamp_ns <= 0) or np.any(np.diff(camera_timestamp_ns) <= 0):
        raise Object6DBoxContractError("camera timestamps must be positive and monotonic")
    if np.max(np.abs(timestamp_ns - camera_timestamp_ns)) > max_timestamp_delta:
        raise Object6DBoxContractError("camera/object timestamp alignment gate failed")
    if not np.allclose(
        timestamp_s,
        timestamp_ns.astype(np.float64) * 1e-9,
        rtol=0.0,
        atol=1e-9,
    ):
        raise Object6DBoxContractError("timestamp_s and timestamp_ns disagree")

    functional_pose = _require_dict(functional.get("object_pose"), field="object_pose")
    if _artifact_identity(
        functional_pose.get("artifact"), field="functional object_pose artifact"
    ) != _artifact_identity(contract["observation_bundle"], field="observation_bundle"):
        raise Object6DBoxContractError("functional object_pose does not bind the bundle")
    pose_file = _resolve_project_file(
        project_root, acquisition.get("task_pose_file"), field="task_pose_file"
    )
    bundle_path = _resolve_project_file(
        project_root,
        _require_dict(contract["observation_bundle"], field="observation_bundle")[
            "path"
        ],
        field="observation_bundle.path",
    )
    if pose_file != bundle_path:
        raise Object6DBoxContractError("acquisition task_pose_file does not bind the bundle")

    array_mapping = _require_dict(
        contract["functional_array_mapping"], field="functional_array_mapping"
    )
    expected_mapping = {
        "T_world_object": "T_world_object",
        "object_valid": "valid",
        "object_confidence": "confidence",
        "object_state": "provenance",
        "direct_observation": "direct_observation",
        "timestamp_s": "timestamp_s",
    }
    if array_mapping != expected_mapping:
        raise Object6DBoxContractError("functional array mapping must be explicit/canonical")
    functional_array_specs = _require_dict(
        functional_pose.get("arrays"), field="functional object_pose arrays"
    )
    expected_functional_specs = {
        "T_world_object": ([frame_count, object_count, 4, 4], "float64"),
        "object_valid": ([frame_count, object_count], "bool"),
        "object_confidence": ([frame_count, object_count], "float64"),
        "object_state": ([frame_count, object_count], None),
        "direct_observation": ([frame_count, object_count], "bool"),
        "timestamp_s": ([frame_count], "float64"),
    }
    for functional_name, (expected_shape, expected_dtype) in expected_functional_specs.items():
        spec = _require_dict(
            functional_array_specs.get(functional_name),
            field=f"functional arrays.{functional_name}",
        )
        if spec.get("shape") != expected_shape:
            raise Object6DBoxContractError(
                f"functional arrays.{functional_name} shape mismatch"
            )
        dtype_text = spec.get("dtype")
        if expected_dtype is not None and dtype_text != expected_dtype:
            raise Object6DBoxContractError(
                f"functional arrays.{functional_name} dtype mismatch"
            )
        if expected_dtype is None and (
            type(dtype_text) is not str or not dtype_text.startswith("<U")
        ):
            raise Object6DBoxContractError(
                f"functional arrays.{functional_name} must be Unicode"
            )

    functional_geometry_list = _require_list(
        functional.get("geometry_registry"), field="geometry_registry"
    )
    acquisition_object_list = _require_list(
        acquisition.get("objects"), field="capture objects"
    )
    functional_geometry = {
        item.get("object_id"): item
        for item in functional_geometry_list
        if type(item) is dict
    }
    acquisition_objects = {
        item.get("object_id"): item
        for item in acquisition_object_list
        if type(item) is dict
    }
    if (
        len(functional_geometry) != len(functional_geometry_list)
        or len(acquisition_objects) != len(acquisition_object_list)
    ):
        raise Object6DBoxContractError("source manifests contain duplicate object IDs")
    observed_ids = [str(value) for value in arrays["object_ids"].tolist()]
    mapped_ids: list[str] = []
    prepared_instances: list[dict[str, Any]] = []
    for index, raw_instance in enumerate(instances):
        instance = _require_dict(raw_instance, field=f"instances[{index}]")
        required_instance_keys = {
            "object_id",
            "mask_role",
            "published_union_role",
            "geometry_id",
            "mesh_scale_to_m",
            "origin_definition",
            "axis_definition",
            "bounds_center_object_m",
            "metric_dimensions_m",
            "session_static",
            "area_fraction_min",
            "area_fraction_max",
        }
        _require_exact_keys(
            instance, required=required_instance_keys, field=f"instances[{index}]"
        )
        object_id = _require_string(instance["object_id"], field="object_id")
        mapped_ids.append(object_id)
        if object_id not in functional_geometry or object_id not in acquisition_objects:
            raise Object6DBoxContractError(f"{object_id} is absent from source manifests")
        geometry = _require_dict(functional_geometry[object_id], field="geometry")
        captured = _require_dict(acquisition_objects[object_id], field="capture object")
        if geometry.get("geometry_id") != instance["geometry_id"]:
            raise Object6DBoxContractError(f"{object_id} geometry_id mismatch")
        mesh_ref = geometry.get("mesh")
        _load_artifact(project_root, mesh_ref, field=f"mesh[{object_id}]")
        mesh_path, _, mesh_sha = _artifact_identity(mesh_ref, field=f"mesh[{object_id}]")
        if (
            captured.get("visual_mesh") != mesh_path
            or captured.get("geometry_sha256") != mesh_sha
            or captured.get("origin_definition") != instance["origin_definition"]
            or captured.get("axis_definition") != instance["axis_definition"]
            or captured.get("static_in_session") is not instance["session_static"]
        ):
            raise Object6DBoxContractError(f"{object_id} capture/role geometry mismatch")
        scale = _require_number(
            instance["mesh_scale_to_m"], field="mesh_scale_to_m", minimum=1e-12
        )
        center_values = _require_list(
            instance["bounds_center_object_m"], field="bounds_center_object_m"
        )
        dimension_values = _require_list(
            instance["metric_dimensions_m"], field="metric_dimensions_m"
        )
        if len(center_values) != 3 or len(dimension_values) != 3:
            raise Object6DBoxContractError("object bounds must contain three coordinates")
        center = np.asarray(
            [_require_number(value, field="bounds center") for value in center_values],
            dtype=np.float64,
        )
        dimensions = np.asarray(
            [
                _require_number(value, field="metric dimension", minimum=1e-9)
                for value in dimension_values
            ],
            dtype=np.float64,
        )
        functional_dimensions = np.asarray(geometry.get("metric_dimensions_m"))
        if functional_dimensions.shape != (3,) or not np.allclose(
            functional_dimensions.astype(np.float64), dimensions, rtol=0.0, atol=1e-12
        ):
            raise Object6DBoxContractError(f"{object_id} metric dimensions mismatch")
        if arrays["geometry_sha256"][index] != mesh_sha:
            raise Object6DBoxContractError(f"{object_id} per-instance geometry SHA mismatch")
        if scale <= 0.0:
            raise Object6DBoxContractError(f"{object_id} mesh scale is invalid")
        area_min = _require_number(
            instance["area_fraction_min"],
            field="area_fraction_min",
            minimum=0.0,
            maximum=1.0,
        )
        area_max = _require_number(
            instance["area_fraction_max"],
            field="area_fraction_max",
            minimum=area_min,
            maximum=1.0,
        )
        prepared_instances.append(
            {
                "object_id": object_id,
                "mask_role": _require_string(instance["mask_role"], field="mask_role"),
                "published_union_role": _require_string(
                    instance["published_union_role"], field="published_union_role"
                ),
                "center": center,
                "dimensions": dimensions,
                "area_min": area_min,
                "area_max": area_max,
            }
        )
    mapped_roles = [value["mask_role"] for value in prepared_instances]
    if (
        len(set(mapped_ids)) != object_count
        or len(set(mapped_roles)) != object_count
        or observed_ids != mapped_ids
    ):
        raise Object6DBoxContractError(
            "object identity/role/order must be unique and identical across mapping and bundle"
        )

    allowed_provenance = acquisition.get("allowed_provenance")
    if type(allowed_provenance) is not list or "DIRECT_TRACKED" not in allowed_provenance:
        raise Object6DBoxContractError("capture manifest lacks DIRECT_TRACKED provenance")
    required_pose_fields = acquisition.get("required_pose_fields")
    if type(required_pose_fields) is not list or not {
        "frame_id",
        "timestamp_ns",
        "object_ids",
        "T_object_to_camera",
        "valid",
        "confidence",
        "provenance",
        "geometry_sha256",
    }.issubset(required_pose_fields):
        raise Object6DBoxContractError("capture manifest pose fields are incomplete")

    required_indices_raw = contract["required_frame_indices"]
    if required_indices_raw is None:
        required_indices = set(range(frame_count))
    else:
        required_indices_list = _require_list(
            required_indices_raw, field="required_frame_indices"
        )
        if (
            any(type(value) is not int or isinstance(value, bool) for value in required_indices_list)
            or len(set(required_indices_list)) != len(required_indices_list)
            or any(value < 0 or value >= frame_count for value in required_indices_list)
        ):
            raise Object6DBoxContractError("required_frame_indices are invalid")
        required_indices = set(required_indices_list)

    rows: list[dict[str, Any]] = []
    required_unknown = 0
    known_count = 0
    previous_known: dict[str, tuple[int, np.ndarray, float]] = {}
    for frame_index in range(frame_count):
        world_camera = arrays["T_world_camera"][frame_index]
        _require_se3(world_camera, field=f"T_world_camera[{frame_index}]", atol=se3_atol)
        camera_world = np.linalg.inv(world_camera)
        for object_index, instance in enumerate(prepared_instances):
            reasons: list[str] = []
            valid = bool(arrays["valid"][frame_index, object_index])
            visible = bool(arrays["visible"][frame_index, object_index])
            direct = bool(arrays["direct_observation"][frame_index, object_index])
            identity_valid = bool(arrays["identity_valid"][frame_index, object_index])
            confidence = float(arrays["confidence"][frame_index, object_index])
            identity_score = float(arrays["identity_score"][frame_index, object_index])
            reprojection = float(
                arrays["reprojection_rmse_px"][frame_index, object_index]
            )
            provenance = str(arrays["provenance"][frame_index, object_index])
            source_instance_id = str(
                arrays["source_instance_ids"][frame_index, object_index]
            )
            source_frame_index = int(
                arrays["source_frame_index"][frame_index, object_index]
            )
            if not valid:
                reasons.append("POSE_INVALID")
            if not visible:
                reasons.append("OCCLUDED_OR_NOT_VISIBLE")
            if not direct or provenance != "DIRECT_TRACKED":
                reasons.append("NOT_DIRECT_OBSERVATION")
            if source_frame_index != frame_index:
                reasons.append("NEIGHBOR_OR_PROPAGATED_SOURCE_FORBIDDEN")
            if source_instance_id != instance["object_id"]:
                reasons.append("SOURCE_INSTANCE_IDENTITY_GATE")
            if not identity_valid or not math.isfinite(identity_score) or identity_score < min_identity_score:
                reasons.append("IDENTITY_GATE")
            if not math.isfinite(confidence) or confidence < min_confidence:
                reasons.append("CONFIDENCE_GATE")
            if not math.isfinite(reprojection) or reprojection > max_reprojection:
                reasons.append("REPROJECTION_GATE")

            box: np.ndarray | None = None
            depth: float | None = None
            area_fraction: float | None = None
            edge_clip_fraction: float | None = None
            if not reasons:
                world_object = arrays["T_world_object"][frame_index, object_index]
                camera_object = arrays["T_object_to_camera"][frame_index, object_index]
                _require_se3(
                    world_object,
                    field=f"T_world_object[{frame_index},{object_index}]",
                    atol=se3_atol,
                )
                _require_se3(
                    camera_object,
                    field=f"T_object_to_camera[{frame_index},{object_index}]",
                    atol=se3_atol,
                )
                composed = camera_world @ world_object
                if not np.allclose(
                    composed, camera_object, rtol=0.0, atol=composition_atol
                ):
                    reasons.append("C2W_POSE_COMPOSITION_GATE")
                else:
                    projected = _project_box(
                        _corners(instance["center"], instance["dimensions"]),
                        camera_object,
                        intrinsic,
                        min_depth_m=min_depth,
                    )
                    if projected is None:
                        reasons.append("DEPTH_OR_PROJECTION_GATE")
                    else:
                        raw_box, depth = projected
                        box, edge_clip_fraction = _clip_box(raw_box, width, height)
                        area_fraction = float(
                            max(0.0, box[2] - box[0])
                            * max(0.0, box[3] - box[1])
                            / float(width * height)
                        )
                        if edge_clip_fraction > max_edge_clip:
                            reasons.append("EDGE_GATE")
                        if not instance["area_min"] <= area_fraction <= instance["area_max"]:
                            reasons.append("AREA_GATE")
                        previous = previous_known.get(instance["object_id"])
                        if previous is not None and previous[0] == frame_index - 1:
                            delta_s = float(timestamp_s[frame_index] - timestamp_s[previous[0]])
                            center_now = np.asarray(
                                [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
                            )
                            speed = float(np.linalg.norm(center_now - previous[1]) / delta_s)
                            area_ratio = area_fraction / previous[2]
                            if speed > max_speed:
                                reasons.append("IDENTITY_CENTER_SPEED_GATE")
                            if not area_ratio_min <= area_ratio <= area_ratio_max:
                                reasons.append("IDENTITY_AREA_CONTINUITY_GATE")

            known = not reasons and box is not None and area_fraction is not None
            if known:
                known_count += 1
                center_px = np.asarray(
                    [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5]
                )
                previous_known[instance["object_id"]] = (
                    frame_index,
                    center_px,
                    area_fraction,
                )
            elif frame_index in required_indices:
                required_unknown += 1
            rows.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ns": int(timestamp_ns[frame_index]),
                    "object_index": object_index,
                    "object_id": instance["object_id"],
                    "mask_role": instance["mask_role"],
                    "published_union_role": instance["published_union_role"],
                    "status": "KNOWN_OBJECT6D_DIRECT" if known else "UNKNOWN",
                    "reasons": reasons,
                    "bbox_xyxy": [float(value) for value in box] if known else None,
                    "mean_depth_m": depth if known else None,
                    "area_fraction": area_fraction if known else None,
                    "edge_clip_fraction": edge_clip_fraction if known else None,
                    "confidence": confidence if math.isfinite(confidence) else None,
                    "identity_score": (
                        identity_score if math.isfinite(identity_score) else None
                    ),
                    "source_instance_id": source_instance_id,
                    "source_frame_index": source_frame_index,
                    "reprojection_rmse_px": (
                        reprojection if math.isfinite(reprojection) else None
                    ),
                    "authority_source": "OBJECT6D_DIRECT_ONLY",
                }
            )

    total_rows = frame_count * object_count
    status = (
        "PASS_OBJECT6D_INDEPENDENT_BOX_AUTHORITY"
        if required_unknown == 0
        else "HOLD_OBJECT6D_BOX_AUTHORITY_UNKNOWN_ROWS"
    )
    return {
        "schema_version": OUTPUT_SCHEMA,
        "task_id": task_id,
        "session_id": session_id,
        "status": status,
        "consumption_authorized": required_unknown == 0,
        "claim_limit": (
            "Only KNOWN_OBJECT6D_DIRECT rows are independent box prompts for a "
            "separate SAM refine gate; this result never authorizes a mask by itself."
        ),
        "fallback_policy": contract["fallback_policy"],
        "provider": contract["provider"],
        "frame_count": frame_count,
        "object_count": object_count,
        "row_count": total_rows,
        "known_row_count": known_count,
        "required_unknown_row_count": required_unknown,
        "required_frame_indices": sorted(required_indices),
        "pins": pins,
        "rows": rows,
    }


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(value)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Object6D and emit per-frame independent box authority."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = build_box_authority(arguments.contract)
    except Object6DBoxContractError as error:
        print(json.dumps({"status": "CONTRACT_FAILED", "error": str(error)}))
        return 2
    _atomic_write_json(arguments.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "known_rows": result["known_row_count"],
                "required_unknown_rows": result["required_unknown_row_count"],
                "output": str(arguments.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if result["consumption_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
