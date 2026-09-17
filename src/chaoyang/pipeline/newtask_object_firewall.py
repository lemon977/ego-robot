#!/usr/bin/env python3
"""Fail-closed geometry and multi-object contract for poker/chips tasks.

This module is deliberately a thin CPU-only adapter.  It validates the new
task authority before callers translate ``CARD_THIN_BOX`` to the existing
``box_xyz`` depth route or translate curved parametric models to externally
rendered ``mask_depth``.  It never guesses a geometry from legacy Object6D
fields and never opens a renderer.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_VERSION = "newtask-object-contract-v2.2"
VALIDATION_MODES = ("DRY_RUN", "FORMAL_CAPTURE")
REAL_MEASUREMENT_EVIDENCE_KINDS = ("RULER_PHOTO", "CALIPER_PHOTO")
PINNED_PILLOW_VERSION = "11.3.0"
ACCEPTED_DECODED_FORMATS = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}}

TASK_ALIASES = {
    "POKER": "POKER",
    "poker": "POKER",
    "puke": "POKER",
    "扑克": "POKER",
    "CHIPS": "CHIPS",
    "chips": "CHIPS",
    "shupian": "CHIPS",
    "薯片": "CHIPS",
}

EXPECTED_OBJECTS: dict[str, dict[str, tuple[str, str]]] = {
    "POKER": {
        "card_0": ("OPERATED_OBJECT", "CARD_THIN_BOX"),
        "card_1": ("OPERATED_OBJECT", "CARD_THIN_BOX"),
        "card_2": ("OPERATED_OBJECT", "CARD_THIN_BOX"),
        "rack": ("STATIC_FURNITURE", "RACK_BOX"),
        "mat_plane": ("SUPPORT_SURFACE", "SUPPORT_PLANE"),
        "rack_top_plane": ("SUPPORT_SURFACE", "SUPPORT_PLANE"),
    },
    "CHIPS": {
        "chip_0": ("OPERATED_OBJECT", "CHIP_SADDLE"),
        "chip_1": ("OPERATED_OBJECT", "CHIP_SADDLE"),
        "chip_2": ("OPERATED_OBJECT", "CHIP_SADDLE"),
        "bowl": ("CONTAINER", "BOWL_REVOLVE"),
        "mat_plane": ("SUPPORT_SURFACE", "SUPPORT_PLANE"),
    },
}

TRACKED_OBJECT_IDS = {
    "POKER": ("card_0", "card_1", "card_2"),
    "CHIPS": ("chip_0", "chip_1", "chip_2"),
}

SUPPORT_SURFACE_IDS = {
    "POKER": ("mat_plane", "rack_top_plane"),
    "CHIPS": ("mat_plane",),
}

ALLOWED_STATES = {
    "POKER": (
        "ON_RACK",
        "IN_HAND",
        "AIRBORNE",
        "ON_MAT",
        "OUT_OF_VIEW",
        "UNKNOWN_TRANSIENT",
    ),
    "CHIPS": (
        "ON_MAT",
        "IN_HAND",
        "AIRBORNE",
        "IN_BOWL",
        "ON_BOWL_RIM",
        "OUT_OF_VIEW",
        "UNKNOWN_TRANSIENT",
    ),
}

GEOMETRY_TO_DEPTH_ADAPTER = {
    "CARD_THIN_BOX": "box_xyz",
    "RACK_BOX": "box_xyz",
    "CHIP_SADDLE": "mask_depth",
    "BOWL_REVOLVE": "mask_depth",
    "SUPPORT_PLANE": "not_rendered_as_operated_object",
}

FORBIDDEN_GEOMETRY_VALUES = {"CYLINDER", "cylinder", "cylinder_y"}
FORBIDDEN_LEGACY_KEYS = {
    "cylinder_radius_m",
    "cylinder_height_m",
    "cylinder_axis_object_unit",
    "cylinder_bottom_center_m",
    "cylinder_top_center_m",
}


class NewTaskObjectContractError(ValueError):
    """Raised when a new-task object payload fails closed."""


class WrongGeometryError(NewTaskObjectContractError):
    """Raised for cylinder or another geometry outside the task whitelist."""


class ContractSchemaError(NewTaskObjectContractError):
    """Raised for structural or cross-document contract violations."""


def canonical_task(task: str) -> str:
    """Return the canonical task name, rejecting unknown task labels."""

    try:
        return TASK_ALIASES[task]
    except (KeyError, TypeError) as exc:
        raise NewTaskObjectContractError(f"unsupported new task: {task!r}") from exc


def canonical_validation_mode(validation_mode: str) -> str:
    if validation_mode not in VALIDATION_MODES:
        raise NewTaskObjectContractError(
            f"validation_mode must be one of {VALIDATION_MODES!r}, got {validation_mode!r}"
        )
    return validation_mode


def _walk(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def reject_legacy_cylinder(payload: Any) -> None:
    """Reject cylinder declarations and legacy cylinder authority anywhere."""

    for path, value in _walk(payload):
        if isinstance(value, Mapping):
            bad_keys = sorted(FORBIDDEN_LEGACY_KEYS.intersection(value))
            if bad_keys:
                raise WrongGeometryError(
                    f"legacy cylinder authority at {path}: {', '.join(bad_keys)}"
                )
            for selector in ("geometry_type", "geometry_kind", "kind", "depth_adapter"):
                if value.get(selector) in FORBIDDEN_GEOMETRY_VALUES:
                    raise WrongGeometryError(
                        f"cylinder is forbidden for poker/chips at {path}.{selector}"
                    )


def _reject_nonfinite_numbers(payload: Any) -> None:
    for path, value in _walk(payload):
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractSchemaError(f"non-finite number at {path}")


def authorise_geometry(task: str, object_id: str, geometry_type: str) -> str:
    """Authorise one exact object ID/type pair and return its depth adapter."""

    canonical = canonical_task(task)
    if geometry_type in FORBIDDEN_GEOMETRY_VALUES:
        raise WrongGeometryError(f"{canonical} object {object_id!r} cannot use CYLINDER")
    expected = EXPECTED_OBJECTS[canonical].get(object_id)
    if expected is None:
        raise WrongGeometryError(f"unexpected {canonical} object ID: {object_id!r}")
    expected_geometry = expected[1]
    if geometry_type != expected_geometry:
        raise WrongGeometryError(
            f"{canonical} object {object_id!r} requires {expected_geometry}, "
            f"got {geometry_type!r}"
        )
    return GEOMETRY_TO_DEPTH_ADAPTER[geometry_type]


def _schema_validate(document: Any, schema: Mapping[str, Any], label: str) -> None:
    try:
        import jsonschema

        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(document, schema)
    except ImportError as exc:  # pragma: no cover - runtime environment provides it
        raise ContractSchemaError("jsonschema is required for contract validation") from exc
    except jsonschema.ValidationError as exc:
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in exc.absolute_path
        )
        raise ContractSchemaError(f"{label} schema violation at {location}: {exc.message}") from exc
    except jsonschema.SchemaError as exc:
        raise ContractSchemaError(f"invalid {label} schema: {exc.message}") from exc


def _unique_records(records: Sequence[Mapping[str, Any]], key: str, label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        value = record[key]
        if value in indexed:
            raise ContractSchemaError(f"duplicate {label} {key}: {value!r}")
        indexed[value] = record
    return indexed


def _validate_metric_dimensions(record: Mapping[str, Any]) -> None:
    geometry_type = record["geometry_type"]
    if geometry_type in {"CARD_THIN_BOX", "CHIP_SADDLE"}:
        if not (record["length_m"] > record["width_m"] > record["thickness_m"]):
            raise ContractSchemaError(
                f"{geometry_type} requires length_m > width_m > thickness_m"
            )
    if geometry_type == "BOWL_REVOLVE":
        if not (record["inner_rim_diameter_m"] < record["outer_rim_diameter_m"]):
            raise ContractSchemaError("bowl inner rim must be smaller than outer rim")
        if not (
            record["inner_bottom_diameter_m"] < record["outer_bottom_diameter_m"]
        ):
            raise ContractSchemaError("bowl inner bottom must be smaller than outer bottom")
        if record["inner_depth_m"] >= record["outer_height_m"]:
            raise ContractSchemaError("bowl inner depth must be smaller than outer height")
        if record["wall_thickness_m"] * 2 >= record["outer_rim_diameter_m"]:
            raise ContractSchemaError("bowl wall thickness is physically inconsistent")


def _safe_evidence_path(evidence_root: Path, relative_path: str) -> Path:
    candidate_relative = Path(relative_path)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ContractSchemaError("formal evidence path must be relative and cannot traverse parents")
    root = evidence_root.resolve()
    unresolved = root / candidate_relative
    cursor = root
    for part in candidate_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ContractSchemaError(f"formal evidence cannot use symlinks: {relative_path!r}")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ContractSchemaError("formal evidence path escapes evidence_root") from exc
    if not candidate.is_file():
        raise ContractSchemaError(f"formal evidence is not an ordinary file: {relative_path!r}")
    return candidate


def _decode_image_evidence(path: Path, data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fully decode bound bytes using the pinned CPU Pillow authority."""

    try:
        runtime_version = package_version("Pillow")
        from PIL import Image, ImageFile, UnidentifiedImageError
    except (ImportError, PackageNotFoundError) as exc:
        raise ContractSchemaError(
            f"formal evidence requires Pillow=={PINNED_PILLOW_VERSION}"
        ) from exc
    if runtime_version != PINNED_PILLOW_VERSION:
        raise ContractSchemaError(
            f"formal evidence decoder drift: requires Pillow=={PINNED_PILLOW_VERSION}, "
            f"got {runtime_version}"
        )
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ContractSchemaError("Pillow truncated-image loading must remain disabled")

    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()
        with Image.open(BytesIO(data)) as decoded:
            decoded.load()
            decoded_format = decoded.format
            decoded_mode = decoded.mode
            width, height = decoded.size
            import numpy as np

            pixels = np.asarray(decoded)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ContractSchemaError(f"formal evidence cannot be fully decoded: {path.name!r}") from exc

    if decoded_format not in ACCEPTED_DECODED_FORMATS:
        raise ContractSchemaError(
            f"formal evidence decoded format {decoded_format!r} is not authorised"
        )
    suffix = path.suffix.lower()
    if suffix not in ACCEPTED_DECODED_FORMATS[decoded_format]:
        raise ContractSchemaError(
            f"formal evidence extension {suffix!r} disagrees with decoded {decoded_format}"
        )
    if width <= 0 or height <= 0:
        raise ContractSchemaError("formal evidence decoded dimensions must be positive")
    if pixels.ndim not in (2, 3) or pixels.shape[:2] != (height, width):
        raise ContractSchemaError("formal evidence decoded pixel layout is invalid")
    if pixels.ndim == 3 and not 1 <= pixels.shape[2] <= 4:
        raise ContractSchemaError("formal evidence decoded channel count is invalid")
    if pixels.dtype.kind not in "buif" or not np.isfinite(pixels).all():
        raise ContractSchemaError("formal evidence decoded pixels must be finite numeric values")

    authority = {
        "implementation": "Pillow",
        "pinned_version": PINNED_PILLOW_VERSION,
        "runtime_version": runtime_version,
        "route": "VERIFY_THEN_FULL_LOAD_FROM_SHA_BOUND_BYTES",
        "accepted_formats": sorted(ACCEPTED_DECODED_FORMATS),
        "truncated_images_allowed": False,
        "cpu_only": True,
    }
    decoded_record = {
        "path": str(path),
        "decoded_format": decoded_format,
        "decoded_mode": decoded_mode,
        "decoded_width_px": width,
        "decoded_height_px": height,
        "pixels_finite": True,
    }
    return authority, decoded_record


def _validate_measurement_authority(
    measurements: Mapping[str, Any],
    validation_mode: str,
    evidence_root: str | Path | None,
) -> tuple[str, str, bool, dict[str, Any] | None, list[dict[str, Any]]]:
    status = measurements["measurement_status"]
    records = measurements["measurements"]
    if validation_mode == "DRY_RUN":
        if status != "EXAMPLE_NOT_CAPTURE_AUTHORITY":
            raise ContractSchemaError(
                "DRY_RUN requires measurement_status=EXAMPLE_NOT_CAPTURE_AUTHORITY"
            )
        for record in records:
            if any(item["kind"] != "ILLUSTRATIVE_DRY_RUN" for item in record["evidence"]):
                raise ContractSchemaError("DRY_RUN evidence must be explicitly illustrative")
        return "ILLUSTRATIVE_DRY_RUN_ONLY", "ILLUSTRATIVE_ONLY", False, None, []

    if status != "CAPTURED_METRIC":
        raise ContractSchemaError("FORMAL_CAPTURE requires measurement_status=CAPTURED_METRIC")
    if evidence_root is None:
        raise ContractSchemaError("FORMAL_CAPTURE requires evidence_root")
    root = Path(evidence_root)
    decoder_authority: dict[str, Any] | None = None
    decoded_evidence: list[dict[str, Any]] = []
    for record in records:
        evidence_items = record["evidence"]
        if not evidence_items:
            raise ContractSchemaError(
                f"measurement {record['measurement_id']!r} requires photo evidence"
            )
        for evidence in evidence_items:
            if evidence["kind"] not in REAL_MEASUREMENT_EVIDENCE_KINDS:
                raise ContractSchemaError(
                    f"formal measurement {record['measurement_id']!r} has non-authority evidence"
                )
            required = {"path", "bytes", "sha256", "provenance"}
            missing = sorted(required - set(evidence))
            if missing:
                raise ContractSchemaError(
                    f"formal evidence for {record['measurement_id']!r} lacks {missing}"
                )
            provenance = evidence["provenance"]
            if not provenance.get("is_capture_authority") or not provenance.get("original_file"):
                raise ContractSchemaError("formal evidence provenance is not capture authority")
            captured_at = provenance.get("captured_at_utc", "")
            if not captured_at.endswith("Z"):
                raise ContractSchemaError("captured_at_utc must be an explicit UTC timestamp")
            try:
                datetime.fromisoformat(captured_at.removesuffix("Z") + "+00:00")
            except ValueError as exc:
                raise ContractSchemaError("captured_at_utc is not a valid timestamp") from exc
            expected_method = "RULER" if evidence["kind"] == "RULER_PHOTO" else "CALIPER"
            if provenance.get("measurement_method") != expected_method:
                raise ContractSchemaError("evidence kind and measurement_method disagree")
            path = _safe_evidence_path(root, evidence["path"])
            data = path.read_bytes()
            if len(data) != evidence["bytes"]:
                raise ContractSchemaError(f"formal evidence byte count mismatch: {evidence['path']!r}")
            if hashlib.sha256(data).hexdigest() != evidence["sha256"]:
                raise ContractSchemaError(f"formal evidence sha256 mismatch: {evidence['path']!r}")
            current_authority, decoded_record = _decode_image_evidence(path, data)
            if decoder_authority is not None and current_authority != decoder_authority:
                raise ContractSchemaError("formal evidence decoder authority changed within validation")
            decoder_authority = current_authority
            decoded_evidence.append(
                {
                    "measurement_id": record["measurement_id"],
                    "evidence_kind": evidence["kind"],
                    "sha256": evidence["sha256"],
                    **decoded_record,
                    "path": evidence["path"],
                }
            )
    if decoder_authority is None:
        raise ContractSchemaError("FORMAL_CAPTURE decoded no measurement evidence")
    return (
        "CAPTURED_METRIC_DECODED_EVIDENCE_VALIDATED",
        "FORMAL_CAPTURE_DECODED_FILE_AUTHORITY_MANUAL_SCENE_REVIEW_REQUIRED",
        True,
        decoder_authority,
        decoded_evidence,
    )


def validate_measurements(
    task: str,
    measurements: Mapping[str, Any],
    *,
    validation_mode: str,
    evidence_root: str | Path | None = None,
) -> tuple[
    dict[str, Mapping[str, Any]],
    str,
    str,
    bool,
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    canonical = canonical_task(task)
    mode = canonical_validation_mode(validation_mode)
    reject_legacy_cylinder(measurements)
    _reject_nonfinite_numbers(measurements)
    if measurements.get("task") != canonical:
        raise ContractSchemaError("measurement task does not match requested task")
    if measurements.get("validation_mode") != mode:
        raise ContractSchemaError("measurement validation_mode does not match requested mode")
    indexed = _unique_records(measurements["measurements"], "measurement_id", "measurement")
    required_types = (
        {"CARD_THIN_BOX", "RACK_BOX"}
        if canonical == "POKER"
        else {"CHIP_SADDLE", "BOWL_REVOLVE"}
    )
    found_types = {record["geometry_type"] for record in indexed.values()}
    if found_types != required_types:
        raise ContractSchemaError(
            f"measurement geometry types must be exactly {sorted(required_types)}; "
            f"got {sorted(found_types)}"
        )
    for record in indexed.values():
        _validate_metric_dimensions(record)
    authority = _validate_measurement_authority(measurements, mode, evidence_root)
    return indexed, *authority


def validate_registry(
    task: str,
    registry: Mapping[str, Any],
    measurements_by_id: Mapping[str, Mapping[str, Any]],
    *,
    validation_mode: str,
) -> dict[str, Mapping[str, Any]]:
    canonical = canonical_task(task)
    mode = canonical_validation_mode(validation_mode)
    reject_legacy_cylinder(registry)
    _reject_nonfinite_numbers(registry)
    if registry.get("task") != canonical:
        raise ContractSchemaError("registry task does not match requested task")
    if registry.get("validation_mode") != mode:
        raise ContractSchemaError("registry validation_mode does not match requested mode")
    indexed = _unique_records(registry["objects"], "object_id", "registry")
    expected = EXPECTED_OBJECTS[canonical]
    if set(indexed) != set(expected):
        raise ContractSchemaError(
            f"{canonical} registry IDs must be exactly {sorted(expected)}; got {sorted(indexed)}"
        )
    for object_id, (role, geometry_type) in expected.items():
        record = indexed[object_id]
        if record["role"] != role:
            raise ContractSchemaError(f"{object_id} role must be {role}")
        authorise_geometry(canonical, object_id, record["geometry_type"])
        if record["geometry_type"] != geometry_type:  # defensive clarity
            raise ContractSchemaError(f"{object_id} geometry must be {geometry_type}")
        measurement_id = record.get("measurement_id")
        if role == "SUPPORT_SURFACE":
            if measurement_id is not None:
                raise ContractSchemaError(f"{object_id} support must not claim an object measurement")
        else:
            measurement = measurements_by_id.get(measurement_id)
            if measurement is None:
                raise ContractSchemaError(f"{object_id} has unknown measurement_id {measurement_id!r}")
            if measurement["geometry_type"] != geometry_type:
                raise ContractSchemaError(f"{object_id} measurement geometry does not match registry")
        expected_adapter = GEOMETRY_TO_DEPTH_ADAPTER[geometry_type]
        if record["depth_adapter"] != expected_adapter:
            raise ContractSchemaError(
                f"{object_id} depth_adapter must be {expected_adapter}, got {record['depth_adapter']!r}"
            )

        expected_allowed_supports = {
            "POKER": {
                "card_0": ("mat_plane", "rack_top_plane"),
                "card_1": ("mat_plane", "rack_top_plane"),
                "card_2": ("mat_plane", "rack_top_plane"),
                "rack": ("mat_plane",),
                "mat_plane": (),
                "rack_top_plane": (),
            },
            "CHIPS": {
                "chip_0": ("mat_plane", "bowl"),
                "chip_1": ("mat_plane", "bowl"),
                "chip_2": ("mat_plane", "bowl"),
                "bowl": ("mat_plane",),
                "mat_plane": (),
            },
        }[canonical][object_id]
        if tuple(record["allowed_support_ids"]) != expected_allowed_supports:
            raise ContractSchemaError(
                f"{object_id} allowed_support_ids must be {expected_allowed_supports!r}"
            )

        expected_modes = {
            "OPERATED_OBJECT": ("PER_FRAME_SE3", "STATE_AWARE_LAYER"),
            "STATIC_FURNITURE": ("STATIC_SCENE", "PRESERVE"),
            "CONTAINER": ("STATIC_SCENE", "PRESERVE"),
            "SUPPORT_SURFACE": (
                "DERIVED_FROM_PARENT" if object_id == "rack_top_plane" else "STATIC_SCENE",
                "SUPPORT_AUTHORITY",
            ),
        }[role]
        if (record["pose_mode"], record["clean_policy"]) != expected_modes:
            raise ContractSchemaError(
                f"{object_id} pose_mode/clean_policy must be {expected_modes!r}"
            )

        expected_parent = "rack" if object_id == "rack_top_plane" else None
        if record.get("parent_object_id") != expected_parent:
            raise ContractSchemaError(f"{object_id} has invalid parent_object_id")

    declared_supports = tuple(registry["support_surface_ids"])
    if declared_supports != SUPPORT_SURFACE_IDS[canonical]:
        raise ContractSchemaError(
            f"support_surface_ids must be {SUPPORT_SURFACE_IDS[canonical]!r} in fixed order"
        )
    if canonical == "POKER" and indexed["rack_top_plane"].get("parent_object_id") != "rack":
        raise ContractSchemaError("rack_top_plane must name rack as parent_object_id")
    return indexed


def _validate_state_support(task: str, object_id: str, state: str, support_id: Any) -> None:
    required_support = {
        ("POKER", "ON_RACK"): "rack_top_plane",
        ("POKER", "ON_MAT"): "mat_plane",
        ("CHIPS", "ON_MAT"): "mat_plane",
        ("CHIPS", "IN_BOWL"): "bowl",
        ("CHIPS", "ON_BOWL_RIM"): "bowl",
    }.get((task, state))
    if required_support is not None and support_id != required_support:
        raise ContractSchemaError(
            f"{object_id} state {state} requires support_id={required_support!r}"
        )
    if required_support is None and support_id is not None:
        raise ContractSchemaError(f"{object_id} state {state} must have null support_id")


def validate_timeline(task: str, timeline: Mapping[str, Any], *, validation_mode: str) -> None:
    canonical = canonical_task(task)
    mode = canonical_validation_mode(validation_mode)
    reject_legacy_cylinder(timeline)
    _reject_nonfinite_numbers(timeline)
    if timeline.get("task") != canonical:
        raise ContractSchemaError("timeline task does not match requested task")
    if timeline.get("validation_mode") != mode:
        raise ContractSchemaError("timeline validation_mode does not match requested mode")
    expected_ids = set(TRACKED_OBJECT_IDS[canonical])
    if set(timeline["tracked_object_ids"]) != expected_ids:
        raise ContractSchemaError("tracked_object_ids must contain every operated object exactly once")
    frames = timeline["frames"]
    if timeline["frame_count"] != len(frames):
        raise ContractSchemaError("frame_count does not match frames length")
    previous_timestamp = None
    for expected_index, frame in enumerate(frames):
        if frame["frame_index"] != expected_index:
            raise ContractSchemaError("frame_index must be contiguous and start at zero")
        timestamp = frame["timestamp_s"]
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ContractSchemaError("timeline timestamps must be strictly increasing")
        previous_timestamp = timestamp
        states = _unique_records(frame["states"], "object_id", f"frame {expected_index} state")
        if set(states) != expected_ids:
            raise ContractSchemaError(
                f"frame {expected_index} must explicitly state all operated objects"
            )
        for object_id, record in states.items():
            state = record["state"]
            if state not in ALLOWED_STATES[canonical]:
                raise ContractSchemaError(f"state {state!r} is not allowed for {canonical}")
            if mode == "FORMAL_CAPTURE" and record["source"] == "DRY_RUN_EXAMPLE":
                raise ContractSchemaError(
                    "FORMAL_CAPTURE timeline cannot use DRY_RUN_EXAMPLE state authority"
                )
            _validate_state_support(canonical, object_id, state, record["support_id"])


def validate_contract(
    *,
    task: str,
    measurements: Mapping[str, Any],
    registry: Mapping[str, Any],
    timeline: Mapping[str, Any],
    schema_dir: str | Path,
    validation_mode: str,
    evidence_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate all three schemas plus cross-document semantics."""

    canonical = canonical_task(task)
    mode = canonical_validation_mode(validation_mode)
    schema_root = Path(schema_dir)
    documents = (
        (measurements, "OBJECT_MEASUREMENT.schema.json", "measurement"),
        (registry, "OBJECT_REGISTRY.schema.json", "registry"),
        (timeline, "OBJECT_STATE_TIMELINE.schema.json", "timeline"),
    )
    for document, schema_name, label in documents:
        reject_legacy_cylinder(document)
        with (schema_root / schema_name).open("r", encoding="utf-8") as handle:
            _schema_validate(document, json.load(handle), label)
    (
        measurements_by_id,
        authority_status,
        claim_limit,
        formal_allowed,
        decoder_authority,
        decoded_evidence,
    ) = validate_measurements(
        canonical,
        measurements,
        validation_mode=mode,
        evidence_root=evidence_root,
    )
    registry_by_id = validate_registry(
        canonical, registry, measurements_by_id, validation_mode=mode
    )
    validate_timeline(canonical, timeline, validation_mode=mode)
    return {
        "contract_version": CONTRACT_VERSION,
        "task": canonical,
        "validation_mode": mode,
        "status": "PASS",
        "object_ids": sorted(registry_by_id),
        "tracked_object_ids": list(TRACKED_OBJECT_IDS[canonical]),
        "support_surface_ids": list(SUPPORT_SURFACE_IDS[canonical]),
        "frame_count": timeline["frame_count"],
        "cylinder_authority_accepted": False,
        "authority_status": authority_status,
        "claim_limit": claim_limit,
        "formal_capture_allowed": formal_allowed,
        "decoder_authority": decoder_authority,
        "decoded_evidence": decoded_evidence,
        "manual_ruler_and_object_same_frame_review_required": True,
        "manual_scene_review_completed_by_machine": False,
        "gpu_used": False,
    }


def inspect_npz_authority(task: str, object_id: str, path: str | Path) -> dict[str, Any]:
    """Inspect an NPZ's scalar metadata without pickle and fail on old cylinders."""

    import numpy as np

    canonical = canonical_task(task)
    with np.load(Path(path), allow_pickle=False) as archive:
        keys = sorted(archive.files)
        bad_keys = sorted(FORBIDDEN_LEGACY_KEYS.intersection(keys))
        if bad_keys:
            raise WrongGeometryError(
                f"{canonical} NPZ contains forbidden cylinder fields: {bad_keys}"
            )
        if "geometry_type" not in archive.files:
            raise NewTaskObjectContractError("NPZ lacks explicit geometry_type; guessing is forbidden")
        value = archive["geometry_type"]
        if value.size != 1:
            raise NewTaskObjectContractError("NPZ geometry_type must be scalar")
        scalar = value.reshape(-1)[0]
        if isinstance(scalar, bytes):
            geometry_type = scalar.decode("utf-8", errors="strict")
        else:
            geometry_type = str(scalar)
        if geometry_type in FORBIDDEN_GEOMETRY_VALUES:
            raise WrongGeometryError(f"{canonical} NPZ cannot use {geometry_type}")
        adapter = authorise_geometry(canonical, object_id, geometry_type)
        if "object_id" in archive.files:
            object_scalar = archive["object_id"]
            if object_scalar.size != 1:
                raise NewTaskObjectContractError("NPZ object_id must be scalar")
            raw_id = object_scalar.reshape(-1)[0]
            archived_id = (
                raw_id.decode("utf-8", errors="strict")
                if isinstance(raw_id, bytes)
                else str(raw_id)
            )
            if archived_id != object_id:
                raise WrongGeometryError(
                    f"NPZ object_id {archived_id!r} does not match requested {object_id!r}"
                )
        return {
            "task": canonical,
            "object_id": object_id,
            "geometry_type": geometry_type,
            "depth_adapter": adapter,
            "keys": keys,
        }


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--measurement", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--schema-dir", required=True)
    parser.add_argument("--validation-mode", choices=VALIDATION_MODES, required=True)
    parser.add_argument("--evidence-root")
    args = parser.parse_args(argv)
    result = validate_contract(
        task=args.task,
        measurements=_read_json(args.measurement),
        registry=_read_json(args.registry),
        timeline=_read_json(args.timeline),
        schema_dir=args.schema_dir,
        validation_mode=args.validation_mode,
        evidence_root=args.evidence_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
