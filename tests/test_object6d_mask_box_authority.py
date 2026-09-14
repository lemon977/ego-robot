from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from pipeline.object6d_mask_box_authority import (
    Object6DBoxContractError,
    build_box_authority,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = (
    REPO_ROOT
    / "tasks/chips/runs/robot/20260903_functional_retarget_devapprox_smoke_v2"
    / "TOMORROW_OBJECT6D_DROPIN.schema.json"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _artifact(root: Path, path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fixture(
    tmp_path: Path,
    *,
    bundle_mutator: Callable[[dict[str, np.ndarray]], None] | None = None,
    contract_mutator: Callable[[dict[str, object]], None] | None = None,
    role_mutator: Callable[[dict[str, object]], None] | None = None,
) -> tuple[Path, Path]:
    root = tmp_path
    root.mkdir(parents=True, exist_ok=True)
    schema_path = root / "schema.json"
    schema_path.write_bytes(SOURCE_SCHEMA.read_bytes())
    object_ids = ["chip_0", "chip_1"]
    frame_count = 3
    object_count = len(object_ids)
    meshes: list[Path] = []
    for object_id in object_ids:
        mesh = root / "models" / f"{object_id}.ply"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(f"ply\ncomment {object_id}\n".encode())
        meshes.append(mesh)

    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    world_camera = np.repeat(np.eye(4)[None, :, :], frame_count, axis=0)
    world_object = np.repeat(
        np.eye(4)[None, None, :, :], frame_count * object_count, axis=0
    ).reshape(frame_count, object_count, 4, 4)
    x_positions = ((-0.15, 0.15), (0.0, 0.0), (0.15, -0.15))
    for frame_index, positions in enumerate(x_positions):
        for object_index, x_value in enumerate(positions):
            world_object[frame_index, object_index, 0, 3] = x_value
            world_object[frame_index, object_index, 2, 3] = 1.0
    timestamp_ns = np.asarray(
        [1_000_000_000, 1_100_000_000, 1_200_000_000], dtype=np.int64
    )
    geometry_sha = np.asarray(
        [hashlib.sha256(path.read_bytes()).hexdigest() for path in meshes]
    )
    arrays: dict[str, np.ndarray] = {
        "K": intrinsic.astype(np.float64),
        "T_world_camera": world_camera.astype(np.float64),
        "frame_id": np.arange(frame_count, dtype=np.int64),
        "camera_timestamp_ns": timestamp_ns.copy(),
        "timestamp_ns": timestamp_ns.copy(),
        "timestamp_s": timestamp_ns.astype(np.float64) * 1e-9,
        "object_ids": np.asarray(object_ids),
        "source_instance_ids": np.tile(np.asarray(object_ids), (frame_count, 1)),
        "source_frame_index": np.tile(
            np.arange(frame_count, dtype=np.int64)[:, None], (1, object_count)
        ),
        "geometry_sha256": geometry_sha,
        "T_world_object": world_object.astype(np.float64),
        "T_object_to_camera": world_object.astype(np.float64),
        "valid": np.ones((frame_count, object_count), dtype=np.bool_),
        "visible": np.ones((frame_count, object_count), dtype=np.bool_),
        "confidence": np.full((frame_count, object_count), 0.95, dtype=np.float64),
        "provenance": np.full(
            (frame_count, object_count), "DIRECT_TRACKED", dtype="<U14"
        ),
        "direct_observation": np.ones(
            (frame_count, object_count), dtype=np.bool_
        ),
        "identity_valid": np.ones((frame_count, object_count), dtype=np.bool_),
        "identity_score": np.full(
            (frame_count, object_count), 0.99, dtype=np.float64
        ),
        "reprojection_rmse_px": np.full(
            (frame_count, object_count), 0.5, dtype=np.float64
        ),
    }
    if bundle_mutator is not None:
        bundle_mutator(arrays)
    bundle_path = root / "session" / "object6d_bundle.npz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(bundle_path, **arrays)
    bundle_ref = _artifact(root, bundle_path)

    contact_path = root / "session" / "contact.csv"
    contact_path.write_text("frame,physical_side\n", encoding="utf-8")
    contact_ref = _artifact(root, contact_path)
    geometry_registry = []
    acquisition_objects = []
    role_instances = []
    for index, (object_id, mesh) in enumerate(zip(object_ids, meshes, strict=True)):
        mesh_ref = _artifact(root, mesh)
        geometry_id = f"geometry_{object_id}"
        dimensions = [0.04, 0.04, 0.004]
        geometry_registry.append(
            {
                "object_id": object_id,
                "geometry_id": geometry_id,
                "role": "operated_object",
                "mesh": mesh_ref,
                "metric_dimensions_m": dimensions,
                "geometry_authority": "MEASURED_CAD",
            }
        )
        acquisition_objects.append(
            {
                "object_id": object_id,
                "geometry_type": "CHIP_SADDLE",
                "visual_mesh": mesh_ref["path"],
                "collision_mesh": mesh_ref["path"],
                "geometry_sha256": mesh_ref["sha256"],
                "origin_definition": "metric_obb_center",
                "axis_definition": "right_handed_object_xyz",
                "measurement_photo_ids": [f"photo_{index}"],
                "dimensions_m": {"x": 0.04, "y": 0.04, "z": 0.004},
                "static_in_session": False,
            }
        )
        role_instances.append(
            {
                "object_id": object_id,
                "mask_role": f"chip_slot_{index}",
                "published_union_role": "chip_visible_union",
                "geometry_id": geometry_id,
                "mesh_scale_to_m": 1.0,
                "origin_definition": "metric_obb_center",
                "axis_definition": "right_handed_object_xyz",
                "bounds_center_object_m": [0.0, 0.0, 0.0],
                "metric_dimensions_m": dimensions,
                "session_static": False,
                "area_fraction_min": 0.00001,
                "area_fraction_max": 0.05,
            }
        )

    functional = {
        "schema_version": "object6d-functional-retarget-input-v1",
        "task_id": "chips",
        "session_id": "synthetic_crossing",
        "frame_count": frame_count,
        "fps": 10.0,
        "units": {"length": "m", "angle": "rad", "time": "s"},
        "coordinate_frames": {
            "camera_frame": "camera_rectified",
            "world_frame": "world",
            "T_world_camera_convention": "column_vector_left_multiply",
            "camera_axis_convention": "+X right, +Y down, +Z forward",
        },
        "provenance": {
            "producer": "synthetic_object6d",
            "checkpoint_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "source_media_sha256": "3" * 64,
            "direct_measurement": True,
            "interpolation_policy": "NONE",
        },
        "geometry_registry": geometry_registry,
        "object_pose": {
            "artifact": bundle_ref,
            "arrays": {
                "T_world_object": {"dtype": "float64", "shape": [3, 2, 4, 4]},
                "object_valid": {"dtype": "bool", "shape": [3, 2]},
                "object_confidence": {"dtype": "float64", "shape": [3, 2]},
                "object_state": {"dtype": "<U14", "shape": [3, 2]},
                "direct_observation": {"dtype": "bool", "shape": [3, 2]},
                "timestamp_s": {"dtype": "float64", "shape": [3]},
            },
            "se3_validation": {
                "bottom_row_max_abs_error": 0.0,
                "rotation_orthogonality_max_abs_error": 0.0,
                "rotation_det_min": 1.0,
            },
        },
        "contact_evidence": {
            "artifact": contact_ref,
            "record_format": "frame,physical_side,human_finger,object_id,active,score,direct_observation,source",
            "identity_contract": "anatomical_human_side_explicit_then_frozen_human_to_physical_mapping",
            "allowed_sources": ["TRACKER_CONTACT_SENSOR"],
        },
        "pad_registry": [
            {
                "physical_side": side,
                "human_finger": "index",
                "kai_pad_id": f"{side}_index",
                "link_name": f"{side}_index_link",
                "patch_authority_sha256": "4" * 64,
            }
            for side in ("left", "right")
        ],
        "formal_gates": {
            "contact_window_object6d_valid_required": True,
            "active_pad_signed_sdf_m": [0.0, 0.003],
            "inactive_finger_policy": "NATURAL_PRIOR",
            "dense_mesh_nonpenetration_required": True,
            "distinct_finger_self_sat_required": True,
            "joint_limits_velocity_acceleration_true_dt_required": True,
            "mask_clean_object_occlusion_required_before_video": True,
        },
    }
    functional_path = root / "session" / "functional.json"
    _write_json(functional_path, functional)

    acquisition = {
        "schema_version": "chaoyang-object6d-capture-v1",
        "task_id": "chips",
        "session_id": "synthetic_crossing",
        "metric_unit": "meter",
        "coordinate_convention": {
            "pose_field": "T_object_to_camera",
            "camera_axes": "+X right, +Y down, +Z forward",
            "matrix_convention": "column vectors, p_camera = T_object_to_camera @ p_object",
        },
        "objects": acquisition_objects,
        "task_pose_file": bundle_ref["path"],
        "required_pose_fields": [
            "frame_id",
            "timestamp_ns",
            "object_ids",
            "T_object_to_camera",
            "valid",
            "confidence",
            "provenance",
            "geometry_sha256",
        ],
        "allowed_provenance": ["DIRECT_TRACKED", "MODEL_FIT", "INTERPOLATED", "MISSING"],
        "contact_canary_file": contact_ref["path"],
        "manual_ruler_and_object_same_frame_review_required": True,
        "capture_operator": "synthetic",
        "capture_utc": "2026-09-03T00:00:00Z",
        "notes": "synthetic crossing fixture",
    }
    acquisition_path = root / "session" / "capture.json"
    _write_json(acquisition_path, acquisition)

    role_mapping: dict[str, object] = {
        "schema_version": "object6d-mask-role-mapping-v1",
        "task_id": "chips",
        "template_only": False,
        "instances": role_instances,
    }
    if role_mutator is not None:
        role_mutator(role_mapping)
    role_path = root / "session" / "roles.json"
    _write_json(role_path, role_mapping)

    contract: dict[str, object] = {
        "schema_version": "object6d-mask-box-adapter-v1",
        "project_root": str(root),
        "task_id": "chips",
        "session_id": "synthetic_crossing",
        "frame_count": frame_count,
        "image_size": {"width": 100, "height": 100},
        "source_contracts": {
            "functional_dropin_schema": _artifact(root, schema_path),
            "functional_dropin_manifest": _artifact(root, functional_path),
            "acquisition_manifest": _artifact(root, acquisition_path),
        },
        "role_mapping": _artifact(root, role_path),
        "observation_bundle": bundle_ref,
        "provider": {
            "producer": "synthetic_object6d",
            "version": "1.0",
            "provider_family": "independent_pose_estimator",
            "code_sha256": "5" * 64,
            "config_sha256": "6" * 64,
            "checkpoint_sha256": "7" * 64,
            "license_sha256": "8" * 64,
            "independent_from_sam3": True,
            "temporal_policy": "PER_FRAME_DIRECT_OBSERVATION_CURRENT_FRAME_ONLY",
        },
        "functional_array_mapping": {
            "T_world_object": "T_world_object",
            "object_valid": "valid",
            "object_confidence": "confidence",
            "object_state": "provenance",
            "direct_observation": "direct_observation",
            "timestamp_s": "timestamp_s",
        },
        "gates": {
            "max_timestamp_delta_ns": 5_000_000,
            "min_confidence": 0.7,
            "min_identity_score": 0.8,
            "max_reprojection_rmse_px": 4.0,
            "max_edge_clip_fraction": 0.05,
            "min_depth_m": 0.05,
            "se3_atol": 1e-6,
            "pose_composition_atol": 1e-6,
            "max_center_speed_px_per_s": 5000.0,
            "adjacent_area_ratio_min": 0.5,
            "adjacent_area_ratio_max": 2.0,
        },
        "required_frame_indices": None,
        "fallback_policy": "UNKNOWN_NO_STATIC_NO_PICO_NO_PROPAGATION",
    }
    if contract_mutator is not None:
        contract_mutator(contract)
    contract_path = root / "contract.json"
    _write_json(contract_path, contract)
    return contract_path, meshes[0]


def test_crossing_instances_keep_frozen_identity_and_pass(tmp_path: Path) -> None:
    contract, _ = _fixture(tmp_path)
    result = build_box_authority(contract)
    assert result["status"] == "PASS_OBJECT6D_INDEPENDENT_BOX_AUTHORITY"
    assert result["consumption_authorized"] is True
    frame_two = [row for row in result["rows"] if row["frame_index"] == 2]
    assert [row["object_id"] for row in frame_two] == ["chip_0", "chip_1"]
    assert frame_two[0]["bbox_xyxy"][0] > frame_two[1]["bbox_xyxy"][0]


@pytest.mark.parametrize(
    ("mutator", "expected_reason"),
    [
        (
            lambda arrays: arrays["source_instance_ids"].__setitem__(
                (2, slice(None)), ["chip_1", "chip_0"]
            ),
            "SOURCE_INSTANCE_IDENTITY_GATE",
        ),
        (
            lambda arrays: arrays["source_frame_index"].__setitem__((1, 0), 0),
            "NEIGHBOR_OR_PROPAGATED_SOURCE_FORBIDDEN",
        ),
        (
            lambda arrays: arrays["direct_observation"].__setitem__((1, 0), False),
            "NOT_DIRECT_OBSERVATION",
        ),
        (
            lambda arrays: arrays["visible"].__setitem__((1, 0), False),
            "OCCLUDED_OR_NOT_VISIBLE",
        ),
    ],
)
def test_identity_directness_and_visibility_fail_closed(
    tmp_path: Path,
    mutator: Callable[[dict[str, np.ndarray]], None],
    expected_reason: str,
) -> None:
    contract, _ = _fixture(tmp_path, bundle_mutator=mutator)
    result = build_box_authority(contract)
    assert result["status"] == "HOLD_OBJECT6D_BOX_AUTHORITY_UNKNOWN_ROWS"
    assert result["consumption_authorized"] is False
    assert any(expected_reason in row["reasons"] for row in result["rows"])


def test_sam_self_provider_is_rejected(tmp_path: Path) -> None:
    def mutate(contract: dict[str, object]) -> None:
        provider = contract["provider"]
        assert isinstance(provider, dict)
        provider["provider_family"] = "SAM3_text_detector"

    contract, _ = _fixture(tmp_path, contract_mutator=mutate)
    with pytest.raises(Object6DBoxContractError, match="independent from SAM"):
        build_box_authority(contract)


def test_timestamp_and_missing_array_contracts_fail(tmp_path: Path) -> None:
    def bad_timestamp(arrays: dict[str, np.ndarray]) -> None:
        arrays["camera_timestamp_ns"][1] += 6_000_000

    contract, _ = _fixture(tmp_path / "timestamp", bundle_mutator=bad_timestamp)
    with pytest.raises(Object6DBoxContractError, match="timestamp alignment"):
        build_box_authority(contract)

    def missing_visibility(arrays: dict[str, np.ndarray]) -> None:
        del arrays["visible"]

    contract, _ = _fixture(tmp_path / "missing", bundle_mutator=missing_visibility)
    with pytest.raises(Object6DBoxContractError, match="bundle keys"):
        build_box_authority(contract)


def test_mesh_byte_drift_and_uninstantiated_role_template_fail(tmp_path: Path) -> None:
    contract, mesh = _fixture(tmp_path / "drift")
    mesh.write_bytes(mesh.read_bytes() + b"drift")
    with pytest.raises(Object6DBoxContractError, match="byte identity"):
        build_box_authority(contract)

    def template(role: dict[str, object]) -> None:
        role["template_only"] = True

    contract, _ = _fixture(tmp_path / "template", role_mutator=template)
    with pytest.raises(Object6DBoxContractError, match="instantiated/frozen"):
        build_box_authority(contract)


def test_intrinsics_c2w_scale_and_axes_are_mandatory(tmp_path: Path) -> None:
    def bad_k(arrays: dict[str, np.ndarray]) -> None:
        arrays["K"][0, 0] = 0.0

    contract, _ = _fixture(tmp_path / "k", bundle_mutator=bad_k)
    with pytest.raises(Object6DBoxContractError, match="pinhole intrinsic"):
        build_box_authority(contract)

    def bad_c2w(arrays: dict[str, np.ndarray]) -> None:
        arrays["T_world_camera"][0, 0, 0] = -1.0

    contract, _ = _fixture(tmp_path / "c2w", bundle_mutator=bad_c2w)
    with pytest.raises(Object6DBoxContractError, match="right handed"):
        build_box_authority(contract)

    def bad_scale(role: dict[str, object]) -> None:
        instances = role["instances"]
        assert isinstance(instances, list)
        instances[0]["mesh_scale_to_m"] = 0.0

    contract, _ = _fixture(tmp_path / "scale", role_mutator=bad_scale)
    with pytest.raises(Object6DBoxContractError, match="mesh_scale_to_m"):
        build_box_authority(contract)

    def bad_axes(role: dict[str, object]) -> None:
        instances = role["instances"]
        assert isinstance(instances, list)
        instances[0]["axis_definition"] = "left_handed"

    contract, _ = _fixture(tmp_path / "axes", role_mutator=bad_axes)
    with pytest.raises(Object6DBoxContractError, match="capture/role geometry"):
        build_box_authority(contract)


def test_reprojection_gate_marks_row_unknown_without_fallback(tmp_path: Path) -> None:
    def bad_reprojection(arrays: dict[str, np.ndarray]) -> None:
        arrays["reprojection_rmse_px"][1, 1] = 4.01

    contract, _ = _fixture(tmp_path, bundle_mutator=bad_reprojection)
    result = build_box_authority(contract)
    failed = [row for row in result["rows"] if row["reasons"]]
    assert result["consumption_authorized"] is False
    assert len(failed) == 1
    assert failed[0]["reasons"] == ["REPROJECTION_GATE"]
    assert failed[0]["bbox_xyxy"] is None
