from __future__ import annotations

import argparse
import ast
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.deterministic_table_reprojection_clean import (  # noqa: E402
    Camera,
    DONOR_ARM_MARGIN_METRIC,
    DONOR_ARM_MARGIN_MIN_DISTANCE_PX,
    DONOR_ARM_MARGIN_NEAREST_PIXEL_QUANTIZATION_MAX_PX,
    DONOR_ARM_MARGIN_PRIOR_SUM_PX,
    DONOR_ARM_MARGIN_RULE,
    MAGENTA_RGB,
    StaticPlane,
    TableReprojectionError,
    _donor_pixels_meet_global_arm_margin,
    apply_batch,
    decode_pixel_provenance_npz,
    deterministic_donor_order,
    donor_arm_margin_contract,
    encode_pixel_provenance_npz,
    initialize_clean,
    plane_homography,
    reproject_unresolved_from_donor,
    reproject_unresolved_from_verified_table_donor,
    target_xy_with_verified_table_support,
)
from tools import run_deterministic_table_reprojection_clean as runner  # noqa: E402


class BadArrayConversion:
    def __array__(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise ValueError("synthetic array conversion failure")


def camera(*, center_x: float = 0.0, width: int = 8, height: int = 6) -> Camera:
    intrinsic = np.asarray(
        [[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[0, 3] = center_x
    return Camera(intrinsic, camera_to_world, width, height)


def plane() -> StaticPlane:
    return StaticPlane(np.asarray([0.0, 0.0, 1.0]), -2.0)


def _write_payload(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: object) -> dict[str, object]:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return _write_payload(path, payload)


def _write_png(path: Path, value: np.ndarray, mode: str) -> dict[str, object]:
    output = BytesIO()
    image = Image.fromarray(value)
    if image.mode != mode:
        image = image.convert(mode)
    image.save(output, format="PNG")
    return _write_payload(path, output.getvalue())


def _formal_fixture(
    root: Path,
    *,
    overlap_exclusion: bool = False,
    lineage_raw_mismatch: bool = False,
    residual_p95_m: float = 0.005,
) -> tuple[Path, dict[str, object]]:
    session = "grap_a_cap_004"
    cohort = "synthetic-cohort-v1"
    frame_count = 15
    binding = {
        "session_id": session,
        "physical_capture_id": session,
        "cohort_id": cohort,
    }
    plane_measurement_target = {
        "schema_version": runner.FORMAL_PLANE_TAG_BINDING_SCHEMA,
        "physical_binding": binding,
        "tag_id": 7,
        "tag_size_m": 0.16,
        "mounting_semantics": runner.FORMAL_PLANE_TAG_MOUNTING,
    }
    zero_counters = {
        "morphology_calls": 0,
        "fill_convex_poly_calls": 0,
        "erosion_calls": 0,
        "polygon_rasterization_calls": 0,
        "temporal_propagation_calls": 0,
        "heuristic_calls": 0,
        "hidden_fallback_calls": 0,
    }
    source_code_ref = _write_payload(
        root / "final_h_producer.py", b"print('synthetic final H producer')\n"
    )
    source_path = root / "FINAL_H_SOURCE.json"
    source_evidence: list[dict[str, object]] = []
    for side in ("left", "right"):
        for frame in range(frame_count):
            source_evidence.append(
                {
                    "absolute_path": str(
                        root / f"frame-{frame:05d}" / f"final_h_{side}.png"
                    ),
                    "minimum_bytes": 1,
                    "verification": "PNG_FULL_DECODE_BINARY",
                    "expected_width": 8,
                    "expected_height": 6,
                }
            )
    source_evidence.append(
        {
            "absolute_path": str(source_path),
            "minimum_bytes": 1,
            "verification": "JSON_PARSE_EXPECTED_KEYS",
            "expected_keys": ["status", "session_id", "frame_count"],
        }
    )
    source_manifest = {
        "schema_version": "eligible68-mask-producer-adapter-v1",
        "status": "ARTIFACT_EXISTS",
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": session,
        "physical_binding": binding,
        "frame_count": frame_count,
        "left_right_png_count": 2 * frame_count,
        "formal_admission_eligible": True,
        "model_called": True,
        "gpu_started": True,
        "pixels_created_or_modified": True,
        "strict_dual": True,
        "d4_applied": True,
        "d4_scope": "GLOBAL_ALL_REQUESTED_SESSIONS_NO_SESSION_BRANCH",
        "d3_candidate_provenance_complete": True,
        "wearable_w1_w4_applied": True,
        "wearable_combination_rule": runner.FORMAL_WEARABLE_COMBINATION,
        "wearable_detection_method": runner.FORMAL_WEARABLE_DETECTION,
        "prompt_contract_sha256": runner.FORMAL_PROMPT_CONTRACT_SHA256,
        "morphology_operations": 0,
        "fill_operations": 0,
        "fallback_operations": 0,
        "governance_bypassed": [],
        "executed_source": source_code_ref,
        "command_manifest": {
            "command": [sys.executable, str(source_code_ref["path"])],
            "completion_mode": "ARTIFACT_EXISTS",
            "success_evidence": source_evidence,
        },
        "success_evidence": source_evidence,
    }
    source_ref = _write_json(source_path, source_manifest)
    source_digest = str(source_ref["sha256"])

    raster_code_ref = _write_payload(
        root / "raster_producer.py", b"print('synthetic raster producer')\n"
    )
    raster_evidence: list[dict[str, object]] = []
    for frame in range(frame_count):
        for name in (
            "target_support.png",
            "visible_table_identity.png",
            "object_unknown.png",
        ):
            raster_evidence.append(
                {
                    "absolute_path": str(root / f"frame-{frame:05d}" / name),
                    "minimum_bytes": 1,
                    "verification": "PNG_FULL_DECODE_BINARY",
                    "expected_width": 8,
                    "expected_height": 6,
                }
            )
    raster_producer = {
        "schema_version": runner.FORMAL_DIRECT_PRODUCER_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "completion_mode": "ARTIFACT_EXISTS",
        "producer_kind": "PER_FRAME_TABLE_RASTER_DIRECT_OBSERVATION",
        "physical_binding": binding,
        "executed_source": raster_code_ref,
        "command_manifest": {
            "command": [sys.executable, str(raster_code_ref["path"])],
            "completion_mode": "ARTIFACT_EXISTS",
            "success_evidence": raster_evidence,
        },
        "success_evidence": raster_evidence,
        "operation_counters": zero_counters,
    }
    raster_producer_ref = _write_json(root / "RASTER_PRODUCER.json", raster_producer)

    frames: list[dict[str, object]] = []
    corner_observations: list[dict[str, object]] = []
    for frame in range(frame_count):
        frame_root = root / f"frame-{frame:05d}"
        raw = np.full((6, 8, 3), 20 + frame, dtype=np.uint8)
        raw_ref = _write_png(frame_root / "raw.png", raw, "RGB")
        metadata = {
            "metadata": {
                "idx": frame,
                "camera_model": "rectified_pinhole",
                "w": 8,
                "h": 6,
                "fps": 30,
                "k": [[4.0, 0.0, 3.5], [0.0, 4.0, 2.5], [0.0, 0.0, 1.0]],
                "c2w": [
                    [1.0, 0.0, 0.0, frame * 0.001],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        }
        metadata_ref = _write_json(frame_root / "camera.json", metadata)
        intrinsic_source = {
            "artifact": metadata_ref,
            "json_pointer": "/metadata/k",
            "session_id": session,
            "frame_index": frame,
        }
        pose_source = {
            "artifact": metadata_ref,
            "json_pointer": "/metadata/c2w",
            "session_id": session,
            "frame_index": frame,
        }
        left = np.zeros((6, 8), dtype=np.uint8)
        right = np.zeros((6, 8), dtype=np.uint8)
        left[2, 2] = 255
        right[3, 5] = 255
        left_ref = _write_png(frame_root / "final_h_left.png", left, "L")
        right_ref = _write_png(frame_root / "final_h_right.png", right, "L")
        lineage = {
            "schema_version": runner.FORMAL_FINAL_H_LINEAGE_SCHEMA,
            "status": "ARTIFACT_EXISTS",
            "completion_mode": "ARTIFACT_EXISTS",
            "session_id": session,
            "frame_index": frame,
            "physical_binding": binding,
            "source_run_manifest_sha256": source_digest,
            "raw_rgb_sha256": (
                "0" * 64 if lineage_raw_mismatch and frame == 0 else raw_ref["sha256"]
            ),
            "left_final_h_mask_sha256": left_ref["sha256"],
            "right_final_h_mask_sha256": right_ref["sha256"],
            "side_combination": runner.FORMAL_FINAL_H_COMBINATION,
            "wearable_combination_rule": runner.FORMAL_WEARABLE_COMBINATION,
            "wearable_detection_method": runner.FORMAL_WEARABLE_DETECTION,
        }
        lineage_ref = _write_json(frame_root / "final_h_lineage.json", lineage)

        support = np.ones((6, 8), dtype=np.uint8) * 255
        donor_identity = np.ones((6, 8), dtype=np.uint8) * 255
        donor_identity[left > 0] = 0
        donor_identity[right > 0] = 0
        exclusion = np.zeros((6, 8), dtype=np.uint8)
        if overlap_exclusion and frame == 0:
            exclusion[0, 0] = 255
        support_ref = _write_png(frame_root / "target_support.png", support, "L")
        identity_ref = _write_png(
            frame_root / "visible_table_identity.png", donor_identity, "L"
        )
        exclusion_ref = _write_png(frame_root / "object_unknown.png", exclusion, "L")
        role_artifacts = {
            "target_table_support": support_ref,
            "visible_table_donor_identity": identity_ref,
            "object_unknown_exclusion": exclusion_ref,
        }
        raster_authority = {
            "schema_version": runner.FORMAL_TABLE_RASTER_AUTHORITY_SCHEMA,
            "status": "ARTIFACT_EXISTS",
            "completion_mode": "ARTIFACT_EXISTS",
            "session_id": session,
            "frame_index": frame,
            "physical_binding": binding,
            "target_table_support_sha256": support_ref["sha256"],
            "visible_table_donor_identity_sha256": identity_ref["sha256"],
            "object_unknown_exclusion_sha256": exclusion_ref["sha256"],
            "target_support_semantics": runner.FORMAL_TARGET_SUPPORT,
            "donor_identity_semantics": runner.FORMAL_DONOR_IDENTITY,
            "object_unknown_exclusion_semantics": runner.FORMAL_OBJECT_UNKNOWN_EXCLUSION,
            "operation_counters": zero_counters,
            "producer_manifest": raster_producer_ref,
            "role_provenance": {
                role: {
                    "schema_version": runner.FORMAL_RASTER_ROLE_PROVENANCE_SCHEMA,
                    "source_kind": "DIRECT_FRAME_OBSERVATION_NO_INFERENCE",
                    "session_id": session,
                    "frame_index": frame,
                    "physical_binding": binding,
                    "role": role,
                    "raw_rgb_sha256": raw_ref["sha256"],
                    "raster_sha256": artifact["sha256"],
                    "producer_manifest_sha256": raster_producer_ref["sha256"],
                    "operation_counters": zero_counters,
                }
                for role, artifact in role_artifacts.items()
            },
        }
        raster_authority_ref = _write_json(
            frame_root / "table_raster_authority.json", raster_authority
        )
        frame_record = {
            "frame_index": frame,
            "physical_binding": binding,
            "raw_rgb": raw_ref,
            "camera_intrinsics": intrinsic_source,
            "camera_to_world": pose_source,
            "final_h_left": left_ref,
            "final_h_right": right_ref,
            "final_h_lineage": lineage_ref,
            "target_table_support": support_ref,
            "visible_table_donor_identity": identity_ref,
            "object_unknown_exclusion": exclusion_ref,
            "table_raster_authority": raster_authority_ref,
        }
        frames.append(frame_record)
        camera_value = np.asarray(metadata["metadata"]["c2w"], dtype=np.float64)
        intrinsic_value = np.asarray(metadata["metadata"]["k"], dtype=np.float64)
        for corner_id, (x_value, z_value) in enumerate(
            ((-0.08, 4.0), (0.08, 4.0), (0.08, 4.16), (-0.08, 4.16))
        ):
            point_world = np.asarray(
                [x_value, -2.0 - residual_p95_m, z_value], dtype=np.float64
            )
            point_camera = camera_value[:3, :3].T @ (point_world - camera_value[:3, 3])
            projected = intrinsic_value @ point_camera
            observed_xy = projected[:2] / projected[2]
            corner_observations.append(
                {
                    "frame_index": frame,
                    "tag_id": 7,
                    "corner_id": corner_id,
                    "raw_rgb_sha256": raw_ref["sha256"],
                    "camera_intrinsics_sha256": metadata_ref["sha256"],
                    "camera_to_world_sha256": metadata_ref["sha256"],
                    "observed_image_xy": observed_xy.tolist(),
                    "point_world_m": point_world.tolist(),
                    "reprojection_error_px": 0.0,
                    "residual_abs_m": residual_p95_m,
                }
            )

    plane_measurement = {
        "schema_version": runner.FORMAL_PLANE_MEASUREMENT_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": session,
        "physical_binding": binding,
        "plane_id": "session_static_table_v1",
        "measurement_frame_selection_rule": "FROZEN_PREDECLARED_15_FRAME_INVENTORY_NO_RESULT_SELECTION",
        "measurement_target": plane_measurement_target,
        "corner_observations": corner_observations,
    }
    plane_measurement_ref = _write_json(
        root / "PLANE_CORNER_EVIDENCE.json", plane_measurement
    )
    plane_code_ref = _write_payload(
        root / "plane_producer.py", b"print('synthetic plane producer')\n"
    )
    plane_success_evidence = [
        {
            "absolute_path": str(plane_measurement_ref["path"]),
            "minimum_bytes": 1,
            "verification": "JSON_PARSE_EXPECTED_KEYS",
            "expected_keys": [
                "schema_version",
                "status",
                "completion_mode",
                "session_id",
                "corner_observations",
            ],
        }
    ]
    plane_producer = {
        "schema_version": runner.FORMAL_DIRECT_PRODUCER_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "completion_mode": "ARTIFACT_EXISTS",
        "producer_kind": "SESSION_STATIC_TABLE_PLANE_DIRECT_MEASUREMENT",
        "physical_binding": binding,
        "executed_source": plane_code_ref,
        "command_manifest": {
            "command": [sys.executable, str(plane_code_ref["path"])],
            "completion_mode": "ARTIFACT_EXISTS",
            "success_evidence": plane_success_evidence,
        },
        "success_evidence": plane_success_evidence,
        "operation_counters": zero_counters,
    }
    plane_producer_ref = _write_json(root / "PLANE_PRODUCER.json", plane_producer)
    success_evidence = [
        {
            "absolute_path": str(source_ref["path"]),
            "minimum_bytes": 1,
            "verification": "JSON_PARSE_EXPECTED_KEYS",
            "expected_keys": ["status", "session_id", "frame_count"],
        }
    ]
    manifest = {
        "schema_version": runner.FORMAL_AUTHORITY_SCHEMA,
        "status": "ARTIFACT_EXISTS",
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": session,
        "session_frame_count": frame_count,
        "physical_binding": binding,
        "final_h_source_manifest": source_ref,
        "plane_authority": {
            "source_kind": runner.FORMAL_PLANE_SOURCE,
            "session_id": session,
            "physical_binding": binding,
            "coordinate_frame": runner.FORMAL_PLANE_COORDINATE_FRAME,
            "orientation_policy": runner.FORMAL_PLANE_ORIENTATION,
            "plane_id": "session_static_table_v1",
            "normal_world": [0.0, -1.0, 0.0],
            "offset_m": -2.0,
            "cardinality": 1,
            "per_frame_offsets_present": False,
            "producer_manifest": plane_producer_ref,
            "measurement_evidence": plane_measurement_ref,
            "measurement_target": plane_measurement_target,
            "quality_gate": {
                "pass": True,
                "thresholds": runner.FORMAL_PLANE_QUALITY_GATES,
                "metrics": {
                    "anchor_frame_count": 15,
                    "anchor_point_count": 60,
                    "tag_corner_reprojection_max_px": 0.0,
                    "residual_p95_abs_m": residual_p95_m,
                    "residual_max_abs_m": residual_p95_m,
                },
            },
        },
        "frames": frames,
        "operation_contract": {
            "donor_order": "ABS_FRAME_DISTANCE_THEN_FRAME_INDEX",
            "sampling": "NEAREST_PIXEL_NO_INTERPOLATION",
            "target_support_semantics": runner.FORMAL_TARGET_SUPPORT,
            "donor_identity_semantics": runner.FORMAL_DONOR_IDENTITY,
            "object_unknown_exclusion_semantics": runner.FORMAL_OBJECT_UNKNOWN_EXCLUSION,
            "donor_arm_margin": donor_arm_margin_contract(),
            "session_parameter_overrides": [],
            "operation_counters": zero_counters,
        },
        "command_manifest": {
            "command": ["/usr/bin/python3", "/synthetic/authority_producer.py"],
            "completion_mode": "ARTIFACT_EXISTS",
            "success_evidence": success_evidence,
        },
        "success_evidence": success_evidence,
    }
    manifest_path = root / "FORMAL_AUTHORITY.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _rewrite_source_and_lineages(
    manifest_path: Path,
    manifest: dict[str, object],
    mutate: object,
) -> None:
    source_path = Path(manifest["final_h_source_manifest"]["path"])  # type: ignore[index]
    source = json.loads(source_path.read_text())
    assert callable(mutate)
    mutate(source)
    source_ref = _write_json(source_path, source)
    manifest["final_h_source_manifest"] = source_ref
    for frame_record in manifest["frames"]:  # type: ignore[union-attr]
        lineage_path = Path(frame_record["final_h_lineage"]["path"])
        lineage = json.loads(lineage_path.read_text())
        lineage["source_run_manifest_sha256"] = source_ref["sha256"]
        frame_record["final_h_lineage"] = _write_json(lineage_path, lineage)
    _write_json(manifest_path, manifest)


def _rewrite_plane_measurement(
    manifest_path: Path,
    manifest: dict[str, object],
    mutate: object,
) -> None:
    evidence_path = Path(
        manifest["plane_authority"]["measurement_evidence"]["path"]  # type: ignore[index]
    )
    evidence = json.loads(evidence_path.read_text())
    assert callable(mutate)
    mutate(evidence)
    manifest["plane_authority"]["measurement_evidence"] = _write_json(  # type: ignore[index]
        evidence_path, evidence
    )
    _write_json(manifest_path, manifest)


def test_same_camera_reprojection_is_identity_and_donor_h_is_fail_closed() -> None:
    source = camera()
    rgb = np.arange(source.width * source.height * 3, dtype=np.uint8).reshape(
        source.height, source.width, 3
    )
    donor_h = np.zeros((source.height, source.width), dtype=np.bool_)
    donor_h[2, 3] = True
    target_xy = np.asarray([[0, 0], [3, 2], [7, 5]], dtype=np.int64)

    batch = reproject_unresolved_from_donor(
        target_xy=target_xy,
        target_camera=source,
        donor_camera=source,
        donor_rgb=rgb,
        donor_h_mask=donor_h,
        plane=plane(),
    )

    assert np.array_equal(batch.target_xy, np.asarray([[0, 0], [7, 5]]))
    assert np.array_equal(batch.donor_xy, np.asarray([[0, 0], [7, 5]]))
    assert np.array_equal(batch.donor_rgb, rgb[[0, 5], [0, 7]])
    assert batch.margin_rejected_candidate_pixel_count == 1
    assert np.allclose(batch.homography / batch.homography[2, 2], np.eye(3))
    assert len(batch.homography_sha256) == 64


def test_global_arm_margin_accepts_n_and_n_plus_but_rejects_n_epsilon() -> None:
    source = camera(width=12, height=12)
    rgb = np.arange(source.width * source.height * 3, dtype=np.uint8).reshape(
        source.height, source.width, 3
    )
    donor_h = np.zeros((source.height, source.width), dtype=np.bool_)
    donor_h[4, 4] = True
    # Distances from (4, 4): sqrt(8) = N-epsilon, 3 = N, sqrt(10) = N+.
    target_xy = np.asarray([[6, 6], [7, 4], [7, 5]], dtype=np.int64)

    batch = reproject_unresolved_from_donor(
        target_xy=target_xy,
        target_camera=source,
        donor_camera=source,
        donor_rgb=rgb,
        donor_h_mask=donor_h,
        plane=plane(),
    )

    assert DONOR_ARM_MARGIN_MIN_DISTANCE_PX == 3
    assert np.array_equal(batch.target_xy, target_xy[1:])
    assert np.array_equal(batch.donor_xy, target_xy[1:])
    assert batch.margin_rejected_candidate_pixel_count == 1


@pytest.mark.parametrize(
    "bad_xy",
    [
        np.asarray([[2, 2]], dtype=np.uint64),
        np.asarray([[True, False]], dtype=np.bool_),
        np.asarray([[2, 2]], dtype=object),
        [[1, 2], [3]],
        np.asarray([[-1, 2]], dtype=np.int64),
        np.asarray([[12, 2]], dtype=np.int64),
        np.asarray([[np.iinfo(np.int64).max, 2]], dtype=np.int64),
    ],
)
def test_arm_margin_coordinate_validation_is_fail_closed(bad_xy: object) -> None:
    donor_h = np.zeros((12, 12), dtype=np.bool_)
    donor_h[4, 4] = True

    with pytest.raises(TableReprojectionError, match="donor_xy"):
        _donor_pixels_meet_global_arm_margin(bad_xy, donor_h)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_mask",
    [
        [[True, False], [False]],
        BadArrayConversion(),
    ],
)
def test_arm_margin_mask_conversion_is_fail_closed(bad_mask: object) -> None:
    with pytest.raises(TableReprojectionError, match="donor H mask"):
        _donor_pixels_meet_global_arm_margin(  # type: ignore[arg-type]
            np.asarray([[1, 1]], dtype=np.int64), bad_mask
        )


@pytest.mark.parametrize(
    "bad_target_xy",
    [
        np.asarray([[2**64 - 1, 2]], dtype=np.uint64),
        np.asarray([[1.0, 2.0]], dtype=np.float64),
        np.asarray([[True, False]], dtype=np.bool_),
        np.asarray([[1, 2]], dtype=object),
        [[1, 2], [3]],
        np.asarray([[-1, 2]], dtype=np.int64),
        np.asarray([[8, 2]], dtype=np.int64),
        np.asarray([[1, 6]], dtype=np.int64),
        np.asarray([[np.iinfo(np.int64).max, 2]], dtype=np.int64),
    ],
)
def test_public_reprojection_target_coordinates_are_fail_closed(
    bad_target_xy: object,
) -> None:
    source = camera()
    donor_rgb = np.zeros((source.height, source.width, 3), dtype=np.uint8)
    donor_h = np.zeros((source.height, source.width), dtype=np.bool_)

    with pytest.raises(TableReprojectionError, match="target_xy"):
        reproject_unresolved_from_donor(
            target_xy=bad_target_xy,  # type: ignore[arg-type]
            target_camera=source,
            donor_camera=camera(center_x=-2.0),
            donor_rgb=donor_rgb,
            donor_h_mask=donor_h,
            plane=plane(),
        )


def test_unsigned_target_wrap_cannot_write_the_last_target_column() -> None:
    raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    target_h = np.zeros((6, 8), dtype=np.bool_)
    target_h[2, 7] = True
    clean, unresolved = initialize_clean(raw, target_h)
    clean_before = clean.copy()
    unresolved_before = unresolved.copy()

    with pytest.raises(TableReprojectionError, match="target_xy"):
        reproject_unresolved_from_donor(
            target_xy=np.asarray([[2**64 - 1, 2]], dtype=np.uint64),
            target_camera=camera(),
            donor_camera=camera(center_x=-2.0),
            donor_rgb=np.full((6, 8, 3), 91, dtype=np.uint8),
            donor_h_mask=np.zeros((6, 8), dtype=np.bool_),
            plane=plane(),
        )

    assert np.array_equal(clean, clean_before)
    assert np.array_equal(unresolved, unresolved_before)


def test_public_donor_array_conversion_errors_are_fail_closed() -> None:
    source = camera()
    valid_rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    valid_mask = np.zeros((6, 8), dtype=np.bool_)
    target_xy = np.asarray([[1, 1]], dtype=np.int64)

    with pytest.raises(TableReprojectionError, match="donor RGB"):
        reproject_unresolved_from_donor(
            target_xy=target_xy,
            target_camera=source,
            donor_camera=source,
            donor_rgb=BadArrayConversion(),  # type: ignore[arg-type]
            donor_h_mask=valid_mask,
            plane=plane(),
        )
    with pytest.raises(TableReprojectionError, match="donor H mask"):
        reproject_unresolved_from_donor(
            target_xy=target_xy,
            target_camera=source,
            donor_camera=source,
            donor_rgb=valid_rgb,
            donor_h_mask=BadArrayConversion(),  # type: ignore[arg-type]
            plane=plane(),
        )
    with pytest.raises(TableReprojectionError, match="donor table identity"):
        reproject_unresolved_from_verified_table_donor(
            target_xy=target_xy,
            target_camera=source,
            donor_camera=source,
            donor_rgb=valid_rgb,
            donor_h_mask=valid_mask,
            donor_table_identity_mask=BadArrayConversion(),  # type: ignore[arg-type]
            plane=plane(),
        )


def test_margin_uses_current_donor_frame_left_right_union(
    tmp_path: Path,
) -> None:
    mask_root = tmp_path / "donor-mask-frames"
    left = np.zeros((12, 12), dtype=np.uint8)
    right = np.zeros_like(left)
    left[1, 1] = 255
    right[8, 8] = 255
    _write_png(mask_root / "left" / "frame_00007.png", left, "L")
    _write_png(mask_root / "right" / "frame_00007.png", right, "L")

    donor_h, _ = runner.combined_mask(7, mask_root, (12, 12))
    source = camera(width=12, height=12)
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    target_xy = np.asarray([[3, 1], [8, 6], [4, 1]], dtype=np.int64)
    batch = reproject_unresolved_from_donor(
        target_xy=target_xy,
        target_camera=source,
        donor_camera=source,
        donor_rgb=rgb,
        donor_h_mask=donor_h,
        plane=plane(),
    )

    assert np.array_equal(donor_h, (left | right).astype(np.bool_))
    assert np.array_equal(batch.target_xy, np.asarray([[4, 1]], dtype=np.int64))
    assert batch.margin_rejected_candidate_pixel_count == 2


def test_homography_matches_direct_plane_projection_for_translated_camera() -> None:
    target = camera(center_x=0.0)
    donor = camera(center_x=0.25)
    point = np.asarray([4.0, 3.0, 1.0])
    matrix = plane_homography(target, donor, plane())
    mapped = matrix @ point
    mapped /= mapped[2]

    rgb = np.zeros((donor.height, donor.width, 3), dtype=np.uint8)
    donor_h = np.zeros((donor.height, donor.width), dtype=np.bool_)
    batch = reproject_unresolved_from_donor(
        target_xy=np.asarray([[4, 3]], dtype=np.int64),
        target_camera=target,
        donor_camera=donor,
        donor_rgb=rgb,
        donor_h_mask=donor_h,
        plane=plane(),
    )

    assert len(batch.target_xy) == 1
    assert np.array_equal(batch.donor_xy[0], np.rint(mapped[:2]).astype(np.int64))


def test_unresolved_stays_exact_magenta_and_apply_changes_only_target() -> None:
    raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    target_h = np.zeros((6, 8), dtype=np.bool_)
    target_h[1, 1] = True
    target_h[4, 5] = True
    clean, unresolved = initialize_clean(raw, target_h)
    assert np.array_equal(clean[~target_h], raw[~target_h])
    assert np.all(clean[target_h] == MAGENTA_RGB)

    donor_rgb = np.full((6, 8, 3), 17, dtype=np.uint8)
    donor_h = np.zeros((6, 8), dtype=np.bool_)
    donor_h[4, 5] = True
    batch = reproject_unresolved_from_donor(
        target_xy=np.asarray([[1, 1], [5, 4]], dtype=np.int64),
        target_camera=camera(),
        donor_camera=camera(),
        donor_rgb=donor_rgb,
        donor_h_mask=donor_h,
        plane=plane(),
    )
    apply_batch(clean, unresolved, batch)

    assert np.array_equal(clean[~target_h], raw[~target_h])
    assert np.array_equal(clean[1, 1], [17, 17, 17])
    assert np.array_equal(clean[4, 5], MAGENTA_RGB)
    assert not unresolved[1, 1]
    assert unresolved[4, 5]
    assert batch.margin_rejected_candidate_pixel_count == 1


def test_all_margin_rejected_donors_leave_unresolved_magenta_without_fallback() -> None:
    raw = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    target_h = np.zeros((6, 8), dtype=np.bool_)
    target_h[2, 3] = True
    clean, unresolved = initialize_clean(raw, target_h)

    rejected_count = 0
    for donor_value, arm_xy in ((19, (3, 2)), (23, (4, 2))):
        donor_rgb = np.full((6, 8, 3), donor_value, dtype=np.uint8)
        donor_h = np.zeros((6, 8), dtype=np.bool_)
        donor_h[arm_xy[1], arm_xy[0]] = True
        batch = reproject_unresolved_from_donor(
            target_xy=np.asarray([[3, 2]], dtype=np.int64),
            target_camera=camera(),
            donor_camera=camera(),
            donor_rgb=donor_rgb,
            donor_h_mask=donor_h,
            plane=plane(),
        )
        rejected_count += batch.margin_rejected_candidate_pixel_count
        apply_batch(clean, unresolved, batch)

    assert rejected_count == 2
    assert unresolved[2, 3]
    assert np.array_equal(clean[2, 3], MAGENTA_RGB)
    assert np.array_equal(clean[~target_h], raw[~target_h])


def test_donor_order_is_fixed_before_pixel_results() -> None:
    assert deterministic_donor_order(10, [13, 8, 9, 11, 10, 7]) == (9, 11, 8, 7, 13)


def test_global_arm_margin_contract_records_frozen_apriori_basis() -> None:
    contract = donor_arm_margin_contract()

    assert contract["rule"] == DONOR_ARM_MARGIN_RULE
    assert contract["metric"] == DONOR_ARM_MARGIN_METRIC
    assert contract["minimum_distance_px"] == 3
    assert contract["basis"] == {
        "selection": "A_PRIORI_GEOMETRY_UNCERTAINTY_NOT_CLEAN_RESULT_SCAN",
        "mask_boundary_tolerance_px": 1.0,
        "formal_reprojection_max_px": 1.0,
        "nearest_pixel_quantization_max_px": (
            DONOR_ARM_MARGIN_NEAREST_PIXEL_QUANTIZATION_MAX_PX
        ),
        "prior_sum_px": DONOR_ARM_MARGIN_PRIOR_SUM_PX,
        "rounding": "CEIL_TO_INTEGER_PIXEL_DISTANCE",
    }
    assert DONOR_ARM_MARGIN_PRIOR_SUM_PX == pytest.approx(2.7071067811865475)


def test_compressed_pixel_provenance_is_lossless_and_non_pickle() -> None:
    target_xy = np.asarray([[4, 1], [7, 5], [2, 3]], dtype=np.int64)
    donor_frame = np.asarray([17, 9, 17], dtype=np.int64)
    donor_xy = np.asarray([[5, 1], [6, 5], [3, 3]], dtype=np.int64)

    payload = encode_pixel_provenance_npz(target_xy, donor_frame, donor_xy)
    decoded = decode_pixel_provenance_npz(payload)

    assert len(payload) > 0
    assert len(hashlib.sha256(payload).hexdigest()) == 64
    assert decoded.target_xy.dtype == np.dtype("<i4")
    assert decoded.donor_frame.dtype == np.dtype("<i4")
    assert decoded.donor_xy.dtype == np.dtype("<i4")
    assert np.array_equal(decoded.target_xy, target_xy)
    assert np.array_equal(decoded.donor_frame, donor_frame)
    assert np.array_equal(decoded.donor_xy, donor_xy)
    assert not decoded.target_xy.flags.writeable
    assert not decoded.donor_frame.flags.writeable
    assert not decoded.donor_xy.flags.writeable


def test_compressed_pixel_provenance_rejects_duplicate_targets() -> None:
    target_xy = np.asarray([[4, 1], [4, 1]], dtype=np.int64)
    donor_frame = np.asarray([17, 18], dtype=np.int64)
    donor_xy = np.asarray([[5, 1], [6, 1]], dtype=np.int64)

    with np.testing.assert_raises_regex(TableReprojectionError, "duplicate pixels"):
        encode_pixel_provenance_npz(target_xy, donor_frame, donor_xy)


def test_formal_target_pixels_cannot_exceed_verified_table_support() -> None:
    unresolved = np.zeros((6, 8), dtype=np.bool_)
    unresolved[1, 1] = True
    unresolved[2, 3] = True
    support = np.zeros_like(unresolved)
    support[2, 3] = True

    selected = target_xy_with_verified_table_support(unresolved, support)

    assert np.array_equal(selected, np.asarray([[3, 2]], dtype=np.int64))


def test_formal_donor_rejects_outside_identity_and_human_pixels() -> None:
    source = camera()
    rgb = np.arange(source.width * source.height * 3, dtype=np.uint8).reshape(
        source.height, source.width, 3
    )
    donor_h = np.zeros((source.height, source.width), dtype=np.bool_)
    donor_h[2, 3] = True
    donor_identity = np.zeros_like(donor_h)
    donor_identity[0, 0] = True
    donor_identity[2, 3] = True
    target_xy = np.asarray([[0, 0], [3, 2], [7, 5]], dtype=np.int64)

    batch = reproject_unresolved_from_verified_table_donor(
        target_xy=target_xy,
        target_camera=source,
        donor_camera=source,
        donor_rgb=rgb,
        donor_h_mask=donor_h,
        donor_table_identity_mask=donor_identity,
        plane=plane(),
    )

    assert np.array_equal(batch.target_xy, np.asarray([[0, 0]], dtype=np.int64))
    assert np.array_equal(batch.donor_xy, np.asarray([[0, 0]], dtype=np.int64))
    assert batch.margin_rejected_candidate_pixel_count == 1


def test_formal_authority_whole_session_preflight_accepts_exact_fixture(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _formal_fixture(tmp_path / "valid")

    bundle = runner.preflight_formal_authority(
        manifest_path,
        expected_session_id="grap_a_cap_004",
        expected_session_frame_count=15,
        expected_cohort_id="synthetic-cohort-v1",
    )

    assert bundle.session_id == "grap_a_cap_004"
    assert bundle.cohort_id == "synthetic-cohort-v1"
    assert len(bundle.frames) == 15
    assert [frame.frame_index for frame in bundle.frames] == list(range(15))
    assert bundle.plane.plane_id == "session_static_table_v1"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("minimum_distance_px", 2),
        ("minimum_distance_px", 3.0),
        ("metric", "PIXEL_CENTER_L1"),
        ("rule", "SESSION_SELECTED_MARGIN"),
    ],
)
def test_formal_authority_rejects_any_donor_margin_contract_override(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / field)
    manifest["operation_contract"]["donor_arm_margin"][field] = bad_value  # type: ignore[index]
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="operation contract drift"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_formal_authority_rejects_numeric_bool_in_margin_basis(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "bool-basis")
    manifest["operation_contract"]["donor_arm_margin"]["basis"][  # type: ignore[index]
        "mask_boundary_tolerance_px"
    ] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="operation contract drift"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_formal_authority_rejects_session_parameter_override(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "session-override")
    manifest["operation_contract"]["session_parameter_overrides"] = [  # type: ignore[index]
        {"donor_arm_margin_minimum_distance_px": 4}
    ]
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="operation contract drift"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_pre_margin_formal_authority_schema_fails_closed(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "old-schema")
    manifest["schema_version"] = "deterministic-table-clean-formal-authority-v1"
    del manifest["operation_contract"]["donor_arm_margin"]  # type: ignore[index]
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="identity/status mismatch"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"overlap_exclusion": True}, "overlaps object/unknown"),
        ({"lineage_raw_mismatch": True}, "final-H lineage mismatch"),
        ({"residual_p95_m": 0.011}, "quality gates"),
    ],
)
def test_formal_authority_rejects_semantically_invalid_inputs(
    tmp_path: Path, fixture_kwargs: dict[str, object], message: str
) -> None:
    manifest_path, _ = _formal_fixture(
        tmp_path / message.replace("/", "-"), **fixture_kwargs
    )

    with pytest.raises(runner.CleanRunError, match=message):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_formal_authority_rejects_hard_linked_reference(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "hard-link")
    source = Path(manifest["frames"][0]["target_table_support"]["path"])  # type: ignore[index]
    os.link(source, source.with_name("second-link.png"))

    with pytest.raises(runner.CleanRunError, match="single-link"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_formal_authority_rejects_symlinked_reference(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "symbolic-link")
    source = Path(manifest["frames"][0]["final_h_right"]["path"])  # type: ignore[index]
    moved = source.with_name("real-right.png")
    source.rename(moved)
    source.symlink_to(moved)

    with pytest.raises(runner.CleanRunError, match="cannot be opened no-follow"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_invalid_formal_authority_fails_before_output_directory(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "tampered")
    source = Path(manifest["frames"][0]["final_h_left"]["path"])  # type: ignore[index]
    source.write_bytes(source.read_bytes() + b"tamper")
    output = tmp_path / "must-not-exist"
    args = argparse.Namespace(
        execution_mode="FORMAL_AUTHORITY",
        formal_authority_manifest=manifest_path,
        expected_cohort_id="synthetic-cohort-v1",
        raw_root=None,
        mask_root=None,
        mask_source_manifest=None,
        plane=None,
        session_id="grap_a_cap_004",
        output_root=output,
        start_frame=0,
        frame_count=15,
        session_frame_count=15,
    )

    with pytest.raises(runner.CleanRunError, match="bytes/SHA-256 mismatch"):
        runner.run(args)
    assert not output.exists()


def test_missing_formal_authority_fails_before_output_directory(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "missing")
    source = Path(manifest["frames"][0]["object_unknown_exclusion"]["path"])  # type: ignore[index]
    source.unlink()
    output = tmp_path / "must-not-exist-missing"
    args = argparse.Namespace(
        execution_mode="FORMAL_AUTHORITY",
        formal_authority_manifest=manifest_path,
        expected_cohort_id="synthetic-cohort-v1",
        raw_root=None,
        mask_root=None,
        mask_source_manifest=None,
        plane=None,
        session_id="grap_a_cap_004",
        output_root=output,
        start_frame=0,
        frame_count=15,
        session_frame_count=15,
    )

    with pytest.raises(runner.CleanRunError, match="cannot be opened no-follow"):
        runner.run(args)
    assert not output.exists()


def test_formal_authority_requires_exact_ordered_frame_coverage(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "coverage")
    manifest["frames"] = manifest["frames"][:-1]  # type: ignore[index]
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="coverage count"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_formal_authority_rejects_empty_payload_command(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "empty-command")
    manifest["command_manifest"]["command"] = ["/usr/bin/false"]  # type: ignore[index]
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="forbidden empty payload"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_final_h_source_evidence_must_cover_exact_artifacts(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "source-evidence")

    def mutate(source: dict[str, object]) -> None:
        source["success_evidence"] = [
            {"absolute_path": "/synthetic-does-not-exist", "minimum_bytes": 1}
        ]
        source["command_manifest"]["success_evidence"] = source[  # type: ignore[index]
            "success_evidence"
        ]

    _rewrite_source_and_lineages(manifest_path, manifest, mutate)

    with pytest.raises(runner.CleanRunError, match="does not cover its exact masks"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_final_h_source_evidence_requires_explicit_verification_method(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "source-evidence-method")

    def mutate(source: dict[str, object]) -> None:
        del source["success_evidence"][0]["verification"]  # type: ignore[index]
        source["command_manifest"]["success_evidence"] = source[  # type: ignore[index]
            "success_evidence"
        ]

    _rewrite_source_and_lineages(manifest_path, manifest, mutate)
    with pytest.raises(runner.CleanRunError, match="fields mismatch"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_final_h_source_command_must_bind_its_executed_code(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "source-code")

    def mutate(source: dict[str, object]) -> None:
        source["command_manifest"]["command"] = [  # type: ignore[index]
            sys.executable,
            "/different/unbound.py",
        ]

    _rewrite_source_and_lineages(manifest_path, manifest, mutate)

    with pytest.raises(runner.CleanRunError, match="does not execute its bound source"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_final_h_source_physical_cohort_binding_is_not_self_selectable(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "source-binding")

    def mutate(source: dict[str, object]) -> None:
        source["physical_binding"]["cohort_id"] = "different-cohort"  # type: ignore[index]

    _rewrite_source_and_lineages(manifest_path, manifest, mutate)

    with pytest.raises(runner.CleanRunError, match="physical/cohort binding mismatch"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_plane_corner_metrics_must_be_reproducible_from_bound_raw_k_c2w(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "plane-recompute")
    evidence_path = Path(
        manifest["plane_authority"]["measurement_evidence"]["path"]  # type: ignore[index]
    )
    evidence = json.loads(evidence_path.read_text())
    evidence["corner_observations"][0]["reprojection_error_px"] = 0.75
    evidence_ref = _write_json(evidence_path, evidence)
    manifest["plane_authority"]["measurement_evidence"] = evidence_ref  # type: ignore[index]
    producer_path = Path(
        manifest["plane_authority"]["producer_manifest"]["path"]  # type: ignore[index]
    )
    producer = json.loads(producer_path.read_text())
    producer["success_evidence"][0]["absolute_path"] = str(evidence_path)
    producer["command_manifest"]["success_evidence"] = producer["success_evidence"]
    manifest["plane_authority"]["producer_manifest"] = _write_json(  # type: ignore[index]
        producer_path, producer
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="not reproducible"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_plane_accepts_one_globally_bound_nonzero_tag_id(tmp_path: Path) -> None:
    manifest_path, _ = _formal_fixture(tmp_path / "plane-tag-seven")
    runner.preflight_formal_authority(
        manifest_path,
        expected_session_id="grap_a_cap_004",
        expected_session_frame_count=15,
        expected_cohort_id="synthetic-cohort-v1",
    )


def test_plane_rejects_mixed_tag_ids(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "plane-tag-mixed")

    def mutate(evidence: dict[str, object]) -> None:
        evidence["corner_observations"][0]["tag_id"] = 8  # type: ignore[index]

    _rewrite_plane_measurement(manifest_path, manifest, mutate)
    with pytest.raises(runner.CleanRunError, match="one bound tag"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


@pytest.mark.parametrize("invalid_tag_id", [-1, True])
def test_plane_rejects_negative_or_boolean_tag_id(
    tmp_path: Path, invalid_tag_id: object
) -> None:
    manifest_path, manifest = _formal_fixture(
        tmp_path / f"plane-tag-invalid-{invalid_tag_id!r}"
    )

    def mutate(evidence: dict[str, object]) -> None:
        evidence["corner_observations"][0]["tag_id"] = invalid_tag_id  # type: ignore[index]

    _rewrite_plane_measurement(manifest_path, manifest, mutate)
    with pytest.raises(runner.CleanRunError, match="tag/corner identity is malformed"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_plane_tag_physical_cohort_binding_cannot_drift(tmp_path: Path) -> None:
    manifest_path, _ = _formal_fixture(tmp_path / "plane-tag-binding")
    manifest = json.loads(manifest_path.read_text())
    target = manifest["plane_authority"]["measurement_target"]
    target["physical_binding"] = {
        **target["physical_binding"],
        "cohort_id": "different-cohort",
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="physical/cohort binding mismatch"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_plane_requires_exactly_sixty_bound_corner_observations(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "plane-cardinality")
    evidence_path = Path(
        manifest["plane_authority"]["measurement_evidence"]["path"]  # type: ignore[index]
    )
    evidence = json.loads(evidence_path.read_text())
    evidence["corner_observations"] = evidence["corner_observations"][:-1]
    manifest["plane_authority"]["measurement_evidence"] = _write_json(  # type: ignore[index]
        evidence_path, evidence
    )
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="exactly 60 corner observations"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_raster_roles_reject_same_path_and_inode_alias(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "raster-alias")
    frame = manifest["frames"][0]  # type: ignore[index]
    frame["visible_table_donor_identity"] = frame["target_table_support"]
    authority_path = Path(frame["table_raster_authority"]["path"])
    authority = json.loads(authority_path.read_text())
    support_sha = frame["target_table_support"]["sha256"]
    authority["visible_table_donor_identity_sha256"] = support_sha
    authority["role_provenance"]["visible_table_donor_identity"]["raster_sha256"] = (
        support_sha
    )
    frame["table_raster_authority"] = _write_json(authority_path, authority)
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="unique paths and inodes"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_visible_donor_identity_must_not_overlap_final_h(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "donor-visible")
    frame = manifest["frames"][0]  # type: ignore[index]
    identity_path = Path(frame["visible_table_donor_identity"]["path"])
    identity = np.ones((6, 8), dtype=np.uint8) * 255
    identity_ref = _write_png(identity_path, identity, "L")
    frame["visible_table_donor_identity"] = identity_ref
    authority_path = Path(frame["table_raster_authority"]["path"])
    authority = json.loads(authority_path.read_text())
    authority["visible_table_donor_identity_sha256"] = identity_ref["sha256"]
    authority["role_provenance"]["visible_table_donor_identity"]["raster_sha256"] = (
        identity_ref["sha256"]
    )
    frame["table_raster_authority"] = _write_json(authority_path, authority)
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="overlaps final H"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_raster_semantic_strings_without_direct_observation_provenance_are_rejected(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "raster-provenance")
    frame = manifest["frames"][0]  # type: ignore[index]
    authority_path = Path(frame["table_raster_authority"]["path"])
    authority = json.loads(authority_path.read_text())
    authority["role_provenance"]["target_table_support"]["source_kind"] = (
        "SELF_ASSERTED_SEMANTIC_STRING"
    )
    frame["table_raster_authority"] = _write_json(authority_path, authority)
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match="lacks exact direct-observation"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_formal_camera_rejects_left_handed_non_se3_c2w(tmp_path: Path) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "reflection")
    frame = manifest["frames"][0]  # type: ignore[index]
    camera_path = Path(frame["camera_to_world"]["artifact"]["path"])
    camera_value = json.loads(camera_path.read_text())
    camera_value["metadata"]["c2w"][0][0] = -1.0
    camera_ref = _write_json(camera_path, camera_value)
    frame["camera_intrinsics"]["artifact"] = camera_ref
    frame["camera_to_world"]["artifact"] = camera_ref
    _write_json(manifest_path, manifest)

    with pytest.raises(runner.CleanRunError, match=r"right-handed SE\(3\)"):
        runner.preflight_formal_authority(
            manifest_path,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=15,
            expected_cohort_id="synthetic-cohort-v1",
        )


def test_ctime_detects_post_eof_mutation_before_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, manifest = _formal_fixture(tmp_path / "ctime")
    target = Path(manifest["frames"][0]["raw_rgb"]["path"])  # type: ignore[index]
    original = target.read_bytes()
    before = target.stat()
    real_read = os.read
    fired = False

    def attacked_read(descriptor: int, count: int) -> bytes:
        nonlocal fired
        payload = real_read(descriptor, count)
        if not payload and not fired:
            try:
                descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            except OSError:
                descriptor_path = Path("/")
            if descriptor_path == target:
                fired = True
                with target.open("r+b") as handle:
                    handle.write(b"X" * len(original))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        return payload

    monkeypatch.setattr(runner.os, "read", attacked_read)
    output = tmp_path / "must-not-exist-ctime"
    args = argparse.Namespace(
        execution_mode="FORMAL_AUTHORITY",
        formal_authority_manifest=manifest_path,
        expected_cohort_id="synthetic-cohort-v1",
        raw_root=None,
        mask_root=None,
        mask_source_manifest=None,
        plane=None,
        session_id="grap_a_cap_004",
        output_root=output,
        start_frame=0,
        frame_count=15,
        session_frame_count=15,
    )

    with pytest.raises(runner.CleanRunError, match="changed during same-FD read"):
        runner.run(args)
    assert fired
    assert not output.exists()


def test_formal_read_rejects_ancestor_directory_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "live"
    replacement = tmp_path / "replacement"
    live.mkdir()
    replacement.mkdir()
    target = live / "artifact.bin"
    target.write_bytes(b"verified bytes")
    real_read = os.read
    fired = False

    def attacked_read(descriptor: int, count: int) -> bytes:
        nonlocal fired
        payload = real_read(descriptor, count)
        if not payload and not fired:
            try:
                descriptor_path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            except OSError:
                descriptor_path = Path("/")
            if descriptor_path == target:
                fired = True
                old = tmp_path / "old-live"
                live.rename(old)
                replacement.rename(live)
                (old / target.name).rename(live / target.name)
        return payload

    monkeypatch.setattr(runner.os, "read", attacked_read)
    with pytest.raises(
        runner.CleanRunError,
        match="ancestor path changed|cannot be opened no-follow|changed during same-FD",
    ):
        runner.read_formal_bytes(target, "ancestor-race")
    assert fired


def test_stable_output_tree_rejects_swapped_component_without_escape(
    tmp_path: Path,
) -> None:
    tree = runner.StableOutputTree(tmp_path / "formal-output")
    victim = tmp_path / "victim"
    victim.mkdir()
    mask_path = tree.path / "MASK"
    held_path = tree.path / "MASK-held"
    mask_path.rename(held_path)
    mask_path.symlink_to(victim, target_is_directory=True)
    try:
        with pytest.raises(runner.CleanRunError, match="component changed"):
            tree.write("MASK", "00000.png", b"synthetic")
        assert not (victim / "00000.png").exists()
    finally:
        tree.close(check_named=False)


def test_mp4_encoder_rejects_preexisting_symlink_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = runner.StableOutputTree(tmp_path / "formal-mp4-output")
    victim = tmp_path / "victim.mp4"
    victim.write_bytes(b"must remain unchanged")
    (tree.path / "CLEAN_AUTHORITY_BOUND.mp4").symlink_to(victim)
    called = False

    def forbidden_popen(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("ffmpeg must not start before an exclusive leaf is bound")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden_popen)
    try:
        with pytest.raises(runner.CleanRunError, match="cannot be created exclusively"):
            runner.encode_video_bound(
                tree,
                [],
                output_name="CLEAN_AUTHORITY_BOUND.mp4",
                width=8,
                height=6,
                fps=30,
            )
        assert not called
        assert victim.read_bytes() == b"must remain unchanged"
    finally:
        tree.close()


def test_mp4_encoder_uses_prebound_leaf_and_full_decode(tmp_path: Path) -> None:
    tree = runner.StableOutputTree(tmp_path / "formal-mp4-smoke")
    try:
        for frame_index in range(2):
            image = np.full((6, 8, 3), frame_index * 127, dtype=np.uint8)
            tree.write(
                "CLEAN",
                f"{frame_index:05d}.png",
                runner.encode_png(image, "RGB"),
            )
        reference = runner.encode_video_bound(
            tree,
            ["00000.png", "00001.png"],
            output_name="CLEAN_AUTHORITY_BOUND.mp4",
            width=8,
            height=6,
            fps=30,
        )
        output = tree.path / "CLEAN_AUTHORITY_BOUND.mp4"
        info = output.stat()
        assert info.st_nlink == 1
        assert stat.S_ISREG(info.st_mode)
        assert reference["path"] == str(output)
        assert reference["bytes"] == info.st_size
        assert reference["frames"] == 2
        assert reference["full_decode"] is True
    finally:
        tree.close()


def test_output_success_evidence_rejects_directory_placeholders(tmp_path: Path) -> None:
    tree = runner.StableOutputTree(tmp_path / "formal-evidence")
    evidence = [
        {
            "absolute_path": str(tree.path / "CLEAN"),
            "minimum_bytes": 1,
            "verification": "JSON_PARSE_EXPECTED_KEYS",
            "expected_keys": ["status"],
        },
        {
            "absolute_path": str(tree.path / "CLEAN_MANIFEST.json"),
            "minimum_bytes": 1,
            "verification": "JSON_PARSE_EXPECTED_KEYS",
            "expected_keys": ["status"],
        },
    ]
    try:
        with pytest.raises(runner.CleanRunError):
            runner._verify_output_success_evidence(
                tree, evidence, expected_frames=0, expected_hw=(6, 8)
            )
    finally:
        tree.close()


def test_implementation_imports_no_morphology_or_inpainting_stack() -> None:
    imported: set[str] = set()
    for relative in (
        "pipeline/deterministic_table_reprojection_clean.py",
        "tools/run_deterministic_table_reprojection_clean.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        )
    assert not imported.intersection(
        {"cv2", "skimage", "scipy", "propainter", "diffusers", "torch"}
    )


def test_global_donor_margin_has_no_cli_or_session_override_surface() -> None:
    source = (ROOT / "tools/run_deterministic_table_reprojection_clean.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    cli_options = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("--")
    }

    assert not {option for option in cli_options if "margin" in option.lower()}
    assert donor_arm_margin_contract()["minimum_distance_px"] == 3


def test_margin_rejection_is_quality_provenance_not_an_operation_counter() -> None:
    source = (ROOT / "tools/run_deterministic_table_reprojection_clean.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    output_counter_key_sets: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if "deterministic_table_reprojection_pixels" in keys:
            output_counter_key_sets.append(keys)

    assert len(output_counter_key_sets) == 2
    assert all(
        "margin_rejected_candidate_pixel_count" not in keys
        for keys in output_counter_key_sets
    )
    assert source.count('"margin_rejected_candidate_pixel_count"') == 4
