from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import pytest

from pipeline.depth_occlusion_v3 import decode_s8_frame_bundle, supersampled_intrinsics
from pipeline.robot_renderer_eevee_fullchain import (
    ARM_JOINT_NAMES,
    CONTACT_INFEASIBLE,
    ENGINE,
    MOUNT_PROVENANCE,
    OUTPUT_SCHEMA,
    SCENE_MANIFEST_DEVELOPMENT_STATUS,
    SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS,
)
from tools import build_robot_s8_frame_bundle_input as subject
from tools import immutable_artifact_io as strict
from tools.run_depth_occlusion_v3_s8_t1 import (
    _read_manifest_record,
    validate_input_manifest,
)


PROJECT = Path(__file__).resolve().parents[1]
SESSION = "grap_a_cap_004"
FRAME_INDEX = 0
FRAME_NAME = "00000"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path.resolve(strict=True)


def _record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    info = path.stat()
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": len(payload),
        "sha256": _sha(payload),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
    }


def _declared(path: Path, label: str) -> subject.DeclaredInput:
    record = _record(path)
    return subject.DeclaredInput.create(
        label=label,
        path=path,
        bytes_count=record["bytes"],
        sha256=record["sha256"],
    )


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _npz_bytes(**values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez(output, **values)
    return output.getvalue()


def _rgb_png(rgb: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", np.asarray(rgb)[..., ::-1])
    assert ok
    return bytes(encoded)


def _rgba_png(rgba: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", np.asarray(rgba)[..., (2, 1, 0, 3)])
    assert ok
    return bytes(encoded)


def _exr(value: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".exr", np.asarray(value, dtype=np.float32))
    assert ok
    return bytes(encoded)


def _scene_state_bytes(camera_values: np.ndarray) -> bytes:
    count = 1
    wrists = np.repeat(np.eye(4, dtype=np.float64)[None, None], 2, axis=1)
    hand_names = np.asarray(
        [
            [f"left_hand_joint_{index:02d}" for index in range(22)],
            [f"right_hand_joint_{index:02d}" for index in range(22)],
        ]
    )
    return _npz_bytes(
        schema_version=np.asarray("robot-fullchain-scene-state-v1"),
        session_id=np.asarray(SESSION),
        mount_provenance=np.asarray(MOUNT_PROVENANCE),
        contact_infeasible=np.asarray(CONTACT_INFEASIBLE),
        frame_names=np.asarray([FRAME_NAME]),
        q_arm=np.zeros((count, 2, 7), dtype=np.float64),
        q_hand=np.zeros((count, 2, 22), dtype=np.float64),
        valid=np.ones((count, 2), dtype=np.bool_),
        wrist_T_camera=wrists,
        T_camera_base=np.eye(4, dtype=np.float64),
        T_tool_hand=np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0),
        camera_intrinsics=np.asarray([camera_values], dtype=np.float64),
        source_resolution=np.asarray([3, 2], dtype=np.int32),
        arm_joint_names=np.asarray(ARM_JOINT_NAMES),
        hand_joint_names=hand_names,
    )


def _object6d_bytes(transforms: np.ndarray) -> bytes:
    value = np.asarray(transforms, dtype=np.float64)
    count = value.shape[0]
    return _npz_bytes(
        T_object_to_world=value.copy(),
        T_object_to_camera=value.copy(),
        T_object_to_world_raw_center_corrected=value.copy(),
        confidence=np.linspace(0.9, 0.8, count, dtype=np.float64),
        valid=np.ones(count, dtype=np.bool_),
        visual_observed=np.ones(count, dtype=np.bool_),
        timestamp_ns=np.arange(count, dtype=np.int64),
        cylinder_radius_m=np.asarray(0.1, dtype=np.float64),
        cylinder_height_m=np.asarray(0.2, dtype=np.float64),
    )


def _tool_definition(
    project: Path,
    asset_pin_ref: dict[str, Any],
    urdf_ref: dict[str, Any],
) -> dict[str, Any]:
    joint = {
        "joint_type": "fixed",
        "xyz_m": [0.0, 0.0, 0.145],
        "rpy_rad": [0.0, 0.0, 0.0],
    }
    return {
        "schema_version": subject.TOOL_DEFINITION_SCHEMA,
        "source_policy": subject.TOOL_DEFINITION_SOURCE_POLICY,
        "asset_pin": {
            "path": Path(asset_pin_ref["path"]).relative_to(project).as_posix(),
            "bytes": asset_pin_ref["bytes"],
            "sha256": asset_pin_ref["sha256"],
        },
        "urdf": {
            "path": Path(urdf_ref["path"]).relative_to(project).as_posix(),
            "bytes": urdf_ref["bytes"],
            "sha256": urdf_ref["sha256"],
        },
        "tool_joints": {
            "left": {
                **joint,
                "side": "left",
                "name": "left_tool_joint",
                "parent": "flange_L",
                "child": "left_tool",
            },
            "right": {
                **joint,
                "side": "right",
                "name": "right_tool_joint",
                "parent": "flange_R",
                "child": "right_tool",
            },
        },
        "external_mjcf_consumed": False,
        "world_translation_interpretation_forbidden": True,
        "axis_note": "synthetic development-only parent-flange-frame fixture",
    }


def _scene_manifest(
    state_ref: dict[str, Any], tool_definition: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "robot-fullchain-scene-manifest-v1",
        "status": SCENE_MANIFEST_DEVELOPMENT_STATUS,
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": SESSION,
        "frame_count": 1,
        "mount_provenance": MOUNT_PROVENANCE,
        "contact_infeasible": CONTACT_INFEASIBLE,
        "visual_only": True,
        "development_only": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "baseline_frozen": False,
        "ik_residual_is_independent_accuracy_evidence": False,
        "q_arm_and_camera_base_authoritative": False,
        "scene_state": state_ref,
        "mount_authority": {
            "session_constant": True,
            "per_frame_mount_forbidden": True,
            "independent_of_r2_wrist_targets": True,
            "selected_by_ik_residual": False,
        },
        "solver": {
            "mount_optimized_or_selected": False,
            "objective_and_reported_residual_use_same_r2_wrist_target": True,
            "reported_residual_is_independent_validation": False,
            "independent_camera_base_or_arm_image_authority_consumed": False,
            "q_arm_and_camera_base_unique_or_authoritative": False,
        },
        "tool_definition": tool_definition,
    }


def _eevee_manifest(
    *,
    cpu4_ref: dict[str, Any],
    asset_pin_ref: dict[str, Any],
    scene_manifest_ref: dict[str, Any],
    scene_state_ref: dict[str, Any],
    beauty_ref: dict[str, Any],
    range_ref: dict[str, Any],
    index_ref: dict[str, Any],
    scaled_k: np.ndarray,
    tool_definition: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": "CANDIDATE_REQUIRES_HUMAN_REVIEW",
        "engine": ENGINE,
        "session_id": SESSION,
        "frame_start": 0,
        "frame_count": 1,
        "resolution": [6, 4],
        "single_process_multi_frame": True,
        "mount_provenance": MOUNT_PROVENANCE,
        "contact_infeasible": CONTACT_INFEASIBLE,
        "visual_only": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "baseline_frozen": False,
        "tool_definition": tool_definition,
        "geometry_contract": dict(subject.EEVEE_GEOMETRY_CONTRACT),
        "buffers": {
            "beauty": "RGBA_PNG_TRANSPARENT",
            "range": "OPENEXR_DEPTH_PASS_EUCLIDEAN_CAMERA_RANGE_METRES",
            "object_index": "OPENEXR_QA_ONLY_NOT_COMPOSITOR_DECISION",
        },
        "claim_limits": list(subject.EEVEE_CLAIM_LIMITS),
        "source_records": {
            "cpu4_tool_consistency": cpu4_ref,
            "scene_manifest": scene_manifest_ref,
            "scene_state": scene_state_ref,
            "asset_pin": asset_pin_ref,
        },
        "outputs": {
            FRAME_NAME: {
                "beauty": beauty_ref,
                "range": range_ref,
                "object_index": index_ref,
                "valid_by_side": [True, True],
                "scaled_k": scaled_k.tolist(),
            }
        },
        "gpu_execution_proven": False,
    }


@dataclass
class Fixture:
    project: Path
    scene_manifest: Path
    scene_state: Path
    eevee_manifest: Path
    clean: Path
    object_texture: Path
    object6d: Path
    beauty: Path
    range_exr: Path
    index_exr: Path
    camera: np.ndarray
    clean_array: np.ndarray
    object_array: np.ndarray
    robot_rgba: np.ndarray
    range_array: np.ndarray

    def declarations(self) -> dict[str, subject.DeclaredInput]:
        return {
            "scene_manifest": _declared(self.scene_manifest, "scene manifest"),
            "scene_state": _declared(self.scene_state, "scene state"),
            "eevee_run_manifest": _declared(self.eevee_manifest, "EEVEE RUN_MANIFEST"),
            "clean_rgb": _declared(self.clean, "direct CLEAN RGB"),
            "object_texture_2x": _declared(
                self.object_texture, "direct 2x object texture"
            ),
            "object6d": _declared(self.object6d, "Object6D"),
        }

    def rewrite_eevee(self, mutation: Any) -> None:
        value = json.loads(self.eevee_manifest.read_bytes())
        mutation(value)
        self.eevee_manifest.write_bytes(_json_bytes(value))

    def refresh_eevee_range_ref(self) -> None:
        self.rewrite_eevee(
            lambda value: value["outputs"][FRAME_NAME].__setitem__(
                "range", _record(self.range_exr)
            )
        )


def _fixture(tmp_path: Path, *, worker_two_c: bool = False) -> Fixture:
    project = tmp_path / "project"
    (project / "_run").mkdir(parents=True)
    evidence = project / "evidence" / SESSION
    camera_values = np.asarray([3.0, 3.5, 1.0, 0.5], dtype=np.float64)
    camera = np.asarray(
        [[3.0, 0.0, 1.0], [0.0, 3.5, 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scene_state = _write(
        evidence / "scene" / "SCENE_STATE.npz",
        _scene_state_bytes(camera_values),
    )
    urdf = _write(
        project / subject.TOOL_URDF_RELATIVE,
        b'<robot name="synthetic-development-only"/>\n',
    )
    urdf_ref = _record(urdf)
    asset_pin = _write(
        project / subject.ASSET_PIN_RELATIVE,
        _json_bytes(
            {
                "schema_version": subject.ASSET_PIN_SCHEMA,
                "status": subject.ASSET_PIN_STATUS,
                "scope": {
                    "mutation_permitted": False,
                    "scale_topology_urdf_fk_frozen": True,
                },
                "files": [
                    {
                        "path": urdf.relative_to(project).as_posix(),
                        "bytes": urdf_ref["bytes"],
                        "sha256": urdf_ref["sha256"],
                    }
                ],
                "synthetic_test_fixture": True,
            }
        ),
    )
    tool_definition = _tool_definition(project, _record(asset_pin), urdf_ref)
    cpu4 = _write(
        evidence / "eevee_inputs" / "CPU4_TOOL_CONSISTENCY.json",
        _json_bytes(
            {
                "schema_version": subject.CPU4_TOOL_SCHEMA,
                "auth_tier": subject.CPU4_TOOL_AUTH_TIER,
                "status": subject.CPU4_TOOL_STATUS,
                "a_class_p0": False,
                "consumed_tool_definition": tool_definition,
                "synthetic_test_fixture": True,
            }
        ),
    )
    scene_manifest = _write(
        evidence / "scene" / "SCENE_MANIFEST.json",
        _json_bytes(_scene_manifest(_record(scene_state), tool_definition)),
    )

    clean_array = np.asarray(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )
    object_array = np.full((4, 6, 3), [101, 102, 103], dtype=np.uint8)
    robot_rgba = np.full((4, 6, 4), [201, 202, 203, 255], dtype=np.uint8)
    robot_rgba[0, 0, 3] = 0
    range_array = np.linspace(1.0, 2.0, 24, dtype=np.float32).reshape(4, 6)
    clean = _write(evidence / "clean" / f"{FRAME_NAME}.png", _rgb_png(clean_array))
    object_texture = _write(
        evidence / "object_texture_2x" / f"{FRAME_NAME}.png",
        _rgb_png(object_array),
    )
    beauty = _write(
        evidence / "eevee" / "beauty" / f"{FRAME_NAME}.png",
        _rgba_png(robot_rgba),
    )
    range_exr = _write(
        evidence / "eevee" / "range" / f"{FRAME_NAME}.exr", _exr(range_array)
    )
    index_exr = _write(
        evidence / "eevee" / "object_index" / f"{FRAME_NAME}.exr",
        _exr(np.ones((4, 6), dtype=np.float32)),
    )
    scaled_k = supersampled_intrinsics(camera)
    if worker_two_c:
        scaled_k = camera.copy()
        scaled_k[0] *= 2.0
        scaled_k[1] *= 2.0
    eevee_manifest = _write(
        evidence / "eevee" / "RUN_MANIFEST.json",
        _json_bytes(
            _eevee_manifest(
                cpu4_ref=_record(cpu4),
                asset_pin_ref=_record(asset_pin),
                scene_manifest_ref=_record(scene_manifest),
                scene_state_ref=_record(scene_state),
                beauty_ref=_record(beauty),
                range_ref=_record(range_exr),
                index_ref=_record(index_exr),
                scaled_k=scaled_k,
                tool_definition=tool_definition,
            )
        ),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[2, 3] = 2.0
    object6d = _write(
        evidence / "object6d" / "object_6dof_v2.npz",
        _object6d_bytes(transform[None]),
    )
    return Fixture(
        project=project,
        scene_manifest=scene_manifest,
        scene_state=scene_state,
        eevee_manifest=eevee_manifest,
        clean=clean,
        object_texture=object_texture,
        object6d=object6d,
        beauty=beauty,
        range_exr=range_exr,
        index_exr=index_exr,
        camera=camera,
        clean_array=clean_array,
        object_array=object_array,
        robot_rgba=robot_rgba,
        range_array=range_array,
    )


def _build(fixture: Fixture, output_name: str = "s8_frame_00000") -> dict[str, Any]:
    return dict(
        subject.build_frame_bundle_input(
            project_root=fixture.project,
            output_name=output_name,
            session_id=SESSION,
            frame_index=FRAME_INDEX,
            object_texture_source="raw_observation",
            **fixture.declarations(),
        )
    )


def _expand_fixture_to_two_frames(fixture: Fixture) -> None:
    with np.load(
        io.BytesIO(fixture.scene_state.read_bytes()), allow_pickle=False
    ) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["frame_names"] = np.asarray(["00000", "00001"])
    for name in (
        "q_arm",
        "q_hand",
        "valid",
        "wrist_T_camera",
        "camera_intrinsics",
    ):
        arrays[name] = np.repeat(arrays[name], 2, axis=0)
    fixture.scene_state.write_bytes(_npz_bytes(**arrays))

    scene = json.loads(fixture.scene_manifest.read_bytes())
    scene["frame_count"] = 2
    scene["scene_state"] = _record(fixture.scene_state)
    fixture.scene_manifest.write_bytes(_json_bytes(scene))

    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    transforms[:, 2, 3] = 2.0
    fixture.object6d.write_bytes(_object6d_bytes(transforms))

    beauty = _write(fixture.beauty.with_name("00001.png"), fixture.beauty.read_bytes())
    range_exr = _write(
        fixture.range_exr.with_name("00001.exr"), fixture.range_exr.read_bytes()
    )
    index_exr = _write(
        fixture.index_exr.with_name("00001.exr"), fixture.index_exr.read_bytes()
    )
    eevee = json.loads(fixture.eevee_manifest.read_bytes())
    eevee["frame_count"] = 2
    eevee["source_records"]["scene_state"] = _record(fixture.scene_state)
    eevee["source_records"]["scene_manifest"] = _record(fixture.scene_manifest)
    eevee["outputs"]["00001"] = {
        "beauty": _record(beauty),
        "range": _record(range_exr),
        "object_index": _record(index_exr),
        "valid_by_side": [True, True],
        "scaled_k": supersampled_intrinsics(fixture.camera).tolist(),
    }
    fixture.eevee_manifest.write_bytes(_json_bytes(eevee))


def test_author_serializes_exact_existing_arrays_and_candidate_manifest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _build(fixture)
    output = fixture.project / "_run" / "s8_frame_00000"
    assert result["status"] == subject.DEVELOPMENT_STATUS
    assert result["completion_mode"] == "ARTIFACT_EXISTS"
    assert result["formal_consumer_allowed"] is False
    assert result["next_bucket_blocked"] is True
    assert result["camera_intrinsics_gate"]["exact_equal"] is True
    assert result["finalization_contract"] == subject.FINALIZATION_CONTRACT
    assert (
        result["finalization_contract"]["cross_file_atomic_snapshot_claimed"] is False
    )
    assert result["finalization_contract"]["consumer_revalidation_required"] is True
    assert set(result["pixel_operations"].values()) == {0}
    assert set(path.name for path in output.iterdir()) == {
        "FRAME_BUNDLE.npz",
        "INPUT_MANIFEST.json",
        "RUN_MANIFEST.json",
    }

    bundle = decode_s8_frame_bundle((output / "FRAME_BUNDLE.npz").read_bytes())
    assert np.array_equal(bundle.clean_rgb, fixture.clean_array)
    assert np.array_equal(bundle.object_rgb_2x, fixture.object_array)
    assert np.array_equal(bundle.robot_rgb_2x, fixture.robot_rgba[..., :3])
    assert np.array_equal(bundle.robot_range_m_2x, fixture.range_array)
    assert np.array_equal(
        bundle.robot_alpha_2x,
        fixture.robot_rgba[..., 3].astype(np.float32) / np.float32(255.0),
    )
    assert np.array_equal(bundle.camera_intrinsics, fixture.camera)
    assert bundle.object_index_2x is None

    input_manifest = json.loads((output / "INPUT_MANIFEST.json").read_bytes())
    validate_input_manifest(input_manifest)
    for record in (
        input_manifest["frame_bundle"],
        input_manifest["object6d"],
        *input_manifest["source_lineage"].values(),
    ):
        strict.read_bytes_nofollow(
            Path(record["path"]),
            expected_bytes=record["bytes"],
            expected_sha256=record["sha256"],
            allowed_root=fixture.project,
        )
    assert input_manifest["development_only"] is True
    assert input_manifest["formal_consumer_allowed"] is False
    assert input_manifest["eligible68_publisher"] is False
    assert input_manifest["formal_publisher"] is False
    assert input_manifest["finalization_contract"] == subject.FINALIZATION_CONTRACT
    assert input_manifest["object_index_role"] == (
        "QA_ONLY_NOT_SERIALIZED_NOT_DECISION"
    )
    assert input_manifest["mount_provenance"] == MOUNT_PROVENANCE
    assert set(input_manifest["pixel_operations"].values()) == {0}
    assert len(result["success_evidence"]) == 2
    assert all(item["bytes"] > 0 for item in result["success_evidence"])


def test_post_author_input_or_output_drift_is_rejected_by_s8_consumer_read(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _build(fixture, "post_author_consumer_revalidation")
    output = fixture.project / "_run" / "post_author_consumer_revalidation"
    input_manifest = json.loads((output / "INPUT_MANIFEST.json").read_bytes())
    cases = (
        (fixture.clean, input_manifest["source_lineage"]["clean"]),
        (output / "FRAME_BUNDLE.npz", input_manifest["frame_bundle"]),
    )
    for index, (path, record) in enumerate(cases):
        replacement = path.with_name(f"{path.name}.consumer-race-{index}")
        replacement.write_bytes(path.read_bytes() + b"post-author-drift")
        os.replace(replacement, path)
        with pytest.raises(strict.StrictIOError, match="byte count mismatch"):
            _read_manifest_record(fixture.project, record)


def test_current_worker_two_c_is_explicit_hold_and_publishes_no_bundle(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, worker_two_c=True)
    result = _build(fixture, "s8_k_hold")
    output = fixture.project / "_run" / "s8_k_hold"
    assert result["status"] == subject.CAMERA_K_HOLD_STATUS
    assert result["camera_intrinsics_gate"]["exact_equal"] is False
    assert result["frame_bundle_published"] is False
    assert result["input_manifest_published"] is False
    assert result["eligible68_publisher"] is False
    assert result["formal_publisher"] is False
    assert result["finalization_contract"] == subject.FINALIZATION_CONTRACT
    assert set(result["pixel_operations"].values()) == {0}
    assert set(path.name for path in output.iterdir()) == {
        "CAMERA_INTRINSICS_HOLD.json",
        "RUN_MANIFEST.json",
    }
    assert not (output / "FRAME_BUNDLE.npz").exists()
    assert not (output / "INPUT_MANIFEST.json").exists()
    hold = json.loads((output / "CAMERA_INTRINSICS_HOLD.json").read_bytes())
    assert hold["candidate_requires_human_review"] is True
    assert hold["formal_consumer_allowed"] is False
    assert hold["eligible68_publisher"] is False
    assert hold["finalization_contract"] == subject.FINALIZATION_CONTRACT
    expected = np.asarray(result["camera_intrinsics_gate"]["expected_s8_scaled_k"])
    observed = np.asarray(result["camera_intrinsics_gate"]["observed_eevee_scaled_k"])
    assert expected[0, 2] - observed[0, 2] == pytest.approx(0.5)
    assert expected[1, 2] - observed[1, 2] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.__setitem__("session_id", "grap_a_cap_002"),
            "session",
        ),
        (
            lambda value: value["outputs"][FRAME_NAME].__setitem__(
                "valid_by_side", [True, False]
            ),
            "valid_by_side",
        ),
        (
            lambda value: value.__setitem__("resolution", [3, 2]),
            "2x grid",
        ),
        (
            lambda value: value.__setitem__("resolution", [6.0, 4.0]),
            "2x grid",
        ),
        (
            lambda value: value["outputs"][FRAME_NAME].__setitem__(
                "scaled_k", [[6.0, 0.0, True], [0.0, 7.0, 1.5], [0.0, 0.0, 1.0]]
            ),
            "numeric JSON",
        ),
        (
            lambda value: value["buffers"].__setitem__(
                "range", "OPENEXR_OPTICAL_AXIS_Z_METRES"
            ),
            "buffer semantics",
        ),
        (
            lambda value: value.__setitem__("single_process_multi_frame", False),
            "single-process",
        ),
        (
            lambda value: value["source_records"].pop("asset_pin"),
            "source-record closure",
        ),
        (
            lambda value: value.__setitem__(
                "tool_definition", {"synthetic_test_fixture": False}
            ),
            "tool-definition",
        ),
        (
            lambda value: value.__setitem__("formal_consumer_allowed", True),
            "formal_consumer_allowed",
        ),
    ],
)
def test_eevee_session_frame_geometry_and_governance_join_fail_closed(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.rewrite_eevee(mutation)
    with pytest.raises(subject.FrameBundleAuthorError, match=match):
        _build(fixture)
    assert not (fixture.project / "_run" / "s8_frame_00000").exists()


def test_multiframe_eevee_validates_every_output_entry_and_alias_metadata(
    tmp_path: Path,
) -> None:
    valid_fixture = _fixture(tmp_path / "valid")
    _expand_fixture_to_two_frames(valid_fixture)
    result = _build(valid_fixture, "multiframe_valid")
    assert result["status"] == subject.DEVELOPMENT_STATUS

    missing_fixture = _fixture(tmp_path / "missing")
    _expand_fixture_to_two_frames(missing_fixture)
    missing_fixture.rewrite_eevee(
        lambda value: value["outputs"].__setitem__("00001", {})
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="00001.*keys"):
        _build(missing_fixture, "multiframe_missing")

    alias_fixture = _fixture(tmp_path / "alias")
    _expand_fixture_to_two_frames(alias_fixture)

    def forge_cross_frame_identity(value: dict[str, Any]) -> None:
        source = value["outputs"]["00000"]["beauty"]
        target = value["outputs"]["00001"]["beauty"]
        target["device"] = source["device"]
        target["inode"] = source["inode"]

    alias_fixture.rewrite_eevee(forge_cross_frame_identity)
    with pytest.raises(subject.FrameBundleAuthorError, match="must not alias"):
        _build(alias_fixture, "multiframe_alias")


def test_scene_state_manifest_bytes_join_and_provisional_flags_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    scene = json.loads(fixture.scene_manifest.read_bytes())
    scene["formal_consumer_allowed"] = True
    fixture.scene_manifest.write_bytes(_json_bytes(scene))
    fixture.rewrite_eevee(
        lambda value: value["source_records"].__setitem__(
            "scene_manifest", _record(fixture.scene_manifest)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="formal_consumer_allowed"):
        _build(fixture)


def test_object6d_requires_requested_session_and_valid_exact_frame(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    replacement = (
        fixture.object6d.parent.parent / "grap_a_cap_002" / "object_6dof_v2.npz"
    )
    replacement = _write(replacement, fixture.object6d.read_bytes())
    fixture.object6d = replacement
    with pytest.raises(subject.FrameBundleAuthorError, match="requested session"):
        _build(fixture)


def test_object6d_frame_count_must_close_exactly_with_scene(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    fixture.object6d.write_bytes(_object6d_bytes(transforms))
    with pytest.raises(subject.FrameBundleAuthorError, match="0..N-1 closure"):
        _build(fixture)


def test_object6d_rejects_nonexact_member_closure(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with np.load(io.BytesIO(fixture.object6d.read_bytes()), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    arrays["unexpected_fallback_pose"] = np.eye(4, dtype=np.float64)
    fixture.object6d.write_bytes(_npz_bytes(**arrays))
    with pytest.raises(subject.FrameBundleAuthorError, match="exact frozen nine-array"):
        _build(fixture)


@pytest.mark.parametrize(
    "defect", ["unicode_pose", "float32_confidence", "float32_scalar"]
)
def test_object6d_rejects_convertible_but_nonexact_dtypes(
    tmp_path: Path,
    defect: str,
) -> None:
    fixture = _fixture(tmp_path)
    with np.load(io.BytesIO(fixture.object6d.read_bytes()), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    if defect == "unicode_pose":
        for name in (
            "T_object_to_world",
            "T_object_to_camera",
            "T_object_to_world_raw_center_corrected",
        ):
            arrays[name] = arrays[name].astype("<U32")
    elif defect == "float32_confidence":
        arrays["confidence"] = arrays["confidence"].astype(np.float32)
    else:
        arrays["cylinder_radius_m"] = arrays["cylinder_radius_m"].astype(np.float32)
    fixture.object6d.write_bytes(_npz_bytes(**arrays))
    with pytest.raises(subject.FrameBundleAuthorError, match="exact dtypes"):
        _build(fixture, f"object6d_{defect}")


def test_direct_clean_and_raw_texture_bind_target_session(tmp_path: Path) -> None:
    clean_fixture = _fixture(tmp_path / "clean")
    clean_fixture.clean = _write(
        clean_fixture.project
        / "evidence"
        / "grap_a_cap_002"
        / "clean"
        / f"{FRAME_NAME}.png",
        clean_fixture.clean.read_bytes(),
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="requested session"):
        _build(clean_fixture)

    texture_fixture = _fixture(tmp_path / "texture")
    texture_fixture.object_texture = _write(
        texture_fixture.project
        / "evidence"
        / "grap_a_cap_002"
        / "object_texture_2x"
        / f"{FRAME_NAME}.png",
        texture_fixture.object_texture.read_bytes(),
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="source session"):
        _build(texture_fixture)


def test_object_donor_session_is_explicit_and_remains_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.object_texture = _write(
        fixture.project
        / "evidence"
        / "grap_a_cap_002"
        / "object_texture_2x"
        / f"{FRAME_NAME}.png",
        fixture.object_texture.read_bytes(),
    )
    result = subject.build_frame_bundle_input(
        project_root=fixture.project,
        output_name="donor_candidate",
        session_id=SESSION,
        frame_index=FRAME_INDEX,
        object_texture_source="object_donor",
        **fixture.declarations(),
    )
    assert result["frame_session_join"]["object_texture_source_session_id"] == (
        "grap_a_cap_002"
    )
    assert result["formal_consumer_allowed"] is False
    assert result["next_bucket_blocked"] is True


def test_eevee_qa_object_index_record_cannot_be_forbidden_or_alias(
    tmp_path: Path,
) -> None:
    forbidden_fixture = _fixture(tmp_path / "forbidden")
    forbidden = _write(
        forbidden_fixture.project / "labels" / SESSION / f"{FRAME_NAME}.exr",
        forbidden_fixture.index_exr.read_bytes(),
    )
    forbidden_fixture.rewrite_eevee(
        lambda value: value["outputs"][FRAME_NAME].__setitem__(
            "object_index", _record(forbidden)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="forbidden"):
        _build(forbidden_fixture)

    alias_fixture = _fixture(tmp_path / "alias")
    alias_fixture.rewrite_eevee(
        lambda value: value["outputs"][FRAME_NAME].__setitem__(
            "object_index", _record(alias_fixture.range_exr)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="alias"):
        _build(alias_fixture)


def test_eevee_object_index_qa_is_same_fd_bound_and_fully_decoded(
    tmp_path: Path,
) -> None:
    forged_fixture = _fixture(tmp_path / "forged")

    def forge_record(value: dict[str, Any]) -> None:
        record = value["outputs"][FRAME_NAME]["object_index"]
        record["bytes"] = 1
        record["sha256"] = "0" * 64
        record["inode"] += 999_999

    forged_fixture.rewrite_eevee(forge_record)
    with pytest.raises(subject.FrameBundleAuthorError, match="byte count"):
        _build(forged_fixture)
    assert not (forged_fixture.project / "_run" / "s8_frame_00000").exists()

    hardlink_fixture = _fixture(tmp_path / "hardlink")
    os.link(
        hardlink_fixture.index_exr,
        hardlink_fixture.index_exr.with_name("object_index_alias.exr"),
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="hard-link"):
        _build(hardlink_fixture)

    finite_fixture = _fixture(tmp_path / "finite")
    finite_fixture.index_exr.write_bytes(
        _exr(np.full((4, 6), np.nan, dtype=np.float32))
    )
    finite_fixture.rewrite_eevee(
        lambda value: value["outputs"][FRAME_NAME].__setitem__(
            "object_index", _record(finite_fixture.index_exr)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="must be finite"):
        _build(finite_fixture)


@pytest.mark.parametrize("source_name", ["cpu4_tool_consistency", "asset_pin"])
def test_eevee_transitive_source_records_are_same_fd_bound(
    tmp_path: Path,
    source_name: str,
) -> None:
    fixture = _fixture(tmp_path)

    def forge_record(value: dict[str, Any]) -> None:
        record = value["source_records"][source_name]
        record["bytes"] = 1
        record["sha256"] = "0" * 64
        record["inode"] += 999_999

    fixture.rewrite_eevee(forge_record)
    with pytest.raises(subject.FrameBundleAuthorError, match="byte count"):
        _build(fixture, f"forged_{source_name}")
    assert not (fixture.project / "_run" / f"forged_{source_name}").exists()


def test_eevee_transitive_source_rejects_hardlink_and_invalid_json(
    tmp_path: Path,
) -> None:
    hardlink_fixture = _fixture(tmp_path / "hardlink")
    hardlink_document = json.loads(hardlink_fixture.eevee_manifest.read_bytes())
    cpu4_path = Path(
        hardlink_document["source_records"]["cpu4_tool_consistency"]["path"]
    )
    os.link(cpu4_path, cpu4_path.with_name("CPU4_TOOL_CONSISTENCY_ALIAS.json"))
    with pytest.raises(subject.FrameBundleAuthorError, match="hard-link"):
        _build(hardlink_fixture, "transitive_hardlink")

    invalid_fixture = _fixture(tmp_path / "invalid_json")
    invalid_document = json.loads(invalid_fixture.eevee_manifest.read_bytes())
    asset_path = Path(invalid_document["source_records"]["asset_pin"]["path"])
    asset_path.write_bytes(b"not-json\n")
    invalid_fixture.rewrite_eevee(
        lambda value: value["source_records"].__setitem__(
            "asset_pin", _record(asset_path)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="invalid EEVEE source"):
        _build(invalid_fixture, "transitive_invalid_json")


@pytest.mark.parametrize("defect", ["a_class_p0", "different_tool"])
def test_cpu4_source_semantics_and_tool_join_fail_closed(
    tmp_path: Path,
    defect: str,
) -> None:
    fixture = _fixture(tmp_path)
    eevee = json.loads(fixture.eevee_manifest.read_bytes())
    cpu4_path = Path(eevee["source_records"]["cpu4_tool_consistency"]["path"])
    cpu4 = json.loads(cpu4_path.read_bytes())
    if defect == "a_class_p0":
        cpu4["a_class_p0"] = True
        match = "canonical result"
    else:
        for side in ("left", "right"):
            cpu4["consumed_tool_definition"]["tool_joints"][side]["xyz_m"][2] = 0.05
        match = "tool-definition join failed"
    cpu4_path.write_bytes(_json_bytes(cpu4))
    fixture.rewrite_eevee(
        lambda value: value["source_records"].__setitem__(
            "cpu4_tool_consistency", _record(cpu4_path)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match=match):
        _build(fixture, f"cpu4_{defect}")


def test_asset_pin_status_and_tool_pin_identity_join_fail_closed(
    tmp_path: Path,
) -> None:
    status_fixture = _fixture(tmp_path / "status")
    eevee = json.loads(status_fixture.eevee_manifest.read_bytes())
    asset_path = Path(eevee["source_records"]["asset_pin"]["path"])
    asset = json.loads(asset_path.read_bytes())
    asset["status"] = "EXECUTION_AUTHORIZATION_FORGED"
    asset_path.write_bytes(_json_bytes(asset))
    status_fixture.rewrite_eevee(
        lambda value: value["source_records"].__setitem__(
            "asset_pin", _record(asset_path)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="asset pin status"):
        _build(status_fixture, "asset_status_drift")

    identity_fixture = _fixture(tmp_path / "identity")
    eevee = json.loads(identity_fixture.eevee_manifest.read_bytes())
    cpu4_path = Path(eevee["source_records"]["cpu4_tool_consistency"]["path"])
    scene = json.loads(identity_fixture.scene_manifest.read_bytes())
    cpu4 = json.loads(cpu4_path.read_bytes())
    for tool in (
        eevee["tool_definition"],
        scene["tool_definition"],
        cpu4["consumed_tool_definition"],
    ):
        tool["asset_pin"]["sha256"] = "b" * 64
    cpu4_path.write_bytes(_json_bytes(cpu4))
    identity_fixture.scene_manifest.write_bytes(_json_bytes(scene))
    eevee["source_records"]["cpu4_tool_consistency"] = _record(cpu4_path)
    eevee["source_records"]["scene_manifest"] = _record(identity_fixture.scene_manifest)
    identity_fixture.eevee_manifest.write_bytes(_json_bytes(eevee))
    with pytest.raises(subject.FrameBundleAuthorError, match="do not join held source"):
        _build(identity_fixture, "asset_identity_drift")


@pytest.mark.parametrize("defect", ["directory_path", "pin_bytes_mismatch"])
def test_tool_urdf_must_join_one_held_asset_pin_entry(
    tmp_path: Path,
    defect: str,
) -> None:
    fixture = _fixture(tmp_path)
    eevee = json.loads(fixture.eevee_manifest.read_bytes())
    scene = json.loads(fixture.scene_manifest.read_bytes())
    cpu4_path = Path(eevee["source_records"]["cpu4_tool_consistency"]["path"])
    cpu4 = json.loads(cpu4_path.read_bytes())
    tool_records = (
        eevee["tool_definition"],
        scene["tool_definition"],
        cpu4["consumed_tool_definition"],
    )
    if defect == "directory_path":
        replacement = {"path": ".", "bytes": 7, "sha256": "c" * 64}
        for tool in tool_records:
            tool["urdf"] = dict(replacement)
        match = "URDF.*canonical"
    else:
        for tool in tool_records:
            tool["urdf"]["bytes"] += 1
        match = "do not join one held asset-pin entry"
    cpu4_path.write_bytes(_json_bytes(cpu4))
    fixture.scene_manifest.write_bytes(_json_bytes(scene))
    eevee["source_records"]["cpu4_tool_consistency"] = _record(cpu4_path)
    eevee["source_records"]["scene_manifest"] = _record(fixture.scene_manifest)
    fixture.eevee_manifest.write_bytes(_json_bytes(eevee))
    with pytest.raises(subject.FrameBundleAuthorError, match=match):
        _build(fixture, f"urdf_{defect}")


def test_actual_tool_urdf_is_same_fd_bound_and_hardlink_free(tmp_path: Path) -> None:
    drift_fixture = _fixture(tmp_path / "drift")
    urdf = drift_fixture.project / subject.TOOL_URDF_RELATIVE
    urdf.write_bytes(b"drifted-after-pin\n")
    with pytest.raises(subject.FrameBundleAuthorError, match="byte count"):
        _build(drift_fixture, "urdf_actual_drift")
    assert not (drift_fixture.project / "_run" / "urdf_actual_drift").exists()

    hardlink_fixture = _fixture(tmp_path / "hardlink")
    urdf = hardlink_fixture.project / subject.TOOL_URDF_RELATIVE
    os.link(urdf, urdf.with_name("canonical_urdf_alias.urdf"))
    with pytest.raises(subject.FrameBundleAuthorError, match="hard-link"):
        _build(hardlink_fixture, "urdf_hardlink")


def test_range_exr_nan_is_rejected_after_full_decode(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    unsafe = fixture.range_array.copy()
    unsafe[1, 2] = np.nan
    fixture.range_exr.write_bytes(_exr(unsafe))
    fixture.refresh_eevee_range_ref()
    with pytest.raises(subject.FrameBundleAuthorError, match="finite and positive"):
        _build(fixture)


def test_truncated_rgba_png_is_rejected_after_manifest_rebind(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.beauty.write_bytes(fixture.beauty.read_bytes()[:20])
    fixture.rewrite_eevee(
        lambda value: value["outputs"][FRAME_NAME].__setitem__(
            "beauty", _record(fixture.beauty)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="decode"):
        _build(fixture)


def test_declared_sha_drift_fails_before_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    declarations = fixture.declarations()
    clean = declarations["clean_rgb"]
    declarations["clean_rgb"] = subject.DeclaredInput(
        label=clean.label,
        path=clean.path,
        bytes=clean.bytes,
        sha256="0" * 64,
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="SHA-256"):
        subject.build_frame_bundle_input(
            project_root=fixture.project,
            output_name="sha_drift",
            session_id=SESSION,
            frame_index=FRAME_INDEX,
            object_texture_source="raw_observation",
            **declarations,
        )
    assert not (fixture.project / "_run" / "sha_drift").exists()


def test_direct_dataclass_cannot_bypass_forbidden_or_project_root(
    tmp_path: Path,
) -> None:
    forbidden_fixture = _fixture(tmp_path / "forbidden")
    forbidden = _write(
        forbidden_fixture.project / "labels" / SESSION / "clean" / f"{FRAME_NAME}.png",
        forbidden_fixture.clean.read_bytes(),
    )
    record = _record(forbidden)
    declarations = forbidden_fixture.declarations()
    declarations["clean_rgb"] = subject.DeclaredInput(
        label="direct CLEAN RGB",
        path=forbidden,
        bytes=record["bytes"],
        sha256=record["sha256"],
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="forbidden"):
        subject.build_frame_bundle_input(
            project_root=forbidden_fixture.project,
            output_name="forbidden_declaration",
            session_id=SESSION,
            frame_index=FRAME_INDEX,
            object_texture_source="raw_observation",
            **declarations,
        )

    external_fixture = _fixture(tmp_path / "external")
    external = _write(
        tmp_path / "outside" / SESSION / "clean" / f"{FRAME_NAME}.png",
        external_fixture.clean.read_bytes(),
    )
    external_fixture.clean = external
    with pytest.raises(subject.FrameBundleAuthorError, match="project root"):
        _build(external_fixture, "external_rejected")


def test_symlink_and_hardlink_inputs_are_rejected(tmp_path: Path) -> None:
    symlink_fixture = _fixture(tmp_path / "symlink")
    target = _write(
        symlink_fixture.clean.parent / "clean_target.png",
        symlink_fixture.clean.read_bytes(),
    )
    symlink_fixture.clean.unlink()
    symlink_fixture.clean.symlink_to(target)
    with pytest.raises(subject.FrameBundleAuthorError, match="symbolic-link"):
        _build(symlink_fixture, "symlink_rejected")

    hardlink_fixture = _fixture(tmp_path / "hardlink")
    os.link(hardlink_fixture.clean, hardlink_fixture.clean.with_name("alias.png"))
    with pytest.raises(subject.FrameBundleAuthorError, match="hard-link"):
        _build(hardlink_fixture, "hardlink_rejected")


def test_distinct_inputs_cannot_alias_one_canonical_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    declarations = fixture.declarations()
    scene_state = declarations["scene_state"]
    declarations["object6d"] = subject.DeclaredInput(
        label="Object6D",
        path=scene_state.path,
        bytes=scene_state.bytes,
        sha256=scene_state.sha256,
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="canonical path"):
        subject.build_frame_bundle_input(
            project_root=fixture.project,
            output_name="alias_rejected",
            session_id=SESSION,
            frame_index=FRAME_INDEX,
            object_texture_source="raw_observation",
            **declarations,
        )


def test_same_bytes_path_replacement_is_detected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    replacement = _write(
        fixture.clean.parent / "same_bytes_replacement.png",
        fixture.clean.read_bytes(),
    )
    original_encode = subject._encode_npz

    def encode_then_replace(**values: np.ndarray) -> bytes:
        payload = original_encode(**values)
        os.replace(replacement, fixture.clean)
        return payload

    monkeypatch.setattr(subject, "_encode_npz", encode_then_replace)
    with pytest.raises(subject.FrameBundleAuthorError, match="identity changed"):
        _build(fixture, "toctou_rejected")
    assert not (fixture.project / "_run" / "toctou_rejected").exists()


def test_output_leaf_replacement_is_detected_before_run_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture.project / "_run" / "leaf_replaced"
    original = subject._canonical_json

    def replace_bundle_before_input_manifest(value: dict[str, Any]) -> bytes:
        payload = original(value)
        if value.get("schema_version") == subject.INPUT_MANIFEST_SCHEMA:
            bundle = output / "FRAME_BUNDLE.npz"
            replacement = output / "replacement.npz"
            replacement.write_bytes(bundle.read_bytes())
            os.replace(replacement, bundle)
        return payload

    monkeypatch.setattr(
        subject, "_canonical_json", replace_bundle_before_input_manifest
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="leaf identity changed"):
        _build(fixture, "leaf_replaced")
    assert not (output / "RUN_MANIFEST.json").exists()


def test_late_input_replacement_rolls_back_completed_output_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture.project / "_run" / "late_input_race"
    replacement = _write(
        fixture.clean.parent / "same_bytes_late_replacement.png",
        fixture.clean.read_bytes(),
    )
    original = subject._canonical_json

    def replace_input_before_run_manifest(value: dict[str, Any]) -> bytes:
        payload = original(value)
        if value.get("schema_version") == subject.AUTHOR_RUN_SCHEMA:
            os.replace(replacement, fixture.clean)
        return payload

    monkeypatch.setattr(subject, "_canonical_json", replace_input_before_run_manifest)
    with pytest.raises(subject.FrameBundleAuthorError, match="identity changed"):
        _build(fixture, "late_input_race")
    assert not output.exists()


def test_input_replacement_during_output_exit_rolls_back_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture.project / "_run" / "exit_order_input_race"
    replacement = _write(
        fixture.clean.parent / "same_bytes_exit_replacement.png",
        fixture.clean.read_bytes(),
    )
    original_input_check = subject._HeldInputSet.revalidate_all
    original_output_check = subject._HeldOutputDirectory.revalidate
    armed = False
    replaced = False

    def arm_after_complete_run(self: Any) -> None:
        nonlocal armed
        original_input_check(self)
        if (output / "RUN_MANIFEST.json").exists():
            armed = True

    def replace_after_output_check(self: Any) -> None:
        nonlocal replaced
        original_output_check(self)
        if armed and not replaced:
            os.replace(replacement, fixture.clean)
            replaced = True

    monkeypatch.setattr(subject._HeldInputSet, "revalidate_all", arm_after_complete_run)
    monkeypatch.setattr(
        subject._HeldOutputDirectory, "revalidate", replace_after_output_check
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="identity changed"):
        _build(fixture, "exit_order_input_race")
    assert replaced is True
    assert not output.exists()


def test_output_mutation_during_final_input_check_rolls_back_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture.project / "_run" / "exit_order_output_race"
    original_input_check = subject._HeldInputSet.revalidate_all
    post_run_checks = 0
    mutated = False

    def mutate_after_last_input_check(self: Any) -> None:
        nonlocal post_run_checks, mutated
        original_input_check(self)
        if (output / "RUN_MANIFEST.json").exists():
            post_run_checks += 1
            if post_run_checks == 3:
                with (output / "FRAME_BUNDLE.npz").open("ab") as stream:
                    stream.write(b"synthetic-finalization-race")
                mutated = True

    monkeypatch.setattr(
        subject._HeldInputSet, "revalidate_all", mutate_after_last_input_check
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="leaf identity changed"):
        _build(fixture, "exit_order_output_race")
    assert post_run_checks == 3
    assert mutated is True
    assert not output.exists()


def test_first_output_write_failure_removes_partial_leaf_and_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = fixture.project / "_run" / "early_write_failure"

    def fail_first_write(descriptor: int, payload: bytes) -> int:
        del descriptor, payload
        raise OSError("synthetic first output write failure")

    monkeypatch.setattr(subject.os, "write", fail_first_write)
    with pytest.raises(OSError, match="synthetic first output write failure"):
        _build(fixture, "early_write_failure")
    assert not output.exists()


def test_output_child_initialization_failure_removes_new_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output_name = "mkdir_stat_failure"
    output = fixture.project / "_run" / output_name
    original_stat = subject.os.stat
    injected = False

    def fail_first_new_child_stat(
        path: Any, *args: Any, **kwargs: Any
    ) -> os.stat_result:
        nonlocal injected
        if path == output_name and kwargs.get("dir_fd") is not None and not injected:
            injected = True
            raise OSError("synthetic mkdir-to-stat initialization failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(subject.os, "stat", fail_first_new_child_stat)
    with pytest.raises(OSError, match="synthetic mkdir-to-stat"):
        _build(fixture, output_name)
    assert injected is True
    assert not output.exists()


def test_identity_fixture_scene_cannot_be_promoted_out_of_development(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manifest = json.loads(fixture.scene_manifest.read_bytes())
    manifest["status"] = SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS
    manifest["development_only"] = False
    fixture.scene_manifest.write_bytes(_json_bytes(manifest))
    fixture.rewrite_eevee(
        lambda value: value["source_records"].__setitem__(
            "scene_manifest", _record(fixture.scene_manifest)
        )
    )
    with pytest.raises(subject.FrameBundleAuthorError, match="cannot be promoted"):
        _build(fixture, "identity_promotion_rejected")
    assert not (fixture.project / "_run" / "identity_promotion_rejected").exists()


def test_output_must_be_new_direct_child_of_project_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(subject.FrameBundleAuthorError, match="direct-child"):
        _build(fixture, "nested/output")
    existing = fixture.project / "_run" / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _build(fixture, "existing")


def test_cli_help_bootstraps_from_outside_repository(tmp_path: Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "tools/build_robot_s8_frame_bundle_input.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--scene-manifest-sha256" in completed.stdout
    assert "--object-texture-2x-sha256" in completed.stdout
