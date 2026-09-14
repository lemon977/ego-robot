from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.robot_renderer_cycles import forward_kinematics
from pipeline.robot_renderer_eevee_fullchain import (
    ARM_JOINT_NAMES,
    CONTACT_INFEASIBLE,
    MOUNT_PROVENANCE,
    SCENE_MANIFEST_DEVELOPMENT_STATUS,
    SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS,
    SCENE_STATE_MODE_EXTERNAL_AUTHORITY,
    FullChainRendererError,
    compose_frame_placement,
    decode_scene_state,
    load_pinned_robot_assets,
    load_strict_io,
    output_manifest_skeleton,
    validate_scene_manifest,
)


PROJECT = Path(__file__).resolve().parents[1]


def scene_bytes(
    *,
    session: str = "grap_a_cap_004",
    valid: np.ndarray | None = None,
    external: bool = False,
    measured_mount: bool = False,
    mount_authority_lineage_sha256: str = "0" * 64,
    external_authority_lineage_sha256: str = "b" * 64,
) -> bytes:
    assets = load_pinned_robot_assets(PROJECT)
    count = 3
    if valid is None:
        valid = np.ones((count, 2), dtype=bool)
    q_arm = np.zeros((count, 2, 7), dtype=np.float64)
    if external and np.any(~valid):
        q_arm[~valid] = np.nan
    q_hand = np.zeros((count, 2, 22), dtype=np.float64)
    base = np.eye(4)
    mounts = np.repeat(np.eye(4)[None], 2, axis=0)
    if measured_mount:
        mounts[0, :3, 3] = [0.01, -0.02, 0.03]
        mounts[1, :3, 3] = [-0.01, -0.02, 0.03]
    arm_values = {name: 0.0 for names in ARM_JOINT_NAMES for name in names}
    tianji_fk = forward_kinematics(assets.tianji, arm_values)
    wrists = np.repeat(np.eye(4)[None, None], count * 2, axis=0).reshape(count, 2, 4, 4)
    wrists[:, 0] = base @ tianji_fk["left_tool"] @ mounts[0]
    wrists[:, 1] = base @ tianji_fk["right_tool"] @ mounts[1]
    hand_names = np.asarray(
        [
            [
                joint.name
                for joint in assets.left_hand.joints
                if joint.joint_type != "fixed"
            ],
            [
                joint.name
                for joint in assets.right_hand.joints
                if joint.joint_type != "fixed"
            ],
        ]
    )
    output = io.BytesIO()
    arrays = {
        "schema_version": np.asarray("robot-fullchain-scene-state-v1"),
        "session_id": np.asarray(session),
        "mount_provenance": np.asarray(MOUNT_PROVENANCE),
        "contact_infeasible": np.asarray(CONTACT_INFEASIBLE),
        "frame_names": np.asarray([f"{index:05d}" for index in range(count)]),
        "q_arm": q_arm,
        "q_hand": q_hand,
        "valid": valid,
        "wrist_T_camera": wrists,
        "T_camera_base": base,
        "T_tool_hand": mounts,
        "camera_intrinsics": np.repeat(
            np.asarray([[640.0, 640.0, 639.5, 479.5]]), count, axis=0
        ),
        "source_resolution": np.asarray([1280, 960], dtype=np.int32),
        "arm_joint_names": np.asarray(ARM_JOINT_NAMES),
        "hand_joint_names": hand_names,
    }
    if external:
        arrays.update(
            {
                "solver_schema": np.asarray("robot-scene-state-cpu-solver-v1"),
                "scene_state_mode": np.asarray(SCENE_STATE_MODE_EXTERNAL_AUTHORITY),
                "solver_used": np.asarray(False, dtype=np.bool_),
                "residual_claimed": np.asarray(False, dtype=np.bool_),
                "mount_descriptor_sha256": np.asarray("7" * 64),
                "mount_evidence_mode": np.asarray(
                    "INDEPENDENT_BILATERAL_VISUAL_EVIDENCE"
                    if measured_mount
                    else "DEVELOPMENT_ONLY_SYNTHETIC_FIXTURE"
                ),
                "mount_evidence_sha256_by_side": np.asarray(
                    ["8" * 64, "9" * 64] if measured_mount else ["", ""]
                ),
                "mount_authority_lineage_sha256": np.asarray(
                    mount_authority_lineage_sha256
                ),
                "position_residual_mm": np.full((count, 2), np.nan),
                "rotation_residual_deg": np.full((count, 2), np.nan),
                "ik_function_evaluations": np.zeros((count, 2), dtype=np.int32),
                "external_authority_schema": np.asarray(
                    "robot-explicit-base-q-authority-v1"
                ),
                "external_authority_descriptor_sha256": np.asarray("d" * 64),
                "external_authority_arrays_sha256": np.asarray("e" * 64),
                "external_authority_lineage_sha256": np.asarray(
                    external_authority_lineage_sha256
                ),
                "external_base_evidence_sha256": np.asarray("f" * 64),
                "external_q_arm_evidence_sha256_by_side": np.asarray(
                    ["1" * 64, "2" * 64]
                ),
                "external_urdf_sha256": np.asarray(["3" * 64, "4" * 64]),
                "external_source_sha256": np.asarray(["5" * 64, "6" * 64]),
                "timestamp_ns": np.arange(count, dtype=np.int64) + 1,
            }
        )
    np.savez(output, **arrays)
    return output.getvalue()


def test_shared_io_is_exactly_sha_bound() -> None:
    strict = load_strict_io(PROJECT)
    record = strict.read_bytes_nofollow(PROJECT / "tools/immutable_artifact_io.py")
    assert (
        record.sha256
        == "e9af8d38146dee0813d415330fb626a295482740b160435fac544a045d74e57e"
    )


def test_actual_visual_asset_closure_is_63_and_collision_free() -> None:
    assets = load_pinned_robot_assets(PROJECT)
    assert len(assets.mesh_records) == 63
    assert (
        sum(
            len(model.visuals)
            for model in (assets.tianji, assets.left_hand, assets.right_hand)
        )
        == 63
    )
    assert assets.tool_definition["tool_joints"]["left"]["xyz_m"][2] == 0.145
    assert assets.tool_definition["tool_joints"]["right"]["xyz_m"][2] == 0.145


def test_fullchain_pose_is_base_fk_mount_hand_fk_and_has_zero_residual() -> None:
    assets = load_pinned_robot_assets(PROJECT)
    state = decode_scene_state(scene_bytes())
    placement = compose_frame_placement(state, assets, 1)
    assert len(placement.visuals) == 63
    assert {item.role for item in placement.visuals} == {
        "TIANJI_BASE",
        "TIANJI_ARM",
        "KAIHAND",
    }
    assert all(
        item.mesh_relative.startswith("assets/robot/") for item in placement.visuals
    )
    assert placement.hand_root_residual_by_side["left"]["position_mm"] == pytest.approx(
        0.0
    )
    assert placement.hand_root_residual_by_side["right"][
        "rotation_deg"
    ] == pytest.approx(0.0)


def test_invalid_side_is_visibly_absent_without_interpolation() -> None:
    assets = load_pinned_robot_assets(PROJECT)
    valid = np.ones((3, 2), dtype=bool)
    valid[1, 0] = False
    state = decode_scene_state(scene_bytes(valid=valid))
    placement = compose_frame_placement(state, assets, 1)
    assert not any(item.side == "left" for item in placement.visuals)
    assert any(item.side == "right" for item in placement.visuals)
    assert placement.hand_root_residual_by_side["left"] is None


@pytest.mark.parametrize("session", ["grap_a_cap_025", "cross_session_blind"])
def test_forbidden_partition_fails_closed(session: str) -> None:
    with pytest.raises(FullChainRendererError, match="forbidden"):
        decode_scene_state(scene_bytes(session=session))


@pytest.mark.parametrize("defect", ["finite_residual", "solver_work", "formal_alias"])
def test_external_scene_state_rejects_residual_solver_or_extra_claim(
    defect: str,
) -> None:
    with np.load(io.BytesIO(scene_bytes(external=True)), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if defect == "finite_residual":
        arrays["position_residual_mm"][0, 0] = 0.0
    elif defect == "solver_work":
        arrays["ik_function_evaluations"][0, 0] = 1
    else:
        arrays["formal_consumer_allowed"] = np.asarray(True, dtype=np.bool_)
    output = io.BytesIO()
    np.savez(output, **arrays)
    with pytest.raises(FullChainRendererError):
        decode_scene_state(output.getvalue())


def test_scene_manifest_rejects_005_tool_source_as_a_class_mismatch() -> None:
    assets = load_pinned_robot_assets(PROJECT)
    tool = assets.tool_definition
    state = decode_scene_state(scene_bytes())
    record = {"bytes": 4, "sha256": "a" * 64}
    manifest = {
        "schema_version": "robot-fullchain-scene-manifest-v1",
        "status": SCENE_MANIFEST_DEVELOPMENT_STATUS,
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": state.session_id,
        "frame_count": len(state.frame_names),
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
        "scene_state": record,
        "mount_authority": {
            "session_constant": True,
            "per_frame_mount_forbidden": True,
            "independent_of_r2_wrist_targets": True,
            "selected_by_ik_residual": False,
            "evidence_mode": "DEVELOPMENT_ONLY_SYNTHETIC_FIXTURE",
            "independent_visual_evidence": False,
            "synthetic_fixture": True,
            "evidence_by_side": {},
        },
        "solver": {
            "mount_optimized_or_selected": False,
            "objective_and_reported_residual_use_same_r2_wrist_target": True,
            "reported_residual_is_independent_validation": False,
            "independent_camera_base_or_arm_image_authority_consumed": False,
            "q_arm_and_camera_base_unique_or_authoritative": False,
        },
        "tool_definition": copy.deepcopy(tool),
    }
    validate_scene_manifest(
        manifest,
        state_record=record,
        state=state,
        cpu4_tool_definition=tool,
        renderer_tool_definition=tool,
    )
    manifest["tool_definition"]["tool_joints"]["left"]["xyz_m"] = [0.0, 0.0, 0.05]
    with pytest.raises(Exception, match="disagree"):
        validate_scene_manifest(
            manifest,
            state_record=record,
            state=state,
            cpu4_tool_definition=tool,
            renderer_tool_definition=tool,
        )


def _valid_scene_manifest(
    *, development_only: bool = True, external: bool = False
) -> tuple[dict[str, object], dict[str, object], object]:
    assets = load_pinned_robot_assets(PROJECT)
    state = decode_scene_state(
        scene_bytes(external=external, measured_mount=not development_only)
    )

    def ref(name: str, sha: str, inode: int) -> dict[str, object]:
        return {
            "path": f"/independent-authority/{name}",
            "bytes": 4096,
            "sha256": sha,
            "device": 7,
            "inode": inode,
        }

    record: dict[str, object] = ref("SCENE_STATE.npz", "a" * 64, 100)
    common_mount = {
        "provider": "TEST_MOUNT_AUTHORITY",
        "method": (
            "DIRECT_BILATERAL_VISUAL_MOUNT_MEASUREMENT"
            if not development_only
            else "EXPLICIT_SYNTHETIC_SE3"
        ),
        "coordinate_definition": (
            "Tianji left/right tool link to corresponding KaiHand root link"
        ),
        "matrix_direction": "p_target_link = T_tool_hand @ p_source_link",
        "axes_convention": "RIGHT_HANDED_XYZ",
        "translation_units": "metres",
        "rotation_units": "radians",
        "source_links": {
            "left": "hand_l_base_link",
            "right": "hand_r_base_link",
        },
        "target_links": {"left": "left_tool", "right": "right_tool"},
    }
    if development_only:
        mount_authority: dict[str, object] = {
            **common_mount,
            "session_constant": True,
            "per_frame_mount_forbidden": True,
            "independent_of_r2_wrist_targets": True,
            "selected_by_ik_residual": False,
            "evidence_mode": "DEVELOPMENT_ONLY_SYNTHETIC_FIXTURE",
            "independent_visual_evidence": False,
            "synthetic_fixture": True,
            "evidence_by_side": {},
        }
    else:
        mount_authority = {
            **common_mount,
            "session_constant": True,
            "per_frame_mount_forbidden": True,
            "independent_of_r2_wrist_targets": True,
            "selected_by_ik_residual": False,
            "evidence_mode": "INDEPENDENT_BILATERAL_VISUAL_EVIDENCE",
            "independent_visual_evidence": True,
            "synthetic_fixture": False,
            "evidence_by_side": {
                "left": {
                    **ref("mount-left.json", "8" * 64, 111),
                    "session_id": "grap_a_cap_004",
                    "side": "left",
                    "provider": "OWNER_MOUNT_CAMERA_LEFT",
                    "method": "DIRECT_VISUAL_MOUNT_MEASUREMENT",
                    "timestamp_ns": 1,
                    "timestamp_clock_id": "mount-clock-ns",
                    "time_scope": "SESSION_CONSTANT_FULL_SESSION",
                    "device_id": "mount-camera-left",
                    "calibration_id": "mount-calibration-left",
                    "calibration_sha256": "b" * 64,
                    "source_link": "hand_l_base_link",
                    "target_link": "left_tool",
                    "matrix_direction": "p_target_link = T_tool_hand @ p_source_link",
                    "axes_convention": "RIGHT_HANDED_XYZ",
                    "translation_units": "metres",
                    "rotation_units": "radians",
                    "independent_of_r2_wrist_targets": True,
                    "derived_from_r2_wrist_targets": False,
                    "derived_from_ik_solver": False,
                    "selected_by_ik_residual": False,
                    "circular_selection": False,
                    "synthetic_fixture": False,
                    "identity_fixture": False,
                    "formal_consumer_allowed": False,
                },
                "right": {
                    **ref("mount-right.json", "9" * 64, 112),
                    "session_id": "grap_a_cap_004",
                    "side": "right",
                    "provider": "OWNER_MOUNT_CAMERA_RIGHT",
                    "method": "DIRECT_VISUAL_MOUNT_MEASUREMENT",
                    "timestamp_ns": 2,
                    "timestamp_clock_id": "mount-clock-ns",
                    "time_scope": "SESSION_CONSTANT_FULL_SESSION",
                    "device_id": "mount-camera-right",
                    "calibration_id": "mount-calibration-right",
                    "calibration_sha256": "c" * 64,
                    "source_link": "hand_r_base_link",
                    "target_link": "right_tool",
                    "matrix_direction": "p_target_link = T_tool_hand @ p_source_link",
                    "axes_convention": "RIGHT_HANDED_XYZ",
                    "translation_units": "metres",
                    "rotation_units": "radians",
                    "independent_of_r2_wrist_targets": True,
                    "derived_from_r2_wrist_targets": False,
                    "derived_from_ik_solver": False,
                    "selected_by_ik_residual": False,
                    "circular_selection": False,
                    "synthetic_fixture": False,
                    "identity_fixture": False,
                    "formal_consumer_allowed": False,
                },
            },
        }
    manifest: dict[str, object] = {
        "schema_version": "robot-fullchain-scene-manifest-v1",
        "status": (
            SCENE_MANIFEST_DEVELOPMENT_STATUS
            if development_only
            else SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS
        ),
        "completion_mode": "ARTIFACT_EXISTS",
        "session_id": state.session_id,
        "frame_count": len(state.frame_names),
        "mount_provenance": MOUNT_PROVENANCE,
        "contact_infeasible": CONTACT_INFEASIBLE,
        "visual_only": True,
        "development_only": development_only,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "baseline_frozen": False,
        "ik_residual_is_independent_accuracy_evidence": False,
        "q_arm_and_camera_base_authoritative": False,
        "q_arm_and_camera_base_authority_input_consumed": external,
        "scene_state_mode": state.scene_state_mode,
        "gpu_calls": 0,
        "renderer_calls": 0,
        "pixels_produced": 0,
        "scene_state": record,
        "explicit_mount_input": ref("MOUNT.json", "7" * 64, 101),
        "mount_authority": mount_authority,
        "sources": {},
        "solver": {
            "mount_optimized_or_selected": False,
            "objective_and_reported_residual_use_same_r2_wrist_target": not external,
            "reported_residual_is_independent_validation": False,
            "independent_camera_base_or_arm_image_authority_consumed": external,
            "q_arm_and_camera_base_unique_or_authoritative": False,
            "solver_used": not external,
            "residual_not_claimed": external,
        },
        "tool_definition": copy.deepcopy(assets.tool_definition),
        "ik_tool_definition": copy.deepcopy(assets.tool_definition),
        "visual_fit_tool_definition": copy.deepcopy(assets.tool_definition),
        "command_manifest": {
            "argv": ["/usr/bin/python3", "producer.py"],
            "payload_kind": "PYTHON3_CPU_SCENE_STATE",
        },
        "success_evidence": [
            {
                **record,
                "minimum_bytes": 1024,
                "verification": "SHA256_MATCH_AND_FULL_DECODE_SCENE_STATE",
            }
        ],
        "claim_limits": ["NO_TRAINING_GROUND_TRUTH"],
    }
    if external:
        manifest["solver"].update(
            {
                "schema_version": "robot-scene-state-cpu-solver-v1",
                "backend": "NOT_USED_EXTERNAL_BASE_Q_AUTHORITY",
                "max_position_residual_mm": None,
                "max_rotation_residual_deg": None,
                "mean_position_residual_mm": None,
                "mean_rotation_residual_deg": None,
                "joint_limit_violations": 0,
                "base_is_one_session_constant": True,
                "invalid_sides_interpolated": False,
                "structural_parameter_count": 0,
                "structural_pose_constraint_count": 0,
                "structural_unknown_minus_constraint_count": 0,
            }
        )
        direct_flags = {
            "independent_of_r2_wrist_targets": True,
            "derived_from_r2_wrist_targets": False,
            "derived_from_ik_solver": False,
            "selected_by_ik_residual": False,
            "interpolated_or_filled": False,
            "synthetic_fixture": False,
            "identity_fixture": False,
            "formal_consumer_allowed": False,
        }
        base_evidence = {
            **ref("camera-base-evidence.json", "f" * 64, 104),
            "schema_version": "robot-external-camera-base-evidence-v1",
            "session_id": state.session_id,
            "evidence_kind": "DIRECT_CAMERA_TO_ROBOT_BASE_CALIBRATION",
            "provider": "OWNER_CAMERA_BASE_CALIBRATION",
            "method": "DIRECT_CAMERA_TO_ROBOT_BASE_CALIBRATION",
            "T_camera_base_semantics": "p_camera = T_camera_base @ p_base",
            "transform_convention": "RIGHT_HANDED_XYZ",
            "translation_units": "metres",
            "timestamp_ns": 1,
            "timestamp_clock_id": "camera-monotonic-ns",
            "device_id": "robot-state-recorder-001",
            "calibration_id": "camera-base-calibration-001",
            "calibration_sha256": "c" * 64,
            **direct_flags,
            "T_camera_base": state.T_camera_base.tolist(),
        }
        q_by_side = {}
        for side_index, side in enumerate(("left", "right")):
            q_by_side[side] = {
                **ref(
                    f"q-arm-evidence-{side}.json",
                    ("1" if side == "left" else "2") * 64,
                    105 + side_index,
                ),
                "schema_version": "robot-external-q-arm-side-evidence-v1",
                "session_id": state.session_id,
                "side": side,
                "evidence_kind": "DIRECT_ARM_JOINT_ENCODER_STATE",
                "provider": f"OWNER_{side.upper()}_ARM_ENCODER",
                "method": "DIRECT_ARM_JOINT_ENCODER_READOUT",
                "frame_count": 3,
                "frame_range": {"start": 0, "stop_exclusive": 3, "contiguous": True},
                "frame_timestamp_mapping": (
                    "ONE_TIMESTAMP_PER_ZERO_BASED_SOURCE_FRAME_NO_INTERPOLATION"
                ),
                "timestamp_clock_id": "camera-monotonic-ns",
                "device_id": f"{side}-encoder-001",
                "calibration_id": f"{side}-encoder-calibration-001",
                "calibration_sha256": ("b" if side == "left" else "c") * 64,
                "axes_convention": "RIGHT_HANDED_XYZ",
                "q_arm_units": "radians",
                "array_digest_canonicalization": (
                    "NUMPY_DTYPE_STR_NUL_SHAPE_JSON_NUL_C_ORDER_BYTES_V1"
                ),
                "frame_names_sha256": "a" * 64,
                "timestamp_ns_sha256": "b" * 64,
                "valid_sha256": "c" * 64,
                "q_arm_sha256": "d" * 64,
                "arm_joint_names_sha256": "e" * 64,
                "joint_lower_sha256": "f" * 64,
                "joint_upper_sha256": "0" * 64,
                **direct_flags,
            }
        manifest["external_base_q_authority"] = {
            "schema_version": "robot-explicit-base-q-authority-v1",
            "descriptor": ref("BASE_Q_AUTHORITY.json", "d" * 64, 102),
            "arrays": ref("BASE_Q_AUTHORITY.npz", "e" * 64, 103),
            "direct_evidence": {
                "camera_base": base_evidence,
                "q_arm_by_side": q_by_side,
            },
            "urdf_refs": {
                "robot_asset_pin": ref("ROBOT_ASSET_PIN.json", "3" * 64, 107),
                "tianji_urdf": ref("tianji.urdf", "4" * 64, 108),
            },
            "source_refs": {
                "r2_sidecar": ref("r2-sidecar.npz", "5" * 64, 109),
                "hawor": ref("hawor.npz", "6" * 64, 110),
            },
            "session_id": state.session_id,
            "frame_count": 3,
            "selected_frame_indices": [0, 1, 2],
            "selected_frame_names": ["00000", "00001", "00002"],
            "selected_timestamp_ns": [1, 2, 3],
            "authority_kind": "DIRECT_EXTERNAL_BASE_AND_ARM_STATE",
            "evidence_kind": "DIRECT_EXTERNAL_BASE_AND_ARM_STATE_MEASUREMENT",
            "provider": "OWNER_EXTERNAL_ROBOT_STATE_AUTHORITY",
            "method": "DIRECT_CAMERA_BASE_CALIBRATION_AND_JOINT_ENCODER_STATE",
            "direct_measurement_evidence": True,
            **direct_flags,
            "timestamp_clock_id": "camera-monotonic-ns",
            "frame_timestamp_mapping": (
                "ONE_TIMESTAMP_PER_ZERO_BASED_SOURCE_FRAME_NO_INTERPOLATION"
            ),
            "device_id": "robot-state-recorder-001",
            "calibration_id": "camera-base-calibration-001",
            "calibration_sha256": "c" * 64,
            "T_camera_base_semantics": "p_camera = T_camera_base @ p_base",
            "q_arm_units": "radians",
            "solver_not_used": True,
            "residual_not_claimed": True,
        }
        manifest["sources"] = {
            "r2_sidecar": copy.deepcopy(
                manifest["external_base_q_authority"]["source_refs"]["r2_sidecar"]
            ),
            "hawor": copy.deepcopy(
                manifest["external_base_q_authority"]["source_refs"]["hawor"]
            ),
            "raw_metadata": {},
        }
        canonical = lambda value: hashlib.sha256(  # noqa: E731
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        state = decode_scene_state(
            scene_bytes(
                external=True,
                measured_mount=not development_only,
                mount_authority_lineage_sha256=canonical(mount_authority),
                external_authority_lineage_sha256=canonical(
                    manifest["external_base_q_authority"]
                ),
            )
        )
    return manifest, record, state


def test_external_authority_manifest_mode_is_strictly_joined_and_never_formal() -> None:
    manifest, record, state = _valid_scene_manifest(external=True)
    assets = load_pinned_robot_assets(PROJECT)
    validate_scene_manifest(
        manifest,
        state_record=record,
        state=state,
        cpu4_tool_definition=assets.tool_definition,
        renderer_tool_definition=assets.tool_definition,
    )
    assert manifest["formal_consumer_allowed"] is False
    assert manifest["next_bucket_blocked"] is True

    defects = []
    wrong_mode = copy.deepcopy(manifest)
    wrong_mode["scene_state_mode"] = "DEVELOPMENT_R2_WRIST_IK_SOLVER"
    defects.append(wrong_mode)
    solver_used = copy.deepcopy(manifest)
    solver_used["solver"]["solver_used"] = True
    defects.append(solver_used)
    residual_claimed = copy.deepcopy(manifest)
    residual_claimed["solver"]["residual_not_claimed"] = False
    defects.append(residual_claimed)
    wrong_sha = copy.deepcopy(manifest)
    wrong_sha["external_base_q_authority"]["arrays"]["sha256"] = "0" * 64
    defects.append(wrong_sha)
    missing_authority = copy.deepcopy(manifest)
    del missing_authority["external_base_q_authority"]
    defects.append(missing_authority)
    missing_mount_evidence = copy.deepcopy(manifest)
    del missing_mount_evidence["mount_authority"]["evidence_mode"]
    defects.append(missing_mount_evidence)
    for defect in defects:
        with pytest.raises(FullChainRendererError):
            validate_scene_manifest(
                defect,
                state_record=record,
                state=state,
                cpu4_tool_definition=assets.tool_definition,
                renderer_tool_definition=assets.tool_definition,
            )


@pytest.mark.parametrize(
    "defect",
    [
        "timestamp_join",
        "q_r2_derived",
        "base_calibration",
        "urdf_sha",
        "evidence_inode_alias",
        "top_source_join",
        "extra_formal_claim",
    ],
)
def test_external_consumer_rejects_lineage_or_claim_spoof(
    defect: str,
) -> None:
    manifest, record, state = _valid_scene_manifest(external=True)
    authority = manifest["external_base_q_authority"]
    assert isinstance(authority, dict)
    if defect == "timestamp_join":
        authority["selected_timestamp_ns"] = [1, 2, 4]
    elif defect == "q_r2_derived":
        authority["direct_evidence"]["q_arm_by_side"]["left"][
            "derived_from_r2_wrist_targets"
        ] = True
    elif defect == "base_calibration":
        authority["direct_evidence"]["camera_base"]["calibration_id"] = "other"
    elif defect == "urdf_sha":
        authority["urdf_refs"]["tianji_urdf"]["sha256"] = "0" * 64
    elif defect == "evidence_inode_alias":
        authority["direct_evidence"]["q_arm_by_side"]["right"]["inode"] = authority[
            "direct_evidence"
        ]["q_arm_by_side"]["left"]["inode"]
    elif defect == "top_source_join":
        manifest["sources"]["r2_sidecar"]["sha256"] = "0" * 64
    else:
        manifest["formal"] = True
    assets = load_pinned_robot_assets(PROJECT)
    with pytest.raises(FullChainRendererError):
        validate_scene_manifest(
            manifest,
            state_record=record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("timestamp_clock_id", ""),
        ("time_scope", "PER_FRAME"),
        ("source_link", "wrong_hand_root"),
        ("derived_from_ik_solver", True),
        ("calibration_sha256", "0" * 64),
    ],
)
def test_measured_mount_consumer_joins_side_time_device_and_calibration(
    field: str, unsafe: object
) -> None:
    manifest, record, state = _valid_scene_manifest(
        development_only=False, external=True
    )
    mount = manifest["mount_authority"]
    assert isinstance(mount, dict)
    mount["evidence_by_side"]["left"][field] = unsafe
    assets = load_pinned_robot_assets(PROJECT)
    with pytest.raises(FullChainRendererError):
        validate_scene_manifest(
            manifest,
            state_record=record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("completion_mode", "EXIT_ZERO"),
        ("status", "FORMAL"),
        ("status", "READY"),
        ("visual_only", False),
        ("candidate_requires_human_review", False),
        ("next_bucket_blocked", False),
        ("advancement_authorized", True),
        ("formal_consumer_allowed", True),
        ("baseline_frozen", True),
        ("ik_residual_is_independent_accuracy_evidence", True),
        ("q_arm_and_camera_base_authoritative", True),
    ],
)
def test_scene_manifest_rejects_unsafe_terminal_or_claim_fields(
    field: str, unsafe: object
) -> None:
    manifest, record, state = _valid_scene_manifest()
    manifest[field] = unsafe
    assets = load_pinned_robot_assets(PROJECT)
    with pytest.raises(FullChainRendererError):
        validate_scene_manifest(
            manifest,
            state_record=record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )


@pytest.mark.parametrize(
    ("section", "field", "unsafe"),
    [
        ("mount_authority", "session_constant", False),
        ("mount_authority", "per_frame_mount_forbidden", False),
        ("mount_authority", "independent_of_r2_wrist_targets", False),
        ("mount_authority", "selected_by_ik_residual", True),
        ("solver", "mount_optimized_or_selected", True),
        ("solver", "objective_and_reported_residual_use_same_r2_wrist_target", False),
        ("solver", "reported_residual_is_independent_validation", True),
        ("solver", "independent_camera_base_or_arm_image_authority_consumed", True),
        ("solver", "q_arm_and_camera_base_unique_or_authoritative", True),
    ],
)
def test_scene_manifest_rejects_circular_mount_or_residual_evidence(
    section: str, field: str, unsafe: object
) -> None:
    manifest, record, state = _valid_scene_manifest()
    nested = manifest[section]
    assert isinstance(nested, dict)
    nested[field] = unsafe
    assets = load_pinned_robot_assets(PROJECT)
    with pytest.raises(FullChainRendererError):
        validate_scene_manifest(
            manifest,
            state_record=record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [("session_id", "grap_a_cap_002"), ("frame_count", 2), ("frame_count", True)],
)
def test_scene_manifest_binds_session_and_frame_count_to_decoded_state(
    field: str, unsafe: object
) -> None:
    manifest, record, state = _valid_scene_manifest()
    manifest[field] = unsafe
    assets = load_pinned_robot_assets(PROJECT)
    with pytest.raises(FullChainRendererError, match="decoded state"):
        validate_scene_manifest(
            manifest,
            state_record=record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )


def test_measured_mount_remains_visual_only_candidate_not_formal_consumer() -> None:
    manifest, record, state = _valid_scene_manifest(
        development_only=False, external=True
    )
    assets = load_pinned_robot_assets(PROJECT)
    validate_scene_manifest(
        manifest,
        state_record=record,
        state=state,
        cpu4_tool_definition=assets.tool_definition,
        renderer_tool_definition=assets.tool_definition,
    )
    assert manifest["status"] == SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS
    assert manifest["formal_consumer_allowed"] is False


def test_synthetic_mount_cannot_impersonate_non_development_evidence() -> None:
    manifest, record, state = _valid_scene_manifest()
    manifest["development_only"] = False
    manifest["status"] = SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS
    assets = load_pinned_robot_assets(PROJECT)
    with pytest.raises(FullChainRendererError, match="development-only"):
        validate_scene_manifest(
            manifest,
            state_record=record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )


def test_output_manifest_never_claims_physical_or_formal_use() -> None:
    assets = load_pinned_robot_assets(PROJECT)
    manifest = output_manifest_skeleton(
        session_id="grap_a_cap_004",
        frame_start=0,
        frame_count=460,
        resolution=(320, 240),
        tool_definition=assets.tool_definition,
        source_records={},
    )
    assert manifest["mount_provenance"] == MOUNT_PROVENANCE
    assert manifest["contact_infeasible"] == CONTACT_INFEASIBLE
    assert manifest["formal_consumer_allowed"] is False
    assert manifest["geometry_contract"]["collision_geometry_consumed"] is False
