from __future__ import annotations

import copy
from dataclasses import replace
import errno
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np
import pytest

from pipeline.robot_renderer_eevee_fullchain import (
    ARM_JOINT_NAMES,
    SCENE_MANIFEST_DEVELOPMENT_STATUS,
    SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS,
    SCENE_STATE_MODE_EXTERNAL_AUTHORITY,
    SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER,
    FullChainRendererError,
    compose_frame_placement,
    decode_scene_state,
    load_pinned_robot_assets,
    load_strict_io,
    validate_scene_manifest,
)
from pipeline.robot_scene_state_cpu import (
    BASE_TRANSFORM_SEMANTICS,
    EXTERNAL_BASE_Q_ARRAY_SCHEMA,
    EXTERNAL_BASE_Q_SCHEMA,
    EXTERNAL_BASE_EVIDENCE_KIND,
    EXTERNAL_BASE_EVIDENCE_SCHEMA,
    EXTERNAL_AUTHORITY_EVIDENCE_KIND,
    EXTERNAL_AUTHORITY_METHOD,
    EXTERNAL_ARRAY_DIGEST_CANONICALIZATION,
    EXTERNAL_FRAME_TIMESTAMP_MAPPING,
    EXTERNAL_Q_ARM_EVIDENCE_KIND,
    EXTERNAL_Q_ARM_SIDE_EVIDENCE_SCHEMA,
    FIXED_CAMERA_BASE_CANDIDATE_SCHEMA,
    FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA,
    MOUNT_COORDINATE_DEFINITION,
    MOUNT_MATRIX_DIRECTION,
    MOUNT_SCHEMA,
    MOUNT_SIDE_EVIDENCE_SCHEMA,
    MOUNT_TIME_SCOPE,
    MOUNT_SOURCE_LINKS,
    MOUNT_TARGET_LINKS,
    RIGHT_HANDED_AXES,
    SceneStateProducerError,
    SolverResult,
    SourceBundle,
    _require_session_path,
    _canonical_array_sha256,
    build_manifest,
    encode_scene_state,
    fixed_base_gate_summary,
    load_external_base_q_authority,
    load_explicit_mount,
    load_fixed_camera_base_candidate,
    load_sources,
    preflight_scene_manifest,
    read_bound_bytes,
    sha256_bytes,
    solve_scene_state,
    solve_scene_state_fixed_base,
)
import pipeline.robot_scene_state_cpu as scene_state_module
import tools.run_robot_scene_state_cpu as scene_state_runner
from tools.run_robot_scene_state_cpu import (
    _READ_DIR_FLAGS,
    _create_private_staging_child,
    _held_directory_path_identity,
    _held_file_ref,
    _rename_noreplace,
    _revalidate_fixed_camera_base_descriptor,
    _rollback_scene_child,
    _resolve_frame_indices,
    _write_new_held_file,
    parser,
)


PROJECT = Path(__file__).resolve().parents[1]


def fake_ref(
    path: Path, *, payload_sha256: str, size: int, device: int, inode: int
) -> dict[str, Any]:
    return {
        "path": str(path.absolute()),
        "bytes": size,
        "sha256": payload_sha256,
        "device": device,
        "inode": inode,
    }


def mount_payload(
    *, session: str = "grap_a_cap_004", matrices: np.ndarray | None = None
) -> bytes:
    if matrices is None:
        matrices = np.repeat(np.eye(4)[None], 2, axis=0)
    value = {
        "schema_version": MOUNT_SCHEMA,
        "session_id": session,
        "mount_provenance": "PROVISIONAL_MOUNT_VISUAL_ONLY",
        "contact_infeasible": "UNMEASURED",
        "session_constant": True,
        "per_frame_mount_forbidden": True,
        "independent_of_r2_wrist_targets": True,
        "selected_by_ik_residual": False,
        "formal_consumer_allowed": False,
        "development_only": True,
        "authority": {
            "provider": "TEST_FIXTURE",
            "method": "EXPLICIT_SYNTHETIC_SE3",
            "coordinate_definition": MOUNT_COORDINATE_DEFINITION,
            "matrix_direction": MOUNT_MATRIX_DIRECTION,
            "axes_convention": RIGHT_HANDED_AXES,
            "translation_units": "metres",
            "rotation_units": "radians",
            "source_links": MOUNT_SOURCE_LINKS,
            "target_links": MOUNT_TARGET_LINKS,
            "independent_visual_evidence": False,
            "synthetic_fixture": True,
            "evidence_by_side": {},
        },
        "T_tool_hand": {
            "left": matrices[0].tolist(),
            "right": matrices[1].tolist(),
        },
    }
    return json.dumps(value).encode()


def fixed_camera_base_payload(matrix: np.ndarray | None = None) -> bytes:
    if matrix is None:
        matrix = np.eye(4, dtype=np.float64)
    value = {
        "schema_version": FIXED_CAMERA_BASE_CANDIDATE_SCHEMA,
        "T_camera_base_semantics": BASE_TRANSFORM_SEMANTICS,
        "axes_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "fixed_across_selected_frames": True,
        "global_across_sessions": True,
        "selected_on_session": "grap_a_cap_004",
        "not_per_session_tuned": True,
        "per_frame_override_forbidden": True,
        "per_session_override_forbidden": True,
        "development_only": True,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "T_camera_base": np.asarray(matrix, dtype=np.float64).tolist(),
    }
    return json.dumps(value, sort_keys=True).encode()


def write_fixed_camera_base_candidate(
    tmp_path: Path, matrix: np.ndarray | None = None
) -> tuple[Any, Any]:
    path = (tmp_path / "FIXED_CAMERA_BASE.json").absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = fixed_camera_base_payload(matrix)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o440,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    strict = load_strict_io(PROJECT)
    record = read_bound_bytes(strict, path, name="test fixed camera-base candidate")
    return record, load_fixed_camera_base_candidate(
        record.payload, source_record=record
    )


def test_mount_input_requires_independent_constant_bilateral_se3() -> None:
    mount = load_explicit_mount(mount_payload(), expected_session_id="grap_a_cap_004")
    assert mount.matrices.shape == (2, 4, 4)
    value = json.loads(mount_payload())
    value["independent_of_r2_wrist_targets"] = False
    with pytest.raises(SceneStateProducerError, match="independent"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )


def test_checked_in_mount_fixture_remains_synthetic_development_blocked() -> None:
    strict = load_strict_io(PROJECT)
    path = PROJECT / "tests/fixtures/robot_scene_state_explicit_dev_mount_004.json"
    record = read_bound_bytes(strict, path, name="checked-in mount fixture")
    mount = load_explicit_mount(
        record.payload,
        expected_session_id="grap_a_cap_004",
        strict=strict,
        source_record=record,
    )
    assert mount.development_only is True
    assert mount.evidence_mode == "DEVELOPMENT_ONLY_SYNTHETIC_FIXTURE"
    assert mount.evidence_records == {}
    value = json.loads(mount_payload())
    value["T_tool_hand"]["left"] = [np.eye(4).tolist(), np.eye(4).tolist()]
    with pytest.raises(SceneStateProducerError, match="session-constant"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )


def test_mount_input_rejects_blind_session_before_any_solver() -> None:
    with pytest.raises(SceneStateProducerError, match="session"):
        load_explicit_mount(
            mount_payload(session="grap_a_cap_025"),
            expected_session_id="grap_a_cap_004",
        )


def test_mount_input_rejects_formal_override_bad_axis_and_non_se3() -> None:
    value = json.loads(mount_payload())
    value["formal_consumer_allowed"] = True
    with pytest.raises(SceneStateProducerError, match="formal_consumer_allowed"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )

    value = json.loads(mount_payload())
    value["authority"]["coordinate_definition"] = "KaiHand root to Tianji tool"
    with pytest.raises(SceneStateProducerError, match="coordinate_definition"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )

    value = json.loads(mount_payload())
    value["T_tool_hand"]["left"][0][0] = 2.0
    with pytest.raises(SceneStateProducerError, match="orthonormal"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )


def test_fixed_camera_base_candidate_is_oexcl_same_fd_global_and_blocked(
    tmp_path: Path,
) -> None:
    record, candidate = write_fixed_camera_base_candidate(tmp_path)
    assert candidate.matrix.dtype == np.float64
    assert np.array_equal(candidate.matrix, np.eye(4, dtype=np.float64))
    assert candidate.descriptor_record == record.evidence_ref()
    assert candidate.contract["global_across_sessions"] is True
    assert candidate.contract["selected_on_session"] == "grap_a_cap_004"
    assert candidate.contract["not_per_session_tuned"] is True
    assert candidate.contract["formal_consumer_allowed"] is False
    with pytest.raises(FileExistsError):
        descriptor = os.open(
            record.path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o440,
        )
        os.close(descriptor)


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("global_across_sessions", False),
        ("global_across_sessions", 1),
        ("selected_on_session", "grap_a_cap_002"),
        ("not_per_session_tuned", False),
        ("formal_consumer_allowed", True),
    ],
)
def test_fixed_camera_base_candidate_rejects_session_tuning_or_formal_claim(
    tmp_path: Path, field: str, unsafe: Any
) -> None:
    value = json.loads(fixed_camera_base_payload())
    value[field] = unsafe
    path = (tmp_path / "candidate.json").absolute()
    path.write_text(json.dumps(value, sort_keys=True))
    record = read_bound_bytes(
        load_strict_io(PROJECT), path, name="bad fixed camera-base candidate"
    )
    with pytest.raises(SceneStateProducerError, match="global, development-only"):
        load_fixed_camera_base_candidate(record.payload, source_record=record)


def test_fixed_camera_base_candidate_rejects_payload_ref_drift_and_non_se3(
    tmp_path: Path,
) -> None:
    record, _ = write_fixed_camera_base_candidate(tmp_path / "good")
    changed = np.eye(4, dtype=np.float64)
    changed[0, 3] = 0.1
    with pytest.raises(SceneStateProducerError, match="same-FD"):
        load_fixed_camera_base_candidate(
            fixed_camera_base_payload(changed),
            source_record=record,
        )

    reflected = np.eye(4, dtype=np.float64)
    reflected[0, 0] = -1.0
    path = (tmp_path / "bad" / "candidate.json").absolute()
    path.parent.mkdir()
    path.write_bytes(fixed_camera_base_payload(reflected))
    bad_record = read_bound_bytes(
        load_strict_io(PROJECT), path, name="reflected fixed camera-base candidate"
    )
    with pytest.raises(SceneStateProducerError, match="right-handed"):
        load_fixed_camera_base_candidate(bad_record.payload, source_record=bad_record)

    string_matrix = json.loads(fixed_camera_base_payload())
    string_matrix["T_camera_base"][0][0] = "1.0"
    string_path = (tmp_path / "string" / "candidate.json").absolute()
    string_path.parent.mkdir()
    string_path.write_text(json.dumps(string_matrix, sort_keys=True))
    string_record = read_bound_bytes(
        load_strict_io(PROJECT),
        string_path,
        name="string-valued fixed camera-base candidate",
    )
    with pytest.raises(SceneStateProducerError, match="exact numeric"):
        load_fixed_camera_base_candidate(
            string_record.payload, source_record=string_record
        )


def test_fixed_camera_base_candidate_revalidation_rejects_path_replacement(
    tmp_path: Path,
) -> None:
    record, candidate = write_fixed_camera_base_candidate(tmp_path)
    replacement = tmp_path / "replacement.json"
    replacement_matrix = candidate.matrix.copy()
    replacement_matrix[0, 3] = 0.1
    replacement.write_bytes(fixed_camera_base_payload(replacement_matrix))
    os.replace(replacement, record.path)
    with pytest.raises(SceneStateProducerError, match="identity drift"):
        _revalidate_fixed_camera_base_descriptor(
            load_strict_io(PROJECT), candidate.descriptor_record
        )


def test_cli_has_no_formal_or_mount_fit_override() -> None:
    destinations = {action.dest for action in parser()._actions}
    assert "base_q_authority_json" in destinations
    assert "fixed_camera_base_json" in destinations
    assert "frame_range" in destinations
    assert "check_only" in destinations
    assert "formal_consumer_allowed" not in destinations
    assert "fit_mount" not in destinations
    assert "select_mount_by_residual" not in destinations
    assert "governance_bypassed" not in destinations


def test_cli_fixed_base_is_exclusive_and_frame_ranges_are_exact() -> None:
    common = [
        "--session-id",
        "grap_a_cap_004",
        "--r2-sidecar",
        "/tmp/r2.npz",
        "--hawor-npz",
        "/tmp/hawor.npz",
        "--raw-root",
        "/tmp/raw",
        "--mount-json",
        "/tmp/mount.json",
        "--frame-range",
        "2",
        "5",
        "--frame-range",
        "8",
        "10",
        "--check-only",
    ]
    parsed = parser().parse_args(
        [*common, "--fixed-camera-base-json", "/tmp/base.json"]
    )
    assert _resolve_frame_indices(parsed) == (2, 3, 4, 8, 9)
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                *common,
                "--fixed-camera-base-json",
                "/tmp/base.json",
                "--base-q-authority-json",
                "/tmp/authority.json",
            ]
        )

    overlap = SimpleNamespace(frame_index=None, frame_range=[[2, 5], [4, 7]])
    with pytest.raises(SceneStateProducerError, match="without overlap"):
        _resolve_frame_indices(overlap)


def test_source_paths_require_one_exact_requested_session_component() -> None:
    _require_session_path(
        Path("/root/grap_a_cap_004/source.npz"),
        session_id="grap_a_cap_004",
        name="source",
    )
    with pytest.raises(SceneStateProducerError, match="requested session"):
        _require_session_path(
            Path("/root/grap_a_cap_002/source.npz"),
            session_id="grap_a_cap_004",
            name="source",
        )
    with pytest.raises(SceneStateProducerError, match="requested session"):
        _require_session_path(
            Path("/root/grap_a_cap_004/grap_a_cap_002/source.npz"),
            session_id="grap_a_cap_004",
            name="source",
        )


def test_same_fd_reader_rejects_symlink_and_hardlink_aliases(tmp_path: Path) -> None:
    strict = load_strict_io(PROJECT)
    source = tmp_path / "source.bin"
    source.write_bytes(b"scene-source")
    record = read_bound_bytes(strict, source, name="source")
    assert record.payload == b"scene-source"

    symlink = tmp_path / "symlink.bin"
    symlink.symlink_to(source)
    with pytest.raises(SceneStateProducerError, match="ordinary non-symlink"):
        read_bound_bytes(strict, symlink, name="symlink")

    hardlink = tmp_path / "hardlink.bin"
    os.link(source, hardlink)
    with pytest.raises(SceneStateProducerError, match="hard-link"):
        read_bound_bytes(strict, source, name="hardlink source")


def test_same_fd_reader_rejects_path_replacement_between_preflight_and_open(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    replacement = tmp_path / "replacement.json"
    path.write_bytes(b'{"old":true}')
    replacement.write_bytes(b'{"new":true}')

    class SwapBeforeOpen:
        @staticmethod
        def read_bytes_nofollow(requested: Path) -> Any:
            os.replace(replacement, requested)
            payload = requested.read_bytes()
            info = requested.stat()
            return SimpleNamespace(
                path=requested,
                payload=payload,
                bytes=len(payload),
                sha256=sha256_bytes(payload),
                device=info.st_dev,
                inode=info.st_ino,
            )

    with pytest.raises(SceneStateProducerError, match="identity changed"):
        read_bound_bytes(SwapBeforeOpen(), path, name="authority")


def _write_real_mount_bundle(
    tmp_path: Path,
    *,
    development_only: bool = False,
    mutate_evidence: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Any, np.ndarray, dict[str, Path]]:
    strict = load_strict_io(PROJECT)
    root = tmp_path / "grap_a_cap_004"
    root.mkdir(parents=True, exist_ok=True)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    matrices[0, :3, 3] = [0.01, -0.02, 0.03]
    matrices[1, :3, 3] = [-0.01, -0.02, 0.03]
    evidence_paths: dict[str, Path] = {}
    evidence_refs: dict[str, Mapping[str, Any]] = {}
    for index, side in enumerate(("left", "right")):
        evidence = {
            "schema_version": MOUNT_SIDE_EVIDENCE_SCHEMA,
            "session_id": "grap_a_cap_004",
            "side": side,
            "evidence_kind": "DIRECT_VISUAL_MOUNT_MEASUREMENT",
            "provider": f"OWNER_MOUNT_CAMERA_{side.upper()}",
            "method": "DIRECT_VISUAL_MOUNT_MEASUREMENT",
            "source_link": MOUNT_SOURCE_LINKS[side],
            "target_link": MOUNT_TARGET_LINKS[side],
            "matrix_direction": MOUNT_MATRIX_DIRECTION,
            "axes_convention": RIGHT_HANDED_AXES,
            "translation_units": "metres",
            "rotation_units": "radians",
            "timestamp_ns": 1_000_000 + index,
            "timestamp_clock_id": "mount-calibration-monotonic-ns",
            "time_scope": MOUNT_TIME_SCOPE,
            "device_id": f"camera-{side}",
            "calibration_id": f"mount-calibration-{side}",
            "calibration_sha256": ("a" if side == "left" else "b") * 64,
            "independent_of_r2_wrist_targets": True,
            "derived_from_r2_wrist_targets": False,
            "derived_from_ik_solver": False,
            "selected_by_ik_residual": False,
            "circular_selection": False,
            "synthetic_fixture": False,
            "identity_fixture": False,
            "formal_consumer_allowed": False,
            "T_tool_hand": matrices[index].tolist(),
        }
        if mutate_evidence is not None:
            mutate_evidence(side, evidence)
        path = root / f"MOUNT_EVIDENCE_{side.upper()}.json"
        path.write_text(json.dumps(evidence, sort_keys=True))
        evidence_paths[side] = path
        evidence_refs[side] = strict.read_bytes_nofollow(path).evidence_ref()
    mount = json.loads(mount_payload(matrices=matrices))
    mount["development_only"] = development_only
    mount["authority"].update(
        {
            "provider": "INDEPENDENT_VISUAL_CALIBRATION_RIG",
            "method": "DIRECT_BILATERAL_VISUAL_MOUNT_MEASUREMENT",
            "independent_visual_evidence": True,
            "synthetic_fixture": False,
            "evidence_by_side": evidence_refs,
        }
    )
    path = root / "MOUNT.json"
    path.write_text(json.dumps(mount, sort_keys=True))
    return strict.read_bytes_nofollow(path), matrices, evidence_paths


def test_real_mount_requires_two_independent_same_fd_evidence_files(
    tmp_path: Path,
) -> None:
    strict = load_strict_io(PROJECT)
    mount_record, matrices, _ = _write_real_mount_bundle(tmp_path)
    mount = load_explicit_mount(
        mount_record.payload,
        expected_session_id="grap_a_cap_004",
        strict=strict,
        source_record=mount_record,
    )
    assert np.array_equal(mount.matrices, matrices)
    assert mount.evidence_mode == "INDEPENDENT_BILATERAL_VISUAL_EVIDENCE"
    assert set(mount.evidence_records) == {"left", "right"}
    assert (
        mount.evidence_records["left"]["inode"]
        != mount.evidence_records["right"]["inode"]
    )


def test_real_mount_rejects_evidence_byte_drift_and_bilateral_alias(
    tmp_path: Path,
) -> None:
    strict = load_strict_io(PROJECT)
    mount_record, _, evidence_paths = _write_real_mount_bundle(tmp_path)
    evidence_paths["left"].write_text("{}")
    with pytest.raises(SceneStateProducerError, match="declared bytes/SHA"):
        load_explicit_mount(
            mount_record.payload,
            expected_session_id="grap_a_cap_004",
            strict=strict,
            source_record=mount_record,
        )

    alias_root = tmp_path / "alias_case"
    aliased_record, _, _ = _write_real_mount_bundle(alias_root)
    value = json.loads(aliased_record.payload)
    value["authority"]["evidence_by_side"]["right"] = value["authority"][
        "evidence_by_side"
    ]["left"]
    alias_path = alias_root / "grap_a_cap_004" / "MOUNT_ALIAS.json"
    alias_path.write_text(json.dumps(value, sort_keys=True))
    alias_record = strict.read_bytes_nofollow(alias_path)
    with pytest.raises(SceneStateProducerError, match="independent files"):
        load_explicit_mount(
            alias_record.payload,
            expected_session_id="grap_a_cap_004",
            strict=strict,
            source_record=alias_record,
        )


def test_synthetic_or_identity_mount_cannot_impersonate_independent_authority() -> None:
    value = json.loads(mount_payload())
    value["development_only"] = False
    with pytest.raises(SceneStateProducerError, match="development_only"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )

    value = json.loads(mount_payload())
    value["authority"]["synthetic_fixture"] = False
    value["authority"]["independent_visual_evidence"] = True
    value["authority"]["method"] = "DIRECT_BILATERAL_VISUAL_MOUNT_MEASUREMENT"
    with pytest.raises(SceneStateProducerError, match="identity mount"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("timestamp_clock_id", ""),
        ("time_scope", "PER_FRAME"),
        ("derived_from_r2_wrist_targets", True),
        ("derived_from_ik_solver", True),
        ("selected_by_ik_residual", True),
        ("circular_selection", True),
        ("formal_consumer_allowed", True),
        ("source_link", "wrong_hand_root"),
    ],
)
def test_real_mount_evidence_rejects_time_link_or_circular_lineage(
    tmp_path: Path, field: str, unsafe: Any
) -> None:
    strict = load_strict_io(PROJECT)

    def mutate(side: str, evidence: dict[str, Any]) -> None:
        if side == "left":
            evidence[field] = unsafe

    mount_record, _, _ = _write_real_mount_bundle(tmp_path, mutate_evidence=mutate)
    with pytest.raises(SceneStateProducerError):
        load_explicit_mount(
            mount_record.payload,
            expected_session_id="grap_a_cap_004",
            strict=strict,
            source_record=mount_record,
        )


def test_mount_descriptor_rejects_unknown_claim_aliases() -> None:
    value = json.loads(mount_payload())
    value["formal"] = True
    with pytest.raises(SceneStateProducerError, match="keys are not exact"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )

    value = json.loads(mount_payload())
    value["authority"]["formal_consumer_allowed"] = True
    with pytest.raises(SceneStateProducerError, match="keys are not exact"):
        load_explicit_mount(
            json.dumps(value).encode(), expected_session_id="grap_a_cap_004"
        )


def _source_fixture(
    tmp_path: Path, *, mutate_r2: Callable[[dict[str, np.ndarray]], None]
) -> tuple[Path, Path, Path]:
    r2_path = tmp_path / "r2" / "grap_a_cap_004" / "sidecar.npz"
    hawor_path = tmp_path / "hawor" / "grap_a_cap_004" / "joints.npz"
    raw_root = tmp_path / "raw" / "grap_a_cap_004"
    r2_path.parent.mkdir(parents=True)
    hawor_path.parent.mkdir(parents=True)
    raw_root.mkdir(parents=True)
    assets = load_pinned_robot_assets(PROJECT)
    count = 3
    hand_models = (assets.left_hand, assets.right_hand)
    hand_names = np.asarray(
        [
            [joint.name for joint in model.joints if joint.joint_type != "fixed"]
            for model in hand_models
        ]
    )
    lower = np.asarray(
        [
            [joint.lower for joint in model.joints if joint.joint_type != "fixed"]
            for model in hand_models
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            [joint.upper for joint in model.joints if joint.joint_type != "fixed"]
            for model in hand_models
        ],
        dtype=np.float64,
    )
    arrays = {
        "schema_version": np.asarray("humanego-robot-sidecar-v1"),
        "frame_names": np.asarray([f"{index:05d}" for index in range(count)]),
        "q": np.zeros((count, 2, 22), dtype=np.float64),
        "valid": np.ones((count, 2), dtype=np.bool_),
        "wrist_T_camera": np.repeat(
            np.eye(4, dtype=np.float64)[None, None], count * 2, axis=0
        ).reshape(count, 2, 4, 4),
        "joint_names": hand_names,
        "joint_lower": lower,
        "joint_upper": upper,
        "canonical_relative_path": np.asarray(
            "common/grap_a_cap_004/canonical_source.npz"
        ),
        "embodiment": np.asarray("kai22"),
        "q_unit": np.asarray("radian"),
        "translation_unit": np.asarray("metre"),
    }
    mutate_r2(arrays)
    np.savez_compressed(r2_path, **arrays)
    intrinsics = np.repeat(
        np.asarray([[640.0, 640.0, 639.5, 479.5]], dtype=np.float64),
        count,
        axis=0,
    )
    np.savez_compressed(
        hawor_path,
        valid=np.ones((2, count), dtype=np.bool_),
        camera_intrinsics=intrinsics,
    )
    for index in range(count):
        frame = raw_root / f"{index:05d}"
        frame.mkdir()
        frame.joinpath("training_data.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "k": [
                            [640.0, 0.0, 639.5],
                            [0.0, 640.0, 479.5],
                            [0.0, 0.0, 1.0],
                        ],
                        "w": 1280,
                        "h": 960,
                    }
                }
            )
        )
    return r2_path, hawor_path, raw_root


def _load_local_sources(tmp_path: Path) -> tuple[SourceBundle, Any]:
    r2_path, hawor_path, raw_root = _source_fixture(
        tmp_path, mutate_r2=lambda arrays: None
    )
    assets = load_pinned_robot_assets(PROJECT)
    return (
        load_sources(
            strict=load_strict_io(PROJECT),
            session_id="grap_a_cap_004",
            sidecar_path=r2_path,
            hawor_path=hawor_path,
            raw_root=raw_root,
            frame_indices=[0, 1, 2],
            assets=assets,
        ),
        assets,
    )


def test_multiframe_manifest_preflight_then_binds_exact_published_inode(
    tmp_path: Path,
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")
    frame_count = len(sources.frame_names)
    solver = SolverResult(
        q_arm=np.zeros((frame_count, 2, 7), dtype=np.float64),
        T_camera_base=np.eye(4, dtype=np.float64),
        position_residual_mm=np.zeros((frame_count, 2), dtype=np.float64),
        rotation_residual_deg=np.zeros((frame_count, 2), dtype=np.float64),
        evaluations=np.ones((frame_count, 2), dtype=np.int32),
    )
    mount_bytes = mount_payload()
    mount = load_explicit_mount(mount_bytes, expected_session_id="grap_a_cap_004")
    state_payload = encode_scene_state(
        session_id="grap_a_cap_004",
        sources=sources,
        mounts=mount,
        solver=solver,
    )
    state = decode_scene_state(state_payload)
    mount_record = fake_ref(
        tmp_path / "MOUNT.json",
        payload_sha256=sha256_bytes(mount_bytes),
        size=len(mount_bytes),
        device=910,
        inode=920,
    )
    state_path = tmp_path / "output" / "SCENE_STATE.npz"
    command = ["/usr/bin/python3", "producer.py"]

    with pytest.raises(FullChainRendererError, match="session"):
        preflight_scene_manifest(
            session_id="grap_a_cap_002",
            state_path=state_path,
            state_payload=state_payload,
            state=state,
            mount_record=mount_record,
            mount=mount,
            sources=sources,
            solver=solver,
            assets=assets,
            command=command,
        )
    assert not state_path.exists()

    preflight_scene_manifest(
        session_id="grap_a_cap_004",
        state_path=state_path,
        state_payload=state_payload,
        state=state,
        mount_record=mount_record,
        mount=mount,
        sources=sources,
        solver=solver,
        assets=assets,
        command=command,
    )
    assert not state_path.exists()

    state_path.parent.mkdir()
    strict = load_strict_io(PROJECT)
    writer_record = strict.write_new_bytes(
        state_path, state_payload, allowed_root=state_path.parent
    )
    assert set(writer_record) == {"path", "bytes", "sha256"}
    with pytest.raises(
        FullChainRendererError,
        match="manifest scene state evidence ref keys are not exact",
    ):
        build_manifest(
            session_id="grap_a_cap_004",
            state_record=writer_record,
            state=state,
            mount_record=mount_record,
            mount=mount,
            sources=sources,
            solver=solver,
            assets=assets,
            command=command,
        )

    state_record = read_bound_bytes(
        strict, state_path, name="test scene state"
    ).evidence_ref()
    manifest = build_manifest(
        session_id="grap_a_cap_004",
        state_record=state_record,
        state=state,
        mount_record=mount_record,
        mount=mount,
        sources=sources,
        solver=solver,
        assets=assets,
        command=command,
    )
    assert manifest["frame_count"] == 3
    assert set(manifest["scene_state"]) == {
        "path",
        "bytes",
        "sha256",
        "device",
        "inode",
    }
    assert manifest["scene_state"] == state_record
    assert "governance_bypassed" not in manifest


def test_manifest_failure_rollback_removes_only_held_files_and_new_child(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    child_fd = -1
    held_files: list[tuple[str, int]] = []
    try:
        os.mkdir("scene-child", dir_fd=parent_fd)
        child = parent / "scene-child"
        child_fd = os.open("scene-child", _READ_DIR_FLAGS, dir_fd=parent_fd)
        child_identity = os.fstat(child_fd)
        state_fd, _ = _write_new_held_file(
            child_fd,
            path=child / "SCENE_STATE.npz",
            payload=b"scene-state-payload",
        )
        held_files.append(("SCENE_STATE.npz", state_fd))
        manifest_fd, _ = _write_new_held_file(
            child_fd,
            path=child / "SCENE_MANIFEST.json",
            payload=b"{}\n",
        )
        held_files.append(("SCENE_MANIFEST.json", manifest_fd))

        _rollback_scene_child(
            parent_fd=parent_fd,
            child_fd=child_fd,
            child_name="scene-child",
            child_identity=child_identity,
            held_files=held_files,
        )
        assert not child.exists()
    finally:
        for _, descriptor in held_files:
            os.close(descriptor)
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def test_child_open_failure_rolls_back_exact_empty_created_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    real_open = os.open
    try:
        os.mkdir("scene-child", dir_fd=parent_fd)
        child_identity = os.stat("scene-child", dir_fd=parent_fd, follow_symlinks=False)
        failed = False

        def fail_first_child_open(
            path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal failed
            if path == "scene-child" and dir_fd == parent_fd and not failed:
                failed = True
                raise OSError("injected child open failure")
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", fail_first_child_open)
        with pytest.raises(OSError, match="injected child open failure"):
            os.open("scene-child", _READ_DIR_FLAGS, dir_fd=parent_fd)
        _rollback_scene_child(
            parent_fd=parent_fd,
            child_fd=-1,
            child_name="scene-child",
            child_identity=child_identity,
            held_files=[],
        )
        assert not (parent / "scene-child").exists()
    finally:
        os.close(parent_fd)


def test_child_identity_drift_rolls_back_owned_name_not_foreign_descriptor(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "foreign").mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    foreign_fd = os.open(parent / "foreign", _READ_DIR_FLAGS)
    try:
        os.mkdir("scene-child", dir_fd=parent_fd)
        child_identity = os.stat("scene-child", dir_fd=parent_fd, follow_symlinks=False)
        _rollback_scene_child(
            parent_fd=parent_fd,
            child_fd=foreign_fd,
            child_name="scene-child",
            child_identity=child_identity,
            held_files=[],
        )
        assert not (parent / "scene-child").exists()
        assert (parent / "foreign").is_dir()
    finally:
        os.close(foreign_fd)
        os.close(parent_fd)


def test_child_replacement_is_never_deleted_by_exact_rollback(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    child_fd = -1
    try:
        os.mkdir("scene-child", dir_fd=parent_fd)
        child_fd = os.open("scene-child", _READ_DIR_FLAGS, dir_fd=parent_fd)
        child_identity = os.fstat(child_fd)
        os.rename(
            "scene-child", "owned-moved", src_dir_fd=parent_fd, dst_dir_fd=parent_fd
        )
        os.mkdir("scene-child", dir_fd=parent_fd)

        with pytest.raises(SceneStateProducerError, match="replaced scene child"):
            _rollback_scene_child(
                parent_fd=parent_fd,
                child_fd=child_fd,
                child_name="scene-child",
                child_identity=child_identity,
                held_files=[],
            )
        assert (parent / "scene-child").is_dir()
        assert (parent / "owned-moved").is_dir()
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def test_parent_replacement_before_open_fails_expected_identity_binding(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    expected_identity = parent.lstat()
    parent.rename(tmp_path / "owned-moved")
    parent.mkdir()
    replacement_fd = os.open(parent, _READ_DIR_FLAGS)
    try:
        with pytest.raises(SceneStateProducerError, match="binding drift"):
            _held_directory_path_identity(
                replacement_fd,
                path=parent,
                name="test parent",
                expected_identity=expected_identity,
            )
        assert not (parent / "scene-child").exists()
        assert not (tmp_path / "owned-moved" / "scene-child").exists()
    finally:
        os.close(replacement_fd)


def test_parent_replacement_after_open_fails_lexical_binding(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    expected_identity = os.fstat(parent_fd)
    try:
        parent.rename(tmp_path / "owned-moved")
        parent.mkdir()
        with pytest.raises(SceneStateProducerError, match="binding drift"):
            _held_directory_path_identity(
                parent_fd,
                path=parent,
                name="test parent",
                expected_identity=expected_identity,
            )
        assert not (parent / "scene-child").exists()
        assert not (tmp_path / "owned-moved" / "scene-child").exists()
    finally:
        os.close(parent_fd)


def test_private_staging_binds_files_before_atomic_noreplace_publish(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    child_fd = -1
    held_files: list[int] = []
    staging_name = f".robot-scene-state-{'a' * 64}.tmp"
    try:
        parent_identity = parent.lstat()
        child_fd, _ = _create_private_staging_child(
            parent_fd=parent_fd,
            parent_path=parent,
            parent_identity=parent_identity,
            staging_name=staging_name,
        )
        staging = parent / staging_name
        output = parent / "scene-final"
        state_payload = b"state-payload"
        state_fd, state_ref = _write_new_held_file(
            child_fd,
            path=staging / "SCENE_STATE.npz",
            evidence_path=output / "SCENE_STATE.npz",
            payload=state_payload,
        )
        held_files.append(state_fd)
        manifest_payload = b"{}\n"
        manifest_fd, manifest_ref = _write_new_held_file(
            child_fd,
            path=staging / "SCENE_MANIFEST.json",
            evidence_path=output / "SCENE_MANIFEST.json",
            payload=manifest_payload,
        )
        held_files.append(manifest_fd)
        os.fsync(child_fd)

        _rename_noreplace(
            source_fd=parent_fd,
            source_name=staging_name,
            target_fd=parent_fd,
            target_name="scene-final",
        )

        assert not staging.exists()
        assert output.is_dir()
        assert (
            _held_file_ref(
                state_fd,
                path=output / "SCENE_STATE.npz",
                expected_payload=state_payload,
            )
            == state_ref
        )
        assert (
            _held_file_ref(
                manifest_fd,
                path=output / "SCENE_MANIFEST.json",
                expected_payload=manifest_payload,
            )
            == manifest_ref
        )
    finally:
        for descriptor in held_files:
            os.close(descriptor)
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def test_atomic_publish_never_replaces_existing_final_child(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    final = parent / "scene-final"
    final.mkdir()
    marker = final / "FOREIGN"
    marker.write_bytes(b"foreign")
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    child_fd = -1
    staging_name = f".robot-scene-state-{'b' * 64}.tmp"
    try:
        parent_identity = parent.lstat()
        child_fd, child_identity = _create_private_staging_child(
            parent_fd=parent_fd,
            parent_path=parent,
            parent_identity=parent_identity,
            staging_name=staging_name,
        )
        with pytest.raises(FileExistsError):
            _rename_noreplace(
                source_fd=parent_fd,
                source_name=staging_name,
                target_fd=parent_fd,
                target_name="scene-final",
            )
        assert marker.read_bytes() == b"foreign"
        _rollback_scene_child(
            parent_fd=parent_fd,
            child_fd=child_fd,
            child_name=staging_name,
            child_identity=child_identity,
            held_files=[],
        )
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def test_cpfs_unsupported_renameat2_uses_locked_same_parent_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    staging_name = f".robot-scene-state-{'c' * 64}.tmp"
    try:
        (parent / staging_name).mkdir()
        source = (parent / staging_name).stat()

        def unsupported_renameat2(**kwargs: Any) -> None:
            assert kwargs["source_fd"] == kwargs["target_fd"] == parent_fd
            raise OSError(errno.EINVAL, "synthetic CPFS renameat2 rejection")

        monkeypatch.setattr(
            scene_state_runner, "_renameat2_noreplace", unsupported_renameat2
        )
        _rename_noreplace(
            source_fd=parent_fd,
            source_name=staging_name,
            target_fd=parent_fd,
            target_name="scene-final",
        )
        target = (parent / "scene-final").stat()
        assert (source.st_dev, source.st_ino) == (target.st_dev, target.st_ino)
        assert not (parent / staging_name).exists()
    finally:
        os.close(parent_fd)


def test_cpfs_compatibility_publication_never_overwrites_injected_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    staging_name = f".robot-scene-state-{'d' * 64}.tmp"
    try:
        (parent / staging_name).mkdir()

        def unsupported_after_target_injection(**kwargs: Any) -> None:
            os.mkdir("scene-final", dir_fd=kwargs["target_fd"])
            (parent / "scene-final" / "FOREIGN").write_bytes(b"foreign")
            raise OSError(errno.EINVAL, "synthetic CPFS renameat2 rejection")

        monkeypatch.setattr(
            scene_state_runner,
            "_renameat2_noreplace",
            unsupported_after_target_injection,
        )
        with pytest.raises(FileExistsError):
            _rename_noreplace(
                source_fd=parent_fd,
                source_name=staging_name,
                target_fd=parent_fd,
                target_name="scene-final",
            )
        assert (parent / "scene-final" / "FOREIGN").read_bytes() == b"foreign"
        assert (parent / staging_name).is_dir()
    finally:
        os.close(parent_fd)


def test_private_staging_stat_failure_publishes_nothing_and_refuses_guess_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    staging_name = f".robot-scene-state-{'c' * 64}.tmp"
    real_stat = os.stat

    def fail_staging_stat(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        if path == staging_name and dir_fd == parent_fd:
            raise OSError("injected post-mkdir stat failure")
        return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    try:
        monkeypatch.setattr(os, "stat", fail_staging_stat)
        with pytest.raises(
            SceneStateProducerError, match="published zero bytes.*left.*untouched"
        ):
            _create_private_staging_child(
                parent_fd=parent_fd,
                parent_path=parent,
                parent_identity=parent.lstat(),
                staging_name=staging_name,
            )
        staging = parent / staging_name
        assert staging.is_dir()
        assert list(staging.iterdir()) == []
        assert not (parent / "scene-final").exists()
    finally:
        os.close(parent_fd)


def test_private_staging_prebind_replacement_is_not_written_or_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    staging_name = f".robot-scene-state-{'d' * 64}.tmp"
    real_open = os.open
    replaced = False

    def replace_before_staging_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if path == staging_name and dir_fd == parent_fd and not replaced:
            replaced = True
            os.rename(
                staging_name,
                "owned-moved",
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.mkdir(staging_name, dir_fd=parent_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    try:
        monkeypatch.setattr(os, "open", replace_before_staging_open)
        with pytest.raises(
            SceneStateProducerError, match="published zero bytes.*left.*untouched"
        ):
            _create_private_staging_child(
                parent_fd=parent_fd,
                parent_path=parent,
                parent_identity=parent.lstat(),
                staging_name=staging_name,
            )
        foreign = parent / staging_name
        moved = parent / "owned-moved"
        assert foreign.is_dir() and list(foreign.iterdir()) == []
        assert moved.is_dir() and list(moved.iterdir()) == []
        assert not (parent / "scene-final").exists()
    finally:
        os.close(parent_fd)


def test_moved_held_leaf_is_not_falsely_reported_as_rolled_back(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    child = parent / "scene-child"
    child.mkdir(parents=True)
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    child_fd = os.open(child, _READ_DIR_FLAGS)
    held_fd = -1
    try:
        child_identity = os.fstat(child_fd)
        held_fd, _ = _write_new_held_file(
            child_fd,
            path=child / "SCENE_STATE.npz",
            payload=b"state",
        )
        os.rename(
            "SCENE_STATE.npz",
            "moved-state",
            src_dir_fd=child_fd,
            dst_dir_fd=parent_fd,
        )
        with pytest.raises(SceneStateProducerError, match="non-owned scene output"):
            _rollback_scene_child(
                parent_fd=parent_fd,
                child_fd=child_fd,
                child_name="scene-child",
                child_identity=child_identity,
                held_files=[("SCENE_STATE.npz", held_fd)],
            )
        assert child.is_dir()
        assert (parent / "moved-state").read_bytes() == b"state"
    finally:
        if held_fd >= 0:
            os.close(held_fd)
        os.close(child_fd)
        os.close(parent_fd)


def test_hardlink_added_during_unlink_is_not_falsely_reported_as_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    child = parent / "scene-child"
    child.mkdir(parents=True)
    parent_fd = os.open(parent, _READ_DIR_FLAGS)
    child_fd = os.open(child, _READ_DIR_FLAGS)
    held_fd = -1
    real_unlink = os.unlink
    linked = False

    def link_then_unlink(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal linked
        if path == "SCENE_STATE.npz" and dir_fd == child_fd and not linked:
            linked = True
            os.link(
                "SCENE_STATE.npz",
                "held-alias",
                src_dir_fd=child_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        real_unlink(path, dir_fd=dir_fd)

    try:
        child_identity = os.fstat(child_fd)
        held_fd, _ = _write_new_held_file(
            child_fd,
            path=child / "SCENE_STATE.npz",
            payload=b"state",
        )
        monkeypatch.setattr(os, "unlink", link_then_unlink)
        with pytest.raises(SceneStateProducerError, match="non-owned scene output"):
            _rollback_scene_child(
                parent_fd=parent_fd,
                child_fd=child_fd,
                child_name="scene-child",
                child_identity=child_identity,
                held_files=[("SCENE_STATE.npz", held_fd)],
            )
        assert child.is_dir()
        assert (parent / "held-alias").read_bytes() == b"state"
        assert os.fstat(held_fd).st_nlink == 1
    finally:
        if held_fd >= 0:
            os.close(held_fd)
        os.close(child_fd)
        os.close(parent_fd)


def _write_external_authority(
    tmp_path: Path,
    *,
    sources: SourceBundle,
    assets: Any,
    mutate_arrays: Callable[[dict[str, np.ndarray]], None] | None = None,
    mutate_descriptor: Callable[[dict[str, Any]], None] | None = None,
    mutate_base_evidence: Callable[[dict[str, Any]], None] | None = None,
    mutate_q_evidence: Callable[[str, dict[str, Any]], None] | None = None,
    development_only: bool = False,
) -> Any:
    strict = load_strict_io(PROJECT)
    root = tmp_path / "external" / "grap_a_cap_004"
    root.mkdir(parents=True, exist_ok=True)
    by_name = {joint.name: joint for joint in assets.tianji.joints}
    joint_lower = np.asarray(
        [[by_name[name].lower for name in row] for row in ARM_JOINT_NAMES],
        dtype=np.float64,
    )
    joint_upper = np.asarray(
        [[by_name[name].upper for name in row] for row in ARM_JOINT_NAMES],
        dtype=np.float64,
    )
    count = sources.source_frame_count
    base = np.eye(4, dtype=np.float64)
    base[:3, 3] = [0.1, -0.2, 0.3]
    arrays = {
        "schema_version": np.asarray(EXTERNAL_BASE_Q_ARRAY_SCHEMA),
        "session_id": np.asarray("grap_a_cap_004"),
        "frame_names": np.asarray([f"{index:05d}" for index in range(count)]),
        "timestamp_ns": np.arange(count, dtype=np.int64) * 33_333_333 + 1,
        "valid": sources.full_valid.copy(),
        "T_camera_base": base,
        "q_arm": np.zeros((count, 2, 7), dtype=np.float64),
        "arm_joint_names": np.asarray(ARM_JOINT_NAMES),
        "joint_lower": joint_lower,
        "joint_upper": joint_upper,
    }
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    arrays_path = root / "BASE_Q_AUTHORITY.npz"
    np.savez_compressed(arrays_path, **arrays)
    arrays_record = strict.read_bytes_nofollow(arrays_path)
    tianji_record = strict.read_bytes_nofollow(assets.tianji.path)
    timestamp_clock_id = "camera-monotonic-ns"
    device_id = "robot-state-recorder-001"
    calibration_id = "camera-base-calibration-001"
    calibration_sha256 = "c" * 64
    base_evidence = {
        "schema_version": EXTERNAL_BASE_EVIDENCE_SCHEMA,
        "session_id": "grap_a_cap_004",
        "evidence_kind": EXTERNAL_BASE_EVIDENCE_KIND,
        "provider": "OWNER_CAMERA_BASE_CALIBRATION",
        "method": "DIRECT_CAMERA_TO_ROBOT_BASE_CALIBRATION",
        "T_camera_base_semantics": BASE_TRANSFORM_SEMANTICS,
        "transform_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "timestamp_ns": 1,
        "timestamp_clock_id": timestamp_clock_id,
        "device_id": device_id,
        "calibration_id": calibration_id,
        "calibration_sha256": calibration_sha256,
        "independent_of_r2_wrist_targets": True,
        "derived_from_r2_wrist_targets": False,
        "derived_from_ik_solver": False,
        "selected_by_ik_residual": False,
        "interpolated_or_filled": False,
        "synthetic_fixture": False,
        "identity_fixture": False,
        "formal_consumer_allowed": False,
        "T_camera_base": base.tolist(),
    }
    if mutate_base_evidence is not None:
        mutate_base_evidence(base_evidence)
    base_evidence_path = root / "CAMERA_BASE_EVIDENCE.json"
    base_evidence_path.write_text(json.dumps(base_evidence, sort_keys=True))
    base_evidence_record = strict.read_bytes_nofollow(base_evidence_path)
    q_evidence_refs: dict[str, Mapping[str, Any]] = {}
    for side_index, side in enumerate(("left", "right")):
        q_evidence = {
            "schema_version": EXTERNAL_Q_ARM_SIDE_EVIDENCE_SCHEMA,
            "session_id": "grap_a_cap_004",
            "side": side,
            "evidence_kind": EXTERNAL_Q_ARM_EVIDENCE_KIND,
            "provider": f"OWNER_{side.upper()}_ARM_ENCODER",
            "method": "DIRECT_ARM_JOINT_ENCODER_READOUT",
            "frame_count": count,
            "frame_range": {"start": 0, "stop_exclusive": count, "contiguous": True},
            "frame_timestamp_mapping": EXTERNAL_FRAME_TIMESTAMP_MAPPING,
            "timestamp_clock_id": timestamp_clock_id,
            "device_id": f"{side}-arm-encoder-001",
            "calibration_id": f"{side}-arm-zero-calibration-001",
            "calibration_sha256": ("1" if side == "left" else "2") * 64,
            "axes_convention": RIGHT_HANDED_AXES,
            "q_arm_units": "radians",
            "array_digest_canonicalization": (EXTERNAL_ARRAY_DIGEST_CANONICALIZATION),
            "frame_names_sha256": _canonical_array_sha256(arrays["frame_names"]),
            "timestamp_ns_sha256": _canonical_array_sha256(arrays["timestamp_ns"]),
            "valid_sha256": _canonical_array_sha256(arrays["valid"][:, side_index]),
            "q_arm_sha256": _canonical_array_sha256(arrays["q_arm"][:, side_index]),
            "arm_joint_names_sha256": _canonical_array_sha256(
                arrays["arm_joint_names"][side_index]
            ),
            "joint_lower_sha256": _canonical_array_sha256(
                arrays["joint_lower"][side_index]
            ),
            "joint_upper_sha256": _canonical_array_sha256(
                arrays["joint_upper"][side_index]
            ),
            "independent_of_r2_wrist_targets": True,
            "derived_from_r2_wrist_targets": False,
            "derived_from_ik_solver": False,
            "selected_by_ik_residual": False,
            "interpolated_or_filled": False,
            "synthetic_fixture": False,
            "identity_fixture": False,
            "formal_consumer_allowed": False,
        }
        if mutate_q_evidence is not None:
            mutate_q_evidence(side, q_evidence)
        evidence_path = root / f"Q_ARM_EVIDENCE_{side.upper()}.json"
        evidence_path.write_text(json.dumps(q_evidence, sort_keys=True))
        q_evidence_refs[side] = strict.read_bytes_nofollow(evidence_path).evidence_ref()
    descriptor = {
        "schema_version": EXTERNAL_BASE_Q_SCHEMA,
        "session_id": "grap_a_cap_004",
        "authority_kind": "DIRECT_EXTERNAL_BASE_AND_ARM_STATE",
        "evidence_kind": EXTERNAL_AUTHORITY_EVIDENCE_KIND,
        "provider": "OWNER_EXTERNAL_ROBOT_STATE_AUTHORITY",
        "method": EXTERNAL_AUTHORITY_METHOD,
        "development_only": development_only,
        "formal_consumer_allowed": False,
        "direct_measurement_evidence": True,
        "independent_of_r2_wrist_targets": True,
        "derived_from_r2_wrist_targets": False,
        "derived_from_ik_solver": False,
        "selected_by_ik_residual": False,
        "interpolated_or_filled": False,
        "synthetic_fixture": False,
        "identity_fixture": False,
        "solver_not_used": True,
        "residual_not_claimed": True,
        "T_camera_base_semantics": BASE_TRANSFORM_SEMANTICS,
        "transform_convention": RIGHT_HANDED_AXES,
        "translation_units": "metres",
        "q_arm_units": "radians",
        "side_order": ["left", "right"],
        "frame_count": count,
        "frame_range": {"start": 0, "stop_exclusive": count, "contiguous": True},
        "frame_timestamp_mapping": EXTERNAL_FRAME_TIMESTAMP_MAPPING,
        "timestamp_clock_id": timestamp_clock_id,
        "device_id": device_id,
        "calibration_id": calibration_id,
        "calibration_sha256": calibration_sha256,
        "urdf_refs": {
            "robot_asset_pin": dict(assets.asset_pin_record),
            "tianji_urdf": tianji_record.evidence_ref(),
        },
        "source_refs": {
            "r2_sidecar": dict(sources.records["r2_sidecar"]),
            "hawor": dict(sources.records["hawor"]),
        },
        "base_evidence_ref": base_evidence_record.evidence_ref(),
        "q_arm_evidence_by_side": q_evidence_refs,
        "arrays_ref": arrays_record.evidence_ref(),
    }
    if mutate_descriptor is not None:
        mutate_descriptor(descriptor)
    descriptor_path = root / "BASE_Q_AUTHORITY.json"
    descriptor_path.write_text(json.dumps(descriptor, sort_keys=True))
    return strict.read_bytes_nofollow(descriptor_path)


def test_external_base_q_authority_bypasses_solver_and_claims_no_residual(
    tmp_path: Path,
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")
    descriptor_record = _write_external_authority(
        tmp_path, sources=sources, assets=assets
    )
    external = load_external_base_q_authority(
        strict=load_strict_io(PROJECT),
        descriptor_record=descriptor_record,
        expected_session_id="grap_a_cap_004",
        sources=sources,
        assets=assets,
    )
    assert external.scene_state_mode == SCENE_STATE_MODE_EXTERNAL_AUTHORITY
    assert external.solver_used is False
    assert external.residual_claimed is False
    assert np.count_nonzero(external.evaluations) == 0
    assert np.isnan(external.position_residual_mm).all()
    assert np.isnan(external.rotation_residual_deg).all()
    assert external.timestamp_ns is not None
    assert external.timestamp_ns.tolist() == [1, 33_333_334, 66_666_667]

    mount = load_explicit_mount(mount_payload(), expected_session_id="grap_a_cap_004")
    state_payload = encode_scene_state(
        session_id="grap_a_cap_004",
        sources=sources,
        mounts=mount,
        solver=external,
    )
    state = decode_scene_state(state_payload)
    with np.load(io.BytesIO(state_payload), allow_pickle=False) as archive:
        float32_base_arrays = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    float32_base_arrays["T_camera_base"] = np.asarray(
        float32_base_arrays["T_camera_base"], dtype=np.float32
    )
    float32_payload = io.BytesIO()
    np.savez_compressed(float32_payload, **float32_base_arrays)
    with pytest.raises(FullChainRendererError, match="exact float64"):
        decode_scene_state(float32_payload.getvalue())
    state_record = fake_ref(
        tmp_path / "SCENE_STATE.npz",
        payload_sha256=sha256_bytes(state_payload),
        size=len(state_payload),
        device=700,
        inode=701,
    )
    mount_bytes = mount_payload()
    mount_record = fake_ref(
        tmp_path / "MOUNT.json",
        payload_sha256=sha256_bytes(mount_bytes),
        size=len(mount_bytes),
        device=700,
        inode=702,
    )
    manifest = build_manifest(
        session_id="grap_a_cap_004",
        state_record=state_record,
        state=state,
        mount_record=mount_record,
        mount=mount,
        sources=sources,
        solver=external,
        assets=assets,
        command=["/usr/bin/python3", "producer.py"],
    )
    assert manifest["formal_consumer_allowed"] is False
    assert manifest["status"] == SCENE_MANIFEST_DEVELOPMENT_STATUS
    assert manifest["q_arm_and_camera_base_authority_input_consumed"] is True
    assert manifest["q_arm_and_camera_base_authoritative"] is False
    assert manifest["solver"]["solver_used"] is False
    assert manifest["solver"]["residual_not_claimed"] is True
    assert manifest["solver"]["max_position_residual_mm"] is None


def test_real_mount_plus_external_base_q_is_visual_candidate_never_formal(
    tmp_path: Path,
) -> None:
    strict = load_strict_io(PROJECT)
    sources, assets = _load_local_sources(tmp_path / "sources")
    descriptor_record = _write_external_authority(
        tmp_path, sources=sources, assets=assets
    )
    external = load_external_base_q_authority(
        strict=strict,
        descriptor_record=descriptor_record,
        expected_session_id="grap_a_cap_004",
        sources=sources,
        assets=assets,
    )
    mount_record, _, _ = _write_real_mount_bundle(tmp_path / "mount")
    mount = load_explicit_mount(
        mount_record.payload,
        expected_session_id="grap_a_cap_004",
        strict=strict,
        source_record=mount_record,
    )
    state_payload = encode_scene_state(
        session_id="grap_a_cap_004",
        sources=sources,
        mounts=mount,
        solver=external,
    )
    state = decode_scene_state(state_payload)
    manifest = build_manifest(
        session_id="grap_a_cap_004",
        state_record=fake_ref(
            tmp_path / "candidate.npz",
            payload_sha256=sha256_bytes(state_payload),
            size=len(state_payload),
            device=800,
            inode=801,
        ),
        state=state,
        mount_record=mount_record.evidence_ref(),
        mount=mount,
        sources=sources,
        solver=external,
        assets=assets,
        command=["/usr/bin/python3", "producer.py"],
    )
    assert manifest["status"] == SCENE_MANIFEST_VISUAL_CANDIDATE_STATUS
    assert manifest["development_only"] is False
    assert manifest["formal_consumer_allowed"] is False
    assert manifest["next_bucket_blocked"] is True


@pytest.mark.parametrize(
    "defect",
    ["timestamp", "joint_limit", "valid_join", "semantics", "urdf_sha"],
)
def test_external_base_q_rejects_timestamp_joint_valid_semantic_and_urdf_drift(
    tmp_path: Path, defect: str
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")

    def mutate_arrays(arrays: dict[str, np.ndarray]) -> None:
        if defect == "timestamp":
            arrays["timestamp_ns"][1] = arrays["timestamp_ns"][0]
        elif defect == "joint_limit":
            arrays["q_arm"][0, 0, 0] = 1_000_000.0
        elif defect == "valid_join":
            arrays["valid"][0, 0] = False
            arrays["q_arm"][0, 0] = np.nan

    def mutate_descriptor(value: dict[str, Any]) -> None:
        if defect == "semantics":
            value["T_camera_base_semantics"] = "p_base = T_camera_base @ p_camera"
        elif defect == "urdf_sha":
            value["urdf_refs"]["tianji_urdf"]["sha256"] = "0" * 64

    descriptor_record = _write_external_authority(
        tmp_path,
        sources=sources,
        assets=assets,
        mutate_arrays=mutate_arrays,
        mutate_descriptor=mutate_descriptor,
    )
    with pytest.raises(SceneStateProducerError):
        load_external_base_q_authority(
            strict=load_strict_io(PROJECT),
            descriptor_record=descriptor_record,
            expected_session_id="grap_a_cap_004",
            sources=sources,
            assets=assets,
        )


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("direct_measurement_evidence", False),
        ("derived_from_r2_wrist_targets", True),
        ("derived_from_ik_solver", True),
        ("selected_by_ik_residual", True),
        ("interpolated_or_filled", True),
        ("synthetic_fixture", True),
        ("identity_fixture", True),
        ("method", "R2_WRIST_IK_RESIDUAL_MINIMIZATION"),
    ],
)
def test_external_base_q_descriptor_rejects_synthetic_or_circular_provenance(
    tmp_path: Path, field: str, unsafe: Any
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")
    descriptor_record = _write_external_authority(
        tmp_path,
        sources=sources,
        assets=assets,
        mutate_descriptor=lambda value: value.__setitem__(field, unsafe),
    )
    with pytest.raises(SceneStateProducerError):
        load_external_base_q_authority(
            strict=load_strict_io(PROJECT),
            descriptor_record=descriptor_record,
            expected_session_id="grap_a_cap_004",
            sources=sources,
            assets=assets,
        )


@pytest.mark.parametrize("defect", ["base_transform", "q_digest", "q_derived"])
def test_external_base_q_actual_evidence_must_match_direct_arrays(
    tmp_path: Path, defect: str
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")

    def mutate_base(value: dict[str, Any]) -> None:
        if defect == "base_transform":
            value["T_camera_base"] = np.eye(4).tolist()

    def mutate_q(side: str, value: dict[str, Any]) -> None:
        if side != "left":
            return
        if defect == "q_digest":
            value["q_arm_sha256"] = "0" * 64
        elif defect == "q_derived":
            value["derived_from_ik_solver"] = True

    descriptor_record = _write_external_authority(
        tmp_path,
        sources=sources,
        assets=assets,
        mutate_base_evidence=mutate_base,
        mutate_q_evidence=mutate_q,
    )
    with pytest.raises(SceneStateProducerError):
        load_external_base_q_authority(
            strict=load_strict_io(PROJECT),
            descriptor_record=descriptor_record,
            expected_session_id="grap_a_cap_004",
            sources=sources,
            assets=assets,
        )


def test_external_base_q_rejects_evidence_alias_and_wrong_source_join(
    tmp_path: Path,
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")

    def alias_q(value: dict[str, Any]) -> None:
        value["q_arm_evidence_by_side"]["right"] = value["q_arm_evidence_by_side"][
            "left"
        ]

    aliased = _write_external_authority(
        tmp_path / "alias",
        sources=sources,
        assets=assets,
        mutate_descriptor=alias_q,
    )
    with pytest.raises(SceneStateProducerError):
        load_external_base_q_authority(
            strict=load_strict_io(PROJECT),
            descriptor_record=aliased,
            expected_session_id="grap_a_cap_004",
            sources=sources,
            assets=assets,
        )

    def wrong_source(value: dict[str, Any]) -> None:
        value["source_refs"]["r2_sidecar"] = value["source_refs"]["hawor"]

    wrong = _write_external_authority(
        tmp_path / "source",
        sources=sources,
        assets=assets,
        mutate_descriptor=wrong_source,
    )
    with pytest.raises(SceneStateProducerError, match="pinned input record"):
        load_external_base_q_authority(
            strict=load_strict_io(PROJECT),
            descriptor_record=wrong,
            expected_session_id="grap_a_cap_004",
            sources=sources,
            assets=assets,
        )


def test_sources_reject_wrong_radian_unit_before_solver(tmp_path: Path) -> None:
    r2_path, hawor_path, raw_root = _source_fixture(
        tmp_path,
        mutate_r2=lambda arrays: arrays.__setitem__("q_unit", np.asarray("degree")),
    )
    with pytest.raises(SceneStateProducerError, match="units"):
        load_sources(
            strict=load_strict_io(PROJECT),
            session_id="grap_a_cap_004",
            sidecar_path=r2_path,
            hawor_path=hawor_path,
            raw_root=raw_root,
            frame_indices=[2],
            assets=load_pinned_robot_assets(PROJECT),
        )


def test_sources_reject_non_se3_wrist_before_solver(tmp_path: Path) -> None:
    def mutate(arrays: dict[str, np.ndarray]) -> None:
        arrays["wrist_T_camera"][2, 0, 0, 0] = 2.0

    r2_path, hawor_path, raw_root = _source_fixture(tmp_path, mutate_r2=mutate)
    with pytest.raises(SceneStateProducerError, match=r"right-handed SE\(3\)"):
        load_sources(
            strict=load_strict_io(PROJECT),
            session_id="grap_a_cap_004",
            sidecar_path=r2_path,
            hawor_path=hawor_path,
            raw_root=raw_root,
            frame_indices=[2],
            assets=load_pinned_robot_assets(PROJECT),
        )


def test_manifest_cannot_claim_independent_or_authoritative_ik() -> None:
    assets = load_pinned_robot_assets(PROJECT)
    hand_names = np.asarray(
        [
            [joint.name for joint in model.joints if joint.joint_type != "fixed"]
            for model in (assets.left_hand, assets.right_hand)
        ]
    )
    sources = SourceBundle(
        frame_indices=(0,),
        source_frame_count=1,
        full_valid=np.ones((1, 2), dtype=bool),
        frame_names=("00000",),
        q_hand=np.zeros((1, 2, 22), dtype=np.float64),
        wrist_T_camera=np.repeat(np.eye(4)[None, None], 2, axis=1),
        valid=np.ones((1, 2), dtype=bool),
        camera_intrinsics=np.asarray([[640.0, 640.0, 639.5, 479.5]]),
        source_resolution=(1280, 960),
        hand_joint_names=hand_names,
        records={
            "r2_sidecar": {},
            "hawor": {},
            "raw_metadata": {},
        },
    )
    solver = SolverResult(
        q_arm=np.zeros((1, 2, 7), dtype=np.float64),
        T_camera_base=np.eye(4),
        position_residual_mm=np.zeros((1, 2), dtype=np.float64),
        rotation_residual_deg=np.zeros((1, 2), dtype=np.float64),
        evaluations=np.ones((1, 2), dtype=np.int32),
    )
    mount = load_explicit_mount(mount_payload(), expected_session_id="grap_a_cap_004")
    state = decode_scene_state(
        encode_scene_state(
            session_id="grap_a_cap_004",
            sources=sources,
            mounts=mount,
            solver=solver,
        )
    )
    manifest = build_manifest(
        session_id="grap_a_cap_004",
        state_record=fake_ref(
            Path("/tmp/SCENE_STATE.npz"),
            payload_sha256="a" * 64,
            size=2048,
            device=900,
            inode=901,
        ),
        state=state,
        mount_record=fake_ref(
            Path("/tmp/mount.json"),
            payload_sha256=sha256_bytes(mount_payload()),
            size=len(mount_payload()),
            device=900,
            inode=902,
        ),
        mount=mount,
        sources=sources,
        solver=solver,
        assets=assets,
        command=["/usr/bin/python3", "producer.py"],
    )
    assert manifest["formal_consumer_allowed"] is False
    assert manifest["status"] == SCENE_MANIFEST_DEVELOPMENT_STATUS
    assert manifest["completion_mode"] == "ARTIFACT_EXISTS"
    assert manifest["mount_authority"]["session_constant"] is True
    assert manifest["mount_authority"]["per_frame_mount_forbidden"] is True
    assert manifest["mount_authority"]["independent_of_r2_wrist_targets"] is True
    assert manifest["mount_authority"]["selected_by_ik_residual"] is False
    assert manifest["ik_residual_is_independent_accuracy_evidence"] is False
    assert manifest["q_arm_and_camera_base_authoritative"] is False
    assert manifest["solver"]["reported_residual_is_independent_validation"] is False
    assert manifest["solver"]["q_arm_and_camera_base_unique_or_authoritative"] is False
    assert manifest["solver"]["structural_unknown_minus_constraint_count"] == 8


def test_cpu_solver_recovers_synthetic_two_arm_targets_with_one_constant_base() -> None:
    from pipeline.robot_renderer_cycles import forward_kinematics
    from pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES

    assets = load_pinned_robot_assets(PROJECT)
    base = np.eye(4)
    base[:3, 3] = [0.1, -0.2, 0.3]
    mounts = np.repeat(np.eye(4)[None], 2, axis=0)
    wrists = np.empty((1, 2, 4, 4), dtype=np.float64)
    q_true = np.asarray(
        [
            [0.15, -0.25, 0.10, -0.35, 0.20, 0.10, -0.10],
            [-0.12, -0.20, -0.15, -0.30, -0.20, 0.05, 0.08],
        ],
        dtype=np.float64,
    )
    for side in range(2):
        values = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
        values.update(dict(zip(ARM_JOINT_NAMES[side], q_true[side], strict=True)))
        wrists[0, side] = (
            base
            @ forward_kinematics(assets.tianji, values)[
                f"{('left', 'right')[side]}_tool"
            ]
        )
    solved = solve_scene_state(
        assets,
        wrist_T_camera=wrists,
        valid=np.ones((1, 2), dtype=bool),
        mounts=mounts,
    )
    assert np.max(solved.position_residual_mm) <= 10.0
    assert np.max(solved.rotation_residual_deg) <= 5.0
    assert solved.T_camera_base.shape == (4, 4)


def _fixed_solver_targets(
    *, count: int = 2
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    from pipeline.robot_renderer_cycles import forward_kinematics

    assets = load_pinned_robot_assets(PROJECT)
    base = np.eye(4, dtype=np.float64)
    base[:3, 3] = [0.1, -0.2, 0.3]
    mounts = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
    wrists = np.empty((count, 2, 4, 4), dtype=np.float64)
    for frame in range(count):
        for side in range(2):
            q = np.asarray(
                [
                    0.03 * (frame + 1) * (1 if side == 0 else -1),
                    -0.15,
                    0.04 * (frame + 1),
                    -0.25,
                    0.05,
                    0.02,
                    -0.03,
                ],
                dtype=np.float64,
            )
            values = {name: 0.0 for row in ARM_JOINT_NAMES for name in row}
            values.update(dict(zip(ARM_JOINT_NAMES[side], q, strict=True)))
            wrists[frame, side] = (
                base
                @ forward_kinematics(assets.tianji, values)[
                    f"{('left', 'right')[side]}_tool"
                ]
                @ mounts[side]
            )
    return assets, base, mounts, wrists


def test_fixed_base_solver_keeps_known_base_bit_exact_and_never_refines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets, base, mounts, wrists = _fixed_solver_targets(count=1)

    def forbidden_refinement(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fixed-base route reached joint base refinement")

    monkeypatch.setattr(
        scene_state_module, "_joint_base_refinement", forbidden_refinement
    )
    solved = solve_scene_state_fixed_base(
        assets,
        wrist_T_camera=wrists,
        valid=np.ones((1, 2), dtype=np.bool_),
        mounts=mounts,
        T_camera_base=base,
    )
    assert solved.scene_state_mode == SCENE_STATE_MODE_FIXED_CAMERA_BASE_SOLVER
    assert solved.T_camera_base.dtype == np.float64
    assert np.array_equal(solved.T_camera_base, base)
    assert np.max(solved.position_residual_mm) <= 10.0
    assert np.max(solved.rotation_residual_deg) <= 5.0


def test_fixed_base_solver_is_frame_order_invariant_and_always_zero_initial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets, base, mounts, wrists = _fixed_solver_targets(count=3)
    position_calls: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    full_pose_calls: list[tuple[int, np.ndarray, np.ndarray]] = []

    def deterministic_position_only(
        assets: Any,
        *,
        side: int,
        base: np.ndarray,
        target_tool: np.ndarray,
        initial_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        del assets, base
        signature = float(np.sum(target_tool[:3, 3])) * 1e-3 + side * 1e-4
        solved = np.clip(np.full(7, signature), lower, upper)
        position_calls.append(
            (side, target_tool.copy(), initial_q.copy(), solved.copy())
        )
        return solved, 2

    def deterministic_full_pose(
        assets: Any,
        *,
        side: int,
        base: np.ndarray,
        target_tool: np.ndarray,
        initial_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        del assets, base
        full_pose_calls.append((side, target_tool.copy(), initial_q.copy()))
        return np.clip(initial_q + 1e-5, lower, upper), 3

    monkeypatch.setattr(
        scene_state_module,
        "_solve_one_arm_position_only",
        deterministic_position_only,
    )
    monkeypatch.setattr(scene_state_module, "_solve_one_arm", deterministic_full_pose)
    valid = np.ones((3, 2), dtype=np.bool_)
    forward = solve_scene_state_fixed_base(
        assets,
        wrist_T_camera=wrists,
        valid=valid,
        mounts=mounts,
        T_camera_base=base,
    )
    order = np.asarray([2, 0, 1])
    reordered = solve_scene_state_fixed_base(
        assets,
        wrist_T_camera=wrists[order],
        valid=valid[order],
        mounts=mounts,
        T_camera_base=base,
    )
    assert len(position_calls) == len(full_pose_calls) == 12
    for position_call, full_pose_call in zip(
        position_calls, full_pose_calls, strict=True
    ):
        position_side, position_target, position_initial, position_result = (
            position_call
        )
        full_side, full_target, full_initial = full_pose_call
        assert position_side == full_side
        assert np.array_equal(position_target, full_target)
        assert np.array_equal(position_initial, np.zeros(7))
        assert np.array_equal(full_initial, position_result)
    assert np.all(forward.position_stage_evaluations[valid] == 2)
    assert np.all(forward.full_pose_stage_evaluations[valid] == 3)
    assert np.all(forward.evaluations[valid] == 5)
    assert np.array_equal(reordered.q_arm, forward.q_arm[order])
    assert np.allclose(
        reordered.position_residual_mm,
        forward.position_residual_mm[order],
        atol=0.0,
        rtol=0.0,
    )
    assert np.allclose(
        reordered.rotation_residual_deg,
        forward.rotation_residual_deg[order],
        atol=0.0,
        rtol=0.0,
    )


def test_fixed_base_wrong_ten_metre_candidate_stays_in_denominator_and_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets, _, mounts, wrists = _fixed_solver_targets(count=1)

    def return_zero(
        assets: Any,
        *,
        side: int,
        base: np.ndarray,
        target_tool: np.ndarray,
        initial_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        del assets, side, base, target_tool, initial_q, lower, upper
        return np.zeros(7, dtype=np.float64), 1

    monkeypatch.setattr(scene_state_module, "_solve_one_arm", return_zero)
    monkeypatch.setattr(scene_state_module, "_solve_one_arm_position_only", return_zero)
    wrong = np.eye(4, dtype=np.float64)
    wrong[0, 3] = 10.0
    valid = np.ones((1, 2), dtype=np.bool_)
    solved = solve_scene_state_fixed_base(
        assets,
        wrist_T_camera=wrists,
        valid=valid,
        mounts=mounts,
        T_camera_base=wrong,
    )
    gate = fixed_base_gate_summary(
        position_residual_mm=solved.position_residual_mm,
        rotation_residual_deg=solved.rotation_residual_deg,
        valid=valid,
    )
    assert gate["ik_gate_pair_denominator"] == 2
    assert gate["ik_gate_pair_pass_count"] == 0
    assert gate["ik_gate_frame_denominator"] == 1
    assert gate["ik_gate_frame_all_valid_sides_pass_count"] == 0
    assert gate["failed_valid_pairs_deleted"] is False


def test_fixed_base_gate_is_inclusive_at_10mm_and_5deg_and_requires_both() -> None:
    valid = np.asarray([[True, True], [True, False]], dtype=np.bool_)
    position = np.asarray([[10.0, 10.000001], [0.0, np.nan]], dtype=np.float64)
    rotation = np.asarray([[5.0, 0.0], [5.000001, np.nan]], dtype=np.float64)
    gate = fixed_base_gate_summary(
        position_residual_mm=position,
        rotation_residual_deg=rotation,
        valid=valid,
    )
    assert gate["ik_gate_pair_pass_count"] == 1
    assert gate["ik_gate_pair_denominator"] == 3
    assert gate["ik_gate_pair_pass_rate"] == pytest.approx(1 / 3)
    assert gate["ik_gate_frame_all_valid_sides_pass_count"] == 0
    assert gate["ik_gate_frame_denominator"] == 2


@pytest.mark.parametrize("defect", ["dtype", "reflection", "bottom_row"])
def test_fixed_base_solver_rejects_non_float64_or_invalid_se3(defect: str) -> None:
    assets, base, mounts, wrists = _fixed_solver_targets(count=1)
    if defect == "dtype":
        candidate = np.eye(4, dtype=np.float32)
    else:
        candidate = base.copy()
        if defect == "reflection":
            candidate[0, 0] = -1.0
        else:
            candidate[3, 0] = 1.0
    with pytest.raises(SceneStateProducerError, match="float64|right-handed"):
        solve_scene_state_fixed_base(
            assets,
            wrist_T_camera=wrists,
            valid=np.ones((1, 2), dtype=np.bool_),
            mounts=mounts,
            T_camera_base=candidate,
        )


def test_fixed_base_manifest_binds_candidate_ref_backend_dimensions_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")
    candidate_record, candidate = write_fixed_camera_base_candidate(
        tmp_path / "candidate"
    )

    def return_zero(
        assets: Any,
        *,
        side: int,
        base: np.ndarray,
        target_tool: np.ndarray,
        initial_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        del assets, side, base, target_tool, initial_q, lower, upper
        return np.zeros(7, dtype=np.float64), 1

    monkeypatch.setattr(scene_state_module, "_solve_one_arm", return_zero)
    monkeypatch.setattr(scene_state_module, "_solve_one_arm_position_only", return_zero)
    solver = solve_scene_state_fixed_base(
        assets,
        wrist_T_camera=sources.wrist_T_camera,
        valid=sources.valid,
        mounts=np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0),
        T_camera_base=candidate.matrix,
        candidate=candidate,
    )
    mount_bytes = mount_payload()
    mount = load_explicit_mount(mount_bytes, expected_session_id="grap_a_cap_004")
    state_payload = encode_scene_state(
        session_id="grap_a_cap_004",
        sources=sources,
        mounts=mount,
        solver=solver,
    )
    state = decode_scene_state(state_payload)
    state_record = fake_ref(
        tmp_path / "SCENE_STATE.npz",
        payload_sha256=sha256_bytes(state_payload),
        size=len(state_payload),
        device=980,
        inode=981,
    )
    mount_record = fake_ref(
        tmp_path / "MOUNT.json",
        payload_sha256=sha256_bytes(mount_bytes),
        size=len(mount_bytes),
        device=980,
        inode=982,
    )
    manifest = build_manifest(
        session_id="grap_a_cap_004",
        state_record=state_record,
        state=state,
        mount_record=mount_record,
        mount=mount,
        sources=sources,
        solver=solver,
        assets=assets,
        command=["/usr/bin/python3", "producer.py"],
    )
    assert manifest["status"] == SCENE_MANIFEST_DEVELOPMENT_STATUS
    assert manifest["next_bucket_blocked"] is True
    assert manifest["fixed_camera_base_candidate"]["descriptor"] == (
        candidate_record.evidence_ref()
    )
    assert manifest["solver"]["schema_version"] == (FIXED_BASE_TWO_STAGE_SOLVER_SCHEMA)
    assert manifest["solver"]["backend"] == (
        "SCIPY_CPU_TRF_FIXED_CAMERA_BASE_POSITION_THEN_FULL_POSE_PER_SIDE_FRAME"
    )
    assert manifest["solver"]["outer_iterations"] == 0
    assert manifest["solver"]["ik_stage_count"] == 2
    assert manifest["solver"]["position_stage_initial_q"] == (
        "UNIFORM_ZERO_7DOF_PER_VALID_PAIR"
    )
    assert manifest["solver"]["full_pose_stage_initial_q"] == (
        "SAME_VALID_PAIR_POSITION_STAGE_RESULT"
    )
    assert manifest["solver"]["cross_frame_side_or_candidate_warm_start"] is False
    assert manifest["solver"]["position_stage_function_evaluations_total"] == 6
    assert manifest["solver"]["full_pose_stage_function_evaluations_total"] == 6
    assert manifest["solver"]["ik_function_evaluations_total"] == 12
    assert manifest["solver"]["structural_parameter_count"] == 42
    assert manifest["solver"]["structural_pose_constraint_count"] == 36
    assert manifest["solver"]["structural_unknown_minus_constraint_count"] == 6

    drifted = copy.deepcopy(manifest)
    drifted["fixed_camera_base_candidate"]["descriptor"]["sha256"] = "0" * 64
    with pytest.raises(FullChainRendererError, match="does not join"):
        validate_scene_manifest(
            drifted,
            state_record=state_record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )

    wrong_boolean_type = copy.deepcopy(manifest)
    wrong_boolean_type["fixed_camera_base_candidate"]["global_across_sessions"] = 1
    with pytest.raises(FullChainRendererError, match="development contract"):
        validate_scene_manifest(
            wrong_boolean_type,
            state_record=state_record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )

    drifted_stage = copy.deepcopy(manifest)
    drifted_stage["solver"]["full_pose_stage_initial_q"] = "PREVIOUS_FRAME_RESULT"
    with pytest.raises(FullChainRendererError, match="solver/gate contract"):
        validate_scene_manifest(
            drifted_stage,
            state_record=state_record,
            state=state,
            cpu4_tool_definition=assets.tool_definition,
            renderer_tool_definition=assets.tool_definition,
        )

    with pytest.raises(FullChainRendererError, match="full 6D IK gate"):
        compose_frame_placement(state, assets, 0)


def test_fixed_base_renderer_recomputes_final_q_residual_instead_of_trusting_npz(
    tmp_path: Path,
) -> None:
    assets, base, mounts, wrists = _fixed_solver_targets(count=1)
    valid = np.ones((1, 2), dtype=np.bool_)
    _, candidate = write_fixed_camera_base_candidate(tmp_path / "candidate", base)
    solver = solve_scene_state_fixed_base(
        assets,
        wrist_T_camera=wrists,
        valid=valid,
        mounts=mounts,
        T_camera_base=base,
        candidate=candidate,
    )
    hand_names = np.asarray(
        [
            [joint.name for joint in model.joints if joint.joint_type != "fixed"]
            for model in (assets.left_hand, assets.right_hand)
        ]
    )
    sources = SourceBundle(
        frame_indices=(0,),
        source_frame_count=1,
        full_valid=valid.copy(),
        frame_names=("00000",),
        q_hand=np.zeros((1, 2, 22), dtype=np.float64),
        wrist_T_camera=wrists,
        valid=valid,
        camera_intrinsics=np.asarray([[640.0, 640.0, 639.5, 479.5]]),
        source_resolution=(1280, 960),
        hand_joint_names=hand_names,
        records={},
    )
    mount = load_explicit_mount(
        mount_payload(matrices=mounts), expected_session_id="grap_a_cap_004"
    )
    state = decode_scene_state(
        encode_scene_state(
            session_id="grap_a_cap_004",
            sources=sources,
            mounts=mount,
            solver=solver,
        )
    )
    compose_frame_placement(state, assets, 0)

    forged_q = state.q_arm.copy()
    forged_q[0, 0, 0] += 0.5
    forged = replace(state, q_arm=forged_q)
    with pytest.raises(FullChainRendererError, match="does not match final q_arm FK"):
        compose_frame_placement(forged, assets, 0)


def test_fixed_base_preflight_synthetic_state_identity_cannot_collide_with_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, assets = _load_local_sources(tmp_path / "sources")
    candidate_record, candidate = write_fixed_camera_base_candidate(
        tmp_path / "candidate"
    )

    def return_zero(
        assets: Any,
        *,
        side: int,
        base: np.ndarray,
        target_tool: np.ndarray,
        initial_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        del assets, side, base, target_tool, initial_q, lower, upper
        return np.zeros(7, dtype=np.float64), 1

    monkeypatch.setattr(scene_state_module, "_solve_one_arm", return_zero)
    monkeypatch.setattr(scene_state_module, "_solve_one_arm_position_only", return_zero)
    solver = solve_scene_state_fixed_base(
        assets,
        wrist_T_camera=sources.wrist_T_camera,
        valid=sources.valid,
        mounts=np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0),
        T_camera_base=candidate.matrix,
        candidate=candidate,
    )
    mount_bytes = mount_payload()
    mount = load_explicit_mount(mount_bytes, expected_session_id="grap_a_cap_004")
    state_payload = encode_scene_state(
        session_id="grap_a_cap_004",
        sources=sources,
        mounts=mount,
        solver=solver,
    )
    state = decode_scene_state(state_payload)
    candidate_ref = candidate_record.evidence_ref()
    assert int(candidate_ref["inode"]) > 1
    # Reproduce the old false collision: the fabricated state inode used to be
    # exactly ``mount inode + 1``, which here is the legitimate candidate inode.
    mount_record = fake_ref(
        tmp_path / "MOUNT.json",
        payload_sha256=sha256_bytes(mount_bytes),
        size=len(mount_bytes),
        device=int(candidate_ref["device"]),
        inode=int(candidate_ref["inode"]) - 1,
    )
    preflight_scene_manifest(
        session_id="grap_a_cap_004",
        state_path=tmp_path / "output" / "SCENE_STATE.npz",
        state_payload=state_payload,
        state=state,
        mount_record=mount_record,
        mount=mount,
        sources=sources,
        solver=solver,
        assets=assets,
        command=["/usr/bin/python3", "producer.py"],
    )


def test_fixed_base_check_only_reports_gate_without_reaching_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    r2_path, hawor_path, raw_root = _source_fixture(
        tmp_path / "sources", mutate_r2=lambda arrays: None
    )
    candidate_record, _ = write_fixed_camera_base_candidate(tmp_path / "candidate")

    def return_zero(
        assets: Any,
        *,
        side: int,
        base: np.ndarray,
        target_tool: np.ndarray,
        initial_q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        del assets, side, base, target_tool, initial_q, lower, upper
        return np.zeros(7, dtype=np.float64), 1

    def publisher_forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("check-only reached the staging publisher")

    monkeypatch.setattr(scene_state_module, "_solve_one_arm", return_zero)
    monkeypatch.setattr(scene_state_module, "_solve_one_arm_position_only", return_zero)
    monkeypatch.setattr(
        scene_state_runner, "_create_private_staging_child", publisher_forbidden
    )
    result = scene_state_runner.main(
        [
            "--project-root",
            str(PROJECT),
            "--session-id",
            "grap_a_cap_004",
            "--r2-sidecar",
            str(r2_path),
            "--hawor-npz",
            str(hawor_path),
            "--raw-root",
            str(raw_root),
            "--mount-json",
            str(
                PROJECT / "tests/fixtures/robot_scene_state_explicit_dev_mount_004.json"
            ),
            "--fixed-camera-base-json",
            str(candidate_record.path),
            "--frame-range",
            "0",
            "3",
            "--check-only",
        ]
    )
    assert result == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "CHECK_ONLY_NO_OUTPUT"
    assert report["output_files_created"] == 0
    assert report["ik_gate_pair_denominator"] == 6
    assert report["ik_gate_frame_denominator"] == 3
    assert report["ik_stage_count"] == 2
    assert report["position_stage_function_evaluations_total"] == 6
    assert report["full_pose_stage_function_evaluations_total"] == 6
    assert report["ik_function_evaluations_total"] == 12
