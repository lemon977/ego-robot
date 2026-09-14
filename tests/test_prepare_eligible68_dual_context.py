from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import prepare_eligible68_dual_context as context
from tools import run_002_012_d4_contact_from_shared_inventory as connector


SESSION_ID = "grap_a_cap_004"


def _ref(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path.absolute()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _candidate(stream: str, offset: int) -> dict[str, object]:
    return {
        "source_stream": stream,
        "raw_instance_offset": offset,
        "instance_id": offset + 1,
        "score": 0.75,
        "mask_sha256": "a" * 64,
    }


def test_literal_scope_equals_frozen_plan_and_constituent_manifest() -> None:
    helper, _ = context._load_helper()
    plan, _ = context._verified_json(
        helper,
        context.PLAN_PATH,
        expected_bytes=context.PLAN_BYTES,
        expected_sha256=context.PLAN_SHA256,
    )
    constituent, _ = context._verified_json(
        helper,
        context.CONSTITUENT_PATH,
        expected_bytes=context.CONSTITUENT_BYTES,
        expected_sha256=context.CONSTITUENT_SHA256,
    )
    rows = context.validate_frozen_scope(plan, constituent)
    assert len(rows) == 68
    assert sum(row[1] for row in context.ELIGIBLE68_ROWS) == 28_265
    assert context.ELIGIBLE68_ROWS[0][:3] == ("grap_a_cap_004", 460, "train")


@pytest.mark.parametrize(
    "session_id",
    [
        "grap_a_cap_025",
        "grap_a_cap_149",
        "025",
        "grap_a_cap_004/",
        "../grap_a_cap_004",
        "/mnt/data/grap_a_cap_004",
    ],
)
def test_session_scope_rejects_forbidden_and_alias_forms(session_id: str) -> None:
    with pytest.raises(context.ContextError, match="literal eligible68 allowlist"):
        context.require_eligible_session(session_id)


def test_context_producer_contains_no_directory_discovery() -> None:
    source = Path(context.__file__).read_text(encoding="utf-8")
    for forbidden in ("os.listdir(", "os.scandir(", "os.walk(", ".glob(", ".rglob("):
        assert forbidden not in source


def test_output_directory_default_run_root_remains_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "_run"
    run_root.mkdir()
    monkeypatch.setattr(context, "RUN_ROOT", run_root)
    output = run_root / "legacy_direct_child_name"
    binding = context._create_output_directory(SESSION_ID, output, None)
    try:
        assert binding.output == output
        assert binding.approval_mode == "DEFAULT_PROJECT_RUN_ROOT"
        context._verify_output_directory(binding)
    finally:
        binding.close()


def test_output_directory_explicit_base_writes_same_fd_verified_json(
    tmp_path: Path,
) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    output = base / SESSION_ID
    binding = context._create_output_directory(SESSION_ID, output, base)
    try:
        ref = context._exclusive_json_at(binding, "artifact.json", {"ok": True})
        assert ref["path"] == str(output / "artifact.json")
        assert ref["bytes"] == (output / "artifact.json").stat().st_size
        assert json.loads((output / "artifact.json").read_bytes()) == {"ok": True}
        assert (output / "artifact.json").stat().st_nlink == 1
        assert binding.approval_mode == "EXPLICIT_CALLER_APPROVED_BASE"
    finally:
        binding.close()


def test_explicit_output_base_rejects_relative_root_missing_and_symlink(
    tmp_path: Path,
) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    link = tmp_path / "approved_link"
    link.symlink_to(base, target_is_directory=True)
    cases = [
        (Path("relative"), Path("relative") / SESSION_ID, "must be absolute"),
        (Path("/"), Path("/") / SESSION_ID, "filesystem anchor"),
        (
            tmp_path / "missing",
            tmp_path / "missing" / SESSION_ID,
            "does not exist",
        ),
        (link, link / SESSION_ID, "symlink or non-directory"),
    ]
    for approved, output, message in cases:
        with pytest.raises(context.ContextError, match=message):
            context._create_output_directory(SESSION_ID, output, approved)


@pytest.mark.parametrize(
    "suffix",
    [
        ("wrong-session",),
        ("nested", SESSION_ID),
        ("nested", "..", SESSION_ID),
    ],
)
def test_explicit_output_must_be_exact_session_direct_child(
    tmp_path: Path,
    suffix: tuple[str, ...],
) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    output = base.joinpath(*suffix)
    with pytest.raises(
        context.ContextError,
        match="must equal approved output base|must not contain",
    ):
        context._create_output_directory(SESSION_ID, output, base)


def test_explicit_output_rejects_preexisting_child(tmp_path: Path) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    output = base / SESSION_ID
    output.mkdir()
    with pytest.raises(context.ContextError, match="output already exists"):
        context._create_output_directory(SESSION_ID, output, base)


def test_output_binding_rejects_base_rename_replacement(tmp_path: Path) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    output = base / SESSION_ID
    binding = context._create_output_directory(SESSION_ID, output, base)
    moved = tmp_path / "moved_approved"
    try:
        base.rename(moved)
        base.mkdir()
        with pytest.raises(context.ContextError, match="identity drift"):
            context._verify_output_directory(binding)
    finally:
        binding.close()


def test_output_binding_rejects_child_rename_replacement(tmp_path: Path) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    output = base / SESSION_ID
    binding = context._create_output_directory(SESSION_ID, output, base)
    moved = base / "moved_session"
    try:
        output.rename(moved)
        output.mkdir()
        with pytest.raises(context.ContextError, match="entry identity drift"):
            context._verify_output_directory(binding)
    finally:
        binding.close()


def test_output_json_is_create_once_and_does_not_overwrite(tmp_path: Path) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    output = base / SESSION_ID
    binding = context._create_output_directory(SESSION_ID, output, base)
    artifact = output / "artifact.json"
    artifact.write_text('{"owner":"existing"}\n', encoding="utf-8")
    before = artifact.read_bytes()
    try:
        with pytest.raises(FileExistsError):
            context._exclusive_json_at(binding, artifact.name, {"owner": "new"})
        assert artifact.read_bytes() == before
    finally:
        binding.close()


def test_output_json_rejects_concurrent_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "approved"
    base.mkdir()
    output = base / SESSION_ID
    binding = context._create_output_directory(SESSION_ID, output, base)
    artifact = output / "artifact.json"
    alias = output / "alias.json"
    original_write = context.os.write
    linked = False

    def hostile_write(descriptor: int, payload: bytes) -> int:
        nonlocal linked
        written = original_write(descriptor, payload)
        if not linked:
            linked = True
            context.os.link(artifact, alias)
        return written

    monkeypatch.setattr(context.os, "write", hostile_write)
    try:
        with pytest.raises(context.ContextError, match="identity drift after write"):
            context._exclusive_json_at(binding, artifact.name, {"ok": True})
        assert artifact.stat().st_nlink == 2
    finally:
        binding.close()


def test_dual_execution_contract_requires_both_complete_streams() -> None:
    value = {
        "detection_mode": "dual",
        "session_id": "grap_a_cap_004",
        "streams": {
            "propagation": {"frame_indices": [0, 1]},
            "per_frame": {"frame_indices": [0, 1]},
        },
        "frame_provenance": [
            {
                "frame_index": frame,
                "candidate_inventories": {
                    "propagation": [_candidate("propagation", 0)],
                    "per_frame": [_candidate("per_frame", 1)],
                },
            }
            for frame in range(2)
        ],
    }
    context.validate_dual_execution_summary(
        value, session_id="grap_a_cap_004", frame_count=2
    )

    missing = json.loads(json.dumps(value))
    missing["streams"]["propagation"]["frame_indices"] = [0]
    with pytest.raises(context.ContextError, match="propagation stream incomplete"):
        context.validate_dual_execution_summary(
            missing, session_id="grap_a_cap_004", frame_count=2
        )

    wrong_source = json.loads(json.dumps(value))
    wrong_source["frame_provenance"][1]["candidate_inventories"]["per_frame"][0][
        "source_stream"
    ] = "propagation"
    with pytest.raises(context.ContextError, match="stream provenance mismatch"):
        context.validate_dual_execution_summary(
            wrong_source, session_id="grap_a_cap_004", frame_count=2
        )


def test_contact_geometry_provider_accepts_explicit_eligible_session(
    tmp_path: Path,
) -> None:
    joints = tmp_path / "joints.npz"
    object6d = tmp_path / "object6d.npz"
    cylinder = tmp_path / "cylinder.json"
    joint_points = np.zeros((2, 2, 21, 3), dtype=np.float64)
    joint_points[1, 1] = np.nan
    joint_valid = np.ones((2, 2), dtype=np.bool_)
    joint_valid[1, 1] = False
    np.savez(
        joints,
        joints_3d_camera=joint_points,
        valid=joint_valid,
    )
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
    transforms[1] = np.nan
    np.savez(
        object6d,
        T_object_to_camera=transforms,
        valid=np.asarray([True, False], dtype=np.bool_),
        cylinder_radius_m=np.asarray(0.025, dtype=np.float64),
        cylinder_height_m=np.asarray(0.08, dtype=np.float64),
    )
    cylinder.write_text(
        json.dumps({"radius_m": 0.025, "height_m": 0.08}), encoding="utf-8"
    )
    geometry = {
        "grap_a_cap_004": {
            "joints": _ref(joints),
            "object6d": _ref(object6d),
            "cylinder": _ref(cylinder),
            "physical_hand_axis": "0=left,1=right; source_slot is diagnostic only",
        }
    }
    provider = connector.FrozenContactGeometryProvider(
        geometry,
        session_frame_counts={"grap_a_cap_004": 2},
        allowed_root=tmp_path,
    )
    pair = provider.pair("grap_a_cap_004", 0)
    invalid_pair = provider.pair("grap_a_cap_004", 1)
    assert pair["left"].physical_hand_axis == 0
    assert pair["right"].physical_hand_axis == 1
    assert pair["left"].session_id == "grap_a_cap_004"
    assert invalid_pair["right"].hawor_valid is False
    assert invalid_pair["right"].object6d_valid is False


def test_contact_geometry_provider_fails_closed_on_scope_mismatch() -> None:
    with pytest.raises(connector.D4WiringError, match="session set drift"):
        connector.FrozenContactGeometryProvider(
            {}, session_frame_counts={"grap_a_cap_004": 1}
        )
    with pytest.raises(
        connector.D4WiringError, match="invalid explicit geometry session"
    ):
        connector.FrozenContactGeometryProvider(
            {"grap_a_cap_025": {}},
            session_frame_counts={"grap_a_cap_025": 1},
        )


def test_context_object6d_allows_nan_only_on_invalid_frames() -> None:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
    transforms[1] = np.nan
    context._validate_object6d_arrays(
        transforms,
        np.asarray([True, False], dtype=np.bool_),
        session_id="grap_a_cap_050",
        frame_count=2,
    )


def test_context_object6d_rejects_nan_on_valid_frame() -> None:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, ...], 2, axis=0)
    transforms[1] = np.nan
    with pytest.raises(context.ContextError, match="non-finite valid transform"):
        context._validate_object6d_arrays(
            transforms,
            np.asarray([True, True], dtype=np.bool_),
            session_id="grap_a_cap_050",
            frame_count=2,
        )


def test_contact_geometry_provider_rejects_nan_on_valid_frame(
    tmp_path: Path,
) -> None:
    joints = tmp_path / "joints.npz"
    object6d = tmp_path / "object6d.npz"
    cylinder = tmp_path / "cylinder.json"
    np.savez(
        joints,
        joints_3d_camera=np.zeros((2, 1, 21, 3), dtype=np.float64),
        valid=np.ones((2, 1), dtype=np.bool_),
    )
    np.savez(
        object6d,
        T_object_to_camera=np.full((1, 4, 4), np.nan, dtype=np.float64),
        valid=np.ones(1, dtype=np.bool_),
        cylinder_radius_m=np.asarray(0.025, dtype=np.float64),
        cylinder_height_m=np.asarray(0.08, dtype=np.float64),
    )
    cylinder.write_text(
        json.dumps({"radius_m": 0.025, "height_m": 0.08}), encoding="utf-8"
    )
    geometry = {
        "grap_a_cap_004": {
            "joints": _ref(joints),
            "object6d": _ref(object6d),
            "cylinder": _ref(cylinder),
            "physical_hand_axis": "0=left,1=right; source_slot is diagnostic only",
        }
    }
    with pytest.raises(connector.D4WiringError, match="non-finite valid"):
        connector.FrozenContactGeometryProvider(
            geometry,
            session_frame_counts={"grap_a_cap_004": 1},
            allowed_root=tmp_path,
        )
