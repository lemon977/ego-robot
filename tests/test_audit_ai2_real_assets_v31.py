from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from chaoyang.ops import audit_ai2_real_assets_v31 as op


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fake_assets() -> SimpleNamespace:
    joints = tuple(
        SimpleNamespace(name=f"joint_{index:02d}", joint_type="revolute")
        for index in range(22)
    )
    hand = SimpleNamespace(joints=joints)
    return SimpleNamespace(left_hand=hand, right_hand=hand)


def _patch_robot_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(op, "load_pinned_robot_assets", lambda _root: _fake_assets())
    monkeypatch.setattr(
        op,
        "forward_kinematics",
        lambda _model, _configuration: {"base": np.eye(4, dtype=np.float64)},
    )


def _hawor_fixture(root: Path, cohort: str, session_id: str, frames: int = 5) -> tuple[Path, Path]:
    base = root / op.COHORT_ROOTS[cohort]["hawor"] / "sessions" / session_id
    base.mkdir(parents=True, exist_ok=True)
    npz = base / "HAWOR_RAW_MANO21.npz"
    joints_3d = np.zeros((2, frames, 21, 3), dtype=np.float32)
    joints_2d = np.zeros((2, frames, 21, 2), dtype=np.float32)
    observed = np.ones((2, frames), dtype=bool)
    np.savez_compressed(
        npz,
        joints_3d_camera=joints_3d,
        joints_2d=joints_2d,
        observed=observed,
        original_frame_indices=np.arange(frames, dtype=np.int64),
        anatomical_side_names=np.asarray(["left", "right"]),
        mano_joint_names=np.asarray([f"joint_{index}" for index in range(21)]),
    )
    result = _write_json(
        base / "RESULT.json",
        {
            "session_id": session_id,
            "frame_count": frames,
            "output": op.artifact_ref(npz),
        },
    )
    return npz, result


def _joint_contract(session_id: str, npz: Path) -> dict:
    names = [f"joint_{index:02d}" for index in range(22)]
    return {
        "session_id": session_id,
        "arrays": op.artifact_ref(npz),
        "joint_order": {"left": names, "right": names},
        "side_contract": {"human_to_physical": {"left": "right", "right": "left"}},
        "q22_units": "radians",
    }


def _a1_kai_fixture(root: Path, session_id: str, frames: int = 5) -> tuple[Path, Path]:
    base = root / op.COHORT_ROOTS["A1"]["kai22"] / "sessions" / session_id
    base.mkdir(parents=True, exist_ok=True)
    npz = base / "KAI22_R0_QUALITY_PROPAGATION_V1.npz"
    np.savez_compressed(
        npz,
        q22=np.zeros((frames, 2, 22), dtype=np.float64),
        q22_computed=np.ones((frames, 2), dtype=bool),
        fk_pass=np.ones((frames, 2), dtype=bool),
        r0_static_gate_pass=np.ones((frames, 2), dtype=bool),
        frame_id=np.arange(frames, dtype=np.int64),
        timestamps_s=np.arange(frames, dtype=np.float64) / 30.0,
    )
    return npz, _write_json(base / "RESULT.json", _joint_contract(session_id, npz))


def _w0_kai_fixture(root: Path, session_id: str, frames: int = 5) -> tuple[Path, Path]:
    base = root / op.COHORT_ROOTS["W0"]["kai22"] / "sessions" / session_id
    base.mkdir(parents=True, exist_ok=True)
    npz = base / "KAI22_R0_BASELINE_V1.npz"
    np.savez_compressed(
        npz,
        q22_init=np.zeros((frames, 2, 22), dtype=np.float64),
        valid_side_frame=np.ones((frames, 2), dtype=bool),
        frame_id=np.arange(frames, dtype=np.int64),
        timestamps_s=np.arange(frames, dtype=np.float64) / 30.0,
    )
    document = _joint_contract(session_id, npz)
    document["q22_joint_order"] = document.pop("joint_order")
    return npz, _write_json(base / "RESULT.json", document)


def _schema_validate(value: dict) -> None:
    schema = json.loads(
        (ROOT / "contracts" / "ai2_real_assets_audit_v31.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)


def test_a1_audit_fails_closed_without_independent_quality_or_suffix_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_robot_geometry(monkeypatch)
    session_id = "get_potato_chips_0915_097"
    hawor_npz, hawor_result = _hawor_fixture(tmp_path, "A1", session_id)
    kai_npz, kai_result = _a1_kai_fixture(tmp_path, session_id)

    result = op.build_audit(
        cohort="A1",
        session_id=session_id,
        hawor_npz=hawor_npz,
        hawor_result=hawor_result,
        kai22_npz=kai_npz,
        kai22_result=kai_result,
        root=tmp_path,
    )

    _schema_validate(result)
    assert result["status"] == "COMPLETED_FAIL_CLOSED_AUDIT"
    assert result["part_presence"]["denominator_published"] is False
    assert result["part_presence"]["independent_observation_rows"] == 0
    assert result["temporal_authority"]["status"] == "REJECTED_NONCAUSAL_CURRENT_INPUT"
    assert result["bounded_comparison"]["adoption_authorized"] is False
    assert result["kai22_tier"]["sides"]["left"]["highest_admitted_level"] == "KINEMATIC_ONLY"
    assert result["kai22_tier"]["sides"]["right"]["highest_admitted_level"] == "KINEMATIC_ONLY"
    assert result["training_eligible"] is False
    assert result["quality_authority_promoted"] is False
    assert result["model_calls"] == result["gpu_calls"] == 0
    assert {
        "BLOCKED_INDEPENDENT_PART_OBSERVABILITY",
        "BLOCKED_INDEPENDENT_REPROJECTION",
        "BLOCKED_SUFFIX_PAIR_MATERIALIZATION",
        "BLOCKED_R0_LOCAL_QUALITY_POLICY",
    } <= set(result["blocker_codes"])


def test_w0_hawor_anchored_sam_and_bounded_candidate_cannot_self_authorize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_robot_geometry(monkeypatch)
    session_id = "get_potato_chips_0915_042"
    hawor_npz, hawor_result = _hawor_fixture(tmp_path, "W0", session_id)
    kai_npz, kai_result = _w0_kai_fixture(tmp_path, session_id)
    with np.load(hawor_npz, allow_pickle=False) as archive:
        raw = {key: np.asarray(archive[key]) for key in archive.files}
    raw["joints_3d_camera"] = raw["joints_3d_camera"] + np.float32(0.001)
    raw["joints_2d"] = raw["joints_2d"] + np.float32(0.25)
    bounded_root = tmp_path / op.COHORT_ROOTS["W0"]["kai22"] / "bounded_v2" / session_id
    bounded_root.mkdir(parents=True, exist_ok=True)
    bounded_npz = bounded_root / "HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz"
    np.savez_compressed(bounded_npz, **raw)
    bounded_result = _write_json(
        bounded_root / "RESULT.json",
        {
            "session_id": session_id,
            "candidate": op.artifact_ref(bounded_npz),
            "raw_input": op.artifact_ref(hawor_npz),
        },
    )
    sam_root = tmp_path / op.COHORT_ROOTS["W0"]["sam"] / "sessions" / session_id
    missing_stage = sam_root / ".staging-missing" / "semantic_masks.npz"
    sam_result = _write_json(
        sam_root / "RESULT.json",
        {
            "session_id": session_id,
            "hawor_input": op.artifact_ref(hawor_npz),
            "semantic_output": {
                "path": str(missing_stage),
                "bytes": 0,
                "sha256": "0" * 64,
            },
        },
    )

    result = op.build_audit(
        cohort="W0",
        session_id=session_id,
        hawor_npz=hawor_npz,
        hawor_result=hawor_result,
        kai22_npz=kai_npz,
        kai22_result=kai_result,
        bounded_npz=bounded_npz,
        bounded_result=bounded_result,
        sam_result=sam_result,
        root=tmp_path,
    )

    provider = result["provider_dependency"]["providers"]["sam31_hawor_anchored_hand"]
    assert provider["admissible_as_hawor_observability_evidence"] is False
    assert "TRANSITIVE_HAWOR_DEPENDENCY:hawor_raw_output" in provider["reason_codes"]
    assert result["provider_dependency"]["sam_admitted_as_hawor_observability_judge"] is False
    assert result["provider_dependency"]["missing_sam_staging_references"] == [
        str(missing_stage)
    ]
    assert result["bounded_comparison"]["status"] == "BLOCKED_INDEPENDENT_QUALITY_EVIDENCE"
    assert result["bounded_comparison"]["structural_pair_check"] == (
        "PASS_SAME_FRAME_AND_OBSERVED_AXES"
    )
    assert result["bounded_comparison"]["adoption_authorized"] is False
    assert result["bounded_comparison"]["per_side_diagnostic"]["left"][
        "camera_correction_p95_mm"
    ] == pytest.approx(np.sqrt(3.0))
    assert result["kai22_tier"]["sides"]["left"]["highest_admitted_level"] == "KINEMATIC_ONLY"
    assert result["model_calls"] == result["gpu_calls"] == 0
    _schema_validate(result)


def test_input_sha_binding_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_robot_geometry(monkeypatch)
    session_id = "play_cards_0915_044"
    hawor_npz, hawor_result = _hawor_fixture(tmp_path, "A1", session_id)
    document = json.loads(hawor_result.read_text(encoding="utf-8"))
    document["output"]["sha256"] = "f" * 64
    _write_json(hawor_result, document)
    kai_npz, kai_result = _a1_kai_fixture(tmp_path, session_id)

    with pytest.raises(op.RealAssetAuditError, match="SHA mismatch"):
        op.build_audit(
            cohort="A1",
            session_id=session_id,
            hawor_npz=hawor_npz,
            hawor_result=hawor_result,
            kai22_npz=kai_npz,
            kai22_result=kai_result,
            root=tmp_path,
        )
