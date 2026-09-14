from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from pipeline.depth_occlusion_v3 import OCCLUSION_EPSILON_M
from tools.diagnose_thumb_causality_d4 import (
    DIAGNOSTIC_EPSILON_GRID_M,
    D4DiagnosticError,
    REQUIRED_CANDIDATE_STATUS,
    STRESS_FRAMES,
    build_candidate_manifest,
    validate_candidate_governance,
    write_candidate_manifest,
)


PROJECT = Path(__file__).resolve().parents[1]
RAW = Path(
    "/mnt/data/egodata/folder/grap_a_cap_0812/grap_a_cap_004/"
    "preprocess/all_data"
)
OBJECT6D = PROJECT / (
    "data/benchmarks/hand/production_runs/final_v3_grap_a_cap_0812/"
    "grap_a_cap_004/10_object6d_v2/v2_cylinder_center_rotation_gated/"
    "object_6dof_v2.npz"
)
Q_HAND = PROJECT / (
    "HumanEgo/data_manifests/robot_sidecars_v4_candidate/"
    "r2_robust24_confidence_temporal/kai22/grap_a_cap_004/sidecar.npz"
)
D2_STATUS = PROJECT / (
    "archive/legacy_runs/incomplete/d2_cycles_renderer_t0/hand_only_frame00228/ATTEMPT_STATUS.json"
)


@pytest.fixture(scope="module")
def current_manifest() -> dict:
    return build_candidate_manifest(
        project_root=PROJECT,
        output_dir=PROJECT / "archive/legacy_runs/unclassified/d4_thumb_causality_candidate_v1",
        raw_root=RAW,
        object6d=OBJECT6D,
        q_hand_sidecar=Q_HAND,
        d2_attempt_status=D2_STATUS,
    )


def test_current_candidate_freezes_exact_historic_stress24(current_manifest: dict) -> None:
    assert tuple(current_manifest["frame_indices"]) == tuple(range(424, 448))
    assert tuple(current_manifest["frame_indices"]) == STRESS_FRAMES
    assert current_manifest["inputs"]["raw"]["ready_frames"] == 24
    assert current_manifest["inputs"]["object6d"]["valid_frames"] == 24
    assert current_manifest["inputs"]["q_hand"]["left_valid_frames"] == 24
    assert current_manifest["inputs"]["q_hand"]["right_valid_frames"] == 24


def test_current_candidate_is_fail_closed_without_formal_d2_chain(
    current_manifest: dict,
) -> None:
    assert current_manifest["status"] == REQUIRED_CANDIDATE_STATUS
    assert current_manifest["next_bucket_blocked"] is True
    assert current_manifest["advancement_authorized"] is False
    assert current_manifest["formal_24_frame_attribution_performed"] is False
    assert current_manifest["synthetic_substitution_used"] is False
    assert current_manifest["per_frame_csv_generated"] is False
    assert current_manifest["nine_panel_generated"] is False
    assert current_manifest["recommended_branch"] is None
    assert current_manifest["branch_decision_status"] == (
        "HOLD_INSUFFICIENT_FROZEN_INPUTS"
    )
    assert set(current_manifest["missing_or_unready_inputs"]) == {
        "mask",
        "clean",
        "object_donor",
        "q_arm",
        "beauty_range_index",
        "table_plane",
        "d2_attempt",
    }
    d2 = current_manifest["inputs"]["d2_attempt"]
    assert d2["ready"] is False
    assert d2["buffers_generated"] == {
        "beauty": False,
        "range": False,
        "object_index": False,
    }
    assert d2["full_chain_gaps"]


def test_current_candidate_preserves_nonnegotiable_depth_rules(
    current_manifest: dict,
) -> None:
    frozen = current_manifest["immutable_diagnostics"]
    assert frozen["primary_compositor_epsilon_m"] == OCCLUSION_EPSILON_M == 0.003
    assert frozen["epsilon_ab_m"] == list(DIAGNOSTIC_EPSILON_GRID_M)
    assert frozen["supersample"] == 2
    assert frozen["object_index_is_qa_only"] is True
    assert "thumb_whitelist" in current_manifest["forbidden_actions"]
    assert "force_thumb_foreground" in current_manifest["forbidden_actions"]
    assert "lower_3mm_epsilon" in current_manifest["forbidden_actions"]


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("status", "complete"),
        ("next_bucket_blocked", False),
        ("advancement_authorized", True),
        ("synthetic_substitution_used", True),
        ("recommended_branch", "COMPOSITOR"),
        ("frame_indices", list(range(423, 447))),
    ],
)
def test_governance_rejects_candidate_advancement_or_evidence_substitution(
    current_manifest: dict, key: str, bad_value: object
) -> None:
    candidate = copy.deepcopy(current_manifest)
    candidate[key] = bad_value
    with pytest.raises(D4DiagnosticError):
        validate_candidate_governance(candidate)


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("primary_compositor_epsilon_m", 0.001),
        ("epsilon_ab_m", [0.0, 0.003]),
    ],
)
def test_governance_rejects_depth_rule_changes(
    current_manifest: dict, key: str, bad_value: object
) -> None:
    candidate = copy.deepcopy(current_manifest)
    candidate["immutable_diagnostics"][key] = bad_value
    with pytest.raises(D4DiagnosticError):
        validate_candidate_governance(candidate)


def test_builder_rejects_output_outside_run() -> None:
    with pytest.raises(D4DiagnosticError, match="below project _run"):
        build_candidate_manifest(
            project_root=PROJECT,
            output_dir=PROJECT / "archive/audits/not_a_candidate_output",
            raw_root=RAW,
            object6d=OBJECT6D,
            q_hand_sidecar=Q_HAND,
            d2_attempt_status=D2_STATUS,
        )


def test_manifest_write_is_idempotent_but_refuses_overwrite(
    tmp_path: Path, current_manifest: dict
) -> None:
    path = tmp_path / "TASK_MANIFEST.json"
    write_candidate_manifest(path, current_manifest)
    write_candidate_manifest(path, current_manifest)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["status"] == REQUIRED_CANDIDATE_STATUS

    changed = copy.deepcopy(current_manifest)
    changed["branch_decision_status"] = "DIFFERENT"
    with pytest.raises(D4DiagnosticError, match="refusing overwrite"):
        write_candidate_manifest(path, changed)
