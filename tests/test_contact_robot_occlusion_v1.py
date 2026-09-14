from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from pipeline.contact_aware_robot_retarget_v1 import (  # noqa: E402
    RetargetCandidate,
    RetargetContractError,
    SolverBudget,
    Stage,
    StageContext,
    audit_geometry,
    run_staged_retarget,
)
from pipeline.human_contact_hypothesis_v1 import (  # noqa: E402
    ContactHypothesisError,
    PoseSource,
    build_contact_pose_hypotheses,
)
from pipeline.occlusion_compositor_v1 import (  # noqa: E402
    DepthQualityEvidence,
    ObjectPixelSource,
    Ownership,
    audit_frame,
    audit_session,
    choose_object_pixels,
    resolve_ownership,
)


def _poses(frames: int, objects: int = 1) -> np.ndarray:
    return np.broadcast_to(np.eye(4), (frames, objects, 4, 4)).copy()


def test_contact_hypothesis_fills_short_two_sided_gap_without_mutating_object6d() -> None:
    poses = _poses(5)
    poses[4, 0, 0, 3] = 0.010
    valid = np.asarray([[True], [False], [False], [False], [True]])
    before = poses.copy()
    result = build_contact_pose_hypotheses(poses, valid, object_ids=["chips_instance_0"], fps=30.0)
    assert np.array_equal(poses, before)
    assert np.array_equal(valid, [[True], [False], [False], [False], [True]])
    assert result.formal_object6d_sha256_before == result.formal_object6d_sha256_after
    assert result.pose_source[:, 0].tolist() == [
        PoseSource.DIRECT_OBJECT6D,
        PoseSource.BIDIRECTIONAL_RIGID_HYPOTHESIS,
        PoseSource.BIDIRECTIONAL_RIGID_HYPOTHESIS,
        PoseSource.BIDIRECTIONAL_RIGID_HYPOTHESIS,
        PoseSource.DIRECT_OBJECT6D,
    ]
    assert result.hypothesis[:, 0].tolist() == [False, True, True, True, False]


def test_contact_hypothesis_keeps_long_or_one_sided_gaps_unknown() -> None:
    poses = _poses(20)
    valid = np.zeros((20, 1), dtype=np.bool_)
    valid[0, 0] = True
    result = build_contact_pose_hypotheses(poses, valid, object_ids=["card_0"], fps=30.0)
    assert np.all(result.pose_source[1:, 0] == PoseSource.UNKNOWN)
    assert not np.any(result.pose_valid[1:, 0])
    assert np.isnan(result.pose_object_to_camera[1:, 0]).all()


def test_contact_hypothesis_rejects_chips_union_identity() -> None:
    with pytest.raises(ContactHypothesisError, match="unique"):
        build_contact_pose_hypotheses(
            _poses(2, 3), np.ones((2, 3), dtype=np.bool_),
            object_ids=["chips_union", "chips_union", "chips_union"], fps=30.0,
        )


def test_attachment_hypothesis_uses_relative_drift_not_global_object_travel() -> None:
    poses = _poses(4)
    poses[3, 0, 0, 3] = 0.1  # direct endpoint disagreement blocks rigid fill
    valid = np.asarray([[True], [False], [False], [True]])
    attachment = _poses(4)
    attachment[:, 0, 0, 3] = np.asarray([0.0, 0.03, 0.06, 0.1])
    attach_valid = np.ones((4, 1), dtype=np.bool_)
    zero = np.zeros((4, 1))
    result = build_contact_pose_hypotheses(
        poses,
        valid,
        object_ids=["card_0"],
        fps=30.0,
        attachment_poses=attachment,
        attachment_valid=attach_valid,
        attachment_entry_distance_m=zero,
        attachment_relative_translation_drift_m=zero,
        attachment_relative_rotation_drift_deg=zero,
    )
    assert result.pose_source[1:3, 0].tolist() == [
        PoseSource.HAND_OBJECT_ATTACHMENT_HYPOTHESIS,
        PoseSource.HAND_OBJECT_ATTACHMENT_HYPOTHESIS,
    ]
    assert np.allclose(result.pose_object_to_camera[1:3, 0, 0, 3], [0.03, 0.06])


def _candidate(*, chirality: tuple[str, ...] = ("left", "right"), gross: float = 0.0) -> RetargetCandidate:
    return RetargetCandidate(
        joint_positions=np.zeros((2, 3)),
        wrist_to_world=_poses(2)[:, 0],
        chirality=chirality,
        objective=1.0,
        iterations=20,
        contact_distance_m=np.asarray([0.001, 0.0015]),
        contact_penetration_m=np.asarray([0.0002, 0.0004]),
        gross_penetration_m=gross,
        noncontact_penetration_m=np.asarray([0.0, 0.0005]),
        arm_reachable=True,
        structure_closed=True,
        coordinate_chain_complete=True,
    )


def _context(task: str = "poker") -> StageContext:
    return StageContext(
        task_id=task,
        expected_chirality=("left", "right"),
        joint_lower=np.full((2, 3), -1.0),
        joint_upper=np.full((2, 3), 1.0),
        contact_hypothesis={"schema_version": "HUMAN_CONTACT_HYPOTHESIS_V1"},
    )


def test_retarget_runs_exact_stage_order_and_never_auto_promotes() -> None:
    seen: list[Stage] = []
    solvers = {}
    for stage in Stage:
        def solve(context, *, initialization, max_iterations, frame_timeout_s, stage=stage):
            seen.append(stage)
            assert max_iterations == 200
            assert frame_timeout_s == 30.0
            return _candidate()
        solvers[stage] = solve
    result = run_staged_retarget(_context(), solvers)
    assert result.status == "PASSED_DIAGNOSTIC"
    assert result.authority_promotable is False
    assert [receipt.stage for receipt in result.receipts] == list(Stage)
    assert seen == [item for stage in Stage for item in (stage, stage)]


def test_retarget_preserves_best_diagnostic_but_fails_chirality_swap() -> None:
    solvers = {stage: (lambda context, **kwargs: _candidate(chirality=("right", "left"))) for stage in Stage}
    result = run_staged_retarget(_context(), solvers)
    assert result.status == "FAILED_QUALITY_C"
    assert result.authority_promotable is False
    assert result.failed_stage == Stage.WRIST_ARM_IK
    assert "CHIRALITY_SWAP" in result.failure_reasons
    assert result.best_diagnostic is not None


def test_retarget_budget_cannot_exceed_frozen_limits() -> None:
    solvers = {stage: (lambda context, **kwargs: _candidate()) for stage in Stage}
    with pytest.raises(RetargetContractError, match="max_iterations"):
        run_staged_retarget(_context(), solvers, budget=SolverBudget(max_iterations=201))


def test_retarget_rejects_compositor_or_untyped_contact_input() -> None:
    solvers = {stage: (lambda context, **kwargs: _candidate()) for stage in Stage}
    context = _context()
    context = StageContext(
        task_id=context.task_id,
        expected_chirality=context.expected_chirality,
        joint_lower=context.joint_lower,
        joint_upper=context.joint_upper,
        contact_hypothesis={"schema_version": "OCCLUSION_COMPOSITOR_V1"},
    )
    with pytest.raises(RetargetContractError, match="independent HUMAN_CONTACT"):
        run_staged_retarget(context, solvers)


def test_task_specific_penetration_gate_is_stricter_for_poker() -> None:
    candidate = _candidate(gross=0.004)
    assert "GROSS_PENETRATION" in audit_geometry(candidate, "poker")
    assert "GROSS_PENETRATION" not in audit_geometry(candidate, "chips")


def _quality(shape: tuple[int, int], *, present: bool = False) -> DepthQualityEvidence:
    yes = np.ones(shape, dtype=np.bool_)
    return DepthQualityEvidence(yes, yes, yes, yes, yes, yes, present)


def test_object_pixel_provenance_has_frozen_precedence() -> None:
    raw = np.full((1, 3, 3), 10, dtype=np.uint8)
    donor = np.full_like(raw, 20)
    renderer = np.full_like(raw, 30)
    pixels, source = choose_object_pixels(
        raw_rgb=raw,
        raw_visible_mask=np.asarray([[True, False, False]]),
        temporal_donor_rgb=donor,
        temporal_donor_valid=np.asarray([[True, True, False]]),
        renderer_rgb=renderer,
        renderer_valid=np.asarray([[True, True, True]]),
    )
    assert pixels[0, :, 0].tolist() == [10, 20, 30]
    assert source[0].tolist() == [
        ObjectPixelSource.RAW_VISIBLE,
        ObjectPixelSource.TEMPORAL_OBJECT_DONOR,
        ObjectPixelSource.TEXTURED_OBJECT_RENDERER,
    ]


def test_object_front_without_appearance_becomes_unknown_and_training_invalid() -> None:
    shape = (1, 2)
    result = resolve_ownership(
        human_mask=np.zeros(shape, dtype=np.bool_),
        object_amodal_mask=np.ones(shape, dtype=np.bool_),
        object_depth_m=np.full(shape, 0.8),
        object_depth_valid=np.ones(shape, dtype=np.bool_),
        robot_alpha_mask=np.ones(shape, dtype=np.bool_),
        robot_depth_m=np.full(shape, 1.0),
        robot_depth_valid=np.ones(shape, dtype=np.bool_),
        stereo_depth_valid=np.ones(shape, dtype=np.bool_),
        depth_quality_evidence=_quality(shape),
        object_rgb=np.zeros((*shape, 3), dtype=np.uint8),
        object_pixel_source=np.zeros(shape, dtype=np.uint8),
        contact_decision_mask=np.ones(shape, dtype=np.bool_),
    )
    assert np.all(result.ownership == Ownership.TIE_UNKNOWN)
    assert not np.any(result.training_valid_mask)


def test_depth_quality_is_evidence_not_native_confidence() -> None:
    with pytest.raises(Exception, match="native confidence"):
        _quality((1, 1), present=True).combined()


def test_unknown_coverage_cannot_pass_by_abstaining_everywhere() -> None:
    shape = (1, 1)
    frame = audit_frame(
        resolve_ownership(
            human_mask=np.zeros(shape, dtype=np.bool_),
            object_amodal_mask=np.ones(shape, dtype=np.bool_),
            object_depth_m=np.ones(shape),
            object_depth_valid=np.ones(shape, dtype=np.bool_),
            robot_alpha_mask=np.ones(shape, dtype=np.bool_),
            robot_depth_m=np.ones(shape),
            robot_depth_valid=np.ones(shape, dtype=np.bool_),
            stereo_depth_valid=np.ones(shape, dtype=np.bool_),
            depth_quality_evidence=_quality(shape),
            object_rgb=np.zeros((*shape, 3), dtype=np.uint8),
            object_pixel_source=np.zeros(shape, dtype=np.uint8),
            contact_decision_mask=np.ones(shape, dtype=np.bool_),
        ),
        reference_ownership=np.full(shape, Ownership.OBJECT_FRONT),
        protected_raw_object_mask=np.ones(shape, dtype=np.bool_),
    )
    audit = audit_session([frame] * 6)
    assert not audit.passed
    assert "KNOWN_DECISION_COVERAGE" in audit.failed_gates
    assert "UNKNOWN_PIXEL_RATIO" in audit.failed_gates
    assert "MAX_UNKNOWN_RUN" in audit.failed_gates


def test_known_accurate_frames_pass_only_with_protected_raw_retention() -> None:
    shape = (1, 2)
    raw = np.full((*shape, 3), 20, dtype=np.uint8)
    pixels, source = choose_object_pixels(
        raw_rgb=raw, raw_visible_mask=np.ones(shape, dtype=np.bool_)
    )
    result = resolve_ownership(
        human_mask=np.zeros(shape, dtype=np.bool_),
        object_amodal_mask=np.ones(shape, dtype=np.bool_),
        object_depth_m=np.full(shape, 0.8),
        object_depth_valid=np.ones(shape, dtype=np.bool_),
        robot_alpha_mask=np.ones(shape, dtype=np.bool_),
        robot_depth_m=np.full(shape, 1.0),
        robot_depth_valid=np.ones(shape, dtype=np.bool_),
        stereo_depth_valid=np.ones(shape, dtype=np.bool_),
        depth_quality_evidence=_quality(shape),
        object_rgb=pixels,
        object_pixel_source=source,
        contact_decision_mask=np.ones(shape, dtype=np.bool_),
    )
    frame = audit_frame(
        result,
        reference_ownership=np.full(shape, Ownership.OBJECT_FRONT),
        protected_raw_object_mask=np.ones(shape, dtype=np.bool_),
    )
    session = audit_session([frame])
    assert session.passed
    assert session.known_decision_coverage == 1.0
    assert session.accuracy_on_known == 1.0
    assert session.protected_retention == 1.0


@pytest.mark.parametrize(
    "schema_name",
    [
        "human_contact_hypothesis_v1.schema.json",
        "contact_aware_robot_retarget_v1.schema.json",
        "occlusion_compositor_v1.schema.json",
    ],
)
def test_v1_contract_schemas_are_valid(schema_name: str) -> None:
    schema = json.loads((PROJECT / "contracts" / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_occlusion_schema_rejects_invented_foundationstereo_confidence() -> None:
    schema = json.loads((PROJECT / "contracts/occlusion_compositor_v1.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    value = {
        "schema_version": "OCCLUSION_COMPOSITOR_V1",
        "session_id": "x",
        "inputs": {name: {"path": "x", "bytes": 1, "sha256": "0" * 64} for name in ["clean_rgb", "stereo_depth", "object_identity_mask", "object6d_or_hypothesis", "robot_render"]},
        "depth_contract": {
            "depth_field": "depth_m", "valid_field": "depth_valid",
            "depth_confidence_present": True,
            "quality_evidence_fields": ["disparity_finite", "registration_pass", "away_from_occlusion_edge", "texture_support", "local_consistency_pass", "in_valid_depth_range"],
        },
        "object_pixel_provenance": ["RAW_VISIBLE", "TEMPORAL_OBJECT_DONOR", "TEXTURED_OBJECT_RENDERER", "NONE_UNKNOWN"],
        "ownership": ["BACKGROUND", "HUMAN_FRONT", "OBJECT_FRONT", "ROBOT_FRONT", "TIE_UNKNOWN"],
        "quality_gates": {"accuracy_on_known_min": 0.95, "known_decision_coverage_min": 0.70, "unknown_pixel_ratio_max": 0.30, "unknown_contact_frame_ratio_max": 0.20, "max_unknown_run": 5, "max_wrong_known_run": 2, "protected_retention_min": 0.99},
        "feedback_to_retarget_allowed": False,
        "authority": {"automatic_promotion_allowed": False, "unknown_training_policy": "TRAINING_VALID_MASK_FALSE_AND_PAIRED_FRAME_DROP_ABOVE_30_PERCENT"},
    }
    assert any("False was expected" in error.message for error in validator.iter_errors(value))
