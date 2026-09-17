from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from chaoyang.pipeline.object_contact_evidence_v1 import validate_evidence_dag
from chaoyang.pipeline.object_pose_hypothesis_v2 import (
    ObjectPoseHypothesisError,
    PoseHypothesisSource,
    build_object_pose_hypotheses,
)


ROOT = Path(__file__).resolve().parents[1]


def _poses(frames: int, objects: int = 1) -> np.ndarray:
    return np.broadcast_to(np.eye(4), (frames, objects, 4, 4)).copy()


def _evidence(valid: np.ndarray) -> np.ndarray:
    result = np.full(valid.shape, None, dtype=object)
    for frame, object_index in zip(*np.nonzero(valid), strict=True):
        result[frame, object_index] = f"direct-{object_index}-{frame}"
    return result


def _seed(evidence_id: str, parent: str) -> dict[str, object]:
    return {
        "evidence_id": evidence_id,
        "evidence_type": "CONTACT_SEED",
        "parent_evidence_ids": [parent],
        "evidence_depth": 1,
        "may_support_contact_authority": True,
        "may_support_object6d_authority": False,
        "may_support_gold_contact_accuracy": False,
        "may_upgrade_tactile_supported_contact": False,
    }


def test_bounded_bidirectional_gap_preserves_formal_object6d_and_parents() -> None:
    poses = _poses(5)
    poses[4, 0, 0, 3] = 0.01
    valid = np.asarray([[True], [False], [False], [False], [True]], dtype=np.bool_)
    before_poses = poses.copy()
    before_valid = valid.copy()
    result = build_object_pose_hypotheses(
        poses,
        valid,
        task_id="poker",
        object_ids=["card_0"],
        fps=30.0,
        formal_evidence_ids=_evidence(valid),
        poker_dimensions_m=[0.063, 0.088, 0.001],
    )
    assert np.array_equal(poses, before_poses)
    assert np.array_equal(valid, before_valid)
    assert result.formal_object6d_sha256_before == result.formal_object6d_sha256_after
    assert result.formal_object6d_mutated is False
    assert result.authority_promotable is False
    assert result.source[:, 0].tolist() == [
        "DIRECT_OBJECT6D", "BIDIRECTIONAL_TRACKED", "BIDIRECTIONAL_TRACKED",
        "BIDIRECTIONAL_TRACKED", "DIRECT_OBJECT6D",
    ]
    for record in result.records[1:4]:
        assert record.parent_evidence_ids == ("direct-0-0", "direct-0-4")
    assert validate_evidence_dag(
        [
            vars(item)
            | {
                "evidence_type": item.evidence_type.value,
                "parent_evidence_ids": list(item.parent_evidence_ids),
            }
            for item in result.evidence_graph.nodes_by_id.values()
        ]
    )


def test_one_sided_or_long_gap_is_unknown() -> None:
    poses = _poses(20)
    valid = np.zeros((20, 1), dtype=np.bool_)
    valid[0, 0] = True
    result = build_object_pose_hypotheses(
        poses, valid, task_id="poker", object_ids=["card_0"], fps=30.0,
        formal_evidence_ids=_evidence(valid), poker_dimensions_m=[0.063, 0.088, 0.001],
    )
    assert np.all(result.source[1:, 0] == PoseHypothesisSource.UNKNOWN.value)
    assert not np.any(result.pose_valid[1:, 0])


def test_attachment_requires_contact_seed_and_never_gains_authority() -> None:
    poses = _poses(4)
    poses[3, 0, 0, 3] = 0.10
    valid = np.asarray([[True], [False], [False], [True]], dtype=np.bool_)
    attached = _poses(4)
    attached[:, 0, 0, 3] = [0.0, 0.03, 0.06, 0.10]
    zeros = np.zeros((4, 1), dtype=np.float64)
    result = build_object_pose_hypotheses(
        poses, valid, task_id="poker", object_ids=["card_0"], fps=30.0,
        formal_evidence_ids=_evidence(valid), poker_dimensions_m=[0.063, 0.088, 0.001],
        contact_seed_nodes=[_seed("seed-card", "direct-0-0")],
        attachment_seed_by_object={"card_0": "seed-card"},
        attachment_poses=attached, attachment_valid=np.ones((4, 1), dtype=np.bool_),
        attachment_entry_distance_m=zeros,
        attachment_relative_translation_drift_m=zeros,
        attachment_relative_rotation_drift_deg=zeros,
    )
    assert result.source[1:3, 0].tolist() == ["HAND_OBJECT_ATTACHMENT"] * 2
    for record in result.records[1:3]:
        assert record.parent_evidence_ids == ("seed-card",)
        node = result.evidence_graph.nodes_by_id[record.evidence_id]
        assert not node.may_support_contact_authority
        assert not node.may_support_object6d_authority
        assert not node.may_support_gold_contact_accuracy
        assert not node.may_upgrade_tactile_supported_contact


def test_attachment_missing_or_illegal_seed_fails_closed() -> None:
    poses = _poses(4)
    poses[3, 0, 0, 3] = 0.10
    valid = np.asarray([[True], [False], [False], [True]], dtype=np.bool_)
    zeros = np.zeros((4, 1))
    with pytest.raises(ObjectPoseHypothesisError, match="missing contact seed evidence"):
        build_object_pose_hypotheses(
            poses, valid, task_id="poker", object_ids=["card_0"], fps=30.0,
            formal_evidence_ids=_evidence(valid), poker_dimensions_m=[0.063, 0.088, 0.001],
            attachment_seed_by_object={"card_0": "missing-seed"},
            attachment_poses=_poses(4), attachment_valid=np.ones((4, 1), dtype=np.bool_),
            attachment_entry_distance_m=zeros,
            attachment_relative_translation_drift_m=zeros,
            attachment_relative_rotation_drift_deg=zeros,
        )


def test_chips_instances_are_independent_and_deformation_becomes_unknown() -> None:
    poses = _poses(3, 3)
    valid = np.ones((3, 3), dtype=np.bool_)
    deformed = np.zeros((3, 3), dtype=np.bool_)
    deformed[1, 1] = True
    result = build_object_pose_hypotheses(
        poses, valid, task_id="chips", object_ids=["chips_0", "chips_1", "chips_2"],
        fps=30.0, formal_evidence_ids=_evidence(valid), visibly_deformed=deformed,
    )
    assert result.source[1, 1] == "UNKNOWN"
    assert not result.pose_valid[1, 1]
    assert result.records[4].reason == "CHIPS_DEFORMED_RIGID_POSE_FORBIDDEN"
    assert result.source[1, 0] == "DIRECT_OBJECT6D"
    assert result.source[1, 2] == "DIRECT_OBJECT6D"


def test_chips_union_and_non_thin_poker_are_rejected() -> None:
    valid = np.ones((2, 3), dtype=np.bool_)
    with pytest.raises(ObjectPoseHypothesisError, match="union"):
        build_object_pose_hypotheses(
            _poses(2, 3), valid, task_id="chips",
            object_ids=["chips_0", "chips_union", "chips_2"], fps=30.0,
            formal_evidence_ids=_evidence(valid),
        )
    poker_valid = np.ones((2, 1), dtype=np.bool_)
    with pytest.raises(ObjectPoseHypothesisError, match="thin two-sided"):
        build_object_pose_hypotheses(
            _poses(2), poker_valid, task_id="poker", object_ids=["card_0"], fps=30.0,
            formal_evidence_ids=_evidence(poker_valid), poker_dimensions_m=[0.06, 0.08, 0.02],
        )


def test_fixture_and_schema_are_well_formed() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/object_pose_hypothesis_v2_fixture.json").read_text())
    assert fixture["expected_sources"][-1] == "UNKNOWN"
    schema = json.loads((ROOT / "contracts/object_pose_hypothesis_v2.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
