from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from chaoyang.pipeline.interaction_evidence_v0a import (
    InteractionEvidenceError,
    estimate_interaction_evidence_v0a,
    interaction_evidence_v0a_to_record,
)


PROJECT = Path(__file__).resolve().parents[1]


def moving_square_masks(frame_count: int = 4) -> np.ndarray:
    masks = np.zeros((frame_count, 1, 24, 32), dtype=bool)
    for frame in range(frame_count):
        masks[frame, 0, 10:13, 10 + frame : 13 + frame] = True
    return masks


def direct_inputs() -> dict[str, np.ndarray]:
    return {
        "fingertip_xy_px": np.asarray(
            [[[2.0, 11.0]], [[7.0, 11.0]], [[11.0, 11.0]], [[13.0, 11.0]]]
        ),
        "fingertip_observation_state": np.full(
            (4, 1), "direct_observed", dtype="<U16"
        ),
        "object_masks": moving_square_masks(),
        "object_mask_state": np.asarray(
            [["seeded"], ["tracked"], ["tracked"], ["reseeded"]]
        ),
        "frame_timestamps_s": np.asarray([0.0, 0.1, 0.2, 0.3]),
    }


def test_2d_adjacency_and_approach_are_direct_observations_only() -> None:
    result = estimate_interaction_evidence_v0a(
        **direct_inputs(), adjacency_threshold_px=1.0, approach_min_delta_px=0.5
    )
    assert result.boundary_distance_px[:, 0, 0].tolist() == [8.0, 4.0, 1.0, 0.0]
    assert result.adjacency_status[:, 0, 0].tolist() == [
        "NOT_ADJACENT_2D",
        "NOT_ADJACENT_2D",
        "ADJACENT_2D",
        "ADJACENT_2D",
    ]
    assert result.approach_status[:, 0, 0].tolist() == [
        "UNKNOWN",
        "APPROACHING_2D",
        "APPROACHING_2D",
        "APPROACHING_2D",
    ]


def test_unknown_mask_and_inferred_hawor_propagate_unknown() -> None:
    inputs = direct_inputs()
    inputs["object_masks"][1, 0] = False
    inputs["object_mask_state"][1, 0] = "unknown"
    inputs["fingertip_observation_state"][2, 0] = "inferred"
    result = estimate_interaction_evidence_v0a(**inputs)

    for frame in (1, 2):
        assert result.adjacency_status[frame, 0, 0] == "UNKNOWN"
        assert result.approach_status[frame, 0, 0] == "UNKNOWN"
        assert result.co_motion_status[frame, 0, 0] == "UNKNOWN"
        assert result.tactile_status[frame, 0, 0] == "UNKNOWN"
        assert np.isnan(result.boundary_distance_px[frame, 0, 0])
    # A direct current frame regains adjacency, but temporal evidence stays
    # unknown because its predecessor was inferred.
    assert result.adjacency_status[3, 0, 0] == "ADJACENT_2D"
    assert result.approach_status[3, 0, 0] == "UNKNOWN"
    assert result.co_motion_status[3, 0, 0] == "UNKNOWN"


def test_co_motion_requires_consistent_direct_motion() -> None:
    inputs = direct_inputs()
    inputs["fingertip_xy_px"] = np.asarray(
        [[[11.0, 11.0]], [[12.0, 11.0]], [[13.0, 11.0]], [[14.0, 11.0]]]
    )
    result = estimate_interaction_evidence_v0a(
        **inputs,
        adjacency_threshold_px=1.0,
        co_motion_min_displacement_px=0.5,
        co_motion_max_residual_px=0.1,
    )
    assert result.co_motion_status[:, 0, 0].tolist() == [
        "UNKNOWN",
        "CO_MOVING_2D",
        "CO_MOVING_2D",
        "CO_MOVING_2D",
    ]


def test_tactile_support_requires_valid_aligned_activity_and_2d_adjacency() -> None:
    inputs = direct_inputs()
    result = estimate_interaction_evidence_v0a(
        **inputs,
        adjacency_threshold_px=1.0,
        tactile_active=np.asarray([[True], [True], [True], [False]], dtype=bool),
        tactile_source_valid=np.asarray([[True], [False], [True], [True]], dtype=bool),
        tactile_timestamps_s=np.asarray([[0.0], [0.1], [0.22], [0.3]]),
        tactile_max_offset_s=0.040,
    )
    assert result.tactile_status[:, 0, 0].tolist() == [
        "NOT_TACTILE_SUPPORTED",  # active but not adjacent
        "UNKNOWN",  # invalid source
        "TACTILE_SUPPORTED_HYPOTHESIS",
        "NOT_TACTILE_SUPPORTED",  # aligned and adjacent, but inactive
    ]
    assert np.isclose(result.tactile_time_offset_ms[2, 0, 0], 20.0)


def test_misaligned_tactile_stays_unknown() -> None:
    inputs = direct_inputs()
    result = estimate_interaction_evidence_v0a(
        **inputs,
        adjacency_threshold_px=1.0,
        tactile_active=np.ones((4, 1), dtype=bool),
        tactile_source_valid=np.ones((4, 1), dtype=bool),
        tactile_timestamps_s=np.asarray([[0.1], [0.2], [0.3], [0.4]]),
        tactile_max_offset_s=0.040,
    )
    assert set(result.tactile_status[:, 0, 0].tolist()) == {"UNKNOWN"}


def test_structural_inconsistency_fails_closed() -> None:
    inputs = direct_inputs()
    inputs["object_mask_state"][0, 0] = "unknown"
    with pytest.raises(InteractionEvidenceError, match="cannot carry mask pixels"):
        estimate_interaction_evidence_v0a(**inputs)

    inputs = direct_inputs()
    inputs["fingertip_observation_state"][0, 0] = "observed_or_inferred"
    with pytest.raises(InteractionEvidenceError, match="unsupported states"):
        estimate_interaction_evidence_v0a(**inputs)

    inputs = direct_inputs()
    with pytest.raises(InteractionEvidenceError, match="supplied together"):
        estimate_interaction_evidence_v0a(
            **inputs, tactile_active=np.ones((4, 1), dtype=bool)
        )


def test_schema_accepts_record_and_hard_codes_non_authority() -> None:
    schema = json.loads(
        (PROJECT / "contracts" / "interaction_evidence_v0a.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    result = estimate_interaction_evidence_v0a(**direct_inputs())
    record = interaction_evidence_v0a_to_record(
        result, object_ids=["card_00"], fingertip_ids=["right_index"]
    )
    jsonschema.Draft202012Validator(schema).validate(record)
    assert record["coordinate_domain"] == "IMAGE_2D_ONLY"
    assert record["authority"] == "DEVELOPMENT_WEAK_EVIDENCE_ONLY"
    assert not any(record["prohibited_claims"].values())
    assert "relative_z" not in record["observations"][0]
    assert "occlusion" not in record["observations"][0]


def test_schema_rejects_contact_or_robot_authority_escalation() -> None:
    schema = json.loads(
        (PROJECT / "contracts" / "interaction_evidence_v0a.schema.json").read_text()
    )
    record = interaction_evidence_v0a_to_record(
        estimate_interaction_evidence_v0a(**direct_inputs()),
        object_ids=["card_00"],
        fingertip_ids=["right_index"],
    )
    record["prohibited_claims"]["contact_ground_truth"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(record)


def test_schema_enforces_unknown_propagation_for_unobserved_inputs() -> None:
    schema = json.loads(
        (PROJECT / "contracts" / "interaction_evidence_v0a.schema.json").read_text()
    )
    inputs = direct_inputs()
    inputs["fingertip_observation_state"][0, 0] = "inferred"
    record = interaction_evidence_v0a_to_record(
        estimate_interaction_evidence_v0a(**inputs),
        object_ids=["card_00"],
        fingertip_ids=["right_index"],
    )
    jsonschema.Draft202012Validator(schema).validate(record)
    record["observations"][0]["adjacency"] = "ADJACENT_2D"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(record)
