from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from chaoyang.pipeline.hand_observability_v1 import (
    HAWOR_SENTINEL,
    PARTS,
    HandObservabilityError,
    audit_provider_independence,
    evaluate_hand_observability,
    mano21_part_presence,
)
from chaoyang.pipeline.hawor_bounded_comparison_v31 import compare_bounded_hawor
from chaoyang.pipeline.kai22_r0_tiered_admission_v1 import evaluate_kai22_r0_tiers
from chaoyang.pipeline.temporal_authority_v1 import (
    audit_suffix_invariance,
    validate_online_current_inputs,
)


ROOT = Path(__file__).resolve().parents[1]


def _validate(schema_name: str, value: dict) -> None:
    schema = json.loads((ROOT / "contracts" / schema_name).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)


def _providers() -> list[dict]:
    return [
        {
            "provider_id": "frozen_rgb_review",
            "family": "FROZEN_RGB_REVIEW",
            "dependencies": [],
            "independent_from_hawor": True,
        },
        {
            "provider_id": "hawor_prompted_sam",
            "family": "SAM31",
            "dependencies": [HAWOR_SENTINEL],
            "independent_from_hawor": False,
        },
        {
            "provider_id": "derived_contour",
            "family": "CONTOUR_QA",
            "dependencies": ["hawor_prompted_sam"],
            "independent_from_hawor": True,
        },
    ]


def test_observability_keeps_visible_missing_hawor_in_failure_denominator() -> None:
    frame_count = 3
    present = np.zeros((frame_count, 2, len(PARTS)), dtype=bool)
    quality = np.zeros_like(present)
    present[2, 0, PARTS.index("palm")] = True
    observations = [
        {
            "frame_id": 0,
            "side": "left",
            "part": "wrist",
            "state": "FULLY_VISIBLE",
            "evidence_provider_ids": ["frozen_rgb_review"],
        },
        {
            "frame_id": 1,
            "side": "left",
            "part": "wrist",
            "state": "FULLY_VISIBLE",
            "evidence_provider_ids": ["derived_contour"],
        },
        {
            "frame_id": 2,
            "side": "left",
            "part": "palm",
            "state": "PARTIALLY_VISIBLE_EVALUABLE",
            "evidence_provider_ids": ["frozen_rgb_review"],
        },
    ]
    result = evaluate_hand_observability(
        frame_count=frame_count,
        observations=observations,
        providers=_providers(),
        hawor_part_output_present=present,
        pose_part_quality_pass=quality,
    )
    wrist = result["per_side_part"]["left"]["wrist"]
    palm = result["per_side_part"]["left"]["palm"]
    assert wrist["observable_total"] == 1
    assert wrist["visible_but_no_hawor_output"] == 1
    assert wrist["observable_pass"] == 0
    assert palm["observable_quality_failure"] == 1
    assert len(result["dependency_demotions"]) == 1
    assert result["dependency_demotions"][0]["frame_id"] == 1
    _validate("hand_observability_v1.schema.json", result)


def test_observability_partial_parts_are_scored_without_promoting_hidden_parts() -> None:
    present = np.zeros((1, 2, len(PARTS)), dtype=bool)
    quality = np.zeros_like(present)
    index = PARTS.index("index")
    present[0, 0, index] = quality[0, 0, index] = True
    result = evaluate_hand_observability(
        frame_count=1,
        observations=[
            {
                "frame_id": 0,
                "side": "left",
                "part": "index",
                "state": "PARTIALLY_VISIBLE_EVALUABLE",
                "evidence_provider_ids": ["frozen_rgb_review"],
            },
            {
                "frame_id": 0,
                "side": "left",
                "part": "middle",
                "state": "OCCLUDED",
                "evidence_provider_ids": ["frozen_rgb_review"],
            },
        ],
        providers=_providers(),
        hawor_part_output_present=present,
        pose_part_quality_pass=quality,
    )
    assert result["per_side_part"]["left"]["index"]["observable_pass"] == 1
    assert result["per_side_part"]["left"]["middle"]["observable_total"] == 0
    assert result["aggregate"]["timeline_pass_rate"] == pytest.approx(1 / 14)


def test_observability_dependency_cycle_is_rejected() -> None:
    providers = [
        {"provider_id": "a", "family": "A", "dependencies": ["b"], "independent_from_hawor": True},
        {"provider_id": "b", "family": "B", "dependencies": ["a"], "independent_from_hawor": True},
    ]
    with pytest.raises(HandObservabilityError, match="cycle"):
        audit_provider_independence(providers)


def test_mano_part_presence_requires_every_joint_in_the_part() -> None:
    joints = np.ones((2, 2, 21), dtype=bool)
    joints[0, 0, 7] = False
    parts = mano21_part_presence(joints)
    assert parts.shape == (2, 2, len(PARTS))
    assert not parts[0, 0, PARTS.index("index")]
    assert parts[0, 0, PARTS.index("middle")]


def test_suffix_invariance_covers_pixels_floats_discrete_and_inferred() -> None:
    full = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "crop": np.asarray([0, 0, 8, 8], dtype=np.int64),
        "robotized_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "state": "OBSERVED",
        "confidence": np.asarray([0.8], dtype=np.float64),
        "offline_track": np.asarray([1.0], dtype=np.float64),
    }
    truncated = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in full.items()}
    truncated["robotized_rgb"][2, 3, 0] = 1
    result = audit_suffix_invariance(
        full_current_inputs=full,
        truncated_current_inputs=truncated,
        declared_authority={"offline_track": "INFERRED"},
        required_fields=("rgb", "crop", "robotized_rgb", "state", "confidence"),
    )
    assert result["fields"]["rgb"]["authority"] == "CAUSAL_CURRENT"
    assert result["fields"]["confidence"]["authority"] == "CAUSAL_CURRENT"
    assert result["fields"]["robotized_rgb"]["authority"] == "OFFLINE_NONCAUSAL"
    assert result["fields"]["offline_track"]["authority"] == "INFERRED"
    assert result["status"] == "REJECTED_NONCAUSAL_CURRENT_INPUT"
    assert validate_online_current_inputs(result, ("rgb", "robotized_rgb")) == [
        "robotized_rgb:OFFLINE_NONCAUSAL"
    ]
    _validate("temporal_authority_audit_v1.schema.json", result)


def test_suffix_invariance_missing_required_field_is_unknown_not_causal() -> None:
    result = audit_suffix_invariance(
        full_current_inputs={"rgb": np.zeros((1,), np.uint8)},
        truncated_current_inputs={},
        required_fields=("rgb",),
    )
    assert result["fields"]["rgb"]["authority"] == "UNKNOWN_TEMPORAL_AUTHORITY"
    assert result["required_field_failures"] == ["rgb"]


def _hand_track(frames: int, noise_scale: float) -> np.ndarray:
    template = np.zeros((21, 3), dtype=np.float64)
    template[:, 0] = np.arange(21) * 0.002
    time = np.arange(frames, dtype=np.float64)
    smooth = 0.0002 * time
    jitter = noise_scale * ((-1.0) ** time)
    result = np.broadcast_to(template, (frames, 21, 3)).copy()
    result[:, :, 1] += (smooth + jitter)[:, None]
    return result


def _bounded_args(frames: int = 30) -> dict:
    raw = _hand_track(frames, 0.002)
    candidate = _hand_track(frames, 0.0002)
    return {
        "raw_joints": raw,
        "candidate_joints": candidate,
        "raw_valid": np.ones(frames, bool),
        "candidate_valid": np.ones(frames, bool),
        "frozen_observable_mask": np.ones(frames, bool),
        "timestamps_s": np.arange(frames, dtype=np.float64) / 30.0,
        "frame_ids": np.arange(frames, dtype=np.int64),
        "raw_reprojection_error_px": np.full((frames, 21), 2.0),
        "candidate_reprojection_error_px": np.full((frames, 21), 1.5),
        "raw_observed": np.ones(frames, bool),
        "candidate_observed": np.ones(frames, bool),
    }


def test_bounded_comparison_adopts_on_same_frozen_set_with_real_timestamps() -> None:
    result = compare_bounded_hawor(**_bounded_args())
    assert result["status"] == "ADOPT"
    assert result["same_frozen_metric_set"] is True
    assert result["raw"]["wrist_step_p95_mm"] > result["candidate"]["wrist_step_p95_mm"]
    assert result["raw"]["all_joint_acceleration_p95_m_per_s2"] > result["candidate"]["all_joint_acceleration_p95_m_per_s2"]
    _validate("hawor_bounded_comparison_v31.schema.json", result)


def test_bounded_comparison_rejects_deleted_and_invented_observed_frames() -> None:
    arguments = _bounded_args()
    arguments["candidate_valid"][5] = False
    arguments["raw_observed"][7] = False
    result = compare_bounded_hawor(**arguments)
    assert result["status"] == "REJECT"
    assert result["dropped_frame_ids"] == [5]
    assert result["invented_observed_frame_ids"] == [7]
    assert "CANDIDATE_DROPPED_FROZEN_EVALUATION_FRAMES" in result["reason_codes"]
    assert "CANDIDATE_INVENTED_OBSERVED_FRAMES" in result["reason_codes"]


def test_bounded_comparison_does_not_bridge_frame_gap() -> None:
    arguments = _bounded_args(20)
    arguments["raw_joints"][:10, :, 0] -= 1.0
    arguments["candidate_joints"][:10, :, 0] -= 1.0
    arguments["frozen_observable_mask"][9:11] = False
    result = compare_bounded_hawor(**arguments)
    assert result["raw"]["wrist_step_p95_mm"] < 10.0


def _causal_audit() -> dict:
    current = {
        "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "crop": np.asarray([0, 0, 4, 4]),
        "state": np.asarray([1.0]),
    }
    return audit_suffix_invariance(
        full_current_inputs=current,
        truncated_current_inputs={key: value.copy() for key, value in current.items()},
        required_fields=tuple(current),
    )


def test_kai22_tiers_allow_h50_only_with_causal_51_frame_window() -> None:
    frames = 60
    result = evaluate_kai22_r0_tiers(
        frame_ids=np.arange(frames),
        q22_valid=np.ones(frames, bool),
        fk_finite=np.ones(frames, bool),
        joint_semantics_pass=True,
        consumer_local_window_mask=np.ones(frames, bool),
        consumer_local_window_quality_pass=True,
        temporal_authority_audit=_causal_audit(),
        h50_current_input_fields=("rgb", "crop", "state"),
        timestamps_valid=True,
    )
    assert result["highest_admitted_level"] == "H50_READY"
    assert result["training_eligible"] is True
    assert result["contact_required"] is False
    assert result["physical_deployment_authorized"] is False
    _validate("kai22_r0_tiered_admission_v1.schema.json", result)


def test_kai22_noncausal_input_blocks_h50_but_preserves_development_r0() -> None:
    audit = _causal_audit()
    audit["fields"]["state"]["authority"] = "OFFLINE_NONCAUSAL"
    audit["fields"]["state"]["online_current_admitted"] = False
    frames = 60
    result = evaluate_kai22_r0_tiers(
        frame_ids=np.arange(frames),
        q22_valid=np.ones(frames, bool),
        fk_finite=np.ones(frames, bool),
        joint_semantics_pass=True,
        consumer_local_window_mask=np.ones(frames, bool),
        consumer_local_window_quality_pass=True,
        temporal_authority_audit=audit,
        h50_current_input_fields=("rgb", "crop", "state"),
        timestamps_valid=True,
    )
    assert result["highest_admitted_level"] == "DEVELOPMENT_R0"
    assert result["levels"]["DEVELOPMENT_R0"]["status"] == "PASS"
    assert result["levels"]["H50_READY"]["status"] == "BLOCKED"
    assert result["training_eligible"] is False


def test_kai22_without_local_window_preserves_kinematic_only() -> None:
    frames = 10
    result = evaluate_kai22_r0_tiers(
        frame_ids=np.arange(frames),
        q22_valid=np.ones(frames, bool),
        fk_finite=np.ones(frames, bool),
        joint_semantics_pass=True,
        consumer_local_window_mask=np.zeros(frames, bool),
        consumer_local_window_quality_pass=False,
        temporal_authority_audit=_causal_audit(),
        h50_current_input_fields=("rgb", "crop", "state"),
        timestamps_valid=True,
    )
    assert result["highest_admitted_level"] == "KINEMATIC_ONLY"
    assert result["levels"]["KINEMATIC_ONLY"]["status"] == "PASS"
    assert result["levels"]["DEVELOPMENT_R0"]["status"] == "BLOCKED"
