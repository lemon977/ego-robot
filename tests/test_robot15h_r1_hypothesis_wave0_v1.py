from __future__ import annotations

import inspect
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from chaoyang.ops import run_0915_robot15h_r1_hypothesis_wave0_v1 as subject


def _contact_documents(
    *, include_hypotheses: bool = True, include_controls: bool = True,
) -> tuple[dict, dict, dict]:
    evidence = {
        "schema_version": "0915-robot15h-contact-evidence-ledger-v1",
        "task_id": subject.CONTACT_TASK_ID,
        "window_run_id": subject.WINDOW_RUN_ID,
        "strict_metric_contact_authorized": False,
        "r1_e_authorized": False,
        "training_eligible": False,
    }
    hypotheses = []
    if include_hypotheses:
        hypotheses = [
            {
                "session_id": subject.HYPOTHESIS_SESSION,
                "frame_id": frame,
                "hand_id": "right",
                "finger_id": "index",
                "object_id": "playing_card_02",
                "status": "HYPOTHESIS_ONLY",
                "metric_contact": False,
                "strict_contact_admitted": False,
                "training_eligible": False,
                "contact_ground_truth": False,
            }
            for frame in subject.CORE_FRAMES
        ]
    controls = []
    if include_controls:
        controls = [
            {
                "session_id": subject.HYPOTHESIS_SESSION,
                "frame_id": 25,
                "hand_id": "right",
                "finger_id": "index",
                "object_id": "playing_card_02",
                "status": "NO_CONTACT",
                "metric_contact": False,
                "strict_contact_admitted": False,
                "training_eligible": False,
                "contact_ground_truth": False,
            }
        ]
    hypothesis = {
        "schema_version": "0915-robot15h-contact-hypothesis-ledger-v1",
        "task_id": subject.CONTACT_TASK_ID,
        "window_run_id": subject.WINDOW_RUN_ID,
        "strict_contact_authorized": False,
        "r1_e_authorized": False,
        "training_eligible": False,
        "contact_ground_truth": False,
        "hypotheses": hypotheses,
        "no_contact_controls": controls,
        "verified_hypothesis_count": len(hypotheses),
        "no_contact_control_count": len(controls),
    }
    batch = {
        "schema_version": "0915-robot15h-contact-wave0-batch-v1",
        "task_id": subject.CONTACT_TASK_ID,
        "window_run_id": subject.WINDOW_RUN_ID,
        "status": "COMPLETED_ALL_TERMINAL",
        "strict_metric_contact_authorized": False,
        "r1_e_authorized": False,
        "sessions": [{"session_id": session_id} for session_id in subject.EXPECTED_SESSIONS],
        "counts": {"r1_e_windows": 0},
    }
    return evidence, hypothesis, batch


def _arrays(frame_count: int = 120) -> dict[str, np.ndarray]:
    # Deterministic high-frequency baseline leaves a strict non-regression margin
    # for the bounded, low-frequency hypothesis taper.
    q = np.random.default_rng(0).uniform(-0.5, 0.5, (frame_count, 2, 22))
    wrist = np.repeat(np.eye(4, dtype=np.float64)[None, None], frame_count * 2, axis=0)
    wrist = wrist.reshape(frame_count, 2, 4, 4)
    valid = np.ones((frame_count, 2), bool)
    timestamps = np.arange(frame_count, dtype=np.float64) / 30.0
    return {
        "q22_init": q,
        "relative_wrist_T": wrist,
        "valid_side_frame": valid,
        "timestamps_s": timestamps,
        "frame_id": np.arange(frame_count, dtype=np.int64),
    }


def _limits() -> tuple[np.ndarray, np.ndarray]:
    return np.full(22, -1.0), np.full(22, 1.0)


def test_formal_contact_ledgers_are_the_only_h1_eligibility_source() -> None:
    documents = _contact_documents()
    value = subject.validate_contact_ledgers(*documents)
    assert value["r1_e_authorized"] is False
    assert value["r1_e_windows"] == 0
    assert value["h1_counterfactual_eligible"] is True
    assert value["hypothesis_frame_ids"] == list(subject.CORE_FRAMES)


@pytest.mark.parametrize(
    ("hypotheses", "controls", "blocker"),
    [
        (False, True, "NO_EXACT_FORMAL_SEVEN_FRAME_HYPOTHESIS"),
        (True, False, "NO_FORMAL_NO_CONTACT_CONTROL"),
    ],
)
def test_h1_fails_closed_without_formal_hypothesis_or_control(
    hypotheses: bool, controls: bool, blocker: str,
) -> None:
    documents = _contact_documents(
        include_hypotheses=hypotheses, include_controls=controls,
    )
    value = subject.validate_contact_ledgers(*documents)
    assert value["h1_counterfactual_eligible"] is False
    assert value["h1_eligibility_blocker"] == blocker


def test_contact_ledger_cannot_authorize_r1_e() -> None:
    evidence, hypothesis, batch = _contact_documents()
    batch["counts"]["r1_e_windows"] = 1
    with pytest.raises(subject.R1HypothesisError, match="zero admitted windows"):
        subject.validate_contact_ledgers(evidence, hypothesis, batch)


def test_taper_is_bounded_by_real_timestamps_and_zero_at_edges() -> None:
    timestamps = np.arange(120, dtype=np.float64) / 30.0
    taper = subject.taper_envelope(timestamps)
    assert taper[86] == 0.0
    assert taper[92:99].tolist() == [1.0] * 7
    assert taper[104] == 0.0
    assert np.all(np.diff(taper[86:93]) >= 0)
    assert np.all(np.diff(taper[98:105]) <= 0)
    timestamps[92:] += 0.01
    with pytest.raises(subject.R1HypothesisError, match="exceeds 0.2"):
        subject.taper_envelope(timestamps)


def test_counterfactual_scope_wrist_freeze_taper_and_bit_exact_boundaries() -> None:
    arrays = _arrays()
    lower, upper = _limits()
    output, diagnostics = subject.build_counterfactual(
        arrays, eligible=True, lower_rad=lower, upper_rad=upper,
    )
    q0, q1 = output["q22_h0"], output["q22_h1_rejected"]
    changed = np.not_equal(q0, q1)
    allowed = np.zeros_like(changed)
    allowed[:, 0, 6:10] = True
    assert not changed[~allowed].any()
    assert changed[:, 1].sum() == 0
    assert changed[:, 0, :6].sum() == 0
    assert changed[:, 0, 10:].sum() == 0
    assert subject.arrays_bit_exact(
        output["relative_wrist_T_h0"], output["relative_wrist_T_h1_rejected"],
    )
    assert diagnostics["max_abs_delta_q_rad"] <= 0.12
    assert diagnostics["transition_zero_at_outer_boundaries"] is True
    assert diagnostics["evaluation_coverage_before"] == 7
    assert diagnostics["evaluation_coverage_after"] == 7


def test_ineligible_counterfactual_is_bit_exact_h0_everywhere() -> None:
    arrays = _arrays()
    lower, upper = _limits()
    output, diagnostics = subject.build_counterfactual(
        arrays, eligible=False, lower_rad=lower, upper_rad=upper,
    )
    assert subject.arrays_bit_exact(output["q22_h0"], output["q22_h1_rejected"])
    assert diagnostics["modified_side_frame_count"] == 0
    assert diagnostics["max_abs_delta_q_rad"] == 0.0


def test_missing_direct_observation_in_core_rejects_hypothesis() -> None:
    arrays = _arrays()
    arrays["valid_side_frame"][95, 0] = False
    arrays["q22_init"][95, 0] = np.nan
    arrays["relative_wrist_T"][95, 0] = np.nan
    lower, upper = _limits()
    with pytest.raises(subject.R1HypothesisError, match="core lacks direct-observed"):
        subject.build_counterfactual(
            arrays, eligible=True, lower_rad=lower, upper_rad=upper,
        )


def test_schema_rejects_adopted_or_control_authority() -> None:
    schema = json.loads(subject.CONTRACT.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    base = {
        "schema_version": "kai22-r1-hypothesis-session-v1",
        "task_id": subject.TASK_ID,
        "session_id": "play_cards_0915_119",
        "status": "COMPLETED_DUAL_TERMINAL",
        "r1_e": {
            "status": "BLOCKED_LOCAL_EVIDENCE", "attempted": False,
            "exported": False, "adopted": False,
            "first_blocker": "NO_STRICT_METRIC_CONTACT_WINDOW",
        },
        "r1_h": {
            "status": "REJECTED_COUNTERFACTUAL_NOT_ADOPTED",
            "attempted": True, "exported": True, "adopted": False,
            "first_blocker": "NO_OBJECT_INDEPENDENT_METRIC_WRIST_PLACEMENT",
            "optimization_semantics": "OPTIMIZATION_ONLY_NOT_ADOPTION_EVIDENCE",
            "pad_or_object_distance_used_as_adoption_evidence": False,
            "wrist_translation_orientation": "FROZEN_BIT_EXACT_R0",
        },
        "diagnostics": {
            "h0_r0_q22_bit_exact": True, "h0_r0_wrist_bit_exact": True,
            "wrist_h1_frozen_bit_exact": True,
            "outside_allowed_q22_scope_bit_exact": True,
            "invalid_rows_bit_exact": True, "max_abs_delta_q_rad": 0.09,
            "evaluation_coverage_not_reduced": True,
            "timestamp_motion_not_degraded": True,
        },
        "robot_self_collision": {
            "status": "PASS_HAND_ONLY_SELF_COLLISION",
            "all_modified_side_frames_checked": True,
            "adoption_authority": False,
        },
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
    }
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(base)
    invalid = {**base, "r1_h": {**base["r1_h"], "adopted": True}}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(invalid)
    invalid = {**base, "control_ground_truth": True}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(invalid)


def test_runner_forbids_self_certifying_object_distance_and_declares_outputs() -> None:
    source = Path(subject.__file__).read_text(encoding="utf-8")
    main_source = inspect.getsource(subject.main)
    for name in (
        "CLAIM.json", "RUN_SIGNATURE.json", "R1_ELIGIBILITY_LEDGER.json",
        "BATCH_RESULT.json", "METRICS.json", "RESULT.json", "RUN_RECEIPT.json",
    ):
        assert name in main_source
    assert '"r1_h_success": 0' in main_source
    assert '"r1_h_adopted": 0' in main_source
    assert "OPTIMIZATION_ONLY_NOT_ADOPTION_EVIDENCE" in source
    assert "NOT_COMPUTED_NOT_ADOPTION_EVIDENCE" in source
    assert "NO_OBJECT_INDEPENDENT_METRIC_WRIST_PLACEMENT" in source
    assert "archive/" not in source
    assert "/mnt/data/egodata" not in source


def test_summarize_separates_attempted_exported_and_adopted() -> None:
    rows = [
        {
            "status": "COMPLETED_DUAL_TERMINAL",
            "r1_e_status": "BLOCKED_LOCAL_EVIDENCE",
            "r1_e_attempted": False, "r1_e_exported": False, "r1_e_adopted": False,
            "r1_h_attempted": index == 0,
            "r1_h_exported": index == 0,
            "r1_h_adopted": False,
        }
        for index in range(4)
    ]
    assert subject.summarize(rows) == {
        "sessions_total": 4,
        "sessions_terminal": 4,
        "failed_runtime": 0,
        "r1_e_attempted": 0,
        "r1_e_exported": 0,
        "r1_e_adopted": 0,
        "r1_e_blocked_local_evidence": 4,
        "r1_h_attempted": 1,
        "r1_h_exported": 1,
        "r1_h_adopted": 0,
    }
