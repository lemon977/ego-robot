from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.sam21_route_b_contract import (
    DECODER_PREFIX,
    DEVELOPMENT_FRAMES,
    EXPECTED_DECODER_PARAMETERS,
    HELD_OUT_FOLDS,
    RouteBContractError,
    U_SEMANTICS,
    asymmetric_houb_loss,
    fixed_oof_plan,
    freeze_except_decoder,
    make_supervision,
    require_development_split,
    validate_candidate_governance,
    validate_u_semantics,
    verify_gradient_boundary,
    verify_weight_delta_boundary,
)


class FakeParameter:
    def __init__(self, numel, *, grad=None):
        self._numel = numel
        self.requires_grad = True
        self.grad = grad

    def numel(self):
        return self._numel

    def requires_grad_(self, value):
        self.requires_grad = bool(value)
        return self


def parameters(*, escaped_grad=False, count=EXPECTED_DECODER_PARAMETERS):
    return [
        (f"{DECODER_PREFIX}weight", FakeParameter(count, grad=object())),
        ("image_encoder.weight", FakeParameter(10, grad=object() if escaped_grad else None)),
    ]


def candidate_record():
    return {
        "AUTH_TIER": "T1",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "labels_read_splits": ["development"],
        "u_semantics": U_SEMANTICS,
        "u_metric_variants_required": ["u_as_h", "u_excluded"],
        "u_gap_review_flag_required": True,
    }


def test_time_spread_folds_are_exact_partition():
    plan = fixed_oof_plan()
    assert tuple(x.held_out_frames for x in plan) == tuple(HELD_OUT_FOLDS.values())
    assert all(len(x.train_frames) == 10 and len(x.held_out_frames) == 5 for x in plan)
    assert sorted(frame for x in plan for frame in x.held_out_frames) == list(DEVELOPMENT_FRAMES)


def test_frame_identity_drift_is_rejected():
    with pytest.raises(RouteBContractError, match="identity drift"):
        fixed_oof_plan(DEVELOPMENT_FRAMES[:-1])


def test_blind_supervised_reads_are_rejected():
    require_development_split("development")
    for split in ("same_session_blind", "cross_session_blind", "test"):
        with pytest.raises(RouteBContractError, match="sealed"):
            require_development_split(split)


def test_houb_mapping_keeps_u_as_separate_directional_evidence():
    h = np.array([[1, 0], [0, 0]], dtype=bool)
    o = np.array([[0, 1], [0, 0]], dtype=bool)
    u = np.array([[0, 0], [1, 0]], dtype=bool)
    b = np.array([[0, 0], [0, 1]], dtype=bool)
    supervision = make_supervision(h, o, u, b)
    np.testing.assert_array_equal(supervision.human_positive, h)
    np.testing.assert_array_equal(
        supervision.object_or_background_negative, np.logical_or(o, b)
    )
    np.testing.assert_array_equal(supervision.uncertain_false_negative_only, u)
    assert not np.any(
        supervision.human_positive & supervision.uncertain_false_negative_only
    )


def test_u_prediction_on_h_side_has_zero_penalty_and_zero_gradient():
    z = np.zeros((1,), dtype=bool)
    u = np.ones((1,), dtype=bool)
    supervision = make_supervision(z, z, u, z)
    logits = torch.tensor([0.75], requires_grad=True)
    loss = asymmetric_houb_loss(logits, supervision)
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None and logits.grad.item() == 0.0


def test_u_prediction_on_background_side_is_penalized_with_corrective_gradient():
    z = np.zeros((1,), dtype=bool)
    u = np.ones((1,), dtype=bool)
    supervision = make_supervision(z, z, u, z)
    logits = torch.tensor([-0.75], requires_grad=True)
    loss = asymmetric_houb_loss(logits, supervision)
    loss.backward()
    assert loss.item() == pytest.approx(0.75)
    assert logits.grad is not None and logits.grad.item() < 0.0


@pytest.mark.parametrize("policy", ["ZERO_WEIGHT", "MERGE_U_INTO_H_HARD_POSITIVE"])
def test_zero_weight_and_hard_positive_u_policies_are_rejected(policy):
    with pytest.raises(RouteBContractError, match="forbidden U-band"):
        validate_u_semantics(policy)
    validate_u_semantics(U_SEMANTICS)


@pytest.mark.parametrize(
    "arrays",
    [
        # H/O overlap.
        (np.ones((1, 1)), np.ones((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))),
        # Unassigned pixel.
        (np.zeros((1, 1)),) * 4,
    ],
)
def test_non_one_hot_houb_is_rejected(arrays):
    with pytest.raises(RouteBContractError, match="exhaustive"):
        make_supervision(*arrays)


def test_exact_decoder_only_freeze_and_gradient_boundary():
    items = parameters()
    names = freeze_except_decoder(items)
    assert names == (f"{DECODER_PREFIX}weight",)
    assert items[0][1].requires_grad is True
    assert items[1][1].requires_grad is False
    verify_gradient_boundary(items)


def test_decoder_parameter_count_drift_is_rejected():
    with pytest.raises(RouteBContractError, match="count drift"):
        freeze_except_decoder(parameters(count=EXPECTED_DECODER_PARAMETERS - 1))


def test_gradient_escape_is_rejected():
    items = parameters(escaped_grad=True)
    freeze_except_decoder(items)
    with pytest.raises(RouteBContractError, match="escaped"):
        verify_gradient_boundary(items)


def test_only_decoder_weight_delta_is_allowed():
    before = {f"{DECODER_PREFIX}weight": "a", "image_encoder.weight": "b"}
    after = {f"{DECODER_PREFIX}weight": "c", "image_encoder.weight": "b"}
    assert verify_weight_delta_boundary(before, after) == (f"{DECODER_PREFIX}weight",)
    after["image_encoder.weight"] = "d"
    with pytest.raises(RouteBContractError, match="frozen weight"):
        verify_weight_delta_boundary(before, after)


def test_no_weight_delta_is_rejected():
    state = {f"{DECODER_PREFIX}weight": "a", "image_encoder.weight": "b"}
    with pytest.raises(RouteBContractError, match="changed no parameter"):
        verify_weight_delta_boundary(state, dict(state))


def test_candidate_review_block_is_mandatory():
    record = candidate_record()
    validate_candidate_governance(record)
    record["next_bucket_blocked"] = False
    with pytest.raises(RouteBContractError, match="governance drift"):
        validate_candidate_governance(record)


@pytest.mark.parametrize(
    "key",
    ["u_semantics", "u_metric_variants_required", "u_gap_review_flag_required"],
)
def test_candidate_manifest_requires_asymmetric_u_and_both_metric_variants(key):
    record = candidate_record()
    del record[key]
    with pytest.raises(RouteBContractError, match="governance drift"):
        validate_candidate_governance(record)


def test_session_or_frame_override_is_rejected():
    record = candidate_record()
    record["session_overrides"] = {"grap_a_cap_004": {}}
    with pytest.raises(RouteBContractError, match="forbidden override"):
        validate_candidate_governance(record)
