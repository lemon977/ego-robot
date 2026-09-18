from __future__ import annotations

from typing import Any

from chaoyang.ops.run_0915_human_stereo_alignment_bound_audit_v1 import (
    audit_alignment_bounds,
)


def _rows(scale: float, offset: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for frame in range(100):
        for hand_index, hand in enumerate(("left", "right")):
            source = 0.25 + 0.002 * frame + 0.0005 * hand_index
            result.append({
                "frame_id": frame,
                "hand_id": hand,
                "whole_frame_hand_contact_excluded": False,
                "support_count": 40,
                "mano_surface_depth_m": source,
                "stereo_surface_depth_m": scale * source + offset,
            })
    return result


def _prior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Generate the exact prior representation from the audit's frozen bounded path.
    # Importing private helpers in a test is deliberate: this fixture validates that
    # prior-result reproduction and saturation are independent decisions.
    from chaoyang.ops import run_0915_human_stereo_alignment_bound_audit_v1 as module

    admitted = module._admitted_rows(rows)
    train = [row for row in admitted if int(row["frame_id"]) % 5 != 0]
    holdout = [row for row in admitted if int(row["frame_id"]) % 5 == 0]
    fit = module._robust_fit(train, bounded=True)
    metrics = module._heldout_metrics(holdout, fit)
    return {
        "fit": {key: fit[key] for key in ("scale", "offset_m", "retained_train_rows")},
        "heldout": metrics,
    }


def test_clipped_scale_cannot_authorize_metric_translation() -> None:
    rows = _rows(0.75, 0.01)
    audit = audit_alignment_bounds(rows, _prior(rows))
    assert audit["bounded_fit"]["scale"] == 0.8
    assert audit["unconstrained_fit"]["scale"] < 0.8
    assert audit["bounded_scale_at_frozen_edge"] is True
    assert audit["bound_saturation_detected"] is True
    assert audit["status"] == "REJECTED_BOUNDED_FIT_SATURATION"
    assert audit["metric_translation_authorized"] is False
    assert audit["contact_or_object_fit_used"] is False
    assert audit["fixed_48mm_bias_used"] is False


def test_interior_optimum_can_pass_when_heldout_is_consistent() -> None:
    rows = _rows(0.95, 0.012)
    audit = audit_alignment_bounds(rows, _prior(rows))
    assert 0.8 < audit["unconstrained_fit"]["scale"] < 1.2
    assert audit["bounded_scale_at_frozen_edge"] is False
    assert audit["bound_saturation_detected"] is False
    assert audit["prior_bounded_result_reproduced"] is True
    assert audit["status"] == "PASS_DEVELOPMENT_ALIGNMENT"
    assert audit["metric_translation_authorized"] is True


def test_holdout_split_and_row_identity_remain_frozen() -> None:
    rows = _rows(0.95, 0.012)
    audit = audit_alignment_bounds(rows, _prior(rows))
    assert audit["train_row_count"] == 160
    assert audit["holdout_row_count"] == 40
    assert audit["bounded_holdout"]["frame_ids"] == list(range(0, 100, 5))
    assert audit["bounded_holdout"]["row_ids_sha256"] == (
        audit["unconstrained_holdout"]["row_ids_sha256"]
    )
