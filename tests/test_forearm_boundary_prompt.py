from __future__ import annotations

import numpy as np

from pipeline.forearm_boundary_prompt import (
    final_forearm_prompt_support,
    propose_boundary_reprompt,
    select_boundary_reprompt_candidate,
)


def _proposal(component: np.ndarray, outward: tuple[float, float] = (-1.0, 0.0)):
    return propose_boundary_reprompt(
        component,
        wrist_width=10.0,
        outward_prior=np.asarray(outward, np.float32),
        max_gap_wrist_ratio=0.4,
        min_edge_ambiguity_ratio=2.0,
        min_edge_direction_cosine=0.5,
    )


def _selector(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    initial: np.ndarray,
    object_core_exclusion: np.ndarray | None = None,
    uncertain_contact_exclusion: np.ndarray | None = None,
    local_support: np.ndarray | None = None,
    min_boundary_width_ratio: float = 0.5,
    max_boundary_width_ratio: float = 1.5,
    max_expansion_ratio: float = 0.5,
):
    shape = initial.shape
    return select_boundary_reprompt_candidate(
        masks,
        scores,
        positive_points=np.asarray([[20, 30], [1, 30]], np.float32),
        negative_points=np.asarray([[1, 10], [1, 54]], np.float32),
        initial_component=initial,
        target_edge="left",
        object_core_exclusion=(
            np.zeros(shape, dtype=bool)
            if object_core_exclusion is None
            else object_core_exclusion
        ),
        uncertain_contact_exclusion=(
            np.zeros(shape, dtype=bool)
            if uncertain_contact_exclusion is None
            else uncertain_contact_exclusion
        ),
        local_added_support=(
            np.ones(shape, dtype=bool) if local_support is None else local_support
        ),
        observed_terminal_width_px=24.0,
        min_initial_recall=0.8,
        max_expansion_ratio=max_expansion_ratio,
        min_boundary_width_ratio=min_boundary_width_ratio,
        max_boundary_width_ratio=max_boundary_width_ratio,
    )


def test_proposal_uses_unambiguous_wrist_normalized_nearest_edge() -> None:
    component = np.zeros((96, 128), dtype=bool)
    component[38:58, 3:60] = True
    proposal = _proposal(component)
    assert proposal["status"] == "REPROMPT_ELIGIBLE"
    assert proposal["edge"] == "left"
    assert proposal["gap_wrist_ratio"] == 0.3
    assert proposal["edge_ambiguity_ratio"] > 2.0
    assert proposal["pca_edge_cosine"] > 0.9
    assert proposal["outward_prior_edge_cosine"] > 0.9
    assert proposal["bridge_pixels"] == 0.0


def test_proposal_holds_ambiguous_corner_without_frame_fallback() -> None:
    component = np.zeros((96, 128), dtype=bool)
    component[4:24, 3:60] = True
    proposal = _proposal(component)
    assert proposal["status"] == "HOLD_EDGE_AMBIGUOUS"


def test_proposal_holds_nearest_edge_incompatible_with_outward_prior() -> None:
    component = np.zeros((96, 128), dtype=bool)
    component[38:58, 3:60] = True
    proposal = _proposal(component, outward=(0.0, 1.0))
    assert proposal["status"] == "HOLD_EDGE_DIRECTION_INCOMPATIBLE"
    assert proposal["pca_edge_cosine"] > 0.9
    assert proposal["outward_prior_edge_cosine"] < 0.5


def test_reprompt_accepts_only_real_sam_component_with_prompt_and_object_gates() -> None:
    shape = (64, 64)
    initial = np.zeros(shape, dtype=bool)
    initial[18:46, 4:34] = True
    good = np.zeros(shape, dtype=bool)
    good[16:48, :34] = True
    table_leak = good.copy()
    table_leak[48:, :] = True

    selected, metrics = _selector(
        np.stack([table_leak, good]),
        np.asarray([0.99, 0.90], np.float32),
        initial=initial,
    )

    assert selected[:, 0].any()
    assert not selected[54, 1]
    assert metrics["held"] == 0.0
    assert metrics["negative_leak"] == 0.0
    assert metrics["target_edge_reached"] == 1.0
    assert 0.5 <= metrics["boundary_width_ratio"] <= 1.5


def test_reprompt_rejects_candidate_that_overlaps_object_before_protection() -> None:
    shape = (64, 64)
    initial = np.zeros(shape, dtype=bool)
    initial[18:46, 4:34] = True
    candidate = np.zeros(shape, dtype=bool)
    candidate[16:48, :34] = True
    object_exclusion = np.zeros(shape, dtype=bool)
    object_exclusion[28:34, 8:14] = True
    selected, metrics = _selector(
        candidate[None],
        np.asarray([0.95], np.float32),
        initial=initial,
        object_core_exclusion=object_exclusion,
    )
    assert not selected.any()
    assert metrics["held"] == 1.0
    assert metrics["object_core_overlap_pixels"] > 0


def test_reprompt_rejects_contact_band_even_with_zero_object_core_overlap() -> None:
    shape = (64, 64)
    initial = np.zeros(shape, dtype=bool)
    initial[18:46, 4:34] = True
    candidate = np.zeros(shape, dtype=bool)
    candidate[16:48, :34] = True
    uncertain = np.zeros(shape, dtype=bool)
    uncertain[28:34, 8:14] = True
    selected, metrics = _selector(
        candidate[None],
        np.asarray([0.95], np.float32),
        initial=initial,
        uncertain_contact_exclusion=uncertain,
    )
    assert not selected.any()
    assert metrics["object_core_overlap_pixels"] == 0
    assert metrics["uncertain_contact_overlap_pixels"] > 0
    assert metrics["held"] == 1.0


def test_connected_table_blob_outside_local_boundary_support_is_held() -> None:
    shape = (64, 64)
    initial = np.zeros(shape, dtype=bool)
    initial[18:46, 4:34] = True
    candidate = np.zeros(shape, dtype=bool)
    candidate[16:48, :34] = True
    candidate[30:40, 34:64] = True
    local_support = np.zeros(shape, dtype=bool)
    local_support[14:50, :35] = True
    selected, metrics = _selector(
        candidate[None],
        np.asarray([0.99], np.float32),
        initial=initial,
        local_support=local_support,
        max_expansion_ratio=2.0,
    )
    assert not selected.any()
    assert metrics["negative_leak"] == 0.0
    assert metrics["outside_local_support_pixels"] > 0
    assert metrics["held"] == 1.0


def test_one_pixel_boundary_filament_fails_normalized_width_lower_bound() -> None:
    shape = (64, 64)
    initial = np.zeros(shape, dtype=bool)
    initial[18:46, 4:34] = True
    candidate = initial.copy()
    candidate[30, :4] = True
    selected, metrics = _selector(
        candidate[None],
        np.asarray([0.99], np.float32),
        initial=initial,
    )
    assert not selected.any()
    assert metrics["target_edge_reached"] == 1.0
    assert metrics["boundary_width_ratio"] < 0.5
    assert metrics["held"] == 1.0


def test_boundary_connected_mask_missing_half_sleeve_prompts_is_held() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, :20] = True
    sleeve_points = np.asarray(
        [[1, 30], [8, 30], [16, 30], [24, 30], [32, 30], [40, 30]],
        np.float32,
    )
    metrics = final_forearm_prompt_support(
        mask, sleeve_points, min_prompt_recall=0.60
    )
    assert metrics["final_boundary_connected"] == 1.0
    assert metrics["final_forearm_prompt_recall"] == 0.5
    assert metrics["final_forearm_supported"] == 0.0
