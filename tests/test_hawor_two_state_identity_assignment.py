from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from pipeline.hawor_two_state_identity_assignment import (
    IdentityAssignmentError,
    assignment_authority_arrays,
    assign_two_state_identity,
    assigned_joints,
)


def joints_from_wrists(wrists: np.ndarray) -> np.ndarray:
    """Expand (2, F, 2) wrists into finite synthetic 21-joint hands."""

    offsets = np.stack([np.linspace(0.0, 20.0, 21), np.linspace(0.0, 10.0, 21)], axis=1)
    return wrists[:, :, None, :] + offsets[None, None, :, :]


def inputs(
    raw_wrists: np.ndarray,
    optimized_wrists: np.ndarray,
) -> dict[str, np.ndarray]:
    frame_count = int(raw_wrists.shape[1])
    return {
        "raw_joints_2d": joints_from_wrists(raw_wrists),
        "raw_observed": np.ones((2, frame_count), dtype=bool),
        "raw_final_valid": np.ones((2, frame_count), dtype=bool),
        "optimized_joints_2d": joints_from_wrists(optimized_wrists),
        "optimized_valid": np.ones((2, frame_count), dtype=bool),
        "optimized_measurement_accepted": np.ones((2, frame_count), dtype=bool),
    }


def test_collapsed_and_later_swapped_optimized_axes_do_not_change_raw_lineage() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0], [102.0, 0.0], [103.0, 0.0]],
        ]
    )
    optimized = np.array(
        [
            [[0.0, 0.0], [101.0, 0.0], [102.0, 0.0], [103.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0], [102.0, 0.0], [3.0, 0.0]],
        ]
    )
    result = assign_two_state_identity(**inputs(raw, optimized))

    assert result.anchor_frame_index == 0
    assert result.anchor_state == "IDENTITY"
    assert result.state_counts() == {"IDENTITY": 4, "SWAP": 0}
    recovered = assigned_joints(inputs(raw, optimized)["raw_joints_2d"], result)
    np.testing.assert_array_equal(recovered[:, :, 0], raw)


def test_raw_slot_reindexing_selects_swap_to_preserve_wrist_continuity() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [100.0, 0.0], [101.0, 0.0]],
            [[100.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        ]
    )
    optimized = raw.copy()
    result = assign_two_state_identity(**inputs(raw, optimized))

    assert [frame.state for frame in result.frames] == [
        "IDENTITY",
        "SWAP",
        "SWAP",
    ]
    recovered = assigned_joints(inputs(raw, optimized)["raw_joints_2d"], result)
    np.testing.assert_array_equal(recovered[0, :, 0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(recovered[1, :, 0, 0], [100.0, 100.0, 101.0])


def test_raw_slot_reindexing_can_swap_and_then_return_to_identity() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [100.0, 0.0], [2.0, 0.0]],
            [[100.0, 0.0], [0.0, 0.0], [102.0, 0.0]],
        ]
    )
    optimized = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
            [[100.0, 0.0], [100.0, 0.0], [102.0, 0.0]],
        ]
    )
    result = assign_two_state_identity(**inputs(raw, optimized))

    assert [frame.state for frame in result.frames] == [
        "IDENTITY",
        "SWAP",
        "IDENTITY",
    ]
    recovered = assigned_joints(inputs(raw, optimized)["raw_joints_2d"], result)
    np.testing.assert_array_equal(recovered[:, :, 0], optimized)


def test_greedy_two_state_path_matches_exhaustive_global_minimum() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [100.0, 0.0], [2.0, 0.0], [103.0, 0.0]],
            [[100.0, 0.0], [0.0, 0.0], [102.0, 0.0], [3.0, 0.0]],
        ]
    )
    optimized = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[100.0, 0.0], [100.0, 0.0], [102.0, 0.0], [103.0, 0.0]],
        ]
    )
    result = assign_two_state_identity(**inputs(raw, optimized))
    chosen = tuple(frame.state for frame in result.frames)

    slots = {"IDENTITY": np.array([0, 1]), "SWAP": np.array([1, 0])}

    def path_cost(states: tuple[str, ...]) -> float:
        return sum(
            float(
                np.square(
                    raw[slots[states[frame - 1]], frame - 1]
                    - raw[slots[states[frame]], frame]
                ).sum()
            )
            for frame in range(1, raw.shape[1])
        )

    candidates = [
        ("IDENTITY", *tail)
        for tail in product(("IDENTITY", "SWAP"), repeat=raw.shape[1] - 1)
    ]
    assert chosen == min(candidates, key=path_cost)
    assert [state for state in chosen] == ["IDENTITY", "SWAP", "IDENTITY", "SWAP"]


@pytest.mark.parametrize(
    ("translation", "scale"),
    [
        (np.array([12345.0, -9876.0]), 1.0),
        (np.array([0.0, 0.0]), 0.125),
        (np.array([-700.0, 300.0]), 8.0),
    ],
)
def test_assignment_is_translation_and_positive_scale_invariant(
    translation: np.ndarray,
    scale: float,
) -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [100.0, 1.0], [101.0, 2.0]],
            [[100.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
        ]
    )
    optimized = np.array(
        [
            [[0.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
            [[100.0, 0.0], [100.0, 1.0], [101.0, 2.0]],
        ]
    )
    baseline = assign_two_state_identity(**inputs(raw, optimized))
    transformed = assign_two_state_identity(
        **inputs(raw * scale + translation, optimized * scale + translation)
    )

    assert [frame.state for frame in transformed.frames] == [
        frame.state for frame in baseline.frames
    ]


def test_final_valid_but_unobserved_raw_pair_is_disclosed_as_filled_input() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0]],
        ]
    )
    optimized = raw.copy()
    values = inputs(raw, optimized)
    values["raw_observed"][1, 1] = False
    result = assign_two_state_identity(**values)

    # ``final_valid`` is the solver's assignment gate.  The observation bit is
    # nevertheless retained so downstream evidence cannot call this a direct
    # two-observation recovery.
    assert result.frames[1].status == "ASSIGNED"
    assert result.frames[1].raw_pair_observed is False
    assert result.frames[1].raw_pair_final_valid is True
    authority = assignment_authority_arrays(result)
    np.testing.assert_array_equal(authority["raw_axis_observed"][1], [True, False])
    np.testing.assert_array_equal(authority["raw_axis_final_valid"][1], [True, True])
    np.testing.assert_array_equal(authority["raw_axis_input_kind_code"][1], [1, 2])
    assert authority["assignment_source_code"][1] == 2


def test_anchor_skips_spatial_alias_and_uses_earliest_unique_bijection() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0]],
        ]
    )
    optimized = np.array(
        [
            [[99.0, 0.0], [1.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0]],
        ]
    )
    result = assign_two_state_identity(**inputs(raw, optimized))

    assert result.anchor_frame_index == 1
    assert result.anchor_state == "IDENTITY"
    assert [frame.state for frame in result.frames] == ["IDENTITY", "IDENTITY"]


def test_exact_continuity_tie_fails_closed() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [50.0, 0.0]],
            [[100.0, 0.0], [50.0, 0.0]],
        ]
    )
    optimized = np.array(
        [
            [[0.0, 0.0], [50.0, 0.0]],
            [[100.0, 0.0], [50.0, 0.0]],
        ]
    )
    values = inputs(raw, optimized)
    values["raw_observed"][:, 1] = False
    values["optimized_valid"][:, 1] = False
    values["optimized_measurement_accepted"][:, 1] = False

    with pytest.raises(IdentityAssignmentError, match="identity cost tie"):
        assign_two_state_identity(**values)


def test_raw_pair_invalid_frame_is_retained_and_continuity_bridges_it() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0], [102.0, 0.0]],
        ]
    )
    optimized = raw.copy()
    values = inputs(raw, optimized)
    values["raw_final_valid"][1, 1] = False
    result = assign_two_state_identity(**values)

    assert result.frames[1].status == "RAW_PAIR_INVALID"
    assert result.frames[1].physical_to_raw_slots is None
    assert result.frames[2].transition_from_frame_index == 0
    assert result.frames[2].state == "IDENTITY"
    authority = assignment_authority_arrays(result)
    assert authority["frame_index"].tolist() == [0, 1, 2]
    assert authority["frame_assignment_status_code"].tolist() == [1, 0, 1]
    assert authority["state_code"].tolist() == [0, -1, 0]
    assert authority["physical_to_raw_slots"].tolist() == [
        [0, 1],
        [-1, -1],
        [0, 1],
    ]
    assert authority["assignment_source_code"].tolist() == [1, 0, 1]
    assert authority["raw_axis_input_kind_code"][1].tolist() == [1, 0]


def test_nonfinite_raw_wrist_is_invalid_even_when_flags_claim_observed() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[100.0, 0.0], [np.nan, 0.0]],
        ]
    )
    optimized = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0]],
        ]
    )
    result = assign_two_state_identity(**inputs(raw, optimized))
    authority = assignment_authority_arrays(result)

    assert result.frames[1].status == "RAW_PAIR_INVALID"
    assert authority["raw_axis_wrist_finite"][1].tolist() == [True, False]
    assert authority["raw_axis_input_kind_code"][1].tolist() == [1, 0]
    assert authority["physical_to_raw_slots"][1].tolist() == [-1, -1]


def test_no_direct_bijective_anchor_fails_closed() -> None:
    raw = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0]],
        ]
    )
    optimized = np.array(
        [
            [[99.0, 0.0], [100.0, 0.0]],
            [[100.0, 0.0], [101.0, 0.0]],
        ]
    )
    with pytest.raises(IdentityAssignmentError, match="no direct unique bijective"):
        assign_two_state_identity(**inputs(raw, optimized))


def test_shape_drift_fails_closed() -> None:
    raw = np.zeros((2, 2, 21, 2), dtype=np.float32)
    optimized = raw.copy()
    values = inputs(raw[:, :, 0], optimized[:, :, 0])
    values["raw_observed"] = np.ones((2, 3), dtype=bool)
    with pytest.raises(IdentityAssignmentError, match="raw_observed"):
        assign_two_state_identity(**values)
