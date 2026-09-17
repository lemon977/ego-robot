"""Deterministic two-state physical identity assignment for frozen HaWoR tracks.

This module does not create masks or pixels.  It recovers a per-frame mapping
from the two raw HaWoR slots to the two physical hand axes.  The only permitted
states are ``IDENTITY`` (physical axes 0/1 consume raw slots 0/1) and ``SWAP``
(physical axes 0/1 consume raw slots 1/0).

The contract intentionally does not consume ``source_slot``.  That field is
diagnostic measurement provenance in the existing production artifacts and can
remain distinct even when the two optimized physical axes alias one hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray


State = Literal["IDENTITY", "SWAP"]
FrameStatus = Literal["ASSIGNED", "RAW_PAIR_INVALID"]

IDENTITY: Final[tuple[int, int]] = (0, 1)
SWAP: Final[tuple[int, int]] = (1, 0)
STATE_TO_SLOTS: Final[dict[State, tuple[int, int]]] = {
    "IDENTITY": IDENTITY,
    "SWAP": SWAP,
}
AUTHORITY_STATUS_CODE: Final[dict[FrameStatus, int]] = {
    "RAW_PAIR_INVALID": 0,
    "ASSIGNED": 1,
}
AUTHORITY_STATE_CODE: Final[dict[State, int]] = {
    "IDENTITY": 0,
    "SWAP": 1,
}
RAW_AXIS_INPUT_KIND_CODE: Final[dict[str, int]] = {
    "INVALID": 0,
    "OBSERVED": 1,
    "FILLED": 2,
}


class IdentityAssignmentError(RuntimeError):
    """A shape, anchor, finiteness, or exact-tie invariant failed."""


@dataclass(frozen=True)
class FrameAssignment:
    """One frame's fail-closed raw-slot to physical-axis assignment."""

    frame_index: int
    status: FrameStatus
    state: State | None
    physical_to_raw_slots: tuple[int, int] | None
    transition_from_frame_index: int | None
    identity_cost_squared_px: float | None
    swap_cost_squared_px: float | None
    raw_pair_observed: bool
    raw_pair_final_valid: bool
    raw_axis_observed: tuple[bool, bool]
    raw_axis_final_valid: tuple[bool, bool]
    raw_axis_wrist_finite: tuple[bool, bool]


@dataclass(frozen=True)
class SessionAssignment:
    """A complete session result with one semantic anchor and no fallback."""

    frame_count: int
    anchor_frame_index: int
    anchor_state: State
    anchor_identity_cost_squared_px: float
    anchor_swap_cost_squared_px: float
    frames: tuple[FrameAssignment, ...]

    def state_counts(self) -> dict[str, int]:
        return {
            state: sum(frame.state == state for frame in self.frames)
            for state in ("IDENTITY", "SWAP")
        }


def _require_bool_matrix(
    name: str, value: NDArray[np.generic], frame_count: int
) -> None:
    if value.shape != (2, frame_count) or value.dtype != np.bool_:
        raise IdentityAssignmentError(
            f"{name} must have shape (2, {frame_count}) and bool dtype"
        )


def _validate_inputs(
    raw_joints_2d: NDArray[np.generic],
    raw_observed: NDArray[np.generic],
    raw_final_valid: NDArray[np.generic],
    optimized_joints_2d: NDArray[np.generic],
    optimized_valid: NDArray[np.generic],
    optimized_measurement_accepted: NDArray[np.generic],
) -> int:
    if raw_joints_2d.ndim != 4 or raw_joints_2d.shape[:1] != (2,):
        raise IdentityAssignmentError("raw_joints_2d must have shape (2, F, 21, 2)")
    if raw_joints_2d.shape[2:] != (21, 2) or raw_joints_2d.shape[1] < 1:
        raise IdentityAssignmentError("raw_joints_2d must have shape (2, F, 21, 2)")
    frame_count = int(raw_joints_2d.shape[1])
    if optimized_joints_2d.shape != (2, frame_count, 21, 2):
        raise IdentityAssignmentError(
            "optimized_joints_2d must match raw shape (2, F, 21, 2)"
        )
    _require_bool_matrix("raw_observed", raw_observed, frame_count)
    _require_bool_matrix("raw_final_valid", raw_final_valid, frame_count)
    _require_bool_matrix("optimized_valid", optimized_valid, frame_count)
    _require_bool_matrix(
        "optimized_measurement_accepted",
        optimized_measurement_accepted,
        frame_count,
    )
    return frame_count


def _state(cost_identity: float, cost_swap: float, *, context: str) -> State:
    if not np.isfinite(cost_identity) or not np.isfinite(cost_swap):
        raise IdentityAssignmentError(f"non-finite identity cost: {context}")
    if cost_identity == cost_swap:
        raise IdentityAssignmentError(f"identity cost tie: {context}")
    return "IDENTITY" if cost_identity < cost_swap else "SWAP"


def _pair_costs(
    previous_physical_wrists: NDArray[np.float64],
    current_raw_wrists: NDArray[np.float64],
) -> tuple[float, float]:
    identity = current_raw_wrists[np.asarray(IDENTITY)]
    swap = current_raw_wrists[np.asarray(SWAP)]
    identity_cost = float(np.square(previous_physical_wrists - identity).sum())
    swap_cost = float(np.square(previous_physical_wrists - swap).sum())
    return identity_cost, swap_cost


def _anchor(
    raw_wrists: NDArray[np.float64],
    optimized_wrists: NDArray[np.float64],
    raw_observed: NDArray[np.bool_],
    raw_pair_valid: NDArray[np.bool_],
    optimized_valid: NDArray[np.bool_],
    optimized_measurement_accepted: NDArray[np.bool_],
) -> tuple[int, State, float, float]:
    """Choose the earliest direct, unique, bijective physical-axis anchor.

    An anchor must have two observed/final-valid raw wrists and two accepted,
    valid optimized physical wrists.  Each optimized wrist must have a unique
    nearest raw wrist, and the two nearest raw slots must differ.  No distance
    threshold is used.  A distance tie is ineligible rather than broken by slot
    order.
    """

    for frame in range(raw_wrists.shape[1]):
        if not (
            bool(raw_observed[:, frame].all())
            and bool(raw_pair_valid[frame])
            and bool(optimized_valid[:, frame].all())
            and bool(optimized_measurement_accepted[:, frame].all())
            and bool(np.isfinite(optimized_wrists[:, frame]).all())
        ):
            continue
        distances = np.linalg.norm(
            optimized_wrists[:, frame, None, :] - raw_wrists[None, :, frame, :],
            axis=-1,
        )
        nearest = np.argmin(distances, axis=1)
        if int(nearest[0]) == int(nearest[1]):
            continue
        if any(
            distances[axis, nearest[axis]] == distances[axis, 1 - nearest[axis]]
            for axis in (0, 1)
        ):
            continue
        identity_cost = float(
            np.square(
                optimized_wrists[:, frame] - raw_wrists[np.asarray(IDENTITY), frame]
            ).sum()
        )
        swap_cost = float(
            np.square(
                optimized_wrists[:, frame] - raw_wrists[np.asarray(SWAP), frame]
            ).sum()
        )
        state = _state(identity_cost, swap_cost, context=f"anchor frame {frame}")
        if STATE_TO_SLOTS[state] != tuple(map(int, nearest)):
            raise IdentityAssignmentError("nearest-slot anchor and pair cost disagree")
        return frame, state, identity_cost, swap_cost
    raise IdentityAssignmentError("no direct unique bijective identity anchor")


def assign_two_state_identity(
    *,
    raw_joints_2d: NDArray[np.generic],
    raw_observed: NDArray[np.generic],
    raw_final_valid: NDArray[np.generic],
    optimized_joints_2d: NDArray[np.generic],
    optimized_valid: NDArray[np.generic],
    optimized_measurement_accepted: NDArray[np.generic],
) -> SessionAssignment:
    """Assign physical identities using only a fixed anchor and raw wrist motion.

    After the anchor, each next raw-valid pair is compared in exactly two
    bijections.  The cost is the unweighted sum of squared 2D wrist displacement
    in pixels.  The lower cost wins; exact ties fail the entire session.  The
    same rule is applied backward from the anchor.  Raw-pair-invalid frames are
    retained in the result with no assignment, and continuity bridges them using
    the nearest previous/next raw-valid pair.  There is no session-specific
    threshold, source-slot input, interpolation, or pixel operation.
    """

    raw = np.asarray(raw_joints_2d)
    observed = np.asarray(raw_observed)
    final_valid = np.asarray(raw_final_valid)
    optimized = np.asarray(optimized_joints_2d)
    opt_valid = np.asarray(optimized_valid)
    accepted = np.asarray(optimized_measurement_accepted)
    frame_count = _validate_inputs(
        raw,
        observed,
        final_valid,
        optimized,
        opt_valid,
        accepted,
    )
    raw_wrists = np.asarray(raw[:, :, 0, :], dtype=np.float64)
    optimized_wrists = np.asarray(optimized[:, :, 0, :], dtype=np.float64)
    raw_pair_valid = final_valid.all(axis=0) & np.isfinite(raw_wrists).all(axis=(0, 2))
    anchor_frame, anchor_state, anchor_identity_cost, anchor_swap_cost = _anchor(
        raw_wrists,
        optimized_wrists,
        np.asarray(observed, dtype=np.bool_),
        np.asarray(raw_pair_valid, dtype=np.bool_),
        np.asarray(opt_valid, dtype=np.bool_),
        np.asarray(accepted, dtype=np.bool_),
    )

    assigned: dict[int, tuple[State, int | None, float | None, float | None]] = {
        anchor_frame: (anchor_state, None, None, None)
    }
    valid_frames = np.flatnonzero(raw_pair_valid)
    anchor_positions = np.flatnonzero(valid_frames == anchor_frame)
    if anchor_positions.size != 1:
        raise IdentityAssignmentError("anchor absent from raw-valid frame set")
    anchor_position = int(anchor_positions[0])

    def propagate(indices: NDArray[np.int64]) -> None:
        previous_frame = anchor_frame
        previous_state = anchor_state
        for raw_frame in indices:
            frame = int(raw_frame)
            previous_slots = np.asarray(STATE_TO_SLOTS[previous_state])
            previous_physical = raw_wrists[previous_slots, previous_frame]
            identity_cost, swap_cost = _pair_costs(
                previous_physical,
                raw_wrists[:, frame],
            )
            state = _state(
                identity_cost,
                swap_cost,
                context=f"transition {previous_frame}->{frame}",
            )
            assigned[frame] = (
                state,
                previous_frame,
                identity_cost,
                swap_cost,
            )
            previous_frame = frame
            previous_state = state

    propagate(np.asarray(valid_frames[anchor_position + 1 :], dtype=np.int64))
    propagate(np.asarray(valid_frames[:anchor_position][::-1], dtype=np.int64))

    frames: list[FrameAssignment] = []
    for frame in range(frame_count):
        if frame not in assigned:
            frames.append(
                FrameAssignment(
                    frame_index=frame,
                    status="RAW_PAIR_INVALID",
                    state=None,
                    physical_to_raw_slots=None,
                    transition_from_frame_index=None,
                    identity_cost_squared_px=None,
                    swap_cost_squared_px=None,
                    raw_pair_observed=bool(observed[:, frame].all()),
                    raw_pair_final_valid=bool(final_valid[:, frame].all()),
                    raw_axis_observed=tuple(
                        bool(value) for value in observed[:, frame]
                    ),
                    raw_axis_final_valid=tuple(
                        bool(value) for value in final_valid[:, frame]
                    ),
                    raw_axis_wrist_finite=tuple(
                        bool(value)
                        for value in np.isfinite(raw_wrists[:, frame]).all(axis=1)
                    ),
                )
            )
            continue
        state, previous, identity_cost, swap_cost = assigned[frame]
        frames.append(
            FrameAssignment(
                frame_index=frame,
                status="ASSIGNED",
                state=state,
                physical_to_raw_slots=STATE_TO_SLOTS[state],
                transition_from_frame_index=previous,
                identity_cost_squared_px=identity_cost,
                swap_cost_squared_px=swap_cost,
                raw_pair_observed=bool(observed[:, frame].all()),
                raw_pair_final_valid=bool(final_valid[:, frame].all()),
                raw_axis_observed=tuple(bool(value) for value in observed[:, frame]),
                raw_axis_final_valid=tuple(
                    bool(value) for value in final_valid[:, frame]
                ),
                raw_axis_wrist_finite=tuple(
                    bool(value)
                    for value in np.isfinite(raw_wrists[:, frame]).all(axis=1)
                ),
            )
        )
    return SessionAssignment(
        frame_count=frame_count,
        anchor_frame_index=anchor_frame,
        anchor_state=anchor_state,
        anchor_identity_cost_squared_px=anchor_identity_cost,
        anchor_swap_cost_squared_px=anchor_swap_cost,
        frames=tuple(frames),
    )


def assigned_joints(
    raw_joints_2d: NDArray[np.generic],
    assignment: SessionAssignment,
) -> NDArray[np.float64]:
    """Materialize assigned numeric joints; unassigned frames remain NaN."""

    raw = np.asarray(raw_joints_2d, dtype=np.float64)
    if raw.shape != (2, assignment.frame_count, 21, 2):
        raise IdentityAssignmentError("raw joints do not match assignment frame count")
    result = np.full_like(raw, np.nan, dtype=np.float64)
    for frame in assignment.frames:
        if frame.physical_to_raw_slots is None:
            continue
        result[:, frame.frame_index] = raw[
            np.asarray(frame.physical_to_raw_slots), frame.frame_index
        ]
    return result


def assignment_authority_arrays(
    assignment: SessionAssignment,
) -> dict[str, NDArray[np.generic]]:
    """Encode every frame of one assignment as optimizer-input authority arrays.

    No frame is dropped.  A physical-to-raw mapping of ``[-1, -1]`` means that
    the raw pair was invalid and must remain unassigned.  ``raw_axis_input_kind``
    is per raw slot: 0=INVALID, 1=OBSERVED, 2=FILLED.  In particular, a filled
    value is disclosed rather than being conflated with a direct observation.
    """

    frame_count = assignment.frame_count
    frame_index = np.arange(frame_count, dtype=np.int32)
    status_code = np.zeros(frame_count, dtype=np.uint8)
    state_code = np.full(frame_count, -1, dtype=np.int8)
    physical_to_raw_slots = np.full((frame_count, 2), -1, dtype=np.int8)
    transition_from_frame_index = np.full(frame_count, -1, dtype=np.int32)
    identity_cost_squared_px = np.full(frame_count, np.nan, dtype=np.float64)
    swap_cost_squared_px = np.full(frame_count, np.nan, dtype=np.float64)
    raw_axis_observed = np.zeros((frame_count, 2), dtype=np.bool_)
    raw_axis_final_valid = np.zeros((frame_count, 2), dtype=np.bool_)
    raw_axis_wrist_finite = np.zeros((frame_count, 2), dtype=np.bool_)
    raw_axis_input_kind = np.zeros((frame_count, 2), dtype=np.uint8)
    assignment_source_code = np.zeros(frame_count, dtype=np.uint8)

    for frame in assignment.frames:
        index = frame.frame_index
        status_code[index] = AUTHORITY_STATUS_CODE[frame.status]
        raw_axis_observed[index] = frame.raw_axis_observed
        raw_axis_final_valid[index] = frame.raw_axis_final_valid
        raw_axis_wrist_finite[index] = frame.raw_axis_wrist_finite
        for raw_slot in (0, 1):
            if not (
                frame.raw_axis_final_valid[raw_slot]
                and frame.raw_axis_wrist_finite[raw_slot]
            ):
                raw_axis_input_kind[index, raw_slot] = RAW_AXIS_INPUT_KIND_CODE[
                    "INVALID"
                ]
            elif frame.raw_axis_observed[raw_slot]:
                raw_axis_input_kind[index, raw_slot] = RAW_AXIS_INPUT_KIND_CODE[
                    "OBSERVED"
                ]
            else:
                raw_axis_input_kind[index, raw_slot] = RAW_AXIS_INPUT_KIND_CODE[
                    "FILLED"
                ]
        if frame.physical_to_raw_slots is None:
            continue
        if frame.state is None:
            raise IdentityAssignmentError("assigned frame has no state")
        state_code[index] = AUTHORITY_STATE_CODE[frame.state]
        physical_to_raw_slots[index] = frame.physical_to_raw_slots
        assignment_source_code[index] = 1 if frame.raw_pair_observed else 2
        if frame.transition_from_frame_index is not None:
            transition_from_frame_index[index] = frame.transition_from_frame_index
        if frame.identity_cost_squared_px is not None:
            identity_cost_squared_px[index] = frame.identity_cost_squared_px
        if frame.swap_cost_squared_px is not None:
            swap_cost_squared_px[index] = frame.swap_cost_squared_px

    return {
        "frame_index": frame_index,
        "frame_assignment_status_code": status_code,
        "state_code": state_code,
        "physical_to_raw_slots": physical_to_raw_slots,
        "transition_from_frame_index": transition_from_frame_index,
        "identity_cost_squared_px": identity_cost_squared_px,
        "swap_cost_squared_px": swap_cost_squared_px,
        "raw_axis_observed": raw_axis_observed,
        "raw_axis_final_valid": raw_axis_final_valid,
        "raw_axis_wrist_finite": raw_axis_wrist_finite,
        "raw_axis_input_kind_code": raw_axis_input_kind,
        "assignment_source_code": assignment_source_code,
    }
