from __future__ import annotations

import numpy as np
import pytest

from tools.render_poker_physical_left_hawor_web_24frame_video import (
    PhysicalLeftVideoError,
    validate_states,
)


def valid_states() -> dict[str, np.ndarray]:
    return {
        "source_frames": np.arange(24),
        "q_arm": np.zeros((24, 2, 7)),
        "q_hand": np.zeros((24, 2, 22)),
        "T_camera_base_session_fit": np.eye(4),
        "T_tool_hand_terminal_chirality": np.repeat(np.eye(4)[None], 2, axis=0),
    }


def test_validate_states_accepts_frozen_24frame_shapes() -> None:
    values = validate_states(valid_states())
    assert np.array_equal(values[0], np.arange(24))


def test_validate_states_rejects_noncanonical_frame_order() -> None:
    states = valid_states()
    states["source_frames"] = np.arange(24)[::-1]
    with pytest.raises(PhysicalLeftVideoError, match="exact 0..23"):
        validate_states(states)


def test_validate_states_rejects_nonfinite_trajectory() -> None:
    states = valid_states()
    states["q_arm"][3, 0, 0] = np.nan
    with pytest.raises(PhysicalLeftVideoError, match="non-finite"):
        validate_states(states)
