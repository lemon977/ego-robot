from __future__ import annotations

import numpy as np
import pytest

from chaoyang.pipeline import exact78_0915_cards001_base_temporal_replay_v1 as replay


def test_top_eight_uses_spatial_score_then_original_index() -> None:
    rows = [{"score": (-1, 4.0)} for _ in range(32)]
    for index in (2, 4, 6, 8, 10, 12, 24, 30):
        rows[index] = {"score": (-8, 1.0)}
    assert replay.select_top_indices(rows) == [2, 4, 6, 8, 10, 12, 24, 30]
    rows[24] = {"score": (-1, 4.0)}
    rows[0] = {"score": (-8, 1.0)}
    with pytest.raises(replay.stage.FreshRobotError, match="ORIGINAL_EXCLUDED"):
        replay.select_top_indices(rows)


def test_replay_holds_last_accepted_q_and_counts_longest_run(monkeypatch: pytest.MonkeyPatch) -> None:
    display = np.zeros((2, replay.FRAMES, 21, 3), dtype=float)
    display[:, :, 0, 0] = np.arange(replay.FRAMES)
    sources = np.full((2, replay.FRAMES), "RGB_HAWOR_PREDICTED")
    identity = np.eye(4)
    row = {"base": identity, "offsets": np.zeros((2, 3)),
           "anchor_q": np.zeros((2, 7)), "neutral_roots": np.stack((identity, identity)),
           "origin_basis": [np.eye(3), np.eye(3)]}
    monkeypatch.setattr(replay.stage, "_palm_basis", lambda *_: np.eye(3))

    def attempt(_assets, _side, target, _lower, _upper, _seeds, previous=None):
        frame = int(target[0, 3])
        q = np.full(7, 0.0 if frame == 0 else (0.3 if frame in (1, 2) else 0.0))
        return q, 0.0, 0.0, 1.0

    monkeypatch.setattr(replay.stage, "_arm_attempt", attempt)
    rows, displayed = replay.replay_candidate(
        row, display, sources, object(), np.stack((identity, identity)),
        np.full((2, 7), -1.0), np.full((2, 7), 1.0), np.zeros((2, 7)))
    assert len(rows) == 300
    left = [item for item in rows if item["side"] == "left"]
    assert left[0]["status"] == "PASS_NUMERIC_CAMERA_DIAGNOSTIC"
    assert left[1]["status"] == "HOLD_ARM_VELOCITY"
    assert left[2]["status"] == "HOLD_ARM_VELOCITY"
    assert np.array_equal(displayed[1, 0], displayed[0, 0])
    assert replay.longest_hold(rows, "left") == {"start_frame": 1, "end_frame": 2, "frames": 2}
    assert replay.summarize(rows)["left"]["pass"] == replay.FRAMES - 2
