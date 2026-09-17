"""Reproduce the bounded-v2.1 all-or-nothing human anchor behavior.

This is a regression fixture for a future successor, not a quality pass for SAM3.1.
"""

import numpy as np

from chaoyang.ops import run_chips001_pico_mask_temporal as legacy


class _State:
    def clear(self):
        pass


class _OneRoleModel:
    def __init__(self):
        self.propagation_calls = 0

    def init_state(self, **_kwargs):
        return _State()

    def add_prompt(self, **_kwargs):
        mask = np.zeros((1, 4, 4), dtype=bool)
        mask[0, 0, 0] = True
        return None, (mask, np.array([7], dtype=np.int32))

    def propagate_in_video(self, **_kwargs):
        self.propagation_calls += 1
        yield 0, None


def test_one_missing_anchor_discards_present_role(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, "FRAME_COUNT", 8)
    monkeypatch.setattr(legacy, "normalize", lambda output, _h, _w: output)
    model = _OneRoleModel()

    streams, meta = legacy.collect_text_roles(
        model,
        tmp_path,
        "a person's hand and forearm",
        0,
        {"left_human": (0, 0), "right_human": (3, 3)},
        4,
        4,
        False,
        0.1,
    )

    assert meta["status"] == "HOLD_ID_SELECTION"
    assert meta["selection"]["left_human"]["chosen_raw_id"] == 7
    assert meta["selection"]["right_human"]["chosen_raw_id"] is None
    assert streams == {"left_human": {}, "right_human": {}}
    assert model.propagation_calls == 0
