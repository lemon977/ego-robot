"""CPU gates for the six-anchor, object-only Mask14 proposal."""

from __future__ import annotations

import numpy as np
import pytest

from chaoyang.governance import exact78_chips_object_prompt_mask14_candidate_v1 as design


def test_exact_sealed_six_and_only_object_delta() -> None:
    rows = design.frozen_six()
    assert len(rows) == 6
    assert [(row["session_id"], row["frame"]) for row in rows] == [
        (session, frame) for session, frames in design.ANCHORS.items() for frame in frames]
    for row in rows:
        new = design.object_only_prompts(row["prompts"])
        assert new["object"] == "a yellow potato chip"
        assert {key: value for key, value in new.items() if key != "object"} == {
            key: value for key, value in row["prompts"].items() if key != "object"}
        assert all(row["masks"][key]["sha256"] for key in design.FROZEN_MASKS)


def test_prompt_set_and_mask_domain_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="PROMPT_SET_DRIFT"):
        design.object_only_prompts({"object": "a potato chip"})
    with pytest.raises(RuntimeError, match="MASK_DOMAIN_DRIFT"):
        design.derive_masks(np.zeros((2, 2), bool), np.zeros((2, 2), bool),
                            np.zeros((2, 2), bool), np.zeros((2, 3), bool))


def test_mask13_algebra_is_preserved() -> None:
    h = np.array([[True, False, False], [False, False, False]])
    e0 = np.array([[False, True, False], [False, False, False]])
    e1 = np.array([[False, False, False], [True, False, False]])
    obj = np.array([[True, True, True], [False, True, False]])
    masks = design.derive_masks(h, e0, e1, obj)
    assert np.array_equal(masks["equipment"], e0 | e1)
    assert np.array_equal(masks["unknown"], (h | e0 | e1) & obj)
    assert np.array_equal(masks["protected"], obj & ~(h | e0 | e1))
    assert np.array_equal(masks["removal"], (h | e0 | e1) & ~obj)
    assert not np.any(masks["removal"] & obj)
