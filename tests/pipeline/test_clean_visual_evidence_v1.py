from __future__ import annotations

import numpy as np
import pytest

from chaoyang.pipeline.clean_visual_evidence_v1 import (
    CleanVisualError,
    compose_clean_visual,
)


def test_clean_removes_roles_but_protects_visible_task_object() -> None:
    rgb = np.full((4, 5, 3), 100, np.uint8)
    hand = np.zeros((4, 5), bool)
    hand[1:3, 1:4] = True
    task = np.zeros_like(hand)
    task[2, 2] = True
    value = compose_clean_visual(rgb, {
        "left_human_skin_forearm": hand,
        "task_object": task,
    })
    assert value["valid_mask"][2, 2]
    assert not value["valid_mask"][1, 1]
    assert value["clean_rgba"][1, 1, 3] == 0
    assert value["clean_rgba"][2, 2, 3] == 255
    assert value["evidence"]["hidden_pixels_synthesized"] == 0


def test_clean_keeps_role_evidence_separate_and_forbids_geometry_feedback() -> None:
    rgb = np.zeros((3, 3, 3), np.uint8)
    cable = np.eye(3, dtype=bool)
    value = compose_clean_visual(rgb, {"right_cable": cable})
    assert value["evidence"]["per_role_mask_pixels"]["right_cable"] == 3
    assert value["evidence"]["geometry_consumers_forbidden"] == [
        "Depth", "Object6D", "Contact",
    ]


def test_clean_rejects_tracker_or_controller_roles() -> None:
    rgb = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(CleanVisualError, match="unknown roles"):
        compose_clean_visual(rgb, {"tracker": np.zeros((2, 2), bool)})
