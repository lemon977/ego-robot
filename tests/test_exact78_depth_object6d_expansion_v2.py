from __future__ import annotations

import pytest

from tools.build_exact78_depth_object6d_expansion_v2 import (
    CHIPS_CANARY,
    adapt_task_object_frames,
    choose_expansion_sessions,
)


CALIBRATED_26 = [
    "get_potato_chips_0903_050",
    "get_potato_chips_0903_085",
    "get_potato_chips_0903_103",
    "get_potato_chips_0903_113",
    "get_potato_chips_0903_140",
    "get_potato_chips_0903_150",
    "get_potato_chips_0903_182",
    "play_cards_0902_024",
    "play_cards_0902_025",
    "play_cards_0902_026",
    "play_cards_0902_031",
    "play_cards_0902_032",
    "play_cards_0902_036",
    "play_cards_0902_039",
    "play_cards_0902_042",
    "play_cards_0902_047",
    "play_cards_0902_049",
    "play_cards_0902_053",
    "play_cards_0903_189",
    "play_cards_0903_195",
    "play_cards_0903_202",
    "play_cards_0903_203",
    "play_cards_0903_224",
    "play_cards_0903_227",
    "play_cards_0903_243",
    "play_cards_0903_245",
]
COMPLETED_4 = {
    "play_cards_0903_203",
    "play_cards_0903_224",
    "play_cards_0903_243",
    "play_cards_0903_245",
}


def test_choose_exact26_excludes_completed_and_pins_chips_canary() -> None:
    plan = choose_expansion_sessions(CALIBRATED_26, COMPLETED_4)
    assert plan["counts"] == {
        "calibrated_ready": 26,
        "already_complete": 4,
        "remaining": 22,
        "chips_remaining": 7,
        "poker_remaining": 15,
    }
    assert plan["canary"] == [CHIPS_CANARY]
    assert len(plan["successors"]) == 21
    assert not COMPLETED_4.intersection(plan["remaining"])


def test_choose_requires_the_frozen_exact26_denominator() -> None:
    with pytest.raises(RuntimeError, match="exactly 26"):
        choose_expansion_sessions(CALIBRATED_26[:-1], COMPLETED_4)


def _mask_ref(name: str) -> dict[str, object]:
    return {"path": f"/{name}.png", "bytes": 1, "sha256": "a" * 64}


def test_poker_adapter_preserves_same_card_and_rebinds_local_id_zero() -> None:
    source = {
        "session": "play_cards_x",
        "task": "poker",
        "semantic_type": "TASK_OBJECT_IDENTITY_MASK_NOT_ROLE_REMOVAL_MASK",
        "frames": [
            {
                "source_frame": 0,
                "selected_rgb_decoded_sha256": "b" * 64,
                "physical_instances": {
                    "0": {"observed": True, "valid": True, "mask": _mask_ref("p0")}
                },
            },
            {
                "source_frame": 1,
                "selected_rgb_decoded_sha256": "c" * 64,
                "physical_instances": {
                    "0": {"observed": False, "valid": False, "mask": _mask_ref("p1")}
                },
            },
        ],
    }
    adapted = adapt_task_object_frames(source, "poker", 0)
    assert adapted["global_physical_instance_id"] == 0
    assert adapted["frames"][0]["physical_instance_id"] == 0
    assert adapted["frames"][0]["observed_identity"] == "ACTION_CONDITIONED_SAME_PHYSICAL_CARD_0"
    assert adapted["frames"][1]["physical_instance_id"] == -1


def test_chips_adapter_keeps_three_independent_global_ids() -> None:
    source = {
        "session": "get_potato_chips_x",
        "task": "chips",
        "semantic_type": "TASK_OBJECT_IDENTITY_MASK_NOT_ROLE_REMOVAL_MASK",
        "frames": [
            {
                "source_frame": 0,
                "selected_rgb_decoded_sha256": "d" * 64,
                "physical_instances": {
                    str(i): {
                        "observed": i != 1,
                        "valid": i != 1,
                        "mask": _mask_ref(f"c{i}"),
                    }
                    for i in range(3)
                },
            }
        ],
    }
    adapters = [adapt_task_object_frames(source, "chips", i) for i in range(3)]
    assert [value["global_physical_instance_id"] for value in adapters] == [0, 1, 2]
    assert [value["frames"][0]["physical_instance_id"] for value in adapters] == [0, -1, 0]
    assert [value["frames"][0]["observed_identity"] for value in adapters] == [
        "PHYSICAL_CHIP_0",
        "PHYSICAL_CHIP_1",
        "PHYSICAL_CHIP_2",
    ]
