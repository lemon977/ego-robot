from __future__ import annotations

import json

import numpy as np
import pytest

from tools import run_clean_medoid_light_canary_v7_adapter as adapter


class Package:
    def __init__(self, payload: object):
        self.payload = payload

    def read(self, _name: str) -> bytes:
        return json.dumps(self.payload).encode()


def hand(offset: float = 0.0) -> dict[str, object]:
    return {
        "joint_names": ["wrist", "palm_center", "index"],
        "keypoints_2d": [
            [30.0 + offset, 30.0],
            [30.0 + offset, 40.0],
            [40.0 + offset, 25.0],
        ],
        "joint_valid": [True, True, True],
        "joint_in_image": [True, True, True],
    }


def config() -> dict[str, int]:
    return {
        "donor_pico_hand_exclusion_dilation_px": 2,
        "donor_pico_tracker_exclusion_radius_px": 4,
        "donor_pico_forearm_exclusion_width_px": 3,
        "donor_pico_forearm_extension_px": 8,
    }


def test_complete_hands_retain_original_geometry_support() -> None:
    package = Package(
        {"entities": {"hands": {"left": hand(), "right": hand(30.0)}}}
    )
    excluded, evidence = adapter.donor_pico_exclusion(
        package, 0, 100, 80, config()
    )
    assert excluded.dtype == np.bool_
    assert int(excluded.sum()) > 0
    assert [row["status"] for row in evidence["hands"]] == [
        "PRESENT_ORIGINAL_GEOMETRY",
        "PRESENT_ORIGINAL_GEOMETRY",
    ]


def test_complete_hands_mask_is_byte_exact_to_frozen_original() -> None:
    payload = {"entities": {"hands": {"left": hand(), "right": hand(30.0)}}}
    original = adapter.load_original_module()
    expected, _ = original.donor_pico_exclusion(
        Package(payload), 0, 100, 80, config()
    )
    observed, _ = adapter.donor_pico_exclusion(
        Package(payload), 0, 100, 80, config()
    )
    assert np.array_equal(observed, expected)


def test_missing_sides_add_zero_fake_support() -> None:
    package = Package({"entities": {"hands": {}}})
    excluded, evidence = adapter.donor_pico_exclusion(
        package, 320, 100, 80, config()
    )
    assert int(excluded.sum()) == 0
    assert evidence["hands"] == [
        {
            "side": "left",
            "status": "ABSENT_NO_AUTHORIZED_SUPPORT",
            "visible_joint_count": 0,
            "authorized_support_pixels": 0,
        },
        {
            "side": "right",
            "status": "ABSENT_NO_AUTHORIZED_SUPPORT",
            "visible_joint_count": 0,
            "authorized_support_pixels": 0,
        },
    ]


@pytest.mark.parametrize("hands", [None, [], "bad", 7])
def test_abnormal_hands_container_fails_closed(hands: object) -> None:
    with pytest.raises(adapter.CleanV7AdapterError, match="hands must be an object"):
        adapter.validate_hands_schema({"entities": {"hands": hands}})


def test_one_missing_side_does_not_block_present_side() -> None:
    package = Package({"entities": {"hands": {"left": hand()}}})
    excluded, evidence = adapter.donor_pico_exclusion(
        package, 40, 100, 80, config()
    )
    assert int(excluded.sum()) > 0
    assert evidence["hands"][0]["status"] == "PRESENT_ORIGINAL_GEOMETRY"
    assert evidence["hands"][1]["status"] == "ABSENT_NO_AUTHORIZED_SUPPORT"
    assert evidence["hands"][1]["visible_joint_count"] == 0
