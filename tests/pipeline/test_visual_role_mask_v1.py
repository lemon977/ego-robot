from __future__ import annotations

import copy

import pytest

from chaoyang.pipeline.visual_role_mask_v1 import (
    VisualRoleContractError,
    clean_removal_roles,
    validate_manifest,
)


def manifest() -> dict:
    roles = [
        "left_human_skin_forearm",
        "right_human_skin_forearm",
        "left_finger_sleeve_attachment",
        "right_finger_sleeve_attachment",
        "left_cable",
        "right_cable",
        "task_object",
    ]
    return {
        "schema_version": "visual-role-mask-v1",
        "session_id": "get_potato_chips_0915_001",
        "frame_count": 379,
        "model": {"identity": "SAM3.1", "weight_sha256": "a" * 64},
        "roles": roles,
        "instances": [
            {
                "instance_id": role,
                "role": role,
                "physical_identity_policy": (
                    "SEPARATE_VISIBLE_PHYSICAL_OBJECT"
                    if role == "task_object" else "SIDE_LOCKED_HUMAN_ROLE"
                ),
                "mask_directory": f"masks/{role}",
            }
            for role in roles
        ],
        "tracker_role_created": False,
        "pico26_consumed": False,
    }


def test_role_manifest_accepts_sam31_without_tracker() -> None:
    validate_manifest(manifest())
    assert "task_object" not in clean_removal_roles()


def test_role_manifest_rejects_tracker_identity() -> None:
    value = copy.deepcopy(manifest())
    value["instances"][0]["instance_id"] = "left_tracker"
    with pytest.raises(VisualRoleContractError, match="forbidden"):
        validate_manifest(value)


def test_role_manifest_rejects_non_sam31() -> None:
    value = copy.deepcopy(manifest())
    value["model"]["identity"] = "SAM2.1"
    with pytest.raises(Exception):
        validate_manifest(value)
