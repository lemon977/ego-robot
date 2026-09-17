from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.pipeline.robot_geometry_v1 import (
    CollisionObservation,
    FrameMode,
    RobotGeometryContractError,
    SceneLayer,
    audit_self_collision,
    audit_unified_zbuffer,
    budget_for_mode,
    rasterize_unified_zbuffer,
    selected_object_world_for_robot,
    validate_frame_budget,
)


def _layer(instance: str, link: str, part: str, z: float) -> SceneLayer:
    return SceneLayer(
        instance,
        link,
        part,
        np.asarray([[1]], dtype=np.int64),
        np.asarray([[z]], dtype=np.float64),
        np.asarray([[True]], dtype=np.bool_),
    )


def test_frame_modes_are_bounded_by_one_total_30_second_budget() -> None:
    for mode in FrameMode:
        budget = budget_for_mode(mode)
        validate_frame_budget(mode, budget)
        assert sum(budget.values()) <= 30.0
    assert sum(budget_for_mode(FrameMode.NON_CONTACT_FRAME).values()) == 3.0
    assert sum(budget_for_mode(FrameMode.CONTACT_WINDOW).values()) == 8.0


def test_noncontact_mode_rejects_contact_refinement() -> None:
    budget = budget_for_mode(FrameMode.NON_CONTACT_FRAME)
    budget["CONTACT_FINGER_REFINEMENT"] = 0.1
    with pytest.raises(RobotGeometryContractError, match="must not run"):
        validate_frame_budget(FrameMode.NON_CONTACT_FRAME, budget)


def test_robot_rejects_legacy_object6d_world_pose_without_selected_adapter() -> None:
    with pytest.raises(RobotGeometryContractError, match="adapter keys missing"):
        selected_object_world_for_robot({"T_object_to_world": np.tile(np.eye(4), (2, 1, 1))})


def test_robot_accepts_explicit_selected_camera_object6d_adapter() -> None:
    pose = np.tile(np.eye(4), (2, 1, 1))
    arrays = {
        "T_object_to_selected_camera": pose,
        "T_object_to_world_corrected": pose.copy(),
        "source_coordinate_domain": np.asarray("STEREO_RECTIFIED_DEPTH_CAMERA"),
        "target_coordinate_domain": np.asarray("SELECTED_LEFT_RGB_CAMERA"),
        "control_ground_truth": np.asarray(False),
    }
    np.testing.assert_array_equal(selected_object_world_for_robot(arrays), pose)


def test_one_zbuffer_resolves_arm_fingers_and_object_by_depth_not_draw_order() -> None:
    layers = [
        _layer("left_arm", "left_forearm", "arm", 1.0),
        _layer("right_finger_2", "right_index_distal", "finger", 0.7),
        _layer("object_card_0", "object_card_0", "object", 0.8),
    ]
    first = rasterize_unified_zbuffer(layers)
    reverse = rasterize_unified_zbuffer(reversed(layers))
    assert first.instance_id[0, 0] == "right_finger_2"
    assert reverse.instance_id[0, 0] == "right_finger_2"
    assert first.part_id[0, 0] == "finger"
    assert first.link_id[0, 0] == "right_index_distal"
    assert first.triangle_id[0, 0] == 1


def test_one_robot_instance_may_have_multiple_links_and_ties_are_draw_order_independent() -> None:
    layers = [
        _layer("left_robot", "palm", "palm", 0.7),
        _layer("left_robot", "index_distal", "finger", 0.7),
    ]
    first = rasterize_unified_zbuffer(layers)
    reverse = rasterize_unified_zbuffer(reversed(layers))
    assert first.instance_id[0, 0] == reverse.instance_id[0, 0] == "left_robot"
    assert first.link_id[0, 0] == reverse.link_id[0, 0] == "index_distal"
    assert first.depth_tie_mask[0, 0]
    audit = audit_unified_zbuffer(first, max_depth_tie_ratio=0.0)
    assert not audit.passed
    assert audit.failures == ("DEPTH_TIE_RATIO",)


def test_zbuffer_audit_rejects_missing_triangle_provenance() -> None:
    layer = SceneLayer(
        "left_robot",
        "palm",
        "palm",
        np.asarray([[-1]], dtype=np.int64),
        np.asarray([[0.7]], dtype=np.float64),
        np.asarray([[True]], dtype=np.bool_),
    )
    audit = audit_unified_zbuffer(rasterize_unified_zbuffer([layer]))
    assert audit.invalid_triangle_ratio == 1.0
    assert "INVALID_TRIANGLE_RATIO" in audit.failures


def test_self_occlusion_does_not_hide_nonadjacent_self_intersection() -> None:
    rows = [
        CollisionObservation("index_1", "index_2", 0.0, 0.002, 1e-9),
        CollisionObservation("index_3", "palm", -0.003, 0.003, 2e-9),
    ]
    audit = audit_self_collision(
        rows,
        adjacent_allowlist={frozenset(("index_1", "index_2"))},
    )
    assert audit.self_collision_count == 1
    assert audit.max_penetration_depth_m == 0.003
    assert audit.failures == ("SELF_INTERSECTION",)


def test_result_schema_keeps_visual_trajectory_out_of_control_truth() -> None:
    schema = json.loads((ROOT / "contracts/robot_geometry_v1.schema.json").read_text())
    value = {
        "schema_version": "ROBOT_GEOMETRY_V1",
        "session_id": "chips_001",
        "mode": "POSE_ONLY_VISUAL_ROBOT",
        "clean_consumed": False,
        "control_ground_truth": False,
        "physical_deployment_authorized": False,
        "frame_budget_s": 3.0,
        "zbuffer_fields": ["depth", "instance_id", "part_id", "link_id", "triangle_id"],
        "collision_qa": {
            "self_collision_count": 0,
            "minimum_non_adjacent_link_distance_m": 0.002,
            "penetration_depth_m": 0.0,
            "penetration_volume_m3": 0.0,
            "joint_limit_violation": 0
        },
        "status": "PASSED"
    }
    jsonschema.validate(value, schema)
    value["control_ground_truth"] = True
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema)
