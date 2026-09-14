import numpy as np

from pipeline.pico_geometry_mask_anchors import (
    apply_homography,
    authorize_object_candidate_rows,
    choose_raw_candidate,
    derive_setup_world_anchors,
    object_candidate_record,
    object_identity_requirements,
    pico_negative_points,
    project_setup_world_anchors,
    select_distinct_object_candidates,
    task_object_candidate_points,
    tracker_candidate_record,
    tracker_candidate_points,
)


def hand(offset=0.0):
    points = np.asarray([[100.0 + offset + index, 200.0 + index] for index in range(21)])
    return {
        "keypoints_2d": points.tolist(),
        "joint_valid": [True] * 21,
        "joint_in_image": [True] * 21,
    }


def test_tracker_lattice_is_deterministic_unique_and_colour_free():
    first = tracker_candidate_points((100, 100), (90, 100), 300, 200, [20, 40], 8)
    second = tracker_candidate_points((100, 100), (90, 100), 300, 200, [20, 40], 8)
    assert first == second
    assert len(first) == len({tuple(point) for point in first}) == 16
    assert [120, 100] in first


def test_task_candidates_combine_setup_geometry_and_pinch():
    config = {"setup_geometry_normalized_xy": {"object": [0.5, 0.5]}}
    rows = task_object_candidate_points(
        config, {"left": hand(), "right": hand(50)}, 400, 300, [[0, 0], [20, 0]]
    )
    assert any(row["source"] == "setup:object" and row["xy"] == [200, 150] for row in rows)
    assert any(row["source"].startswith("pico:left") for row in rows)
    assert any(row["source"].startswith("pico:right") for row in rows)


def test_task_candidates_accept_pose_projected_setup_geometry():
    config = {"setup_geometry_normalized_xy": {"object": [0.5, 0.5]}}
    rows = task_object_candidate_points(
        config,
        {"left": hand(), "right": hand(50)},
        400,
        300,
        [[0, 0]],
        setup_points_xy={"object": [222.0, 111.0]},
    )
    assert any(row["source"] == "setup:object" and row["xy"] == [222, 111] for row in rows)


def test_setup_world_anchor_uses_pico_depth_but_preserves_exact_setup_ray():
    config = {"setup_geometry_normalized_xy": {"fixture": [0.5, 0.5]}}
    reference = {
        "idx": 0,
        "k": [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        "c2w": np.eye(4).tolist(),
    }
    anchors, evidence = derive_setup_world_anchors(
        config,
        reference,
        [{
            "source_frame": 7,
            "side": "right",
            "world_xyz": [0.02, -0.01, 2.0],
            "thumb_index_distance_m": 0.03,
        }],
        100,
        80,
        open_pinch_penalty_px_per_m=150.0,
        closed_pinch_reference_m=0.06,
        reference_match_distance_px_max=10.0,
    )
    assert np.allclose(anchors["fixture"], [0.0, 0.0, 2.0])
    assert evidence["roles"]["fixture"]["depth_proxy_source_frame"] == 7
    current_c2w = np.eye(4)
    current_c2w[0, 3] = 0.2
    projected, projection = project_setup_world_anchors(
        anchors,
        {"idx": 3, "k": reference["k"], "c2w": current_c2w.tolist()},
    )
    assert np.allclose(projected["fixture"], [40.0, 40.0])
    assert projection["rgb_used"] is False


def test_movable_setup_anchor_expires_and_dynamic_seed_needs_active_binding():
    task = {
        "setup_geometry_normalized_xy": {"chip_slot_0": [0.5, 0.5], "bowl": [0.2, 0.5]},
        "role_groups": {
            "task_object": {"setup_roles": ["chip_slot_0"]},
            "fixture": {"setup_roles": ["bowl"]},
        },
    }
    binding = {
        "chip_slot_0": {
            "contact_frame": 10,
            "active_start_frame": 8,
            "active_end_frame": 16,
            "side": "right",
        }
    }
    rows = [
        {"source": "setup:chip_slot_0", "xy": [200, 150]},
        {"source": "setup:bowl", "xy": [80, 150]},
        {"source": "pico:left:thumb_index_pinch_midpoint", "xy": [180, 150]},
        {"source": "pico:right:thumb_index_pinch_midpoint", "xy": [210, 150]},
    ]
    before, before_gate = authorize_object_candidate_rows(rows, task, binding, 9)
    assert any(row["source"] == "setup:chip_slot_0" for row in before)
    assert not any(row["source"].startswith("pico:") for row in before)
    assert before_gate["unresolved_moved_roles"] == []
    during, during_gate = authorize_object_candidate_rows(rows, task, binding, 12)
    assert not any(row["source"] == "setup:chip_slot_0" for row in during)
    dynamic = [row for row in during if row["identity_mode"] == "CONTACT_BOUND_MOVING_INSTANCE"]
    assert len(dynamic) == 1 and dynamic[0]["identity_roles"] == ["chip_slot_0"]
    assert during_gate["unresolved_moved_roles"] == []
    after, after_gate = authorize_object_candidate_rows(rows, task, binding, 20)
    assert all(row["identity_mode"] == "SESSION_STATIC_FIXTURE" for row in after)
    assert after_gate["unresolved_moved_roles"] == ["chip_slot_0"]
    assert object_identity_requirements(task, binding, 20)["pass_without_external_object_authority"] is False


def test_homography_and_negative_points():
    matrix = [[1, 0, 10], [0, 1, -5], [0, 0, 1]]
    assert np.allclose(apply_homography((20, 30), matrix), [30, 25])
    points = pico_negative_points(hand(), hand(50), 400, 300, [[250, 250]])
    assert [100, 200] in points
    assert [250, 250] in points


def test_candidate_choice_rejects_failed_hard_gate_and_is_stable():
    gates = ["positive_inside", "area", "own_wrist"]
    failed = {"candidate_id": 1, "gates": {"positive_inside": True, "area": False, "own_wrist": True}}
    worse = {
        "candidate_id": 3, "gates": {name: True for name in gates},
        "pico_negative_points_inside": 0, "own_human_min_distance_px": 3,
        "centroid_to_own_wrist_px": 30, "largest_component_area_fraction": 0.95,
        "protected_object_overlap_pixels": 0,
    }
    better = {**worse, "candidate_id": 2, "own_human_min_distance_px": 0}
    assert choose_raw_candidate([failed, worse, better], gates)["candidate_id"] == 2


def test_tracker_candidate_hard_gates_accept_only_wrist_adjacent_clean_mask():
    shape = (200, 300)
    mask = np.zeros(shape, bool)
    mask[80:90, 118:138] = True
    own_human = np.zeros(shape, bool)
    own_human[88:150, 120:180] = True
    opposing_human = np.zeros(shape, bool)
    opposing_human[80:150, 240:280] = True
    gate = {
        "positive_candidate_point_inside": True,
        "valid_pico_finger_or_palm_points_inside_max": 0,
        "own_human_min_distance_px_max": 12,
        "centroid_to_own_pico_wrist_px_max": 120,
        "opposing_minus_own_wrist_centroid_px_min": 20,
        "frame_area_fraction_min": 0.0002,
        "frame_area_fraction_max": 0.02,
        "connected_components_over_64px_max": 2,
        "largest_component_area_fraction_min": 0.9,
        "protected_object_overlap_pixels_max": 0,
        "non_wrist_authorized_image_edge_contact_pixels_max": 0,
    }
    record = tracker_candidate_record(
        mask, 7, [125, 84], [[160, 120]], own_human, opposing_human,
        [125, 95], [255, 110], np.zeros(shape, bool), gate,
    )
    assert all(record["gates"].values())
    assert choose_raw_candidate([record], list(record["gates"]))["candidate_id"] == 7


def test_object_candidate_gate_and_raw_iou_deduplication():
    shape = (200, 300)
    first = np.zeros(shape, bool)
    first[70:90, 100:130] = True
    duplicate = first.copy()
    distinct = np.zeros(shape, bool)
    distinct[110:130, 180:210] = True
    gate = {
        "positive_candidate_point_inside": True,
        "valid_pico_joint_points_inside_max": 2,
        "connected_components_over_64px_max": 2,
        "largest_component_area_fraction_min": 0.9,
        "image_edge_contact_pixels_max": 0,
    }
    limits = {"frame_area_fraction_min": 0.001, "frame_area_fraction_max": 0.02}
    rows = [
        (object_candidate_record(first, 1, [110, 80], [], limits, gate), first),
        (object_candidate_record(duplicate, 2, [110, 80], [], limits, gate), duplicate),
        (object_candidate_record(distinct, 3, [190, 120], [], limits, gate), distinct),
    ]
    selected = select_distinct_object_candidates(rows, 4, 0.85)
    assert [row[0]["candidate_id"] for row in selected] == [1, 3]
