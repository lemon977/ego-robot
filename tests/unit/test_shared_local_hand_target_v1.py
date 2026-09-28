from __future__ import annotations

import numpy as np
import pytest

from chaoyang.governance.common import REPO_ROOT
from chaoyang.pipeline.huro_hand_only_retarget_v1 import (
    HuroHandOnlyError,
    keypoints_from_q,
    load_hand_model,
)
from chaoyang.pipeline.pico_manus_motion_v2 import MANUS_NAMES
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets
from chaoyang.pipeline.shared_local_hand_target_v1 import (
    HAWOR21_NAMES,
    SOURCE_TO_PHYSICAL,
    adapt_source_points,
    anatomical_keypoints_from_q,
    build_local_target,
    enumerate_h50,
    palm_width,
    solve_local_frame,
    solve_local_sequence,
)


def fixture_hand() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float64)
    for finger in range(5):
        x = (2 - finger) * 0.018
        for joint in range(4):
            points[1 + 4 * finger + joint] = (
                x + 0.001 * joint * finger,
                0.045 + joint * (0.013 + 0.001 * finger),
                0.002 * joint * joint,
            )
    return points


def test_source_adapters_reject_wrong_order_side_and_units():
    np.testing.assert_array_equal(SOURCE_TO_PHYSICAL, [0, 1])
    manus = np.zeros((2, 3, 25, 3), dtype=np.float64)
    assert adapt_source_points(
        manus, source_kind="MANUS25", joint_names=MANUS_NAMES
    ).shape == (2, 3, 21, 3)
    with pytest.raises(HuroHandOnlyError, match="name/order"):
        adapt_source_points(
            manus, source_kind="MANUS25", joint_names=tuple(reversed(MANUS_NAMES))
        )
    with pytest.raises(HuroHandOnlyError, match="metres"):
        adapt_source_points(
            manus, source_kind="MANUS25", joint_names=MANUS_NAMES, units="mm"
        )
    points = np.broadcast_to(fixture_hand(), (2, 2, 21, 3)).copy()
    observed = np.ones((2, 2), dtype=bool)
    frames = np.arange(2)
    fake_hand = type("Hand", (), {})()
    fake_hand.lower = np.zeros(22)
    fake_hand.upper = np.ones(22)
    fake_hand.joint_names = tuple(str(i) for i in range(22))
    with pytest.raises(HuroHandOnlyError, match="left,right"):
        solve_local_sequence(
            (fake_hand, fake_hand), points, observed, frames,
            source_kind="HAWOR21", joint_names=HAWOR21_NAMES,
            anatomical_side_names=("right", "left"),
        )


def test_target_is_similarity_invariant_and_rejects_degenerate_bone():
    source = fixture_hand()
    robot = fixture_hand() * 1.3
    base = build_local_target(
        source,
        robot,
        source_reference_width_m=palm_width(source),
        robot_reference_width_m=palm_width(robot),
    )
    moved = source * 4.0 + np.asarray([2.0, -3.0, 7.0])
    scaled = build_local_target(
        moved,
        robot,
        source_reference_width_m=4.0 * palm_width(source),
        robot_reference_width_m=palm_width(robot),
    )
    np.testing.assert_allclose(base.directions, scaled.directions, atol=1e-12)
    np.testing.assert_allclose(
        base.normalized_pinch, scaled.normalized_pinch, atol=1e-12
    )
    broken = source.copy()
    broken[2] = broken[1]
    target = build_local_target(
        broken,
        robot,
        source_reference_width_m=palm_width(broken),
        robot_reference_width_m=palm_width(robot),
    )
    assert not target.direction_valid[0]
    assert np.isnan(target.directions[0]).all()


def test_actual_robot_fk_round_trip():
    assets = load_pinned_robot_assets(REPO_ROOT)
    hand = load_hand_model(assets.left_hand.path, "left")
    q = 0.5 * (hand.lower + hand.upper)
    q += 0.04 * (hand.upper - hand.lower) * np.sin(np.arange(22))
    points = anatomical_keypoints_from_q(hand, q)
    target = build_local_target(
        points,
        points,
        source_reference_width_m=palm_width(points),
        robot_reference_width_m=palm_width(points),
    )
    solved, fk, diagnostics = solve_local_frame(hand, target, previous_q=q)
    assert diagnostics.success
    assert diagnostics.max_direction_angle_deg < 0.005
    assert diagnostics.normalized_pinch_error < 5e-5
    # Kai22 is redundant for this local target: target/FK agreement, not byte-
    # identical q, is the round-trip contract.
    assert float(np.max(np.abs(solved - q))) < 0.005
    np.testing.assert_allclose(fk, points, atol=5e-5)


def test_first_anatomical_bone_excludes_fixed_link_offset():
    assets = load_pinned_robot_assets(REPO_ROOT)
    hand = load_hand_model(assets.left_hand.path, "left")
    q = 0.5 * (hand.lower + hand.upper)
    legacy = keypoints_from_q(hand, q)
    semantic = anatomical_keypoints_from_q(hand, q)
    legacy_first = legacy[6] - legacy[5]
    semantic_first = semantic[6] - semantic[5]
    cosine = float(np.dot(legacy_first, semantic_first) / (
        np.linalg.norm(legacy_first) * np.linalg.norm(semantic_first)
    ))
    # This regression is intentionally large: consuming link1->link2 as the
    # MCP->PIP direction is exactly the source-semantics bug being prevented.
    assert np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) > 45.0


def test_h50_requires_51_source_frames_without_gaps():
    quality = np.ones((52, 2), dtype=bool)
    frames = np.arange(52)
    timestamps = np.arange(52, dtype=np.int64) * 16_666_667
    anchors, sides = enumerate_h50(quality, frames, timestamps)
    np.testing.assert_array_equal(anchors, [0, 0, 1, 1])
    np.testing.assert_array_equal(sides, [0, 1, 0, 1])
    frames[25:] += 1
    anchors, sides = enumerate_h50(quality, frames, timestamps)
    assert not len(anchors) and not len(sides)
