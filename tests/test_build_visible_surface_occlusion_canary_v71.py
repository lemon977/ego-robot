from __future__ import annotations

import numpy as np

from chaoyang.pipeline.occlusion_compositor_v1 import DepthQualityEvidence
from chaoyang.ops.build_visible_surface_occlusion_canary_v71 import (
    object_front_edge_support,
    protected_loss_reasons,
)


def test_protected_loss_reasons_are_mutually_exclusive_and_closed() -> None:
    shape = (1, 8)
    true = np.ones(shape, dtype=np.bool_)
    evidence = DepthQualityEvidence(
        disparity_finite=np.array([[False, True, True, True, True, True, True, True]], dtype=np.bool_),
        registration_pass=np.array([[True, False, True, True, True, True, True, True]], dtype=np.bool_),
        away_from_occlusion_edge=np.array([[True, True, False, True, True, True, True, True]], dtype=np.bool_),
        texture_support=np.array([[True, True, True, False, True, True, True, True]], dtype=np.bool_),
        local_consistency_pass=np.array([[True, True, True, True, False, True, True, True]], dtype=np.bool_),
        in_valid_depth_range=np.array([[True, True, True, True, True, False, True, True]], dtype=np.bool_),
        depth_confidence_present=False,
    )
    reasons = protected_loss_reasons(
        protected=true,
        retained=np.array([[False, False, False, False, False, False, False, True]], dtype=np.bool_),
        robot_mask=true,
        robot_depth_valid=np.array([[True, True, True, True, True, True, False, True]], dtype=np.bool_),
        evidence=evidence,
    )
    assert reasons == {
        "outside_robot_unexpected": 0,
        "robot_depth_invalid": 1,
        "disparity_invalid": 1,
        "registration_failed": 1,
        "object_occlusion_edge": 1,
        "low_texture": 1,
        "local_depth_inconsistent": 1,
        "depth_out_of_range": 1,
        "unclassified_ordering": 0,
        "total_lost": 7,
    }


def test_loss_outside_robot_is_reported_as_contract_diagnostic() -> None:
    true = np.ones((1, 1), dtype=np.bool_)
    evidence = DepthQualityEvidence(true, true, true, true, true, true, False)
    reasons = protected_loss_reasons(
        protected=true,
        retained=np.zeros((1, 1), dtype=np.bool_),
        robot_mask=np.zeros((1, 1), dtype=np.bool_),
        robot_depth_valid=np.zeros((1, 1), dtype=np.bool_),
        evidence=evidence,
    )
    assert reasons["outside_robot_unexpected"] == 1
    assert reasons["total_lost"] == 1


def test_edge_support_requires_adjacent_quality_interior_and_object_front_margin() -> None:
    shape = (3, 3)
    true = np.ones(shape, dtype=np.bool_)
    interior = np.zeros(shape, dtype=np.bool_)
    interior[1, 1] = True
    evidence = DepthQualityEvidence(true, true, interior, true, true, true, False)
    supported = object_front_edge_support(
        object_mask=true,
        robot_mask=true,
        object_depth_m=np.full(shape, 0.8),
        robot_depth_m=np.full(shape, 1.0),
        evidence=evidence,
    )
    assert int(supported.sum()) == 8
    assert not supported[1, 1]

    no_margin = object_front_edge_support(
        object_mask=true,
        robot_mask=true,
        object_depth_m=np.full(shape, 1.0),
        robot_depth_m=np.full(shape, 0.8),
        evidence=evidence,
    )
    assert not np.any(no_margin)
