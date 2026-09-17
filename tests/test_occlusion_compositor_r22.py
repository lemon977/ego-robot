import numpy as np
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from chaoyang.pipeline.occlusion_compositor_r22 import CompositePixelSource, derive_composite_domain
from chaoyang.pipeline.occlusion_compositor_v1 import FrameOcclusionResult, ObjectPixelSource, Ownership


def test_composite_domain_is_independent_and_provenance_is_explicit() -> None:
    ownership = np.asarray(
        [[Ownership.BACKGROUND, Ownership.HUMAN_FRONT, Ownership.OBJECT_FRONT,
          Ownership.OBJECT_FRONT, Ownership.ROBOT_FRONT, Ownership.TIE_UNKNOWN]],
        dtype=np.uint8,
    )
    object_source = np.asarray(
        [[ObjectPixelSource.NONE_UNKNOWN, ObjectPixelSource.NONE_UNKNOWN,
          ObjectPixelSource.RAW_VISIBLE, ObjectPixelSource.TEMPORAL_OBJECT_DONOR,
          ObjectPixelSource.NONE_UNKNOWN, ObjectPixelSource.NONE_UNKNOWN]],
        dtype=np.uint8,
    )
    result = FrameOcclusionResult(
        ownership=ownership,
        training_valid_mask=ownership != Ownership.TIE_UNKNOWN,
        object_pixel_source=object_source,
        contact_decision_mask=np.ones_like(ownership, dtype=np.bool_),
        object_rgb=np.zeros((1, 6, 3), dtype=np.uint8),
    )

    domain = derive_composite_domain(result)
    assert domain.m_composite.tolist() == [[False, True, True, True, True, True]]
    assert domain.pixel_source.tolist() == [[
        CompositePixelSource.CLEAN,
        CompositePixelSource.RAW,
        CompositePixelSource.RAW,
        CompositePixelSource.OBJECT_ATLAS,
        CompositePixelSource.ROBOT,
        CompositePixelSource.UNKNOWN,
    ]]


def test_illegal_object_front_is_fail_closed_to_unknown() -> None:
    result = FrameOcclusionResult(
        ownership=np.asarray([[Ownership.OBJECT_FRONT]], dtype=np.uint8),
        training_valid_mask=np.ones((1, 1), dtype=np.bool_),
        object_pixel_source=np.asarray([[ObjectPixelSource.NONE_UNKNOWN]], dtype=np.uint8),
        contact_decision_mask=np.ones((1, 1), dtype=np.bool_),
        object_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
    )
    domain = derive_composite_domain(result)
    assert domain.pixel_source[0, 0] == CompositePixelSource.UNKNOWN
