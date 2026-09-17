"""R2.2 compositing write-domain and final-layer provenance helpers.

This module deliberately wraps, rather than mutates, the current V1 occlusion
authority implementation.  ``M_composite`` is the only area a downstream
diagnostic compositor may write.  The coarse final-layer provenance is kept
separate from ``ObjectPixelSource`` so a temporal object donor is never
mistaken for Clean/background pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from chaoyang.pipeline.occlusion_compositor_v1 import (
    FrameOcclusionResult,
    ObjectPixelSource,
    Ownership,
)


class CompositePixelSource(IntEnum):
    CLEAN = 0
    RAW = 1
    ROBOT = 2
    OBJECT_ATLAS = 3
    UNKNOWN = 4


@dataclass(frozen=True)
class CompositeDomain:
    m_composite: np.ndarray
    pixel_source: np.ndarray


def derive_composite_domain(result: FrameOcclusionResult) -> CompositeDomain:
    """Return an explicit write mask and coarse final-layer provenance.

    ``HUMAN_FRONT`` maps to Raw for an occlusion diagnostic.  A Robotized
    training producer may instead reject such frames; it must never silently
    relabel those pixels as Clean.  Temporal donors and textured renderers are
    grouped as ``OBJECT_ATLAS`` here while their exact origin remains available
    in ``result.object_pixel_source``.
    """

    ownership = np.asarray(result.ownership)
    object_source = np.asarray(result.object_pixel_source)
    if ownership.ndim != 2 or object_source.shape != ownership.shape:
        raise ValueError("ownership and object_pixel_source must be matching 2D images")

    legal_ownership = {int(value) for value in Ownership}
    legal_object_sources = {int(value) for value in ObjectPixelSource}
    if not set(np.unique(ownership).tolist()).issubset(legal_ownership):
        raise ValueError("unknown ownership value")
    if not set(np.unique(object_source).tolist()).issubset(legal_object_sources):
        raise ValueError("unknown object pixel source")

    provenance = np.full(ownership.shape, int(CompositePixelSource.CLEAN), dtype=np.uint8)
    provenance[ownership == int(Ownership.HUMAN_FRONT)] = int(CompositePixelSource.RAW)
    provenance[ownership == int(Ownership.ROBOT_FRONT)] = int(CompositePixelSource.ROBOT)
    provenance[ownership == int(Ownership.TIE_UNKNOWN)] = int(CompositePixelSource.UNKNOWN)

    object_front = ownership == int(Ownership.OBJECT_FRONT)
    raw_object = object_source == int(ObjectPixelSource.RAW_VISIBLE)
    legal_non_raw_object = object_source != int(ObjectPixelSource.NONE_UNKNOWN)
    provenance[object_front & raw_object] = int(CompositePixelSource.RAW)
    provenance[object_front & legal_non_raw_object & ~raw_object] = int(
        CompositePixelSource.OBJECT_ATLAS
    )
    illegal_object = object_front & ~legal_non_raw_object
    provenance[illegal_object] = int(CompositePixelSource.UNKNOWN)

    # Every non-background decision is an explicit compositor write.  Keeping
    # this independent from Clean's M_write prevents accidental mask reuse.
    m_composite = ownership != int(Ownership.BACKGROUND)
    return CompositeDomain(m_composite=m_composite, pixel_source=provenance)
