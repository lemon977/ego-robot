"""Session-independent extraction of the selected V3 *visual* ownership rule.

No model inference, session constants, frame windows, geometry claims, or global
monkeypatches. Callers supply masks for each frame and declare their coverage.
The six/twelve-pixel rule is frozen at the V3 review resolution, not metric Z.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualOwnershipConfig:
    near_px: float = 6.0
    far_px: float = 12.0
    reference_width: int = 640
    reference_height: int = 480

    def __post_init__(self):
        if not (np.isfinite(self.near_px) and np.isfinite(self.far_px)
                and 0 <= self.near_px < self.far_px
                and self.reference_width > 0 and self.reference_height > 0):
            raise ValueError("INVALID_VISUAL_OWNERSHIP_CONFIG")


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"MASK_DOMAIN_OR_TYPE:{name}")
    return value


def compose_visual_frame(
    clean: np.ndarray,
    robot_bgr: np.ndarray,
    robot_alpha: np.ndarray,
    visible_object: np.ndarray,
    human: np.ndarray,
    unknown_object: np.ndarray,
    *,
    robot_observed: bool,
    masks_available: bool,
    config: VisualOwnershipConfig = VisualOwnershipConfig(),
) -> dict[str, object]:
    """Compose supplied images; object restoration belongs to the upstream stage.

    UNKNOWN geometry stays UNKNOWN even when the offline display puts Robot in
    front. Missing motion suppresses the supplied Robot layer; missing object
    masks are explicitly different from an observed empty mask. Inputs are never
    mutated. BGR ordering matches OpenCV and the selected V3 PNGs.
    """
    if not isinstance(robot_observed, bool) or not isinstance(masks_available, bool):
        raise ValueError("EXPLICIT_BOOLEAN_VALIDITY_REQUIRED")
    clean, robot_bgr = np.asarray(clean), np.asarray(robot_bgr)
    if (clean.ndim != 3 or clean.shape[2] != 3 or clean.dtype != np.uint8
            or robot_bgr.shape != clean.shape or robot_bgr.dtype != np.uint8):
        raise ValueError("RGB_DOMAIN_OR_TYPE")
    shape = clean.shape[:2]
    if min(shape) == 0:
        raise ValueError("EMPTY_IMAGE_DOMAIN")
    alpha = _mask(robot_alpha, shape, "robot_alpha").copy()
    visible = _mask(visible_object, shape, "visible_object")
    human = _mask(human, shape, "human")
    unknown = _mask(unknown_object, shape, "unknown_object")
    if not masks_available and (visible.any() or human.any() or unknown.any()):
        raise ValueError("UNAVAILABLE_MASKS_MUST_NOT_CARRY_OBSERVATIONS")
    if not robot_observed:
        alpha[:] = False
    observed = visible & ~human & ~unknown
    overlap = alpha & observed
    robot_front = np.zeros(shape, bool)
    object_front = np.zeros(shape, bool)
    boundary = np.zeros(shape, bool)
    if overlap.any():
        sx = shape[1] / config.reference_width
        sy = shape[0] / config.reference_height
        if not np.isclose(sx, sy):
            raise ValueError("OWNERSHIP_RULE_REQUIRES_ISOTROPIC_DOMAIN")
        distance = cv2.distanceTransform((~human).astype(np.uint8), cv2.DIST_L2, 5)
        robot_front = overlap & (distance <= config.near_px * sx)
        object_front = overlap & (distance >= config.far_px * sx)
        boundary = overlap & ~(robot_front | object_front)
    draw = alpha & ~object_front
    final = clean.copy()
    final[draw] = robot_bgr[draw]
    return {
        "image": final,
        "draw": draw,
        "robot_front": robot_front,
        "object_front": object_front,
        "boundary_unknown": boundary,
        "object_unknown_overlap": alpha & unknown,
        "robot_observed": robot_observed,
        "mask_coverage": "SUPPLIED" if masks_available else "UNAVAILABLE",
        "physical_occlusion_authority": False,
        "training_eligible": False,
    }
