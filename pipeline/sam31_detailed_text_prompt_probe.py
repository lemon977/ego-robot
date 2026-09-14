"""Pure diagnostics for the isolated SAM3.1 detailed-text prompt probe.

This module does not select, combine, edit, or create mask pixels. It only fixes
the prompt inventory and computes evidence over one raw SAM instance at a time.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


PROBE_ID = "sam31_detailed_text_prompt_probe_v1"

GENERAL_PROMPTS = (
    "a hand",
    "a wrist",
    "a forearm",
    "an arm",
    "a sleeve",
    "a wrist cuff",
    "a wristband",
    "a wearable device",
    "a glove cuff",
    "clothing attached to a hand",
    "an arm wearing a sleeve",
)

APPEARANCE_DIAGNOSTIC_PROMPTS = (
    "a black sleeve",
    "a gray wrist cuff",
    "a white circular wrist device",
    "a gray circular object worn on a wrist",
)

OBJECT_CONTROL_PROMPT = "a beverage can"
ALL_PROMPTS = GENERAL_PROMPTS + APPEARANCE_DIAGNOSTIC_PROMPTS + (
    OBJECT_CONTROL_PROMPT,
)
FRAME_INDICES = (232, 235, 242)


class DetailedPromptProbeContractError(RuntimeError):
    pass


def prompt_group(prompt: str) -> str:
    if prompt in GENERAL_PROMPTS:
        return "GENERAL_PROJECT_SEMANTIC"
    if prompt in APPEARANCE_DIAGNOSTIC_PROMPTS:
        return "APPEARANCE_004_DIAGNOSTIC_ONLY"
    if prompt == OBJECT_CONTROL_PROMPT:
        return "OBJECT_CONTROL"
    raise DetailedPromptProbeContractError(f"unknown fixed prompt: {prompt!r}")


def prompt_slug(prompt: str) -> str:
    if prompt not in ALL_PROMPTS:
        raise DetailedPromptProbeContractError(f"unknown fixed prompt: {prompt!r}")
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def validate_prompt_inventory() -> None:
    if len(ALL_PROMPTS) != 16 or len(set(ALL_PROMPTS)) != len(ALL_PROMPTS):
        raise DetailedPromptProbeContractError("fixed prompt inventory drift")
    if set(GENERAL_PROMPTS) & set(APPEARANCE_DIAGNOSTIC_PROMPTS):
        raise DetailedPromptProbeContractError("prompt groups overlap")


def hand_scale_px(joints: np.ndarray) -> float:
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise DetailedPromptProbeContractError("joints must be finite [21,2]")
    scale = float(np.median(np.linalg.norm(points[[4, 8, 12, 16, 20]] - points[0], axis=1)))
    if not np.isfinite(scale) or scale <= 1.0:
        raise DetailedPromptProbeContractError("invalid hand scale")
    return scale


def closest_boundary_point(wrist_xy: np.ndarray, width: int, height: int) -> tuple[float, float]:
    x, y = np.asarray(wrist_xy, dtype=np.float64)
    if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
        raise DetailedPromptProbeContractError("wrist outside image")
    candidates = (
        (x, 0.0, y),
        (width - 1.0 - x, width - 1.0, y),
        (y, x, 0.0),
        (height - 1.0 - y, x, height - 1.0),
    )
    _, bx, by = min(candidates, key=lambda item: item[0])
    return float(bx), float(by)


def point_supported(mask: np.ndarray, point_xy: np.ndarray, radius: float) -> bool:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise DetailedPromptProbeContractError("mask must be 2D")
    height, width = binary.shape
    x, y = np.asarray(point_xy, dtype=np.float64)
    radius = max(float(radius), 1.0)
    x0, x1 = max(int(np.floor(x - radius)), 0), min(int(np.ceil(x + radius)) + 1, width)
    y0, y1 = max(int(np.floor(y - radius)), 0), min(int(np.ceil(y + radius)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        return False
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return bool(binary[y0:y1, x0:x1][disk].any())


def raw_instance_metrics(
    mask: np.ndarray,
    *,
    score: float,
    instance_id: int,
    joints_by_side: np.ndarray,
    cad_object_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute metrics without selecting or changing the supplied SAM mask."""
    binary = np.asarray(mask, dtype=bool)
    cad = np.asarray(cad_object_mask, dtype=bool)
    joints = np.asarray(joints_by_side, dtype=np.float64)
    if binary.ndim != 2 or cad.shape != binary.shape or joints.shape != (2, 21, 2):
        raise DetailedPromptProbeContractError("metric input shape drift")
    height, width = binary.shape
    sides: dict[str, Any] = {}
    for side_index, side_name in enumerate(("left", "right")):
        side_joints = joints[side_index]
        scale = hand_scale_px(side_joints)
        joint_hits = [
            point_supported(binary, point, 0.055 * scale) for point in side_joints
        ]
        boundary = closest_boundary_point(side_joints[0], width, height)
        sides[side_name] = {
            "hand_scale_pixels": scale,
            "joint_support_ratio": float(np.mean(joint_hits)),
            "wrist_supported": point_supported(binary, side_joints[0], 0.12 * scale),
            "boundary_supported": point_supported(binary, np.asarray(boundary), 0.30 * scale),
            "boundary_point_xy": list(boundary),
        }
        sides[side_name]["complete_limb_evidence"] = bool(
            sides[side_name]["joint_support_ratio"] >= 0.20
            and sides[side_name]["wrist_supported"]
            and sides[side_name]["boundary_supported"]
        )
    overlap = binary & cad
    return {
        "instance_id": int(instance_id),
        "score": float(score),
        "area_pixels": int(binary.sum()),
        "area_ratio": float(binary.mean()),
        "large_background_risk_area_ratio_gt_0_30": bool(binary.mean() > 0.30),
        "object6d_overlap_pixels": int(overlap.sum()),
        "object6d_overlap_over_instance": float(overlap.sum() / max(binary.sum(), 1)),
        "sides": sides,
        "pixels_changed_by_diagnostics": 0,
    }


validate_prompt_inventory()
