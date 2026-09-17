"""Dynamic HaWoR-derived prompt contract for the SAM3.1-only 0915 Mask lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MODEL_IDENTITY = "SAM3.1"
PALM_JOINTS = (0, 5, 9, 13, 17)
SIDE_NAMES = ("left", "right")
TASK_TEXT_PROMPTS = {
    "playing_cards": ("a visible playing card", "a visible stack of playing cards"),
    "potato_chips": ("a visible potato chip",),
}
AUXILIARY_TEXT_PROMPTS = {
    "finger_sleeve_attachment": (
        "a finger-worn sleeve or fingertip attachment on a person's hand"
    ),
    "cable": "a data cable attached to a person's hand",
}


class Sam31PromptContractError(ValueError):
    pass


@dataclass(frozen=True)
class SpatialPrompt:
    role: str
    frame_index: int
    positive_points_xy: tuple[tuple[float, float], ...]
    negative_points_xy: tuple[tuple[float, float], ...]
    box_xywh: tuple[float, float, float, float]
    coordinate_domain: str = "RECTIFIED_PHYSICAL_LEFT_PIXELS"
    source: str = "HAWOR_PROJECTED_JOINTS"


def _valid(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        np.isfinite(points).all(axis=-1)
        & (points[..., 0] >= 0) & (points[..., 0] < width)
        & (points[..., 1] >= 0) & (points[..., 1] < height)
    )


def validate_hawor(joints_2d: np.ndarray, observed: np.ndarray) -> None:
    if joints_2d.ndim != 4 or joints_2d.shape[0] != 2 or joints_2d.shape[2:] != (21, 2):
        raise Sam31PromptContractError("HaWoR joints_2d must have shape (2,N,21,2)")
    if observed.shape != joints_2d.shape[:2] or observed.dtype != np.bool_:
        raise Sam31PromptContractError("HaWoR observed must be bool shape (2,N)")


def select_bilateral_anchor(
    joints_2d: np.ndarray, observed: np.ndarray, *, width: int, height: int,
) -> int:
    validate_hawor(joints_2d, observed)
    valid = _valid(joints_2d, width, height)
    candidates = []
    for frame in range(joints_2d.shape[1]):
        if not bool(observed[:, frame].all()):
            continue
        per_side = valid[:, frame].sum(axis=1)
        palm_per_side = valid[:, frame, list(PALM_JOINTS)].sum(axis=1)
        if int(palm_per_side.min()) < 3:
            continue
        candidates.append((int(per_side.min()), int(per_side.sum()), -frame, frame))
    if not candidates:
        raise Sam31PromptContractError(
            "no frame has bilateral observed HaWoR with at least three valid palm joints per side"
        )
    return max(candidates)[-1]


def _box(points: np.ndarray, width: int, height: int) -> tuple[float, float, float, float]:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, np.asarray([8.0, 8.0]))
    padding = 0.15 * span
    x0, y0 = np.maximum(minimum - padding, 0.0)
    x1, y1 = np.minimum(maximum + padding, [width - 1.0, height - 1.0])
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def hand_spatial_prompt(
    joints_2d: np.ndarray, observed: np.ndarray, *, side: str,
    frame_index: int, width: int, height: int,
) -> SpatialPrompt:
    validate_hawor(joints_2d, observed)
    if side not in SIDE_NAMES:
        raise Sam31PromptContractError(f"unknown anatomical side: {side}")
    side_index = SIDE_NAMES.index(side)
    other_index = 1 - side_index
    if not bool(observed[side_index, frame_index]):
        raise Sam31PromptContractError(f"{side} is unobserved at prompt frame")
    own = joints_2d[side_index, frame_index]
    other = joints_2d[other_index, frame_index]
    own_valid = _valid(own, width, height)
    other_valid = _valid(other, width, height)
    positives = own[list(PALM_JOINTS)][own_valid[list(PALM_JOINTS)]]
    negatives = other[list(PALM_JOINTS)][other_valid[list(PALM_JOINTS)]]
    all_own = own[own_valid]
    if len(positives) < 3 or len(all_own) < 5:
        raise Sam31PromptContractError(f"insufficient valid {side} HaWoR prompt geometry")
    return SpatialPrompt(
        role=f"{side}_human_skin_forearm",
        frame_index=frame_index,
        positive_points_xy=tuple(tuple(map(float, point)) for point in positives),
        negative_points_xy=tuple(tuple(map(float, point)) for point in negatives),
        box_xywh=_box(all_own, width, height),
    )


def build_prompt_plan(
    joints_2d: np.ndarray, observed: np.ndarray, *, task: str,
    width: int = 1280, height: int = 960,
) -> dict[str, Any]:
    if task not in TASK_TEXT_PROMPTS:
        raise Sam31PromptContractError(f"unsupported 0915 task: {task}")
    anchor = select_bilateral_anchor(
        joints_2d, observed, width=width, height=height,
    )
    hands = [
        hand_spatial_prompt(
            joints_2d, observed, side=side, frame_index=anchor,
            width=width, height=height,
        )
        for side in SIDE_NAMES
    ]
    return {
        "schema_version": "sam31-0915-prompt-plan-v2",
        "model_identity": MODEL_IDENTITY,
        "task": task,
        "anchor_frame": anchor,
        "image_domain": {
            "width": width, "height": height,
            "identity": "PHYSICAL_LEFT_SOURCE_INDEX_1_EQUIDIS62_TO_PINHOLE_FOV90",
        },
        "hand_spatial_prompts": [prompt.__dict__ for prompt in hands],
        "task_object_text_prompts": list(TASK_TEXT_PROMPTS[task]),
        "auxiliary_text_prompts": dict(AUXILIARY_TEXT_PROMPTS),
        "tracker_role_created": False,
        "controller_role_created": False,
        "pico26_consumed": False,
        "identity_policy": {
            "task_objects": "SEPARATE_VISIBLE_PHYSICAL_OBJECT",
            "after_exit": "UNKNOWN_UNTIL_SAM31_MEMORY_OR_NEW_VISUAL_EVIDENCE",
            "union_mask_for_identity": "FORBIDDEN",
        },
    }
