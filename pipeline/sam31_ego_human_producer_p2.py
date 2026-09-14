"""Fail-closed SAM3.1 ego hand/forearm/sleeve candidate selector.

This module never invokes a model and never creates human pixels. It selects
whole SAM instances using projected ego-hand seeds, image-boundary evidence,
identity continuity, and temporal diagnostics. Analytic Object6D geometry may
select a visible task-object SAM instance, which may only subtract pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw


PRODUCER_VERSION = "producer_p2"
NOT_A_V9_HEURISTIC_PATCH = True
HUMAN_PROMPTS = ("a hand", "a human forearm", "a sleeve")
OBJECT_PROMPT = "a beverage can"


class ProducerP2ContractError(RuntimeError):
    """Raised when an input or pixel-lineage contract is violated."""


@dataclass(frozen=True)
class ProducerP2Config:
    hand_seed_radius_hand_scale: float = 0.055
    wrist_seed_radius_hand_scale: float = 0.12
    boundary_radius_hand_scale: float = 0.30
    minimum_hand_joint_support_ratio: float = 0.24
    minimum_final_joint_support_ratio: float = 0.20
    maximum_instance_area_ratio: float = 0.30
    maximum_final_area_ratio: float = 0.38
    minimum_object_cad_iou: float = 0.05
    minimum_object_cad_coverage: float = 0.18
    minimum_temporal_iou: float = 0.55
    minimum_instance_score: float = 0.0


@dataclass(frozen=True)
class SamInstances:
    masks: np.ndarray
    scores: np.ndarray
    instance_ids: np.ndarray

    def validated(self, height: int, width: int) -> "SamInstances":
        masks = np.asarray(self.masks)
        scores = np.asarray(self.scores, dtype=np.float64)
        instance_ids = np.asarray(self.instance_ids)
        if masks.ndim != 3 or masks.shape[1:] != (height, width):
            raise ProducerP2ContractError(
                f"SAM masks must have shape [N,{height},{width}], got {masks.shape}"
            )
        if scores.shape != (masks.shape[0],):
            raise ProducerP2ContractError("SAM scores do not align with masks")
        if instance_ids.shape != (masks.shape[0],):
            raise ProducerP2ContractError("SAM instance IDs do not align with masks")
        if not np.isfinite(scores).all():
            raise ProducerP2ContractError("SAM scores must be finite")
        if len(set(instance_ids.tolist())) != len(instance_ids):
            raise ProducerP2ContractError("SAM instance IDs must be unique per prompt")
        return SamInstances(masks.astype(bool), scores, instance_ids)


@dataclass(frozen=True)
class SelectionResult:
    final_mask: np.ndarray
    human_union_before_object_protect: np.ndarray
    object_protect_mask: np.ndarray
    selected_ids: dict[str, dict[str, int]]
    diagnostics: dict[str, Any]
    hold_reasons: tuple[str, ...]

    @property
    def sufficient(self) -> bool:
        return not self.hold_reasons


def hand_scale_px(joints: np.ndarray) -> float:
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ProducerP2ContractError("each ego seed set must be finite [21,2]")
    fingertips = points[[4, 8, 12, 16, 20]]
    distances = np.linalg.norm(fingertips - points[0], axis=1)
    scale = float(np.median(distances))
    if not np.isfinite(scale) or scale <= 1:
        raise ProducerP2ContractError("invalid hand scale")
    return scale


def closest_boundary_point(wrist_xy: np.ndarray, width: int, height: int) -> tuple[float, float]:
    x, y = np.asarray(wrist_xy, dtype=np.float64)
    if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
        raise ProducerP2ContractError("wrist seed lies outside the image")
    candidates = (
        (x, 0.0, y),
        (width - 1.0 - x, width - 1.0, y),
        (y, x, 0.0),
        (height - 1.0 - y, x, height - 1.0),
    )
    _, boundary_x, boundary_y = min(candidates, key=lambda item: item[0])
    return float(boundary_x), float(boundary_y)


def _disk_values(mask: np.ndarray, point_xy: np.ndarray, radius: float) -> np.ndarray:
    height, width = mask.shape
    x, y = np.asarray(point_xy, dtype=np.float64)
    radius = max(float(radius), 1.0)
    x0 = max(int(np.floor(x - radius)), 0)
    x1 = min(int(np.ceil(x + radius)) + 1, width)
    y0 = max(int(np.floor(y - radius)), 0)
    y1 = min(int(np.ceil(y + radius)) + 1, height)
    if x0 >= x1 or y0 >= y1:
        return np.zeros(0, dtype=bool)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    return mask[y0:y1, x0:x1][disk]


def point_supported(mask: np.ndarray, point_xy: np.ndarray, radius: float) -> bool:
    values = _disk_values(mask, point_xy, radius)
    return bool(values.size and values.any())


def joint_support_ratio(mask: np.ndarray, joints: np.ndarray, radius: float) -> float:
    return float(
        np.mean([point_supported(mask, point, radius) for point in np.asarray(joints)])
    )


def binary_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ProducerP2ContractError("IoU masks must have equal shape")
    union = np.logical_or(a, b).sum()
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set(map(tuple, np.asarray(points, dtype=np.float64).tolist())))
    if len(unique) < 3:
        raise ProducerP2ContractError("projected cylinder hull has fewer than 3 points")

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def project_cylinder_mask(
    transform_object_to_camera: np.ndarray,
    intrinsics_fx_fy_cx_cy: np.ndarray,
    radius_m: float,
    height_m: float,
    width: int,
    height: int,
    *,
    segments: int = 96,
) -> np.ndarray:
    """Project an analytic Y-axis cylinder for object-instance selection only."""
    transform = np.asarray(transform_object_to_camera, dtype=np.float64)
    intrinsics = np.asarray(intrinsics_fx_fy_cx_cy, dtype=np.float64)
    if transform.shape != (4, 4) or intrinsics.shape != (4,):
        raise ProducerP2ContractError("invalid Object6D projection inputs")
    if not np.isfinite(transform).all() or not np.isfinite(intrinsics).all():
        raise ProducerP2ContractError("Object6D projection inputs must be finite")
    if radius_m <= 0 or height_m <= 0 or segments < 16:
        raise ProducerP2ContractError("invalid cylinder dimensions or sampling")
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = []
    for y in (-height_m / 2, height_m / 2):
        rings.append(
            np.stack(
                [radius_m * np.cos(theta), np.full_like(theta, y), radius_m * np.sin(theta)],
                axis=1,
            )
        )
    points = np.concatenate(rings, axis=0)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    camera = (transform @ homogeneous.T).T[:, :3]
    if np.any(camera[:, 2] <= 1e-5):
        raise ProducerP2ContractError("projected cylinder crosses/behind camera plane")
    fx, fy, cx, cy = intrinsics
    projected = np.stack(
        [fx * camera[:, 0] / camera[:, 2] + cx, fy * camera[:, 1] / camera[:, 2] + cy],
        axis=1,
    )
    hull = _convex_hull(projected)
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in hull], fill=1)
    mask = np.asarray(canvas, dtype=bool)
    if not mask.any():
        raise ProducerP2ContractError("projected cylinder is empty")
    return mask


def _instance_diagnostics(
    mask: np.ndarray,
    score: float,
    joints_by_side: np.ndarray,
    scales: tuple[float, float],
    config: ProducerP2Config,
) -> dict[str, Any]:
    height, width = mask.shape
    sides = {}
    for side_index, side_name in enumerate(("left", "right")):
        scale = scales[side_index]
        joints = joints_by_side[side_index]
        boundary = closest_boundary_point(joints[0], width, height)
        sides[side_name] = {
            "joint_support_ratio": joint_support_ratio(
                mask, joints, config.hand_seed_radius_hand_scale * scale
            ),
            "wrist_supported": point_supported(
                mask, joints[0], config.wrist_seed_radius_hand_scale * scale
            ),
            "boundary_supported": point_supported(
                mask, np.asarray(boundary), config.boundary_radius_hand_scale * scale
            ),
            "boundary_point_xy": list(boundary),
        }
    return {
        "score": float(score),
        "area_pixels": int(mask.sum()),
        "area_ratio": float(mask.mean()),
        "sides": sides,
    }


def _choose_human_instances(
    prompt: str,
    instances: SamInstances,
    joints_by_side: np.ndarray,
    config: ProducerP2Config,
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    height, width = instances.masks.shape[1:]
    scales = (hand_scale_px(joints_by_side[0]), hand_scale_px(joints_by_side[1]))
    diagnostics = []
    eligible: dict[str, list[tuple[float, int]]] = {"left": [], "right": []}
    for offset, (mask, score, instance_id) in enumerate(
        zip(instances.masks, instances.scores, instances.instance_ids, strict=True)
    ):
        info = _instance_diagnostics(mask, float(score), joints_by_side, scales, config)
        info.update({"instance_id": int(instance_id), "eligible_sides": []})
        if info["area_ratio"] <= config.maximum_instance_area_ratio and score >= config.minimum_instance_score:
            for side_name in ("left", "right"):
                side = info["sides"][side_name]
                if prompt == "a hand":
                    accepted = (
                        side["joint_support_ratio"]
                        >= config.minimum_hand_joint_support_ratio
                    )
                    rank = side["joint_support_ratio"] + 0.15 * float(side["wrist_supported"])
                else:
                    accepted = side["wrist_supported"] and side["boundary_supported"]
                    rank = (
                        0.5 * side["joint_support_ratio"]
                        + 0.75 * float(side["wrist_supported"])
                        + 1.0 * float(side["boundary_supported"])
                    )
                if accepted:
                    info["eligible_sides"].append(side_name)
                    eligible[side_name].append((rank + 0.05 * float(score), offset))
        diagnostics.append(info)

    selected: dict[str, int] = {}
    for side_name in ("left", "right"):
        ranked = sorted(eligible[side_name], reverse=True)
        for _, offset in ranked:
            info = diagnostics[offset]
            if len(info["eligible_sides"]) > 1:
                continue
            selected[side_name] = offset
            break
    return selected, diagnostics


def _choose_object_instance(
    instances: SamInstances,
    cad_mask: np.ndarray,
    config: ProducerP2Config,
) -> tuple[int | None, list[dict[str, Any]]]:
    cad_pixels = int(cad_mask.sum())
    if cad_pixels <= 0:
        raise ProducerP2ContractError("Object6D CAD mask must be nonempty")
    diagnostics = []
    ranked = []
    for offset, (mask, score, instance_id) in enumerate(
        zip(instances.masks, instances.scores, instances.instance_ids, strict=True)
    ):
        intersection = int(np.logical_and(mask, cad_mask).sum())
        union = int(np.logical_or(mask, cad_mask).sum())
        iou = intersection / union if union else 0.0
        cad_coverage = intersection / cad_pixels
        info = {
            "instance_id": int(instance_id),
            "score": float(score),
            "area_pixels": int(mask.sum()),
            "cad_intersection_pixels": intersection,
            "cad_iou": float(iou),
            "cad_coverage": float(cad_coverage),
        }
        diagnostics.append(info)
        if (
            score >= config.minimum_instance_score
            and iou >= config.minimum_object_cad_iou
            and cad_coverage >= config.minimum_object_cad_coverage
        ):
            ranked.append((iou + cad_coverage + 0.05 * float(score), offset))
    return (max(ranked)[1] if ranked else None), diagnostics


def select_frame_candidate(
    *,
    prompt_instances: Mapping[str, SamInstances],
    joints_by_side: np.ndarray,
    cad_object_mask: np.ndarray,
    previous_final_mask: np.ndarray | None = None,
    previous_selected_ids: Mapping[str, Mapping[str, int]] | None = None,
    config: ProducerP2Config = ProducerP2Config(),
) -> SelectionResult:
    """Select a candidate while proving that no non-SAM human pixel is added."""
    if set(prompt_instances) != set(HUMAN_PROMPTS + (OBJECT_PROMPT,)):
        raise ProducerP2ContractError("prompt set must exactly match the frozen prompts")
    joints = np.asarray(joints_by_side, dtype=np.float64)
    if joints.shape != (2, 21, 2) or not np.isfinite(joints).all():
        raise ProducerP2ContractError("joints_by_side must be finite [2,21,2]")
    cad = np.asarray(cad_object_mask, dtype=bool)
    if cad.ndim != 2:
        raise ProducerP2ContractError("cad_object_mask must be 2D")
    height, width = cad.shape
    validated = {
        prompt: instances.validated(height, width)
        for prompt, instances in prompt_instances.items()
    }
    human_union = np.zeros((height, width), dtype=bool)
    all_human_sam_pixels = np.zeros_like(human_union)
    selected_ids: dict[str, dict[str, int]] = {}
    prompt_diagnostics: dict[str, Any] = {}
    hold_reasons = []
    side_has_hand = {"left": False, "right": False}
    side_has_boundary_arm = {"left": False, "right": False}

    for prompt in HUMAN_PROMPTS:
        instances = validated[prompt]
        if instances.masks.shape[0]:
            all_human_sam_pixels |= np.any(instances.masks, axis=0)
        selected, diagnostics = _choose_human_instances(
            prompt, instances, joints, config
        )
        selected_ids[prompt] = {}
        for side_name, offset in selected.items():
            mask = instances.masks[offset]
            human_union |= mask
            selected_ids[prompt][side_name] = int(instances.instance_ids[offset])
            if prompt == "a hand":
                side_has_hand[side_name] = True
            else:
                side_has_boundary_arm[side_name] = True
        prompt_diagnostics[prompt] = {
            "instances": diagnostics,
            "selected_ids_by_side": selected_ids[prompt],
        }

    for side_name in ("left", "right"):
        if not side_has_hand[side_name]:
            hold_reasons.append(f"missing_seeded_hand_instance:{side_name}")
        if not side_has_boundary_arm[side_name]:
            hold_reasons.append(f"missing_seeded_boundary_arm_or_sleeve:{side_name}")

    object_instances = validated[OBJECT_PROMPT]
    object_offset, object_diagnostics = _choose_object_instance(
        object_instances, cad, config
    )
    if object_offset is None:
        object_mask = np.zeros_like(human_union)
        hold_reasons.append("missing_object6d_selected_visible_object_instance")
        selected_ids[OBJECT_PROMPT] = {}
    else:
        object_mask = object_instances.masks[object_offset]
        selected_ids[OBJECT_PROMPT] = {
            "task_object": int(object_instances.instance_ids[object_offset])
        }
    pre_overlap = int(np.logical_and(human_union, object_mask).sum())
    final_mask = np.logical_and(human_union, np.logical_not(object_mask))
    post_overlap = int(np.logical_and(final_mask, object_mask).sum())

    scales = (hand_scale_px(joints[0]), hand_scale_px(joints[1]))
    final_joint_support = {}
    final_boundary_support = {}
    for side_index, side_name in enumerate(("left", "right")):
        scale = scales[side_index]
        final_joint_support[side_name] = joint_support_ratio(
            final_mask,
            joints[side_index],
            config.hand_seed_radius_hand_scale * scale,
        )
        boundary = closest_boundary_point(joints[side_index, 0], width, height)
        final_boundary_support[side_name] = point_supported(
            final_mask,
            np.asarray(boundary),
            config.boundary_radius_hand_scale * scale,
        )
        if final_joint_support[side_name] < config.minimum_final_joint_support_ratio:
            hold_reasons.append(f"final_seed_support_insufficient:{side_name}")
        if not final_boundary_support[side_name]:
            hold_reasons.append(f"final_boundary_support_missing:{side_name}")

    if final_mask.mean() > config.maximum_final_area_ratio:
        hold_reasons.append("final_area_ratio_exceeds_dimensionless_limit")
    temporal_iou = None
    if previous_final_mask is not None:
        temporal_iou = binary_iou(previous_final_mask, final_mask)
        if temporal_iou is None or temporal_iou < config.minimum_temporal_iou:
            hold_reasons.append("temporal_iou_below_dimensionless_limit")
    identity_changes = []
    if previous_selected_ids is not None:
        for prompt, by_role in selected_ids.items():
            previous_by_role = previous_selected_ids.get(prompt, {})
            for role, instance_id in by_role.items():
                if role in previous_by_role and previous_by_role[role] != instance_id:
                    identity_changes.append(
                        {
                            "prompt": prompt,
                            "role": role,
                            "previous": int(previous_by_role[role]),
                            "current": int(instance_id),
                        }
                    )
        if identity_changes:
            hold_reasons.append("same_prompt_identity_changed")

    pixel_subset_ok = not np.logical_and(final_mask, ~all_human_sam_pixels).any()
    object_overlap_zero = post_overlap == 0
    if not pixel_subset_ok:
        raise ProducerP2ContractError("human pixel-lineage invariant violated")
    if not object_overlap_zero:
        raise ProducerP2ContractError("object protect subtraction invariant violated")

    diagnostics = {
        "producer_version": PRODUCER_VERSION,
        "not_a_v9_heuristic_patch": True,
        "fixed_prompts": {
            "human": list(HUMAN_PROMPTS),
            "object": OBJECT_PROMPT,
        },
        "prompt_diagnostics": prompt_diagnostics,
        "object_prompt_diagnostics": object_diagnostics,
        "human_union_pixels_before_object_protect": int(human_union.sum()),
        "object_protect_pixels": int(object_mask.sum()),
        "pre_object_overlap_pixels": pre_overlap,
        "post_object_overlap_pixels": post_overlap,
        "final_pixels": int(final_mask.sum()),
        "final_area_ratio": float(final_mask.mean()),
        "final_joint_support_ratio": final_joint_support,
        "final_boundary_support": final_boundary_support,
        "temporal_iou": temporal_iou,
        "identity_changes": identity_changes,
        "pixel_lineage": {
            "final_is_subset_of_current_or_propagated_human_sam_pixels": pixel_subset_ok,
            "geometry_created_human_pixels": False,
            "flow_warp_created_output_pixels": False,
            "post_object_overlap_zero": object_overlap_zero,
        },
    }
    return SelectionResult(
        final_mask=final_mask,
        human_union_before_object_protect=human_union,
        object_protect_mask=object_mask,
        selected_ids=selected_ids,
        diagnostics=diagnostics,
        hold_reasons=tuple(sorted(set(hold_reasons))),
    )
