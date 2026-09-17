"""Official-box SAM3.1 candidate prompt geometry and pixel selector.

Prompt boxes are computed from authoritative ego joints, projected wrist width,
and the closest image boundary.  Boxes can only prompt the official model.  The
selector unions whole returned SAM instances and never rasterizes, fills, crops,
or clips output using prompt geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PRODUCER_VERSION = "producer_p3_geometric_box"
BOX_PROMPT_ROUTE = "official_sam31_positive_limb_boxes"
OBJECT_PROMPT = "a beverage can"
SIDE_NAMES = ("left", "right")


class ProducerP3ContractError(RuntimeError):
    """Raised when prompt, input, or pixel-lineage constraints drift."""


@dataclass(frozen=True)
class ProducerP3Config:
    box_padding_wrist_width: float = 1.25
    maximum_prompt_box_area_ratio: float = 0.65
    hand_seed_radius_hand_scale: float = 0.055
    wrist_seed_radius_hand_scale: float = 0.12
    boundary_radius_hand_scale: float = 0.30
    minimum_hand_joint_support_ratio: float = 0.24
    minimum_final_joint_support_ratio: float = 0.20
    maximum_instance_area_ratio: float = 0.45
    maximum_final_area_ratio: float = 0.55
    minimum_instance_score: float = 0.0
    minimum_object_cad_iou: float = 0.05
    minimum_object_cad_coverage: float = 0.18
    minimum_side_temporal_iou: float = 0.45


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
            raise ProducerP3ContractError(
                f"SAM masks must have shape [N,{height},{width}], got {masks.shape}"
            )
        if scores.shape != (masks.shape[0],):
            raise ProducerP3ContractError("SAM scores do not align with masks")
        if instance_ids.shape != (masks.shape[0],):
            raise ProducerP3ContractError("SAM instance IDs do not align with masks")
        if not np.isfinite(scores).all():
            raise ProducerP3ContractError("SAM scores must be finite")
        if len(set(instance_ids.tolist())) != len(instance_ids):
            raise ProducerP3ContractError("SAM instance IDs must be unique")
        return SamInstances(masks.astype(bool), scores, instance_ids)


@dataclass(frozen=True)
class BoxPromptResult:
    boxes_xywh: np.ndarray
    box_labels: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SelectionResult:
    final_mask: np.ndarray
    side_masks_before_object_protect: dict[str, np.ndarray]
    human_union_before_object_protect: np.ndarray
    object_protect_mask: np.ndarray
    selected_ids: dict[str, int]
    diagnostics: dict[str, Any]
    hold_reasons: tuple[str, ...]

    @property
    def sufficient(self) -> bool:
        return not self.hold_reasons


def _validate_joints(joints: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ProducerP3ContractError("ego joints must be finite [21,2]")
    if width <= 1 or height <= 1:
        raise ProducerP3ContractError("image dimensions must exceed one pixel")
    if (
        (points[:, 0] < 0).any()
        or (points[:, 0] >= width).any()
        or (points[:, 1] < 0).any()
        or (points[:, 1] >= height).any()
    ):
        raise ProducerP3ContractError("ego joints lie outside the image")
    return points


def hand_scale_px(joints: np.ndarray) -> float:
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ProducerP3ContractError("ego joints must be finite [21,2]")
    distances = np.linalg.norm(points[[4, 8, 12, 16, 20]] - points[0], axis=1)
    scale = float(np.median(distances))
    if not np.isfinite(scale) or scale <= 1:
        raise ProducerP3ContractError("invalid projected hand scale")
    return scale


def wrist_width_px(joints: np.ndarray) -> float:
    points = np.asarray(joints, dtype=np.float64)
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise ProducerP3ContractError("ego joints must be finite [21,2]")
    width = float(np.linalg.norm(points[5] - points[17]))
    if not np.isfinite(width) or width <= 1:
        raise ProducerP3ContractError("invalid projected wrist width")
    return width


def closest_boundary_point(
    wrist_xy: np.ndarray, width: int, height: int
) -> tuple[float, float, str]:
    x, y = np.asarray(wrist_xy, dtype=np.float64)
    if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
        raise ProducerP3ContractError("wrist lies outside the image")
    candidates = (
        (x, 0.0, y, "left"),
        (width - 1.0 - x, width - 1.0, y, "right"),
        (y, x, 0.0, "top"),
        (height - 1.0 - y, x, height - 1.0, "bottom"),
    )
    _, bx, by, edge = min(candidates, key=lambda item: item[0])
    return float(bx), float(by), edge


def derive_limb_box(
    joints: np.ndarray,
    width: int,
    height: int,
    *,
    config: ProducerP3Config = ProducerP3Config(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derive one normalized prompt box; this function creates no mask pixels."""
    points = _validate_joints(joints, width, height)
    wrist_width = wrist_width_px(points)
    boundary_x, boundary_y, boundary_edge = closest_boundary_point(
        points[0], width, height
    )
    evidence = np.concatenate(
        [points, np.asarray([[boundary_x, boundary_y]], dtype=np.float64)], axis=0
    )
    padding = config.box_padding_wrist_width * wrist_width
    x0 = max(float(evidence[:, 0].min() - padding), 0.0)
    y0 = max(float(evidence[:, 1].min() - padding), 0.0)
    x1 = min(float(evidence[:, 0].max() + padding), float(width))
    y1 = min(float(evidence[:, 1].max() + padding), float(height))
    if x1 <= x0 or y1 <= y0:
        raise ProducerP3ContractError("derived prompt box is empty")
    normalized = np.asarray(
        [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
        dtype=np.float64,
    )
    area_ratio = float(normalized[2] * normalized[3])
    if area_ratio > config.maximum_prompt_box_area_ratio:
        raise ProducerP3ContractError(
            "derived prompt box exceeds the fixed dimensionless area bound"
        )
    if (
        (normalized < 0).any()
        or normalized[0] + normalized[2] > 1 + 1e-12
        or normalized[1] + normalized[3] > 1 + 1e-12
    ):
        raise ProducerP3ContractError("derived prompt box is outside normalized bounds")
    return normalized, {
        "wrist_width_pixels": wrist_width,
        "padding_pixels": padding,
        "padding_wrist_width_ratio": config.box_padding_wrist_width,
        "closest_boundary_point_xy": [boundary_x, boundary_y],
        "closest_boundary_edge": boundary_edge,
        "box_xywh_normalized": normalized.tolist(),
        "box_area_ratio": area_ratio,
        "geometry_created_human_pixels": 0,
        "hard_crop_applied": False,
    }


def derive_two_limb_boxes(
    joints_by_side: np.ndarray,
    width: int,
    height: int,
    *,
    config: ProducerP3Config = ProducerP3Config(),
) -> BoxPromptResult:
    joints = np.asarray(joints_by_side, dtype=np.float64)
    if joints.shape != (2, 21, 2) or not np.isfinite(joints).all():
        raise ProducerP3ContractError("joints_by_side must be finite [2,21,2]")
    boxes = []
    diagnostics = {}
    for side_index, side_name in enumerate(SIDE_NAMES):
        box, info = derive_limb_box(
            joints[side_index], width, height, config=config
        )
        boxes.append(box)
        diagnostics[side_name] = info
    box_array = np.stack(boxes)
    return BoxPromptResult(
        boxes_xywh=box_array,
        box_labels=np.asarray([1, 1], dtype=np.int64),
        diagnostics={
            "prompt_type": BOX_PROMPT_ROUTE,
            "sides": diagnostics,
            "box_count": 2,
            "box_labels": [1, 1],
            "geometry_created_human_pixels": 0,
            "hard_crop_applied": False,
        },
    )


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
        raise ProducerP3ContractError("IoU masks must have equal shape")
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return float(np.logical_and(a, b).sum() / union)


def _deduplicate(reasons: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(reasons))


def select_frame_candidate(
    *,
    box_prompt_instances: SamInstances,
    object_prompt_instances: SamInstances,
    prompt_boxes: BoxPromptResult,
    joints_by_side: np.ndarray,
    cad_object_mask: np.ndarray,
    previous_side_masks: Mapping[str, np.ndarray] | None = None,
    config: ProducerP3Config = ProducerP3Config(),
) -> SelectionResult:
    """Select whole SAM instances and prove prompt geometry adds zero pixels."""
    joints = np.asarray(joints_by_side, dtype=np.float64)
    if joints.shape != (2, 21, 2) or not np.isfinite(joints).all():
        raise ProducerP3ContractError("joints_by_side must be finite [2,21,2]")
    cad = np.asarray(cad_object_mask, dtype=bool)
    if cad.ndim != 2 or not cad.any():
        raise ProducerP3ContractError("cad_object_mask must be nonempty 2D")
    height, width = cad.shape
    box_instances = box_prompt_instances.validated(height, width)
    object_instances = object_prompt_instances.validated(height, width)
    if np.asarray(prompt_boxes.boxes_xywh).shape != (2, 4):
        raise ProducerP3ContractError("prompt boxes must have shape [2,4]")
    if np.asarray(prompt_boxes.box_labels).tolist() != [1, 1]:
        raise ProducerP3ContractError("prompt labels must remain [1,1]")

    scales = [hand_scale_px(joints[index]) for index in range(2)]
    instance_diagnostics = []
    eligible: dict[str, list[tuple[float, int]]] = {name: [] for name in SIDE_NAMES}
    for offset, (mask, score, instance_id) in enumerate(
        zip(
            box_instances.masks,
            box_instances.scores,
            box_instances.instance_ids,
            strict=True,
        )
    ):
        info: dict[str, Any] = {
            "instance_id": int(instance_id),
            "score": float(score),
            "area_pixels": int(mask.sum()),
            "area_ratio": float(mask.mean()),
            "eligible_sides": [],
            "sides": {},
        }
        for side_index, side_name in enumerate(SIDE_NAMES):
            scale = scales[side_index]
            side_joints = joints[side_index]
            boundary = closest_boundary_point(side_joints[0], width, height)[:2]
            support = {
                "joint_support_ratio": joint_support_ratio(
                    mask,
                    side_joints,
                    config.hand_seed_radius_hand_scale * scale,
                ),
                "wrist_supported": point_supported(
                    mask,
                    side_joints[0],
                    config.wrist_seed_radius_hand_scale * scale,
                ),
                "boundary_supported": point_supported(
                    mask,
                    np.asarray(boundary),
                    config.boundary_radius_hand_scale * scale,
                ),
                "boundary_point_xy": list(boundary),
            }
            info["sides"][side_name] = support
            accepted = (
                score >= config.minimum_instance_score
                and info["area_ratio"] <= config.maximum_instance_area_ratio
                and support["joint_support_ratio"]
                >= config.minimum_hand_joint_support_ratio
                and support["wrist_supported"]
                and support["boundary_supported"]
            )
            if accepted:
                info["eligible_sides"].append(side_name)
                rank = (
                    support["joint_support_ratio"]
                    + float(support["wrist_supported"])
                    + float(support["boundary_supported"])
                    + 0.05 * float(score)
                )
                eligible[side_name].append((rank, offset))
        instance_diagnostics.append(info)

    selected_offsets: dict[str, int] = {}
    hold_reasons: list[str] = []
    for side_name in SIDE_NAMES:
        ranked = sorted(eligible[side_name], reverse=True)
        for _, offset in ranked:
            if len(instance_diagnostics[offset]["eligible_sides"]) == 1:
                selected_offsets[side_name] = offset
                break
        if side_name not in selected_offsets:
            hold_reasons.append(f"missing_box_prompt_complete_limb_instance:{side_name}")

    side_masks = {
        side: box_instances.masks[offset].copy()
        for side, offset in selected_offsets.items()
    }
    human_union = np.zeros((height, width), dtype=bool)
    for mask in side_masks.values():
        human_union |= mask

    object_diagnostics = []
    object_ranked: list[tuple[float, int]] = []
    cad_pixels = int(cad.sum())
    for offset, (mask, score, instance_id) in enumerate(
        zip(
            object_instances.masks,
            object_instances.scores,
            object_instances.instance_ids,
            strict=True,
        )
    ):
        intersection = int(np.logical_and(mask, cad).sum())
        union = int(np.logical_or(mask, cad).sum())
        iou = intersection / union if union else 0.0
        coverage = intersection / cad_pixels
        object_diagnostics.append(
            {
                "instance_id": int(instance_id),
                "score": float(score),
                "area_pixels": int(mask.sum()),
                "cad_intersection_pixels": intersection,
                "cad_iou": float(iou),
                "cad_coverage": float(coverage),
            }
        )
        if (
            score >= config.minimum_instance_score
            and iou >= config.minimum_object_cad_iou
            and coverage >= config.minimum_object_cad_coverage
        ):
            object_ranked.append((iou + coverage + 0.05 * float(score), offset))
    object_offset = max(object_ranked)[1] if object_ranked else None
    if object_offset is None:
        object_mask = np.zeros((height, width), dtype=bool)
        hold_reasons.append("missing_object6d_selected_visible_object_instance")
    else:
        object_mask = object_instances.masks[object_offset].copy()

    pre_overlap = int(np.logical_and(human_union, object_mask).sum())
    final = np.logical_and(human_union, ~object_mask)
    post_overlap = int(np.logical_and(final, object_mask).sum())

    final_support: dict[str, Any] = {}
    temporal_iou: dict[str, float | None] = {}
    for side_index, side_name in enumerate(SIDE_NAMES):
        scale = scales[side_index]
        boundary = closest_boundary_point(joints[side_index, 0], width, height)[:2]
        support = {
            "joint_support_ratio": joint_support_ratio(
                final,
                joints[side_index],
                config.hand_seed_radius_hand_scale * scale,
            ),
            "wrist_supported": point_supported(
                final,
                joints[side_index, 0],
                config.wrist_seed_radius_hand_scale * scale,
            ),
            "boundary_supported": point_supported(
                final,
                np.asarray(boundary),
                config.boundary_radius_hand_scale * scale,
            ),
        }
        final_support[side_name] = support
        if (
            support["joint_support_ratio"] < config.minimum_final_joint_support_ratio
            or not support["wrist_supported"]
            or not support["boundary_supported"]
        ):
            hold_reasons.append(f"final_complete_limb_support_missing:{side_name}")
        previous = None if previous_side_masks is None else previous_side_masks.get(side_name)
        current = side_masks.get(side_name)
        temporal_iou[side_name] = (
            None if previous is None or current is None else binary_iou(previous, current)
        )
        if (
            temporal_iou[side_name] is not None
            and temporal_iou[side_name] < config.minimum_side_temporal_iou
        ):
            hold_reasons.append(f"side_temporal_iou_below_limit:{side_name}")

    if final.mean() > config.maximum_final_area_ratio:
        hold_reasons.append("final_area_ratio_above_dimensionless_limit")
    all_box_pixels = (
        np.logical_or.reduce(box_instances.masks, axis=0)
        if len(box_instances.masks)
        else np.zeros((height, width), dtype=bool)
    )
    if np.logical_and(final, ~all_box_pixels).any():
        raise ProducerP3ContractError("final human pixels escape SAM box output")
    if post_overlap:
        raise ProducerP3ContractError("object protection failed to subtract overlap")

    selected_ids = {
        side: int(box_instances.instance_ids[offset])
        for side, offset in selected_offsets.items()
    }
    return SelectionResult(
        final_mask=final,
        side_masks_before_object_protect=side_masks,
        human_union_before_object_protect=human_union,
        object_protect_mask=object_mask,
        selected_ids=selected_ids,
        diagnostics={
            "producer_version": PRODUCER_VERSION,
            "box_prompt_route": BOX_PROMPT_ROUTE,
            "prompt_boxes": prompt_boxes.diagnostics,
            "box_instance_diagnostics": instance_diagnostics,
            "selected_ids": selected_ids,
            "object_prompt": OBJECT_PROMPT,
            "object_prompt_diagnostics": object_diagnostics,
            "selected_object_instance_id": (
                None
                if object_offset is None
                else int(object_instances.instance_ids[object_offset])
            ),
            "pre_object_overlap_pixels": pre_overlap,
            "post_object_overlap_pixels": post_overlap,
            "final_pixels": int(final.sum()),
            "final_area_ratio": float(final.mean()),
            "final_support": final_support,
            "side_temporal_iou": temporal_iou,
            "pixel_lineage": {
                "final_is_subset_of_current_frame_box_prompt_sam_pixels": True,
                "prompt_box_created_human_pixels": 0,
                "geometry_created_human_pixels": 0,
                "hard_crop_applied": False,
                "flow_warp_created_output_pixels": 0,
                "p2_pixels_inherited": 0,
                "post_object_overlap_zero": post_overlap == 0,
            },
        },
        hold_reasons=_deduplicate(hold_reasons),
    )

