"""Evidence-bound SAM reprompt for short forearm-to-crop gaps."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def propose_boundary_reprompt(
    component: np.ndarray,
    wrist_width: float,
    *,
    outward_prior: np.ndarray,
    max_gap_wrist_ratio: float,
    min_edge_ambiguity_ratio: float,
    min_edge_direction_cosine: float,
) -> dict[str, Any]:
    """Propose one boundary prompt from observed component geometry only."""

    if component.ndim != 2 or component.dtype != bool:
        raise RuntimeError("observed component must be a binary 2D mask")
    if not np.any(component) or not np.isfinite(wrist_width) or wrist_width <= 0:
        return {"status": "HOLD_NO_OBSERVED_COMPONENT"}
    y, x = np.nonzero(component)
    height, width = component.shape
    gaps = {
        "left": int(x.min()),
        "right": int(width - 1 - x.max()),
        "top": int(y.min()),
        "bottom": int(height - 1 - y.max()),
    }
    ranked = sorted(gaps.items(), key=lambda item: (item[1], item[0]))
    edge, gap = ranked[0]
    second_gap = ranked[1][1]
    if gap == 0:
        return {"status": "ALREADY_BOUNDARY_CONNECTED", "edge": edge, "gaps_px": gaps}
    gap_ratio = float(gap / wrist_width)
    ambiguity_ratio = float(second_gap / max(gap, 1))
    if gap_ratio > max_gap_wrist_ratio:
        return {
            "status": "HOLD_GAP_TOO_LONG",
            "edge": edge,
            "gap_wrist_ratio": gap_ratio,
            "edge_ambiguity_ratio": ambiguity_ratio,
            "gaps_px": gaps,
        }
    if ambiguity_ratio <= min_edge_ambiguity_ratio:
        return {
            "status": "HOLD_EDGE_AMBIGUOUS",
            "edge": edge,
            "gap_wrist_ratio": gap_ratio,
            "edge_ambiguity_ratio": ambiguity_ratio,
            "gaps_px": gaps,
        }

    xy = np.column_stack([x, y]).astype(np.float64)
    covariance = np.cov(xy - xy.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    normal = np.asarray([-principal[1], principal[0]], np.float64)
    edge_distance = {
        "left": x,
        "right": width - 1 - x,
        "top": y,
        "bottom": height - 1 - y,
    }[edge]
    terminal_band = max(2.0, 0.75 * wrist_width)
    terminal = xy[edge_distance <= gap + terminal_band]
    if len(terminal) < 8:
        return {"status": "HOLD_TERMINAL_EVIDENCE_SPARSE", "edge": edge, "gaps_px": gaps}
    center = np.median(terminal, axis=0)
    if edge == "left":
        target = np.asarray([1.0, center[1]])
    elif edge == "right":
        target = np.asarray([width - 2.0, center[1]])
    elif edge == "top":
        target = np.asarray([center[0], 1.0])
    else:
        target = np.asarray([center[0], height - 2.0])
    target[0] = np.clip(target[0], 1, width - 2)
    target[1] = np.clip(target[1], 1, height - 2)

    target_direction = target - center
    target_direction /= max(float(np.linalg.norm(target_direction)), 1e-6)
    if float(np.dot(principal, target_direction)) < 0:
        principal = -principal
    prior = np.asarray(outward_prior, np.float64)
    prior /= max(float(np.linalg.norm(prior)), 1e-6)
    pca_edge_cosine = float(np.dot(principal, target_direction))
    prior_edge_cosine = float(np.dot(prior, target_direction))
    if (
        pca_edge_cosine < min_edge_direction_cosine
        or prior_edge_cosine < min_edge_direction_cosine
    ):
        return {
            "status": "HOLD_EDGE_DIRECTION_INCOMPATIBLE",
            "edge": edge,
            "gap_wrist_ratio": gap_ratio,
            "edge_ambiguity_ratio": ambiguity_ratio,
            "pca_edge_cosine": pca_edge_cosine,
            "outward_prior_edge_cosine": prior_edge_cosine,
            "gaps_px": gaps,
        }

    normal_projection = terminal @ normal
    observed_width = float(
        max(np.quantile(normal_projection, 0.90) - np.quantile(normal_projection, 0.10), 1.0)
    )
    negative_offset = 0.75 * observed_width + 0.5 * wrist_width
    negatives = np.stack(
        [target + negative_offset * normal, target - negative_offset * normal], axis=0
    )
    negatives[:, 0] = np.clip(negatives[:, 0], 1, width - 2)
    negatives[:, 1] = np.clip(negatives[:, 1], 1, height - 2)
    support_half_width = 0.5 * observed_width + 0.5 * wrist_width
    polygon = np.stack(
        [
            center + support_half_width * normal,
            center - support_half_width * normal,
            target - support_half_width * normal,
            target + support_half_width * normal,
        ]
    )
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return {
        "status": "REPROMPT_ELIGIBLE",
        "edge": edge,
        "gaps_px": gaps,
        "gap_wrist_ratio": gap_ratio,
        "edge_ambiguity_ratio": ambiguity_ratio,
        "target_positive_xy": target.astype(np.float32),
        "table_negative_xy": negatives.astype(np.float32),
        "terminal_center_xy": center.astype(np.float32),
        "observed_terminal_width_px": observed_width,
        "pca_principal_xy": principal.astype(np.float32),
        "pca_edge_cosine": pca_edge_cosine,
        "outward_prior_edge_cosine": prior_edge_cosine,
        "local_support_polygon_xy": polygon.astype(np.float32),
        "bridge_pixels": 0.0,
    }


def local_support_from_proposal(
    shape: tuple[int, int], proposal: dict[str, Any]
) -> np.ndarray:
    polygon = np.rint(np.asarray(proposal["local_support_polygon_xy"], np.float32)).astype(np.int32)
    support = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(support, polygon, 1)
    return support.astype(bool)


def final_forearm_prompt_support(
    mask: np.ndarray,
    sleeve_points: np.ndarray,
    *,
    min_prompt_recall: float,
) -> dict[str, float]:
    """Require crop connection and normalized sleeve-prompt coverage together."""

    if mask.ndim != 2 or mask.dtype != bool:
        raise RuntimeError("final forearm must be a binary 2D mask")
    points = np.asarray(sleeve_points, np.float32).reshape(-1, 2)
    if len(points) == 0:
        return {
            "final_forearm_prompt_recall": 0.0,
            "final_boundary_connected": 0.0,
            "final_forearm_supported": 0.0,
        }
    height, width = mask.shape
    xy = np.rint(points).astype(int)
    valid = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    if not np.any(valid):
        recall = 0.0
    else:
        valid_xy = xy[valid]
        recall = float(mask[valid_xy[:, 1], valid_xy[:, 0]].mean())
    boundary = float(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )
    return {
        "final_forearm_prompt_recall": recall,
        "final_boundary_connected": boundary,
        "final_forearm_supported": float(
            recall >= min_prompt_recall and boundary == 1.0
        ),
    }


def select_boundary_reprompt_candidate(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    positive_points: np.ndarray,
    negative_points: np.ndarray,
    initial_component: np.ndarray,
    target_edge: str,
    object_core_exclusion: np.ndarray,
    uncertain_contact_exclusion: np.ndarray,
    local_added_support: np.ndarray,
    observed_terminal_width_px: float,
    min_initial_recall: float,
    max_expansion_ratio: float,
    min_boundary_width_ratio: float,
    max_boundary_width_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Accept only a real SAM mask with connected positives and zero negatives."""

    best: tuple[float, np.ndarray, dict[str, float]] | None = None
    best_rejected: tuple[float, dict[str, float]] | None = None
    height, width = initial_component.shape
    positive_xy = np.rint(positive_points).astype(int)
    negative_xy = np.rint(negative_points).astype(int)
    positive_xy[:, 0] = np.clip(positive_xy[:, 0], 0, width - 1)
    positive_xy[:, 1] = np.clip(positive_xy[:, 1], 0, height - 1)
    negative_xy[:, 0] = np.clip(negative_xy[:, 0], 0, width - 1)
    negative_xy[:, 1] = np.clip(negative_xy[:, 1], 0, height - 1)
    initial_pixels = max(int(np.count_nonzero(initial_component)), 1)
    for raw_mask, score in zip(masks.astype(bool), scores, strict=True):
        component_count, labels = cv2.connectedComponents(raw_mask.astype(np.uint8), connectivity=8)
        positive_labels = labels[positive_xy[:, 1], positive_xy[:, 0]]
        connected_positive = bool(
            len(positive_labels)
            and positive_labels[0] != 0
            and np.all(positive_labels == positive_labels[0])
        )
        candidate = (
            labels == int(positive_labels[0])
            if connected_positive
            else np.zeros_like(raw_mask, dtype=bool)
        )
        negative_leak = float(candidate[negative_xy[:, 1], negative_xy[:, 0]].mean())
        initial_recall = float(
            np.count_nonzero(candidate & initial_component) / initial_pixels
        )
        expansion_ratio = float(
            np.count_nonzero(candidate & ~initial_component) / initial_pixels
        )
        added = candidate & ~initial_component
        outside_local_support = float(np.count_nonzero(added & ~local_added_support))
        object_core_overlap = float(
            np.count_nonzero(candidate & object_core_exclusion)
        )
        uncertain_contact_overlap = float(
            np.count_nonzero(candidate & uncertain_contact_exclusion)
        )
        reaches = {
            "left": bool(np.any(candidate[:, 0])),
            "right": bool(np.any(candidate[:, -1])),
            "top": bool(np.any(candidate[0, :])),
            "bottom": bool(np.any(candidate[-1, :])),
        }[target_edge]
        boundary_values = {
            "left": candidate[:, 0],
            "right": candidate[:, -1],
            "top": candidate[0, :],
            "bottom": candidate[-1, :],
        }[target_edge]
        boundary_width_px = float(np.count_nonzero(boundary_values))
        boundary_width_ratio = float(
            boundary_width_px / max(observed_terminal_width_px, 1.0)
        )
        accepted = bool(
            connected_positive
            and negative_leak == 0.0
            and initial_recall >= min_initial_recall
            and expansion_ratio <= max_expansion_ratio
            and object_core_overlap == 0.0
            and uncertain_contact_overlap == 0.0
            and reaches
            and outside_local_support == 0.0
            and boundary_width_ratio >= min_boundary_width_ratio
            and boundary_width_ratio <= max_boundary_width_ratio
        )
        metrics = {
            "sam_score": float(score),
            "connected_positive": float(connected_positive),
            "negative_leak": negative_leak,
            "initial_component_recall": initial_recall,
            "expansion_ratio": expansion_ratio,
            "object_core_overlap_pixels": object_core_overlap,
            "uncertain_contact_overlap_pixels": uncertain_contact_overlap,
            "protected_exclusion_overlap_pixels": float(
                np.count_nonzero(
                    candidate & (object_core_exclusion | uncertain_contact_exclusion)
                )
            ),
            "outside_local_support_pixels": outside_local_support,
            "boundary_width_px": boundary_width_px,
            "boundary_width_ratio": boundary_width_ratio,
            "target_edge_reached": float(reaches),
            "component_count": float(max(component_count - 1, 0)),
            "held": float(not accepted),
        }
        rank = float(score) + initial_recall - expansion_ratio - 2.0 * negative_leak
        if accepted and (best is None or rank > best[0]):
            best = (rank, candidate, metrics)
        if not accepted and (best_rejected is None or rank > best_rejected[0]):
            best_rejected = (rank, metrics)
    if best is None:
        rejected_metrics = best_rejected[1] if best_rejected is not None else {
            "sam_score": 0.0,
            "connected_positive": 0.0,
            "negative_leak": 0.0,
            "initial_component_recall": 0.0,
            "expansion_ratio": 0.0,
            "object_core_overlap_pixels": 0.0,
            "uncertain_contact_overlap_pixels": 0.0,
            "protected_exclusion_overlap_pixels": 0.0,
            "outside_local_support_pixels": 0.0,
            "boundary_width_px": 0.0,
            "boundary_width_ratio": 0.0,
            "target_edge_reached": 0.0,
            "component_count": 0.0,
        }
        return np.zeros_like(initial_component, dtype=bool), {
            **rejected_metrics,
            "held": 1.0,
        }
    return best[1], best[2]
