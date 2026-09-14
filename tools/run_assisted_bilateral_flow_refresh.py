#!/usr/bin/env python3
"""Fixed assisted bilateral mask successor using raw DIS flow and SAM refresh.

The only manual wearable registration is frame 0.  Every later refresh point
is the centroid of the immediately preceding accepted flow warp.  Class masks
are never repaired by colour, morphology, dilation, erosion, fill, or GrabCut.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT = Path("/mnt/workspace/code/chaoyang")
CONTRACT = PROJECT / "_run/gpt_mask_assisted_bilateral_wearable_contract_20260902_v1"
PROTOCOL_PATH = CONTRACT / "FLOW_REFRESH_PROTOCOL_V4.json"
SOURCE_ROOT = PROJECT / "_run/newtask_baseline_mask_probe_20260901_v1/action_input_frames"
OBJECT_ROOTS = {
    "chips": PROJECT / "_run/newtask_baseline_mask_probe_20260901_v1/action/chips/object_chip/raw_single_instance_masks",
    "poker": PROJECT / "_run/newtask_baseline_mask_probe_20260901_v1/action/poker/object_card/raw_single_instance_masks",
}

sys.path.insert(0, str(PROJECT))
from tools import run_assisted_bilateral_sam31_point_canary as frame0
from tools import run_assisted_bilateral_wearable_temporal as temporal


ROLE_COLORS = {
    "upper_human_core": np.asarray([255, 40, 180], np.float32),
    "lower_human_core": np.asarray([30, 155, 255], np.float32),
    "upper_tracker_wearable": np.asarray([255, 245, 30], np.float32),
    "lower_tracker_wearable": np.asarray([255, 145, 25], np.float32),
    "lower_sleeve_cuff": np.asarray([60, 255, 90], np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("chips", "poker"), required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--canary-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    return (float(xs.mean()), float(ys.mean())) if len(xs) else None


def centroid_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_center = centroid(first)
    second_center = centroid(second)
    return float(math.dist(first_center, second_center)) if first_center and second_center else float("inf")


def farthest_mask_point(mask: np.ndarray, reference_xy: tuple[int, int]) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        return reference_xy
    x, y = reference_xy
    position = int(np.argmax((xs - x) ** 2 + (ys - y) ** 2))
    return int(xs[position]), int(ys[position])


def min_distance(mask: np.ndarray, target: np.ndarray) -> float:
    if not mask.any() or not target.any():
        return float("inf")
    distance = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[mask].min())


def boundary_gradient(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    boundary = cv2.Canny(mask.astype(np.uint8) * 255, 64, 128) > 0
    return float(np.median(gradient[boundary])) if boundary.any() else 0.0


class FlowCache:
    def __init__(self, frame_paths: list[Path], half_width: int, half_height: int) -> None:
        self.frame_paths = frame_paths
        self.width = half_width
        self.height = half_height
        self.images: dict[int, np.ndarray] = {}
        self.gray: dict[int, np.ndarray] = {}
        self.flow: dict[tuple[int, int], np.ndarray] = {}

    def image(self, index: int) -> np.ndarray:
        if index not in self.images:
            value = cv2.imread(str(self.frame_paths[index]), cv2.IMREAD_COLOR)
            if value is None:
                raise RuntimeError(f"frame decode failed: {self.frame_paths[index]}")
            self.images[index] = value
        return self.images[index]

    def grayscale_half(self, index: int) -> np.ndarray:
        if index not in self.gray:
            image = cv2.resize(self.image(index), (self.width, self.height), interpolation=cv2.INTER_AREA)
            self.gray[index] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.gray[index]

    def directed_flow(self, source: int, target: int) -> np.ndarray:
        key = (source, target)
        if key not in self.flow:
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            dis.setUseSpatialPropagation(True)
            self.flow[key] = dis.calc(self.grayscale_half(source), self.grayscale_half(target), None)
        return self.flow[key]

    def warp(self, previous_mask: np.ndarray, previous: int, current: int) -> tuple[np.ndarray, dict[str, float]]:
        previous_half = cv2.resize(
            previous_mask.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        # Inverse-sampling field is current->previous; the reverse field is
        # independently estimated and used only for the raw cycle gate.
        backward = self.directed_flow(current, previous)
        forward = self.directed_flow(previous, current)
        yy, xx = np.mgrid[: self.height, : self.width].astype(np.float32)
        map_x = xx + backward[..., 0]
        map_y = yy + backward[..., 1]
        warped_half = cv2.remap(
            previous_half.astype(np.uint8), map_x, map_y,
            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        forward_x = cv2.remap(forward[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        forward_y = cv2.remap(forward[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cycle = np.sqrt((backward[..., 0] + forward_x) ** 2 + (backward[..., 1] + forward_y) ** 2)
        warped = cv2.resize(
            warped_half.astype(np.uint8), (previous_mask.shape[1], previous_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        return warped, {
            "cycle_error_median_px_half": float(np.median(cycle[warped_half])) if warped_half.any() else float("inf"),
            "cycle_error_p90_px_half": float(np.percentile(cycle[warped_half], 90)) if warped_half.any() else float("inf"),
        }

    def warp_bidirectional_consensus(
        self, previous_mask: np.ndarray, previous: int, current: int
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Intersect independent inverse-warp and forward-splat RAW masks.

        This is deliberately a Boolean nearest-neighbour construction.  It
        applies no morphology, dilation, erosion, connected-component repair,
        or fill after either directed DIS estimate.
        """
        previous_half = cv2.resize(
            previous_mask.astype(np.uint8), (self.width, self.height), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        backward = self.directed_flow(current, previous)
        forward = self.directed_flow(previous, current)
        yy, xx = np.mgrid[: self.height, : self.width].astype(np.float32)

        map_x = xx + backward[..., 0]
        map_y = yy + backward[..., 1]
        inverse_warp = cv2.remap(
            previous_half.astype(np.uint8), map_x, map_y,
            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)

        source_y, source_x = np.where(previous_half)
        target_x = np.rint(source_x + forward[source_y, source_x, 0]).astype(np.int32)
        target_y = np.rint(source_y + forward[source_y, source_x, 1]).astype(np.int32)
        valid = (
            (target_x >= 0) & (target_x < self.width)
            & (target_y >= 0) & (target_y < self.height)
        )
        forward_splat = np.zeros((self.height, self.width), dtype=bool)
        forward_splat[target_y[valid], target_x[valid]] = True
        consensus_half = np.logical_and(inverse_warp, forward_splat)

        forward_x = cv2.remap(forward[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        forward_y = cv2.remap(forward[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cycle = np.sqrt((backward[..., 0] + forward_x) ** 2 + (backward[..., 1] + forward_y) ** 2)
        consensus = cv2.resize(
            consensus_half.astype(np.uint8), (previous_mask.shape[1], previous_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        return consensus, {
            "cycle_error_median_px_half": (
                float(np.median(cycle[consensus_half])) if consensus_half.any() else float("inf")
            ),
            "cycle_error_p90_px_half": (
                float(np.percentile(cycle[consensus_half], 90)) if consensus_half.any() else float("inf")
            ),
            "inverse_warp_area_pixels_half": int(inverse_warp.sum()),
            "forward_splat_area_pixels_half": int(forward_splat.sum()),
            "consensus_area_pixels_half": int(consensus_half.sum()),
        }


def stabilize_human_streams(
    raw_streams: dict[str, dict[int, np.ndarray]],
    frame_paths: list[Path],
    flow_cache: FlowCache,
    object_masks: list[np.ndarray],
    protocol: dict,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Keep the frozen 4x human gate by a raw DIS hold, never by relaxing it."""
    rules = protocol["human_core"]
    stabilized: dict[str, dict[int, np.ndarray]] = {}
    metadata: dict[str, Any] = {}
    for name, raw_stream in raw_streams.items():
        initial = raw_stream.get(0)
        if initial is None or not initial.any():
            stabilized[name] = raw_stream
            metadata[name] = {"status": "HOLD", "reason": "missing_initial_human_raw"}
            continue
        initial_area = int(initial.sum())
        accepted = {0: initial.copy()}
        previous = initial.copy()
        fallback_records = []
        missing_raw_frames = []
        for index in range(1, len(frame_paths)):
            candidate = raw_stream.get(index)
            candidate_ratio = float(candidate.sum() / initial_area) if candidate is not None else 0.0
            candidate_pass = bool(
                candidate is not None
                and candidate.any()
                and rules["canary_area_ratio_min"] <= candidate_ratio <= rules["canary_area_ratio_max"]
            )
            if candidate_pass:
                chosen = candidate
            else:
                if candidate is None:
                    missing_raw_frames.append(index)
                warped, flow_record = flow_cache.warp_bidirectional_consensus(previous, index - 1, index)
                step_ratio = float(warped.sum() / max(int(previous.sum()), 1))
                total_ratio = float(warped.sum() / initial_area)
                edge = boundary_gradient(flow_cache.image(index), warped)
                checks = {
                    "present": bool(warped.any()),
                    "flow_cycle": bool(flow_record["cycle_error_p90_px_half"] <= rules["flow_cycle_p90_max_px_half"]),
                    "adjacent_area_ratio": bool(rules["adjacent_area_ratio_min"] <= step_ratio <= rules["adjacent_area_ratio_max"]),
                    "unchanged_total_area_gate": bool(rules["canary_area_ratio_min"] <= total_ratio <= rules["canary_area_ratio_max"]),
                    "edge_boundary": bool(edge >= rules["boundary_gradient_median_min"]),
                    "published_union_object_protection_available": bool(object_masks[index].shape == warped.shape),
                }
                fallback_records.append({
                    "frame_index": index,
                    "raw_candidate_area_ratio": candidate_ratio,
                    "provider": "PREVIOUS_ACCEPTED_HUMAN_RAW_TRUE_BIDIRECTIONAL_DIS_INTERSECTION",
                    "pass": bool(all(checks.values())),
                    "checks": checks,
                    "warped_area_pixels": int(warped.sum()),
                    "warped_area_ratio_to_initial": total_ratio,
                    "warped_area_ratio_to_previous": step_ratio,
                    "boundary_gradient_median": edge,
                    **flow_record,
                })
                chosen = warped
            accepted[index] = chosen.copy()
            previous = chosen
        failed = [row["frame_index"] for row in fallback_records if not row["pass"]]
        stabilized[name] = accepted
        metadata[name] = {
            "status": "PASS" if not failed else "HOLD",
            "raw_provider": "SAM31_TEXT_TEMPORAL",
            "raw_missing_frames": missing_raw_frames,
            "unchanged_area_gate": [rules["canary_area_ratio_min"], rules["canary_area_ratio_max"]],
            "flow_hold_frame_count": len(fallback_records),
            "flow_hold_frames": [row["frame_index"] for row in fallback_records],
            "failed_flow_hold_frames": failed,
            "flow_hold_records": fallback_records,
        }
    return stabilized, metadata


def extract_id(outputs: dict, obj_id: int, height: int, width: int) -> np.ndarray:
    masks, ids = frame0.normalize_masks(outputs, height, width)
    matches = np.flatnonzero(ids == obj_id)
    return masks[int(matches[0])] if len(matches) == 1 else np.zeros((height, width), bool)


def sam_output_summary(outputs: dict | None, height: int, width: int) -> dict[str, Any]:
    """Persist the point-prompt return contract without retaining model tensors."""
    if outputs is None:
        return {"returned_none": True, "keys": [], "out_obj_ids": [], "areas_by_row": [], "out_probs": []}
    masks, ids = frame0.normalize_masks(outputs, height, width)
    probs = outputs.get("out_probs")
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    return {
        "returned_none": False,
        "keys": sorted(str(key) for key in outputs),
        "out_obj_ids": [int(value) for value in ids.tolist()],
        "areas_by_row": [int(mask.sum()) for mask in masks],
        "out_probs": [float(value) for value in np.asarray(probs if probs is not None else []).reshape(-1)],
    }


def role_side(role_name: str) -> str:
    return "upper" if role_name.startswith("upper_") else "lower"


def human_role(side: str) -> str:
    return f"{side}_human_core"


def other_human_role(side: str) -> str:
    return "lower_human_core" if side == "upper" else "upper_human_core"


def wearable_gates(
    mask: np.ndarray,
    previous_mask: np.ndarray,
    own_human: np.ndarray,
    opposing_human: np.ndarray,
    object_mask: np.ndarray,
    image_bgr: np.ndarray,
    flow_record: dict[str, float],
    protocol: dict,
) -> dict[str, Any]:
    rules = protocol["wearables"]
    previous_area = max(int(previous_mask.sum()), 1)
    area_ratio = float(mask.sum() / previous_area)
    own_distance = min_distance(mask, own_human)
    own_centroid_distance = centroid_distance(mask, own_human)
    opposing_centroid_distance = centroid_distance(mask, opposing_human)
    boundary = boundary_gradient(image_bgr, mask)
    overlap = int(np.logical_and(mask, object_mask).sum())
    checks = {
        "present": bool(mask.any()),
        "flow_cycle": bool(flow_record["cycle_error_p90_px_half"] <= rules["flow_cycle_p90_max_px_half"]),
        "adjacent_area_ratio": bool(rules["adjacent_area_ratio_min"] <= area_ratio <= rules["adjacent_area_ratio_max"]),
        "own_human_adjacency": bool(own_distance <= rules["own_human_min_distance_max_px"]),
        "opposing_identity_margin": bool(opposing_centroid_distance - own_centroid_distance >= rules["opposing_human_distance_margin_min_px"]),
        "edge_boundary": bool(boundary >= rules["boundary_gradient_median_min"]),
        "object_overlap_zero": bool(overlap <= protocol["task_object"]["wearable_raw_overlap_max_pixels"]),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "area_pixels": int(mask.sum()),
        "area_ratio_from_previous": area_ratio,
        "own_human_min_distance_px": own_distance,
        "own_human_centroid_distance_px": own_centroid_distance,
        "opposing_human_centroid_distance_px": opposing_centroid_distance,
        "boundary_gradient_median": boundary,
        "object_overlap_pixels": overlap,
        **flow_record,
    }


def refresh_gates(
    proposal: np.ndarray,
    warped: np.ndarray,
    point_xy: tuple[int, int],
    own_human: np.ndarray,
    opposing_human: np.ndarray,
    object_mask: np.ndarray,
    protocol: dict,
) -> dict[str, Any]:
    rules = protocol["wearables"]
    proposal_center = centroid(proposal)
    warped_center = centroid(warped)
    center_distance = (
        float(math.dist(proposal_center, warped_center))
        if proposal_center is not None and warped_center is not None else float("inf")
    )
    area_ratio = float(proposal.sum() / max(int(warped.sum()), 1))
    own_distance = min_distance(proposal, own_human)
    own_centroid_distance = centroid_distance(proposal, own_human)
    opposing_centroid_distance = centroid_distance(proposal, opposing_human)
    x, y = point_xy
    overlap = int(np.logical_and(proposal, object_mask).sum())
    checks = {
        "present": bool(proposal.any()),
        "predicted_centroid_inside": bool(proposal[y, x]) if proposal.any() else False,
        "iou_with_warp": bool(iou(proposal, warped) >= rules["refresh_iou_with_warp_min"]),
        "centroid_distance": bool(center_distance <= rules["refresh_centroid_distance_max_px"]),
        "area_ratio": bool(rules["refresh_area_ratio_min"] <= area_ratio <= rules["refresh_area_ratio_max"]),
        "own_human_adjacency": bool(own_distance <= rules["own_human_min_distance_max_px"]),
        "opposing_identity_margin": bool(opposing_centroid_distance - own_centroid_distance >= rules["opposing_human_distance_margin_min_px"]),
        "object_overlap_zero": bool(overlap <= protocol["task_object"]["wearable_raw_overlap_max_pixels"]),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "iou_with_warp": iou(proposal, warped),
        "centroid_distance_px": center_distance,
        "area_ratio_to_warp": area_ratio,
        "own_human_min_distance_px": own_distance,
        "own_human_centroid_distance_px": own_centroid_distance,
        "opposing_human_centroid_distance_px": opposing_centroid_distance,
        "object_overlap_pixels": overlap,
        "proposal_area_pixels": int(proposal.sum()),
    }


def collect_flow_refresh_role(
    model: Any,
    role: dict,
    case_spec: dict,
    frame_root: Path,
    frame_paths: list[Path],
    human_streams: dict[str, dict[int, np.ndarray]],
    object_masks: list[np.ndarray],
    flow_cache: FlowCache,
    protocol: dict,
    height: int,
    width: int,
    output_root: Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    name = role["name"]
    side = role_side(name)
    raw_id = int(role["id"])
    skip_discarded_priming = bool(case_spec.get("skip_discarded_priming", False))
    tracker_prompt_contract = case_spec.get("tracker_prompt_contract", "SINGLE_FULL_PROMPT")
    if tracker_prompt_contract not in {"SINGLE_FULL_PROMPT", "SAME_ID_POSITIVE_BOOTSTRAP_THEN_FULL_REFINE"}:
        raise RuntimeError(f"unsupported tracker prompt contract: {tracker_prompt_contract}")
    priming_id = None if skip_discarded_priming else raw_id + 10000
    stream: dict[int, np.ndarray] = {}
    step_records: list[dict[str, Any]] = []
    refresh_records: list[dict[str, Any]] = []
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True, async_loading_frames=False)
    try:
        registered_priming = case_spec.get("registered_priming_points_by_frame")
        if registered_priming is not None and len(registered_priming) != len(frame_paths):
            raise RuntimeError("registered priming point timeline length mismatch")
        if not skip_discarded_priming:
            assert priming_id is not None
            prime_x, prime_y = (
                registered_priming[0]
                if registered_priming is not None
                else case_spec["object_protection_points"][0]
            )
            model.add_prompt(
                inference_state=state, frame_idx=0,
                points=torch.tensor([[prime_x / width, prime_y / height]], dtype=torch.float32),
                point_labels=torch.tensor([1], dtype=torch.int32), obj_id=priming_id,
                rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
            )
        bootstrap_summary = None
        if tracker_prompt_contract == "SAME_ID_POSITIVE_BOOTSTRAP_THEN_FULL_REFINE":
            _, bootstrap_outputs = model.add_prompt(
                inference_state=state, frame_idx=0,
                points=torch.tensor(
                    [[x / width, y / height] for x, y in role["positive"]], dtype=torch.float32
                ),
                point_labels=torch.ones(len(role["positive"]), dtype=torch.int32), obj_id=raw_id,
                rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
            )
            bootstrap_summary = sam_output_summary(bootstrap_outputs, height, width)
        initial_points = [*role["positive"], *role["negative"]]
        initial_labels = [1] * len(role["positive"]) + [0] * len(role["negative"])
        _, initial_outputs = model.add_prompt(
            inference_state=state, frame_idx=0,
            points=torch.tensor([[x / width, y / height] for x, y in initial_points], dtype=torch.float32),
            point_labels=torch.tensor(initial_labels, dtype=torch.int32), obj_id=raw_id,
            rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
        )
        current_mask = extract_id(initial_outputs, raw_id, height, width)
        point_checks = {
            "prompt_contract": tracker_prompt_contract,
            "bootstrap_return": bootstrap_summary,
            "refine_return": sam_output_summary(initial_outputs, height, width),
            "positive_coverage": [bool(current_mask[y, x]) for x, y in role["positive"]],
            "negative_exclusion": [not bool(current_mask[y, x]) for x, y in role["negative"]],
            "object_overlap_pixels": int(np.logical_and(current_mask, object_masks[0]).sum()),
            "area_pixels": int(current_mask.sum()),
        }
        initial_pass = bool(
            current_mask.any()
            and all(point_checks["positive_coverage"])
            and all(point_checks["negative_exclusion"])
            and point_checks["object_overlap_pixels"] == 0
        )
        stream[0] = current_mask.copy()
    finally:
        state.clear()
    cadence = int(protocol["wearables"]["refresh_cadence_frames"])
    for index in range(1, len(frame_paths)):
        warped, flow_record = flow_cache.warp(current_mask, index - 1, index)
        own = human_streams[human_role(side)].get(index, np.zeros((height, width), bool))
        opposing = human_streams[other_human_role(side)].get(index, np.zeros((height, width), bool))
        flow_gate = wearable_gates(
            warped, current_mask, own, opposing, object_masks[index],
            flow_cache.image(index), flow_record, protocol,
        )
        step_record: dict[str, Any] = {"frame_index": index, "provider": "RAW_DIS_BIDIRECTIONAL_CYCLE", "flow_gate": flow_gate}
        chosen = warped
        if index % cadence == 0:
            predicted_center = centroid(warped)
            if predicted_center is None:
                point_xy = (0, 0)
                proposal = np.zeros((height, width), bool)
                object_prime_xy = None if skip_discarded_priming else (0, 0)
                priming_point_source = (
                    "SKIPPED_OBJECT_PROTECTION_PENDING"
                    if skip_discarded_priming else "NOT_RUN_EMPTY_TRACKER_FLOW"
                )
            else:
                point_xy = (
                    int(np.clip(round(predicted_center[0]), 0, width - 1)),
                    int(np.clip(round(predicted_center[1]), 0, height - 1)),
                )
                object_center = centroid(object_masks[index])
                if skip_discarded_priming:
                    object_prime_xy = None
                    priming_point_source = "SKIPPED_OBJECT_PROTECTION_PENDING"
                elif object_center is None:
                    if not case_spec.get("allow_registered_fallback_priming", False):
                        raise RuntimeError(f"protected task-object raw mask empty at refresh f{index}")
                    fallback_x, fallback_y = (
                        registered_priming[index]
                        if registered_priming is not None
                        else case_spec["object_protection_points"][0]
                    )
                    object_prime_xy = (
                        int(np.clip(round(fallback_x), 0, width - 1)),
                        int(np.clip(round(fallback_y), 0, height - 1)),
                    )
                    priming_point_source = "REGISTERED_DISCARDED_RUNTIME_PRIME_FALLBACK"
                else:
                    object_prime_xy = (
                        int(np.clip(round(object_center[0]), 0, width - 1)),
                        int(np.clip(round(object_center[1]), 0, height - 1)),
                    )
                    priming_point_source = "current_task_object_raw_centroid"
                one_frame = output_root / "automatic_refresh_inputs" / name / f"{index:05d}"
                one_frame.mkdir(parents=True)
                os.symlink(frame_paths[index], one_frame / "00000.png")
                negative_xy = farthest_mask_point(own, point_xy)
                refresh_state = model.init_state(
                    resource_path=str(one_frame), offload_video_to_cpu=True, async_loading_frames=False
                )
                try:
                    if not skip_discarded_priming:
                        assert object_prime_xy is not None
                        assert priming_id is not None
                        prime_x, prime_y = object_prime_xy
                        model.add_prompt(
                            inference_state=refresh_state, frame_idx=0,
                            points=torch.tensor([[prime_x / width, prime_y / height]], dtype=torch.float32),
                            point_labels=torch.tensor([1], dtype=torch.int32), obj_id=priming_id,
                            rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
                        )
                    x, y = point_xy
                    negative_x, negative_y = negative_xy
                    refresh_bootstrap_summary = None
                    if tracker_prompt_contract == "SAME_ID_POSITIVE_BOOTSTRAP_THEN_FULL_REFINE":
                        _, refresh_bootstrap_outputs = model.add_prompt(
                            inference_state=refresh_state, frame_idx=0,
                            points=torch.tensor([[x / width, y / height]], dtype=torch.float32),
                            point_labels=torch.ones(1, dtype=torch.int32), obj_id=raw_id,
                            rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
                        )
                        refresh_bootstrap_summary = sam_output_summary(
                            refresh_bootstrap_outputs, height, width
                        )
                    _, refresh_outputs = model.add_prompt(
                        inference_state=refresh_state, frame_idx=0,
                        points=torch.tensor(
                            [[x / width, y / height], [negative_x / width, negative_y / height]],
                            dtype=torch.float32,
                        ),
                        point_labels=torch.tensor([1, 0], dtype=torch.int32), obj_id=raw_id,
                        rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
                    )
                    proposal = extract_id(refresh_outputs, raw_id, height, width)
                    refresh_refine_summary = sam_output_summary(refresh_outputs, height, width)
                finally:
                    refresh_state.clear()
            refresh_gate = refresh_gates(
                proposal, warped, point_xy, own, opposing, object_masks[index], protocol
            )
            refresh_record = {
                "frame_index": index,
                "runtime": "independent_single_frame_state",
                "prompt_contract": tracker_prompt_contract,
                "bootstrap_return": refresh_bootstrap_summary if predicted_center is not None else None,
                "refine_return": refresh_refine_summary if predicted_center is not None else None,
                "discarded_priming_raw_id": priming_id,
                "priming_point_source": priming_point_source,
                "priming_point_xy": list(object_prime_xy) if object_prime_xy is not None else None,
                "priming_output_consumed": False,
                "priming_skipped": skip_discarded_priming,
                "point_source": "previous_RAW_DIS_warp_centroid",
                "point_xy": list(point_xy),
                "negative_point_source": "farthest_same_side_human_mask_pixel_from_positive",
                "negative_point_xy": list(negative_xy) if predicted_center is not None else None,
                "gate": refresh_gate,
                "accepted": bool(refresh_gate["pass"] and flow_gate["pass"]),
            }
            refresh_records.append(refresh_record)
            step_record["refresh"] = refresh_record
            if refresh_record["accepted"]:
                chosen = proposal
                step_record["provider"] = "SAM31_INDEPENDENT_SINGLE_FRAME_REFRESH_ACCEPTED"
        stream[index] = chosen.copy()
        current_mask = chosen
        step_records.append(step_record)
    failed_steps = [row["frame_index"] for row in step_records if not row["flow_gate"]["pass"]]
    rejected_refreshes = [row["frame_index"] for row in refresh_records if not row["accepted"]]
    status = "PASS" if initial_pass and not failed_steps and not rejected_refreshes else "HOLD"
    return stream, {
        "status": status,
        "raw_id": raw_id,
        "discarded_priming_raw_id": priming_id,
        "priming_output_consumed": False,
        "priming_skipped": skip_discarded_priming,
        "tracker_prompt_contract": tracker_prompt_contract,
        "manual_anchor_frames": [0],
        "automatic_refresh_cadence_frames": cadence,
        "automatic_refresh_runtime": (
            "independent_single_frame_state_without_object_priming"
            if skip_discarded_priming
            else "independent_single_frame_state_with_discarded_task_object_priming"
        ),
        "initial_gate": {"pass": initial_pass, **point_checks},
        "refresh_attempt_count": len(refresh_records),
        "refresh_accept_count": sum(row["accepted"] for row in refresh_records),
        "refresh_acceptance_rate": float(np.mean([row["accepted"] for row in refresh_records])) if refresh_records else 1.0,
        "rejected_refresh_frames": rejected_refreshes,
        "failed_flow_gate_frames": failed_steps,
        "refresh_records": refresh_records,
        "step_records": step_records,
    }


def canary_indices(frame_count: int, protocol: dict) -> list[int]:
    return sorted({
        min(frame_count - 1, int(round(value * (frame_count - 1))))
        for value in protocol["temporal_canary"]["normalized_positions"]
    })


def evaluate_canary(
    streams: dict[str, dict[int, np.ndarray]],
    wearable_meta: dict[str, dict[str, Any]],
    frame_count: int,
    protocol: dict,
) -> dict[str, Any]:
    indices = canary_indices(frame_count, protocol)
    roles: dict[str, Any] = {}
    for name, stream in streams.items():
        initial = max(int(stream.get(0, np.zeros((1, 1), bool)).sum()), 1)
        frames = []
        for index in indices:
            mask = stream.get(index, np.zeros_like(stream[0]))
            ratio = float(mask.sum() / initial)
            frames.append({
                "frame_index": index,
                "present": bool(mask.any()),
                "area_pixels": int(mask.sum()),
                "area_ratio_to_initial": ratio,
                "area_ratio_pass": bool(0.25 <= ratio <= 4.0),
                "centroid_xy": centroid(mask),
            })
        roles[name] = {
            "present_count": sum(row["present"] for row in frames),
            "area_ratio_pass_count": sum(row["area_ratio_pass"] for row in frames),
            "frames": frames,
        }
    order = []
    for index in indices:
        upper = centroid(streams["upper_human_core"].get(index, np.zeros_like(streams["upper_human_core"][0])))
        lower = centroid(streams["lower_human_core"].get(index, np.zeros_like(streams["lower_human_core"][0])))
        order.append({
            "frame_index": index,
            "upper_centroid_y": upper[1] if upper else None,
            "lower_centroid_y": lower[1] if lower else None,
            "pass": bool(upper and lower and upper[1] < lower[1]),
        })
    role_pass = all(
        row["present_count"] == len(indices) and row["area_ratio_pass_count"] == len(indices)
        for row in roles.values()
    )
    flow_pass = all(meta["status"] == "PASS" for meta in wearable_meta.values())
    status = "PASS" if role_pass and flow_pass and all(row["pass"] for row in order) else "HOLD"
    return {"status": status, "indices": indices, "roles": roles, "upper_before_lower": order, "all_flow_refresh_gates_pass": flow_pass}


def tint(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    image[mask] = (1.0 - alpha) * image[mask] + alpha * color


def write_contact_sheet(
    root: Path,
    case: str,
    frame_paths: list[Path],
    streams: dict[str, dict[int, np.ndarray]],
    object_masks: list[np.ndarray],
    indices: list[int],
    status: str,
) -> Path:
    tiles = []
    for index in indices:
        raw = np.asarray(Image.open(frame_paths[index]).convert("RGB"))[:, :, ::-1]
        overlay = raw.astype(np.float32)
        for name, stream in streams.items():
            tint(overlay, stream.get(index, np.zeros(raw.shape[:2], bool)), ROLE_COLORS[name], 0.64)
        tint(overlay, object_masks[index], np.asarray([20, 255, 20], np.float32), 0.82)
        pair = np.hstack((raw, np.clip(overlay, 0, 255).astype(np.uint8)))
        pair = cv2.resize(pair, (540, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(pair, (0, 0), (540, 30), (0, 0, 0), -1)
        cv2.putText(pair, f"{case} f{index:05d} RAW | classes+object {status}", (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(pair)
    rows = []
    for start in range(0, len(tiles), 2):
        row = tiles[start:start + 2]
        if len(row) == 1:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    path = root / f"{case.upper()}_FLOW_REFRESH_CANARY12_{status}.png"
    cv2.imwrite(str(path), np.vstack(rows))
    return path


def write_full_artifacts(
    args: argparse.Namespace,
    frame_paths: list[Path],
    streams: dict[str, dict[int, np.ndarray]],
    object_masks: list[np.ndarray],
) -> dict[str, Any]:
    role_root = args.output_root / "raw_role_masks"
    class_root = args.output_root / "raw_class_masks"
    union_root = args.output_root / "binary_union_masks"
    for name in streams:
        (role_root / name).mkdir(parents=True, exist_ok=True)
    for name in ("human_core", "tracker_wearable", "sleeve_cuff"):
        (class_root / name).mkdir(parents=True, exist_ok=True)
    union_root.mkdir(parents=True, exist_ok=True)
    video_path = args.output_root / f"{args.case}_RAW_VS_BILATERAL_CLASSES.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (540, 480))
    object_overlap_raw = []
    for index, frame_path in enumerate(frame_paths):
        raw = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        class_masks = {
            "human_core": streams["upper_human_core"][index] | streams["lower_human_core"][index],
            "tracker_wearable": streams["upper_tracker_wearable"][index] | streams["lower_tracker_wearable"][index],
            "sleeve_cuff": streams.get("lower_sleeve_cuff", {}).get(index, np.zeros(raw.shape[:2], bool)),
        }
        raw_union = np.zeros(raw.shape[:2], bool)
        for name, stream in streams.items():
            mask = stream[index]
            Image.fromarray(mask.astype(np.uint8) * 255).save(role_root / name / f"{index:05d}.png")
            raw_union |= mask
        for class_name, mask in class_masks.items():
            Image.fromarray(mask.astype(np.uint8) * 255).save(class_root / class_name / f"{index:05d}.png")
        object_overlap_raw.append(int(np.logical_and(raw_union, object_masks[index]).sum()))
        protected_union = raw_union & ~object_masks[index]
        Image.fromarray(protected_union.astype(np.uint8) * 255).save(union_root / f"{index:05d}.png")
        overlay = raw.astype(np.float32)
        for name, stream in streams.items():
            tint(overlay, stream[index], ROLE_COLORS[name], 0.64)
        tint(overlay, object_masks[index], np.asarray([20, 255, 20], np.float32), 0.82)
        pair = np.hstack((raw, np.clip(overlay, 0, 255).astype(np.uint8)))
        pair = cv2.resize(pair, (540, 480), interpolation=cv2.INTER_AREA)
        cv2.rectangle(pair, (0, 0), (540, 28), (0, 0, 0), -1)
        cv2.putText(pair, f"{args.case} f{index:05d} RAW | bilateral classes", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(pair)
    writer.release()
    return {
        "raw_role_masks": str(role_root),
        "raw_class_masks": str(class_root),
        "binary_union_masks": str(union_root),
        "review_video": str(video_path),
        "object_overlap_raw_pixels_max": max(object_overlap_raw),
        "published_union_object_overlap_pixels_max": 0,
    }


def main() -> int:
    args = parse_args()
    args.frame_root = args.frame_root.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output root required: {args.output_root}")
    frame_paths = sorted(args.frame_root.glob("*.png"))[: args.frame_count]
    if len(frame_paths) != args.frame_count:
        raise RuntimeError("frame count drift")
    first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    height, width = first.shape[:2]
    if (width, height) != (540, 960):
        raise RuntimeError("natural display geometry drift")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["status"] != "FROZEN_BEFORE_CHIPS_V4_CANARY":
        raise RuntimeError("flow-refresh protocol is not frozen")
    lease = json.loads((PROJECT / "_run/GPU_LEASE.json").read_text())
    if lease.get("status") not in {"ACTIVE", "ACQUIRED"} or lease.get("holder") != "mask-bilateral-tracker-final":
        raise RuntimeError("central GPU lease is not active")
    object_paths = sorted(OBJECT_ROOTS[args.case].glob("*.png"))[: args.frame_count]
    if len(object_paths) != args.frame_count:
        raise RuntimeError("object mask inventory drift")
    object_masks = [np.asarray(Image.open(path).convert("L")) > 0 for path in object_paths]
    if any(mask.shape != (height, width) for mask in object_masks):
        raise RuntimeError("object mask geometry drift")
    args.output_root.mkdir(parents=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sys.path.insert(0, str(frame0.CODE_ROOT))
    adapter_module = temporal.baseline.load_adapter_module()
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=frame0.CODE_ROOT, checkpoint_path=frame0.CHECKPOINT
    )
    roles = frame0.SPECS[args.case]["objects"]
    humans = [role for role in roles if role["class"] == "human_core"]
    wearable_roles = [role for role in roles if role["class"] != "human_core"]
    streams: dict[str, dict[int, np.ndarray]] = {}
    wearable_meta: dict[str, dict[str, Any]] = {}
    try:
        human_raw_streams, human_meta = temporal.collect_text_humans(
            adapter.model, args.frame_root, args.frame_count, humans, height, width
        )
        if human_meta["status"] != "COMPLETE":
            raise RuntimeError("human text temporal initialization HOLD")
        half_width, half_height = protocol["wearables"]["flow_resolution"]
        cache = FlowCache(frame_paths, half_width, half_height)
        human_streams, human_stabilization = stabilize_human_streams(
            human_raw_streams, frame_paths, cache, object_masks, protocol
        )
        streams.update(human_streams)
        for role in wearable_roles:
            role_stream, role_meta = collect_flow_refresh_role(
                adapter.model, role, frame0.SPECS[args.case], args.frame_root,
                frame_paths, human_streams, object_masks, cache, protocol, height, width,
                args.output_root,
            )
            streams[role["name"]] = role_stream
            wearable_meta[role["name"]] = role_meta
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        adapter.predictor.shutdown()
    evaluation = evaluate_canary(streams, wearable_meta, args.frame_count, protocol)
    human_flow_pass = all(row["status"] == "PASS" for row in human_stabilization.values())
    evaluation["all_human_flow_hold_gates_pass"] = human_flow_pass
    if not human_flow_pass:
        evaluation["status"] = "HOLD"
    contact = write_contact_sheet(
        args.output_root, args.case, frame_paths, streams, object_masks,
        evaluation["indices"], evaluation["status"],
    )
    pass_gate = evaluation["status"] == "PASS"
    artifacts: dict[str, Any] = {"canary_contact_sheet": str(contact)}
    consumption_authorized = bool(pass_gate and not args.canary_only)
    if consumption_authorized:
        artifacts.update(write_full_artifacts(args, frame_paths, streams, object_masks))
        status = "PASS_CANARY12_FULL_ARTIFACTS_EMITTED"
    elif pass_gate:
        status = "PASS_CANARY12_NO_FULL_ARTIFACT_PROMOTION_BY_REQUEST"
    else:
        status = "HOLD_CANARY12_NO_FULL_ARTIFACT_PROMOTION"
    result = {
        "schema_version": "assisted-bilateral-wearable-flow-refresh-result-v1",
        "status": status,
        "consumption_authorized": consumption_authorized,
        "case": args.case,
        "frame_count": args.frame_count,
        "frame_geometry": [width, height],
        "canary": evaluation,
        "human_initialization": human_meta,
        "human_stabilization": human_stabilization,
        "wearable_tracking": wearable_meta,
        "anchor_burden": {
            "manual_anchor_frames_per_wearable_instance": 1,
            "manual_anchor_frame_indices": [0],
            "manual_point_instances": len(wearable_roles),
            "automatic_refresh_cadence_frames": protocol["wearables"]["refresh_cadence_frames"],
            "automatic_refresh_point_source": "preceding accepted RAW DIS warp centroid",
        },
        "artifacts": artifacts,
        "forbidden_repairs_used": [],
        "pins": {
            "protocol": str(PROTOCOL_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "checkpoint_sha256": frame0.CHECKPOINT_SHA256,
            "runner_sha256": sha256(Path(__file__)),
            "build_evidence": build_evidence,
        },
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "claim_limit": protocol["claim_limit"],
    }
    (args.output_root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "consumption_authorized": consumption_authorized,
        "wall_seconds": result["resource"]["wall_seconds"],
        "peak_cuda_reserved_bytes": result["resource"]["peak_cuda_reserved_bytes"],
    }))
    return 0 if status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
