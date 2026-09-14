#!/usr/bin/env python3
"""Validate the shared Mask successor protocol and its frozen FIT/EVAL plans.

The central invariant is structural: event-stratified spatial samples are
independent images, while temporal propagation may see only unit-stride source
windows.  Metric helpers are model-agnostic and deliberately contain no colour
heuristic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tools.project_guardian import atomic_json


class ProtocolError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON root must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate_shared(shared: Mapping[str, Any]) -> None:
    require(
        shared.get("schema_version") == "shared-mask-spatial-temporal-successor-v1",
        "unexpected shared protocol schema",
    )
    spatial = shared.get("spatial_canary", {})
    temporal = shared.get("temporal_canary", {})
    require(spatial.get("frame_count") == 12, "spatial frame_count must be 12")
    require(
        spatial.get("execution") == "one_independent_single_frame_model_state_per_source_frame",
        "spatial samples must use independent model states",
    )
    require(spatial.get("previous_or_next_sample_visible_to_model") is False, "spatial state leakage")
    require(spatial.get("cross_sample_flow_or_propagation_forbidden") is True, "cross-sample flow not forbidden")
    require(temporal.get("window_count") == 3, "temporal window_count must be 3")
    require(temporal.get("frames_per_window") == 12, "temporal windows must be 12 frames")
    require(temporal.get("source_indices_must_be_unit_stride") is True, "temporal unit-stride not required")
    require(temporal.get("propagation_between_windows_forbidden") is True, "cross-window propagation not forbidden")
    require(shared.get("shared_model", {}).get("checkpoint_write_forbidden") is True, "checkpoint writes not forbidden")
    tracker = shared.get("tracker_spatial_gate", {})
    required_tracker_metrics = {
        "positive_anchor_coverage_fraction_min",
        "negative_anchor_exclusion_fraction_min",
        "own_pico_wrist_min_distance_px_max",
        "centroid_to_own_pico_wrist_px_max",
        "opposing_wrist_centroid_margin_px_min",
        "frame_area_fraction_min",
        "frame_area_fraction_max",
        "connected_components_over_64px_max",
        "largest_component_area_fraction_min",
        "protected_object_overlap_pixels_max",
        "non_wrist_authorized_image_edge_contact_pixels_max",
    }
    require(required_tracker_metrics <= set(tracker), "tracker leakage gates incomplete")


def validate_plan(plan: Mapping[str, Any], shared: Mapping[str, Any]) -> None:
    require(plan.get("schema_version") == "task-mask-successor-evaluation-plan-v1", "unexpected task plan schema")
    split = plan.get("split")
    require(split in {"FIT", "EVAL"}, "split must be FIT or EVAL")
    frame_count = int(plan.get("frame_count", 0))
    spatial = list(map(int, plan.get("spatial_source_frames", [])))
    require(len(spatial) == shared["spatial_canary"]["frame_count"], "wrong spatial sample count")
    require(len(spatial) == len(set(spatial)), "duplicate spatial source frames")
    require(spatial == sorted(spatial), "spatial source frames must be sorted")
    require(all(0 <= frame < frame_count for frame in spatial), "spatial frame outside source")
    windows = plan.get("temporal_windows", [])
    require(len(windows) == shared["temporal_canary"]["window_count"], "wrong temporal window count")
    used: set[int] = set()
    for window in windows:
        frames = list(map(int, window.get("source_frames", [])))
        require(len(frames) == shared["temporal_canary"]["frames_per_window"], "wrong temporal window length")
        require(frames == list(range(frames[0], frames[0] + len(frames))), "temporal window is not unit-stride")
        require(all(0 <= frame < frame_count for frame in frames), "temporal frame outside source")
        require(not (used & set(frames)), "temporal windows overlap")
        used.update(frames)
    require(plan.get("threshold_override") is None, "task-specific threshold override forbidden")
    if split == "EVAL":
        require(plan.get("fit_result_access_for_prompt_design") is False, "EVAL prompt design may not inspect FIT results")
    identity = f"{plan['task_id']}:{plan['session_id']}"
    design = shared["evaluation_design"]
    allowed = design["fit"] if split == "FIT" else design["independent_eval"]
    require(identity in allowed, f"session {identity} not preregistered for {split}")


def tracker_spatial_failures(record: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    """Return stable failure keys for one tracker mask on one source frame."""
    failures: list[str] = []
    checks = [
        (record.get("present") is True, "missing"),
        (float(record.get("positive_anchor_coverage_fraction", -1)) >= float(gate["positive_anchor_coverage_fraction_min"]), "positive_anchor"),
        (float(record.get("negative_anchor_exclusion_fraction", -1)) >= float(gate["negative_anchor_exclusion_fraction_min"]), "negative_anchor"),
        (float(record.get("own_pico_wrist_min_distance_px", 1e12)) <= float(gate["own_pico_wrist_min_distance_px_max"]), "own_wrist_adjacency"),
        (float(record.get("centroid_to_own_pico_wrist_px", 1e12)) <= float(gate["centroid_to_own_pico_wrist_px_max"]), "own_wrist_centroid"),
        (float(record.get("opposing_minus_own_wrist_centroid_px", -1e12)) >= float(gate["opposing_wrist_centroid_margin_px_min"]), "side_identity_margin"),
        (float(gate["frame_area_fraction_min"]) <= float(record.get("frame_area_fraction", -1)) <= float(gate["frame_area_fraction_max"]), "area"),
        (int(record.get("connected_components_over_64px", 10**9)) <= int(gate["connected_components_over_64px_max"]), "components"),
        (float(record.get("largest_component_area_fraction", -1)) >= float(gate["largest_component_area_fraction_min"]), "largest_component"),
        (int(record.get("protected_object_overlap_pixels", 10**9)) <= int(gate["protected_object_overlap_pixels_max"]), "object_overlap"),
        (int(record.get("non_wrist_authorized_image_edge_contact_pixels", 10**9)) <= int(gate["non_wrist_authorized_image_edge_contact_pixels_max"]), "background_edge_leak"),
        (record.get("same_wrist_adjacency") is True, "same_wrist"),
    ]
    failures.extend(name for passed, name in checks if not passed)
    return failures


def temporal_failures(record: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    checks = [
        (record.get("all_frames_pass_spatial") is True, "frame_spatial_gate"),
        (float(record.get("flow_aligned_iou_median", -1)) >= float(gate["flow_aligned_iou_median_min"]), "flow_iou"),
        (float(record.get("flow_cycle_error_p90_px_half", 1e12)) <= float(gate["flow_cycle_error_p90_px_half_max"]), "flow_cycle"),
        (float(record.get("adjacent_area_ratio_min", -1)) >= float(gate["adjacent_area_ratio_min"]), "area_drop"),
        (float(record.get("adjacent_area_ratio_max", 1e12)) <= float(gate["adjacent_area_ratio_max"]), "area_growth"),
        (int(record.get("side_swap_count", 10**9)) <= int(gate["side_swap_count_max"]), "side_swap"),
        (int(record.get("background_leak_frame_count", 10**9)) <= int(gate["background_leak_frame_count_max"]), "background_leak"),
        (int(record.get("unexplained_role_disappearance_count", 10**9)) <= int(gate["unexplained_role_disappearance_count_max"]), "role_disappearance"),
    ]
    failures.extend(name for passed, name in checks if not passed)
    return failures


def validate_files(shared_path: Path, plan_paths: Sequence[Path]) -> dict[str, Any]:
    shared = load_object(shared_path)
    validate_shared(shared)
    plans = []
    identities: set[str] = set()
    for path in plan_paths:
        plan = load_object(path)
        validate_plan(plan, shared)
        identity = f"{plan['split']}:{plan['task_id']}:{plan['session_id']}"
        require(identity not in identities, f"duplicate evaluation identity: {identity}")
        identities.add(identity)
        plans.append({"path": str(path.resolve()), "sha256": sha256(path), "identity": identity})
    require(sum(item["identity"].startswith("FIT:") for item in plans) == 1, "exactly one FIT plan required")
    require(sum(item["identity"].startswith("EVAL:") for item in plans) == 2, "exactly two EVAL plans required")
    return {
        "schema_version": "mask-successor-protocol-validation-v1",
        "status": "PASS_FROZEN_STRUCTURE",
        "shared_protocol": {"path": str(shared_path.resolve()), "sha256": sha256(shared_path)},
        "plans": plans,
        "invariants": {
            "spatial_single_frame_only": True,
            "temporal_unit_stride_only": True,
            "fit_eval_split_frozen": True,
            "task_threshold_overrides_absent": True,
            "colour_specific_mask_logic_absent": True,
        },
        "claim_limit": "Static protocol validation only; no model inference or quality result is implied."
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", type=Path, required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_files(args.shared, args.plan)
    atomic_json(args.output, report)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
