#!/usr/bin/env python3
"""PICO-native bilateral Mask runner for chips session 001.

The algorithm is task-generic and config-like: SAM3.1 raw text instances for
both hands/forearms and protected objects, plus independently prompted tracker
instances carried by bidirectional raw DIS flow with periodic SAM refresh.
Anatomical left/right comes from PICO projections; screen upper/lower is only
an internal ordering at the shared v4 tracker-flow interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
RAW = Path(
    "/mnt/data/egodata/datasets/ego/chips_cards_tracker_0901/"
    "potato_chips/get_potato_chips_0901_001/preprocess/all_data"
)
CODE_ROOT = PROJECT / "third_party/SAM3"
CHECKPOINT = PROJECT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
FRAME_COUNT = 799
ANCHOR_FRAME = 221
CANARY = [0, 120, 168, 221, 290, 363, 435, 491, 577, 692, 749, 798]
SOURCE_INDICES = list(range(FRAME_COUNT))
TASK_ID = "chips"
SESSION_ID = "get_potato_chips_0901_001"
HUMAN_ANCHORS = {"left_human": (515, 762), "right_human": (795, 399)}
OBJECT_ROLE_SPECS = {
    "chip_0": {"id": 301, "positive": [[795, 558]], "negative": [[748, 612], [865, 614], [500, 550]]},
    "chip_1": {"id": 302, "positive": [[748, 612]], "negative": [[795, 558], [865, 614], [500, 550]]},
    "chip_2": {"id": 303, "positive": [[865, 614]], "negative": [[795, 558], [748, 612], [500, 550]]},
    "bowl": {"id": 401, "positive": [[500, 550]], "negative": [[795, 558], [748, 612], [865, 614]]},
}
TRACKER_ROLES = {
    "upper_tracker_wearable": {
        "id": 202,
        "name": "upper_tracker_wearable",
        "positive": [[905, 355], [915, 392]],
        "negative": [[795, 399], [405, 783]],
    },
    "lower_tracker_wearable": {
        "id": 201,
        "name": "lower_tracker_wearable",
        "positive": [[405, 783], [405, 820]],
        "negative": [[515, 762], [905, 355]],
    },
}
TRACKER_ANATOMICAL = {
    "upper_tracker_wearable": "right_tracker",
    "lower_tracker_wearable": "left_tracker",
}
TRACKER_HUMAN_INTERNAL = {
    "upper_human_core": "right_human",
    "lower_human_core": "left_human",
}
OBJECT_PROTECTION_POINTS = [[690, 445]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """Keep result serialization deterministic across NumPy-backed metrics."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def normalize(outputs: dict[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    masks = outputs.get("out_binary_masks")
    ids = outputs.get("out_obj_ids")
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    elif isinstance(masks, (list, tuple)):
        masks = np.stack([
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for value in masks
        ])
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    masks = np.asarray(masks if masks is not None else np.zeros((0, height, width), bool))
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = np.squeeze(masks, axis=1)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[1:] != (height, width):
        raise RuntimeError(f"mask geometry drift: {masks.shape}")
    return masks.astype(bool), np.asarray(ids if ids is not None else [], dtype=np.int64).reshape(-1)


def mask_distance(mask: np.ndarray, point: tuple[int, int]) -> float:
    ys, xs = np.where(mask)
    if not len(xs):
        return float("inf")
    x, y = point
    return float(np.sqrt((xs - x) ** 2 + (ys - y) ** 2).min())


def choose_ids(
    masks: np.ndarray,
    ids: np.ndarray,
    anchors: dict[str, tuple[int, int]],
    max_distance: float,
) -> tuple[dict[str, int], dict[str, Any]]:
    chosen: dict[str, int] = {}
    used: set[int] = set()
    evidence: dict[str, Any] = {}
    for name, point in anchors.items():
        rows = [
            {
                "raw_id": int(ids[index]),
                "distance_px": mask_distance(mask, point),
                "area_pixels": int(mask.sum()),
            }
            for index, mask in enumerate(masks)
        ]
        eligible = sorted(
            (row for row in rows if row["raw_id"] not in used and row["distance_px"] <= max_distance),
            key=lambda row: (row["distance_px"], -row["area_pixels"], row["raw_id"]),
        )
        if eligible:
            chosen[name] = eligible[0]["raw_id"]
            used.add(eligible[0]["raw_id"])
        evidence[name] = {"anchor_xy": list(point), "candidates": rows, "chosen_raw_id": chosen.get(name)}
    return chosen, evidence


def materialize_view(root: Path, indices: list[int]) -> tuple[Path, list[Path]]:
    root.mkdir(parents=True)
    sources = []
    for slot, internal in enumerate(indices):
        original = SOURCE_INDICES[internal]
        source = (RAW / f"{original:05d}" / "rgb.png").resolve(strict=True)
        destination = root / f"{slot:05d}.png"
        os.symlink(source, destination)
        sources.append(source)
    return root, sources


def collect_text_roles(
    model: Any,
    frame_root: Path,
    prompt: str,
    anchor_frame: int,
    role_anchors: dict[str, tuple[int, int]],
    height: int,
    width: int,
    bidirectional: bool,
    max_anchor_distance: float,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True, async_loading_frames=False)
    streams = {name: {} for name in role_anchors}
    try:
        _, initial = model.add_prompt(
            inference_state=state,
            frame_idx=anchor_frame,
            text_str=prompt,
            output_prob_thresh=0.5,
        )
        initial_masks, initial_ids = normalize(initial, height, width)
        ids_by_role, selection = choose_ids(
            initial_masks, initial_ids, role_anchors, max_anchor_distance
        )
        if len(ids_by_role) != len(role_anchors):
            return streams, {"status": "HOLD_ID_SELECTION", "prompt": prompt, "selection": selection}

        directions = [(False, FRAME_COUNT - anchor_frame)]
        if bidirectional and anchor_frame:
            directions.append((True, anchor_frame + 1))
        for reverse, maximum in directions:
            for frame_index, outputs in model.propagate_in_video(
                inference_state=state,
                start_frame_idx=anchor_frame,
                max_frame_num_to_track=maximum,
                reverse=reverse,
                output_prob_thresh=0.5,
            ):
                masks, ids = normalize(outputs, height, width)
                by_id = {int(value): masks[index] for index, value in enumerate(ids)}
                for name, raw_id in ids_by_role.items():
                    if raw_id in by_id:
                        streams[name][int(frame_index)] = by_id[raw_id].copy()
        return streams, {
            "status": "COMPLETE",
            "prompt": prompt,
            "anchor_frame": anchor_frame,
            "selection": selection,
            "raw_ids": ids_by_role,
            "observed_counts": {name: len(stream) for name, stream in streams.items()},
        }
    finally:
        state.clear()


def collect_point_roles(
    model: Any,
    frame_root: Path,
    role_specs: dict[str, dict[str, Any]],
    height: int,
    width: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    """Track several disjoint protected objects in one multiplex state."""
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True, async_loading_frames=False)
    streams = {name: {} for name in role_specs}
    initial_gates: dict[str, Any] = {}
    prime_id = 9901
    try:
        # The pinned multiplex runtime suppresses the first raw object.  Prime
        # it on the unrelated small green table clip and never consume it.
        model.add_prompt(
            inference_state=state,
            frame_idx=0,
            points=torch.tensor([[1070 / width, 415 / height]], dtype=torch.float32),
            point_labels=torch.tensor([1], dtype=torch.int32),
            obj_id=prime_id,
            rel_coordinates=True,
            clear_old_points=True,
            output_prob_thresh=0.5,
        )
        for name, spec in role_specs.items():
            points = [*spec["positive"], *spec["negative"]]
            labels = [1] * len(spec["positive"]) + [0] * len(spec["negative"])
            _, outputs = model.add_prompt(
                inference_state=state,
                frame_idx=0,
                points=torch.tensor([[x / width, y / height] for x, y in points], dtype=torch.float32),
                point_labels=torch.tensor(labels, dtype=torch.int32),
                obj_id=int(spec["id"]),
                rel_coordinates=True,
                clear_old_points=True,
                output_prob_thresh=0.5,
            )
            masks, ids = normalize(outputs, height, width)
            matches = np.flatnonzero(ids == int(spec["id"]))
            mask = masks[int(matches[0])] if len(matches) == 1 else np.zeros((height, width), bool)
            gate = {
                "area_pixels": int(mask.sum()),
                "positive_coverage": [bool(mask[y, x]) for x, y in spec["positive"]],
                "negative_exclusion": [not bool(mask[y, x]) for x, y in spec["negative"]],
            }
            gate["pass"] = bool(
                gate["area_pixels"]
                and all(gate["positive_coverage"])
                and all(gate["negative_exclusion"])
            )
            streams[name][0] = mask.copy()
            initial_gates[name] = gate
        if not all(value["pass"] for value in initial_gates.values()):
            return streams, {
                "status": "HOLD_INITIAL_POINT_GATE",
                "discarded_priming_id": prime_id,
                "priming_output_consumed": False,
                "initial_gates": initial_gates,
            }
        ids_by_role = {name: int(spec["id"]) for name, spec in role_specs.items()}
        for frame_index, outputs in model.propagate_in_video(
            inference_state=state,
            start_frame_idx=0,
            max_frame_num_to_track=FRAME_COUNT,
            reverse=False,
            output_prob_thresh=0.5,
        ):
            masks, ids = normalize(outputs, height, width)
            by_id = {int(value): masks[index] for index, value in enumerate(ids)}
            for name, raw_id in ids_by_role.items():
                if raw_id in by_id:
                    streams[name][int(frame_index)] = by_id[raw_id].copy()
        return streams, {
            "status": "COMPLETE",
            "provider": "SAM31_MULTIPLEX_POINT_RAW_IDS",
            "discarded_priming_id": prime_id,
            "priming_output_consumed": False,
            "initial_gates": initial_gates,
            "raw_ids": ids_by_role,
            "observed_counts": {name: len(stream) for name, stream in streams.items()},
        }
    finally:
        state.clear()


def complete_with_flow(
    raw_streams: dict[str, dict[int, np.ndarray]],
    frame_paths: list[Path],
    flow_cache: Any,
    height: int,
    width: int,
    *,
    conservative_union: bool = False,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    completed: dict[str, dict[int, np.ndarray]] = {}
    evidence: dict[str, Any] = {}
    for name, raw in raw_streams.items():
        forward: dict[int, np.ndarray] = {}
        backward: dict[int, np.ndarray] = {}
        forward_records = []
        backward_records = []
        for index in range(FRAME_COUNT):
            if index in raw and raw[index].any():
                forward[index] = raw[index]
            elif index and forward[index - 1].any():
                warped, record = flow_cache.warp_bidirectional_consensus(forward[index - 1], index - 1, index)
                forward[index] = warped
                forward_records.append({"frame": index, **record})
            else:
                forward[index] = np.zeros((height, width), bool)
        for index in range(FRAME_COUNT - 1, -1, -1):
            if index in raw and raw[index].any():
                backward[index] = raw[index]
            elif index + 1 < FRAME_COUNT and backward[index + 1].any():
                warped, record = flow_cache.warp_bidirectional_consensus(backward[index + 1], index + 1, index)
                backward[index] = warped
                backward_records.append({"frame": index, **record})
            else:
                backward[index] = np.zeros((height, width), bool)
        output: dict[int, np.ndarray] = {}
        merge_counts = {"raw": 0, "forward_only": 0, "backward_only": 0, "both": 0, "empty": 0}
        for index in range(FRAME_COUNT):
            if index in raw and raw[index].any():
                output[index] = raw[index]
                merge_counts["raw"] += 1
                continue
            fmask = forward[index]
            bmask = backward[index]
            if fmask.any() and bmask.any():
                output[index] = (fmask | bmask) if conservative_union else (fmask & bmask)
                if not output[index].any():
                    output[index] = fmask if fmask.sum() <= bmask.sum() else bmask
                merge_counts["both"] += 1
            elif fmask.any():
                output[index] = fmask
                merge_counts["forward_only"] += 1
            elif bmask.any():
                output[index] = bmask
                merge_counts["backward_only"] += 1
            else:
                output[index] = np.zeros((height, width), bool)
                merge_counts["empty"] += 1
        completed[name] = output
        evidence[name] = {
            "raw_frames": len(raw),
            "forward_flow_hold_frames": [row["frame"] for row in forward_records],
            "backward_flow_hold_frames": [row["frame"] for row in backward_records],
            "merge_counts": merge_counts,
            "conservative_union": conservative_union,
            "missing_frames": [index for index, mask in output.items() if not mask.any()],
            "all_frames_present": all(mask.any() for mask in output.values()),
        }
    return completed, evidence


def tracker_protocol() -> dict[str, Any]:
    return {
        "human_core": {
            "canary_area_ratio_min": 0.20,
            "canary_area_ratio_max": 5.0,
            "flow_cycle_p90_max_px_half": 5.0,
            "adjacent_area_ratio_min": 0.65,
            "adjacent_area_ratio_max": 1.35,
            "boundary_gradient_median_min": 1.0,
        },
        "wearables": {
            "refresh_cadence_frames": 20,
            "flow_resolution": [640, 480],
            "flow_cycle_p90_max_px_half": 5.0,
            "adjacent_area_ratio_min": 0.65,
            "adjacent_area_ratio_max": 1.35,
            "own_human_min_distance_max_px": 24.0,
            "opposing_human_distance_margin_min_px": 4.0,
            "boundary_gradient_median_min": 1.0,
            "refresh_iou_with_warp_min": 0.08,
            "refresh_centroid_distance_max_px": 48.0,
            "refresh_area_ratio_min": 0.25,
            "refresh_area_ratio_max": 4.0,
            "refresh_target_point_must_be_inside": True,
            "refresh_acceptance_required": 1.0,
        },
        "task_object": {"wearable_raw_overlap_max_pixels": 0},
    }


def track_bidirectional(
    model: Any,
    output: Path,
    frame_paths: list[Path],
    human: dict[str, dict[int, np.ndarray]],
    objects: list[np.ndarray],
    height: int,
    width: int,
    flow_module: Any,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    protocol = tracker_protocol()
    chronological_roles = TRACKER_ROLES
    merged = {"right_tracker": {}, "left_tracker": {}}
    all_meta: dict[str, Any] = {}
    for direction, originals in {
        "forward": list(range(ANCHOR_FRAME, FRAME_COUNT)),
        "reverse": list(range(ANCHOR_FRAME, -1, -1)),
    }.items():
        view, paths = materialize_view(output / "tracker_views" / direction, originals)
        object_sequence = [objects[index] for index in originals]
        human_sequence = {
            internal: {slot: human[anatomical][original] for slot, original in enumerate(originals)}
            for internal, anatomical in TRACKER_HUMAN_INTERNAL.items()
        }
        cache = flow_module.FlowCache(paths, 640, 480)
        for internal_name, role in chronological_roles.items():
            stream, meta = flow_module.collect_flow_refresh_role(
                model,
                role,
                {
                    "object_protection_points": OBJECT_PROTECTION_POINTS,
                    "allow_registered_fallback_priming": True,
                },
                view,
                paths,
                human_sequence,
                object_sequence,
                cache,
                protocol,
                height,
                width,
                output / "tracker_refresh" / direction,
            )
            anatomical = TRACKER_ANATOMICAL[internal_name]
            for slot, mask in stream.items():
                merged[anatomical][originals[slot]] = mask
            all_meta[f"{direction}_{anatomical}"] = meta
    return merged, all_meta


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.62) -> None:
    value = np.asarray(color, np.float32)
    image[mask] = ((1.0 - alpha) * image[mask] + alpha * value).astype(np.uint8)


def write_outputs(
    output: Path,
    frame_paths: list[Path],
    human: dict[str, dict[int, np.ndarray]],
    tracker: dict[str, dict[int, np.ndarray]],
    objects: dict[str, dict[int, np.ndarray]],
) -> dict[str, Any]:
    role_root = output / "raw_role_masks"
    object_root = output / "protected_object_masks"
    union_root = output / "binary_union_masks"
    for name in [*human, *tracker]:
        (role_root / name).mkdir(parents=True, exist_ok=True)
    for name in [*objects, "object_union"]:
        (object_root / name).mkdir(parents=True, exist_ok=True)
    union_root.mkdir(parents=True)

    canary_tiles = []
    raw_overlap = []
    for index, source in enumerate(frame_paths):
        raw = cv2.imread(str(source), cv2.IMREAD_COLOR)
        object_union = np.zeros(raw.shape[:2], bool)
        for name, stream in objects.items():
            mask = stream.get(index, np.zeros(raw.shape[:2], bool))
            object_union |= mask
            Image.fromarray(mask.astype(np.uint8) * 255).save(object_root / name / f"{SOURCE_INDICES[index]:05d}.png")
        Image.fromarray(object_union.astype(np.uint8) * 255).save(object_root / "object_union" / f"{SOURCE_INDICES[index]:05d}.png")
        removal = np.zeros(raw.shape[:2], bool)
        for name, stream in {**human, **tracker}.items():
            mask = stream.get(index, np.zeros(raw.shape[:2], bool))
            removal |= mask
            Image.fromarray(mask.astype(np.uint8) * 255).save(role_root / name / f"{SOURCE_INDICES[index]:05d}.png")
        raw_overlap.append(int(np.logical_and(removal, object_union).sum()))
        published = removal & ~object_union
        Image.fromarray(published.astype(np.uint8) * 255).save(union_root / f"{SOURCE_INDICES[index]:05d}.png")
        if index in CANARY:
            overlay = raw.copy()
            empty = np.zeros(raw.shape[:2], bool)
            tint(overlay, human.get("left_human", {}).get(index, empty), (255, 50, 180))
            tint(overlay, human.get("right_human", {}).get(index, empty), (40, 180, 255))
            tint(overlay, tracker.get("left_tracker", {}).get(index, empty), (255, 240, 20), 0.72)
            tint(overlay, tracker.get("right_tracker", {}).get(index, empty), (255, 120, 20), 0.72)
            tint(overlay, object_union, (40, 255, 70), 0.76)
            tile = np.hstack((raw, overlay))
            tile = cv2.resize(tile, (640, 240), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (639, 27), (0, 0, 0), -1)
            cv2.putText(tile, f"f{SOURCE_INDICES[index]:05d} RAW | removal+protected", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            canary_tiles.append(tile)
    sheet = np.zeros((4 * 240, 3 * 640, 3), np.uint8)
    for slot, tile in enumerate(canary_tiles):
        row, column = divmod(slot, 3)
        sheet[row * 240:(row + 1) * 240, column * 640:(column + 1) * 640] = tile
    sheet_path = output / "MASK_CANARY12_RAW_VS_CLASSES.png"
    cv2.imwrite(str(sheet_path), sheet)
    return {
        "raw_role_masks": str(role_root),
        "protected_object_masks": str(object_root),
        "binary_union_masks": str(union_root),
        "canary12": str(sheet_path),
        "canary12_sha256": sha256(sheet_path),
        "raw_removal_object_overlap_pixels_max": max(raw_overlap),
        "published_union_object_overlap_pixels_max": 0,
    }


def main() -> int:
    global RAW, FRAME_COUNT, ANCHOR_FRAME, CANARY, SOURCE_INDICES
    global TASK_ID, SESSION_ID, HUMAN_ANCHORS, OBJECT_ROLE_SPECS
    global TRACKER_ROLES, TRACKER_ANATOMICAL, TRACKER_HUMAN_INTERNAL
    global OBJECT_PROTECTION_POINTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lease-holder", required=True)
    parser.add_argument("--task-config", type=Path)
    args = parser.parse_args()
    if args.task_config is not None:
        task = json.loads(args.task_config.read_text())
        RAW = Path(task["raw_all_data"])
        SOURCE_INDICES = [int(value) for value in task["source_frame_indices"]]
        if len(SOURCE_INDICES) != 12 or len(set(SOURCE_INDICES)) != 12:
            raise RuntimeError("formal sampled canary requires 12 unique source frames")
        FRAME_COUNT = len(SOURCE_INDICES)
        ANCHOR_FRAME = int(task["anchor_slot"])
        CANARY = list(range(FRAME_COUNT))
        TASK_ID = str(task["task_id"])
        SESSION_ID = str(task["session_id"])
        HUMAN_ANCHORS = {name: tuple(map(int, point)) for name, point in task["human_anchors"].items()}
        OBJECT_ROLE_SPECS = task["object_role_specs"]
        TRACKER_ROLES = task["tracker_roles"]
        TRACKER_ANATOMICAL = task["tracker_anatomical"]
        TRACKER_HUMAN_INTERNAL = task["tracker_human_internal"]
        OBJECT_PROTECTION_POINTS = task["object_protection_points"]
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"fresh output root required: {output}")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA drift")
    lease = json.loads((PROJECT / "_run/GPU_LEASE.json").read_text())
    if lease.get("status") != "ACQUIRED" or lease.get("holder") != args.lease_holder:
        raise RuntimeError("central GPU lease mismatch")
    output.mkdir(parents=True)
    frame_root, frame_paths = materialize_view(output / "input_frames", list(range(FRAME_COUNT)))
    first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    height, width = first.shape[:2]
    if (height, width) != (960, 1280):
        raise RuntimeError("PICO geometry drift")

    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(CODE_ROOT))
    from tools import run_assisted_bilateral_flow_refresh as flow_module
    from tools import run_newtask_baseline_sam31_mask_probe as baseline

    adapter_module = baseline.load_adapter_module()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT
    )
    try:
        human_raw, human_meta = collect_text_roles(
            adapter.model,
            frame_root,
            "a person's hand and forearm",
            ANCHOR_FRAME,
            HUMAN_ANCHORS,
            height,
            width,
            True,
            20.0,
        )
        flow_cache = flow_module.FlowCache(frame_paths, 640, 480)
        human, human_completion = complete_with_flow(
            human_raw, frame_paths, flow_cache, height, width
        )

        object_raw, object_meta = collect_point_roles(
            adapter.model,
            frame_root,
            OBJECT_ROLE_SPECS,
            height,
            width,
        )
        objects, object_completion = complete_with_flow(
            object_raw,
            frame_paths,
            flow_cache,
            height,
            width,
            conservative_union=True,
        )
        object_union = [
            np.logical_or.reduce([
                stream.get(index, np.zeros((height, width), bool))
                for stream in objects.values()
            ])
            for index in range(FRAME_COUNT)
        ]
        object_union_missing = [index for index, mask in enumerate(object_union) if not mask.any()]

        tracker, tracker_meta = track_bidirectional(
            adapter.model,
            output,
            frame_paths,
            human,
            object_union,
            height,
            width,
            flow_module,
        )
    finally:
        adapter.predictor.shutdown()

    role_presence = {
        **{name: sum(mask.any() for mask in stream.values()) for name, stream in human.items()},
        **{name: sum(mask.any() for mask in stream.values()) for name, stream in tracker.items()},
        **{name: sum(mask.any() for mask in stream.values()) for name, stream in objects.items()},
    }
    tracker_pass = all(meta["status"] == "PASS" for meta in tracker_meta.values())
    human_pass = all(value["all_frames_present"] for value in human_completion.values())
    object_canary_pass = all(
        any(objects[name].get(index, np.zeros((height, width), bool)).any() for name in objects)
        for index in CANARY
    )
    automatic_pass = bool(
        human_meta["status"] == "COMPLETE"
        and object_meta["status"] == "COMPLETE"
        and human_pass
        and tracker_pass
        and object_canary_pass
        and all(value["all_frames_present"] for value in object_completion.values())
    )
    artifacts = write_outputs(output, frame_paths, human, tracker, objects)
    result = {
        "schema_version": "formal-sampled-pico-bilateral-mask-temporal-v1",
        "status": "PENDING_ROOT_VISUAL_REVIEW" if automatic_pass else "HOLD_AUTOMATIC_GATE",
        "automatic_gate_pass": automatic_pass,
        "consumption_authorized": False,
        "task_id": TASK_ID,
        "session_id": SESSION_ID,
        "frame_count": FRAME_COUNT,
        "source_frame_indices": SOURCE_INDICES,
        "geometry": [width, height],
        "anatomical_mapping": {"upper_internal": "right", "lower_internal": "left"},
        "human": {"inference": human_meta, "completion": human_completion},
        "objects": {"inference": object_meta, "completion": object_completion},
        "protected_object_union_missing_frames": object_union_missing,
        "tracker": tracker_meta,
        "role_presence_frames": role_presence,
        "artifacts": artifacts,
        "forbidden_repairs_used": [],
        "pins": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "runner_sha256": sha256(Path(__file__)),
            "build_evidence": build_evidence,
        },
        "resource": {
            "wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "claim_limit": "Event-stratified sampled frames are a bounded FIT canary, not contiguous temporal validation. No Clean consumption until automatic gates and root visual review both pass; object masks are protected raw SAM instances, not Object6D authority.",
    }
    (output / "RESULT.json").write_text(
        json.dumps(result, indent=2, default=json_default) + "\n"
    )
    print(json.dumps({"status": result["status"], "resource": result["resource"], "canary": artifacts["canary12"]}))
    return 0 if automatic_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
