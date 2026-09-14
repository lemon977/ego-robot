#!/usr/bin/env python3
"""Fixed assisted bilateral hand/wearable SAM3.1 temporal runner.

The runner consumes a decoded natural-display frame directory and writes raw
per-role masks, a review-only bilateral union, a class overlay and auditable
canary metrics.  It never changes a SAM mask with morphology, colour, fill or
object subtraction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT = Path("/mnt/workspace/code/chaoyang")
sys.path.insert(0, str(PROJECT))
from tools import run_assisted_bilateral_sam31_point_canary as canary
from tools import run_newtask_baseline_sam31_mask_probe as baseline


PROTOCOL = PROJECT / "_run/gpt_mask_assisted_bilateral_wearable_contract_20260902_v1/PROTOCOL.json"
ANCHORS = PROJECT / "_run/gpt_mask_assisted_bilateral_wearable_contract_20260902_v1/FRAME0_ANCHORS.json"
OBJECT_MASK_ROOTS = {
    "chips": PROJECT / "_run/newtask_baseline_mask_probe_20260901_v1/action/chips/object_chip/raw_single_instance_masks",
    "poker": PROJECT / "_run/newtask_baseline_mask_probe_20260901_v1/action/poker/object_card/raw_single_instance_masks",
}
ROLE_COLORS = {
    "upper_human_core": np.asarray([255, 40, 180], np.float32),
    "lower_human_core": np.asarray([30, 155, 255], np.float32),
    "upper_tracker_wearable": np.asarray([255, 245, 30], np.float32),
    "lower_tracker_wearable": np.asarray([255, 145, 25], np.float32),
    "lower_sleeve_cuff": np.asarray([60, 255, 90], np.float32),
    "object_raw": np.asarray([20, 255, 20], np.float32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("chips", "poker"), required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="Evaluate and emit the frozen 12-frame gate without promoting full masks/video even on PASS.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distance(mask: np.ndarray, point: list[int]) -> float | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    x, y = point
    return float(np.sqrt((xs - x) ** 2 + (ys - y) ** 2).min())


def geometry(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if not len(xs):
        return {"area_pixels": 0, "bbox_xyxy": None, "centroid_xy": None, "touches_right_edge": False}
    return {
        "area_pixels": int(mask.sum()),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "touches_right_edge": bool(xs.max() == mask.shape[1] - 1),
    }


def select_text_ids(outputs: dict, roles: list[dict], height: int, width: int) -> tuple[dict[str, int], dict]:
    masks, ids = canary.normalize_masks(outputs, height, width)
    selected: dict[str, int] = {}
    evidence = {}
    used: set[int] = set()
    for role in roles:
        point = role["positive"][0]
        rows = []
        for index, mask in enumerate(masks):
            value = distance(mask, point)
            rows.append({"raw_id": int(ids[index]), "anchor_distance_px": value, **geometry(mask)})
        eligible = sorted(
            (row for row in rows if row["anchor_distance_px"] is not None and row["anchor_distance_px"] <= 12.0 and row["raw_id"] not in used),
            key=lambda row: (row["anchor_distance_px"], row["area_pixels"], row["raw_id"]),
        )
        if eligible:
            selected[role["name"]] = int(eligible[0]["raw_id"])
            used.add(int(eligible[0]["raw_id"]))
        evidence[role["name"]] = {"anchor_xy": point, "candidates": rows, "chosen_raw_id": selected.get(role["name"])}
    return selected, evidence


def collect_text_humans(model, frame_root: Path, frame_count: int, roles: list[dict], height: int, width: int) -> tuple[dict[str, dict[int, np.ndarray]], dict]:
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True, async_loading_frames=False)
    streams = {role["name"]: {} for role in roles}
    try:
        _, initial = model.add_prompt(inference_state=state, frame_idx=0, text_str="a person's hand and forearm", output_prob_thresh=0.5)
        ids_by_role, selection = select_text_ids(initial, roles, height, width)
        if len(ids_by_role) != len(roles):
            return streams, {"status": "HOLD_TEXT_ID_SELECTION", "selection": selection}
        for frame_idx, outputs in model.propagate_in_video(
            inference_state=state, start_frame_idx=0, max_frame_num_to_track=frame_count,
            reverse=False, output_prob_thresh=0.5,
        ):
            masks, ids = canary.normalize_masks(outputs, height, width)
            by_id = {int(obj_id): masks[index] for index, obj_id in enumerate(ids)}
            for role_name, raw_id in ids_by_role.items():
                if raw_id in by_id:
                    streams[role_name][int(frame_idx)] = by_id[raw_id]
        return streams, {"status": "COMPLETE", "selection": selection, "raw_ids": ids_by_role}
    finally:
        state.clear()


def collect_point_role(model, frame_root: Path, frame_count: int, role: dict, height: int, width: int) -> tuple[dict[int, np.ndarray], dict]:
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True, async_loading_frames=False)
    masks_by_frame: dict[int, np.ndarray] = {}
    points = [*role["positive"], *role["negative"]]
    labels = [1] * len(role["positive"]) + [0] * len(role["negative"])
    points_rel = [[x / width, y / height] for x, y in points]
    raw_id = int(role["id"])
    priming_id = raw_id + 10000
    try:
        prime_x, prime_y = role["priming_point"]
        model.add_prompt(
            inference_state=state, frame_idx=0,
            points=torch.tensor([[prime_x / width, prime_y / height]], dtype=torch.float32),
            point_labels=torch.tensor([1], dtype=torch.int32),
            obj_id=priming_id, rel_coordinates=True, clear_old_points=True,
            output_prob_thresh=0.5,
        )
        _, initial = model.add_prompt(
            inference_state=state, frame_idx=0,
            points=torch.tensor(points_rel, dtype=torch.float32),
            point_labels=torch.tensor(labels, dtype=torch.int32),
            obj_id=raw_id, rel_coordinates=True, clear_old_points=True,
            output_prob_thresh=0.5,
        )
        initial_masks, initial_ids = canary.normalize_masks(initial, height, width)
        initial_by_id = {int(obj_id): initial_masks[index] for index, obj_id in enumerate(initial_ids)}
        initial_mask = initial_by_id.get(raw_id, np.zeros((height, width), bool))
        point_gate = {
            "positive_coverage": [bool(initial_mask[y, x]) for x, y in role["positive"]],
            "negative_exclusion": [not bool(initial_mask[y, x]) for x, y in role["negative"]],
            **geometry(initial_mask),
        }
        if not point_gate["area_pixels"] or not all(point_gate["positive_coverage"]) or not all(point_gate["negative_exclusion"]):
            return masks_by_frame, {"status": "HOLD_INITIAL_POINT_GATE", "point_gate": point_gate}
        for frame_idx, outputs in model.propagate_in_video(
            inference_state=state, start_frame_idx=0, max_frame_num_to_track=frame_count,
            reverse=False, output_prob_thresh=0.5,
        ):
            masks, ids = canary.normalize_masks(outputs, height, width)
            matches = np.flatnonzero(ids == raw_id)
            if len(matches) == 1:
                masks_by_frame[int(frame_idx)] = masks[int(matches[0])]
        return masks_by_frame, {"status": "COMPLETE", "raw_id": raw_id, "discarded_priming_raw_id": priming_id, "priming_output_consumed": False, "point_gate": point_gate}
    finally:
        state.clear()


def adjacent_iou(first: np.ndarray, second: np.ndarray) -> float | None:
    union = int(np.logical_or(first, second).sum())
    if not union:
        return None
    return float(np.logical_and(first, second).sum() / union)


def tint(raw: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.65) -> None:
    raw[mask] = ((1.0 - alpha) * raw[mask].astype(np.float32) + alpha * color).astype(np.uint8)


def evaluate_canary(streams: dict[str, dict[int, np.ndarray]], frame_count: int, protocol: dict) -> dict:
    indices = sorted({min(frame_count - 1, int(round(value * (frame_count - 1)))) for value in protocol["temporal_contract"]["canary_normalized_positions"]})
    roles = {}
    for role_name, stream in streams.items():
        initial = stream.get(0)
        initial_area = int(initial.sum()) if initial is not None else 0
        frame_rows = []
        for index in indices:
            mask = stream.get(index)
            area = int(mask.sum()) if mask is not None else 0
            frame_rows.append({
                "frame_index": index,
                "present": area > 0,
                "area_pixels": area,
                "area_ratio_to_initial": area / max(initial_area, 1),
                "area_ratio_pass": bool(0.25 <= area / max(initial_area, 1) <= 4.0),
                "geometry": geometry(mask) if mask is not None else geometry(np.zeros_like(initial) if initial is not None else np.zeros((1, 1), bool)),
            })
        roles[role_name] = {
            "initial_area_pixels": initial_area,
            "present_count": sum(row["present"] for row in frame_rows),
            "area_ratio_pass_count": sum(row["area_ratio_pass"] for row in frame_rows),
            "frames": frame_rows,
        }
    order_rows = []
    for slot, index in enumerate(indices):
        upper = roles["upper_human_core"]["frames"][slot]["geometry"]["centroid_xy"]
        lower = roles["lower_human_core"]["frames"][slot]["geometry"]["centroid_xy"]
        order_rows.append({
            "frame_index": index,
            "upper_centroid_y": upper[1] if upper else None,
            "lower_centroid_y": lower[1] if lower else None,
            "pass": bool(upper is not None and lower is not None and upper[1] < lower[1]),
        })
    passed = all(
        value["present_count"] == len(indices) and value["area_ratio_pass_count"] == len(indices)
        for value in roles.values()
    ) and all(row["pass"] for row in order_rows)
    return {"status": "PASS" if passed else "HOLD", "indices": indices, "roles": roles, "upper_before_lower": order_rows}


def write_canary_only(args: argparse.Namespace, frame_paths: list[Path], streams: dict[str, dict[int, np.ndarray]], evaluation: dict) -> Path:
    root = args.output_root / "canary_only"
    root.mkdir(parents=True)
    object_root = OBJECT_MASK_ROOTS[args.case]
    pairs = []
    for frame_idx in evaluation["indices"]:
        raw = np.asarray(Image.open(frame_paths[frame_idx]).convert("RGB"))
        overlay = raw.copy()
        for role_name, stream in streams.items():
            mask = stream.get(frame_idx, np.zeros(raw.shape[:2], bool))
            tint(overlay, mask, ROLE_COLORS[role_name])
        object_path = object_root / f"{frame_idx:05d}.png"
        if object_path.is_file():
            tint(overlay, np.asarray(Image.open(object_path).convert("L")) > 0, ROLE_COLORS["object_raw"], alpha=0.78)
        raw_small = cv2.resize(raw, (270, 480), interpolation=cv2.INTER_AREA)
        overlay_small = cv2.resize(overlay, (270, 480), interpolation=cv2.INTER_AREA)
        pair = np.concatenate([raw_small, overlay_small], axis=1)
        cv2.rectangle(pair, (0, 0), (539, 42), (0, 0, 0), -1)
        cv2.putText(pair, f"{args.case} f{frame_idx:04d} | CANARY {evaluation['status']}", (7, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        pairs.append(pair)
    sheet = np.zeros((4 * 480, 3 * 540, 3), np.uint8)
    for slot, tile in enumerate(pairs):
        row, column = divmod(slot, 3)
        sheet[row * 480:(row + 1) * 480, column * 540:(column + 1) * 540] = tile
    output = root / f"{args.case.upper()}_BILATERAL_CANARY12_{evaluation['status']}.png"
    Image.fromarray(sheet).save(output)
    return output


def write_artifacts(args: argparse.Namespace, frame_paths: list[Path], streams: dict[str, dict[int, np.ndarray]], stream_meta: dict, protocol: dict) -> dict:
    root = args.output_root
    role_roots = {}
    for role_name in streams:
        path = root / "raw_class_masks" / role_name
        path.mkdir(parents=True)
        role_roots[role_name] = path
    union_root = root / "binary_union_masks"
    review_root = root / "review_frames"
    union_root.mkdir()
    review_root.mkdir()
    height, width = np.asarray(Image.open(frame_paths[0]).convert("RGB")).shape[:2]
    object_root = OBJECT_MASK_ROOTS[args.case]
    zeros = np.zeros((height, width), bool)
    canary_indices = sorted({min(args.frame_count - 1, int(round(value * (args.frame_count - 1)))) for value in protocol["temporal_contract"]["canary_normalized_positions"]})
    rows = []
    per_role_rows = {name: [] for name in streams}
    canary_pairs = []
    object_overlap = []
    previous = {name: None for name in streams}
    adjacent = {name: [] for name in streams}
    for frame_idx, frame_path in enumerate(frame_paths):
        raw = np.asarray(Image.open(frame_path).convert("RGB"))
        overlay = raw.copy()
        union = np.zeros((height, width), bool)
        role_geometry = {}
        for role_name, role_stream in streams.items():
            mask = role_stream.get(frame_idx, zeros)
            Image.fromarray(mask.astype(np.uint8) * 255).save(role_roots[role_name] / f"{frame_idx:05d}.png")
            union |= mask
            tint(overlay, mask, ROLE_COLORS[role_name])
            geom = geometry(mask)
            role_geometry[role_name] = geom
            per_role_rows[role_name].append({"frame_index": frame_idx, **geom})
            if previous[role_name] is not None:
                value = adjacent_iou(previous[role_name], mask)
                if value is not None:
                    adjacent[role_name].append(value)
            previous[role_name] = mask
        Image.fromarray(union.astype(np.uint8) * 255).save(union_root / f"{frame_idx:05d}.png")
        object_path = object_root / f"{frame_idx:05d}.png"
        object_mask = np.asarray(Image.open(object_path).convert("L")) > 0 if object_path.is_file() else zeros
        overlap = int(np.logical_and(union, object_mask).sum())
        object_overlap.append({"frame_index": frame_idx, "object_raw_available": bool(object_path.is_file()), "raw_union_object_overlap_pixels": overlap})
        if object_mask.any():
            tint(overlay, object_mask, ROLE_COLORS["object_raw"], alpha=0.78)
        cv2.rectangle(overlay, (0, 0), (width - 1, 48), (0, 0, 0), -1)
        cv2.putText(overlay, f"{args.case.upper()} f{frame_idx:04d} | upper+lower hand / tracker / cuff | object raw last", (7, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        raw_small = cv2.resize(raw, (270, 480), interpolation=cv2.INTER_AREA)
        overlay_small = cv2.resize(overlay, (270, 480), interpolation=cv2.INTER_AREA)
        pair = np.concatenate([raw_small, overlay_small], axis=1)
        Image.fromarray(pair).save(review_root / f"{frame_idx:05d}.png")
        if frame_idx in canary_indices:
            canary_pairs.append(pair)
        rows.append({"frame_index": frame_idx, "roles": role_geometry, "raw_union_object_overlap_pixels": overlap})
    contact = root / f"{args.case.upper()}_BILATERAL_CANARY12.png"
    sheet = np.zeros((4 * 480, 3 * 540, 3), np.uint8)
    for slot, tile in enumerate(canary_pairs):
        row, column = divmod(slot, 3)
        sheet[row * 480:(row + 1) * 480, column * 540:(column + 1) * 540] = tile
    Image.fromarray(sheet).save(contact)
    video = root / f"{args.case.upper()}_RAW_VS_BILATERAL_CLASSES.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(args.fps),
        "-i", str(review_root / "%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(video),
    ], check=True)
    role_summary = {}
    for role_name, frame_rows in per_role_rows.items():
        areas = [row["area_pixels"] for row in frame_rows]
        initial = max(areas[0], 1)
        canary_rows = [frame_rows[index] for index in canary_indices]
        role_summary[role_name] = {
            "present_frames": sum(area > 0 for area in areas),
            "frame_count": args.frame_count,
            "area_initial_min_median_max": [areas[0], min(areas), float(np.median(areas)), max(areas)],
            "canary_present": sum(row["area_pixels"] > 0 for row in canary_rows),
            "canary_area_ratio_pass": [bool(0.25 <= row["area_pixels"] / initial <= 4.0) for row in canary_rows],
            "adjacent_iou_median": float(np.median(adjacent[role_name])) if adjacent[role_name] else None,
        }
    order_pass = []
    for index in canary_indices:
        upper = rows[index]["roles"]["upper_human_core"]["centroid_xy"]
        lower = rows[index]["roles"]["lower_human_core"]["centroid_xy"]
        order_pass.append(bool(upper is not None and lower is not None and upper[1] < lower[1]))
    canary_pass = all(
        summary["canary_present"] == len(canary_indices) and all(summary["canary_area_ratio_pass"])
        for summary in role_summary.values()
    ) and all(order_pass)
    result = {
        "schema_version": "assisted-bilateral-wearable-temporal-result-v1",
        "status": "PASS_CANARY12_FULL_ARTIFACTS_EMITTED" if canary_pass else "HOLD_CANARY12_FULL_ARTIFACTS_DIAGNOSTIC_ONLY",
        "consumption_authorized": bool(canary_pass),
        "case": args.case,
        "frame_count": args.frame_count,
        "canary_indices": canary_indices,
        "canary_upper_before_lower_identity_gate": order_pass,
        "stream_initialization": stream_meta,
        "roles": role_summary,
        "object_overlap": object_overlap,
        "object_priority": "raw object mask drawn last; class raw masks remain unmodified",
        "anchor_burden": {
            "anchor_frames": 1,
            "human_text_role_anchors": 2,
            "point_role_instances": len([name for name in streams if not name.endswith("human_core")]),
            "maximum_allowed_anchor_frames_per_role": 4
        },
        "forbidden_repairs_used": [],
        "artifacts": {"contact_sheet": str(contact), "video": str(video)},
        "claim_limit": "assisted phone-baseline development; not zero-click, PICO, blind, or production generalization",
    }
    (root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


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
        raise RuntimeError("natural display shape drift")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["status"] != "FROZEN_BEFORE_TEMPORAL_CANARY":
        raise RuntimeError("protocol status drift")
    lease = json.loads((PROJECT / "_run/GPU_LEASE.json").read_text())
    if lease.get("status") not in {"ACTIVE", "ACQUIRED"} or lease.get("holder") != "mask-bilateral-tracker-final":
        raise RuntimeError("central GPU lease is not active")
    args.output_root.mkdir(parents=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sys.path.insert(0, str(canary.CODE_ROOT))
    adapter_module = baseline.load_adapter_module()
    adapter, build_evidence = adapter_module.build_pinned_adapter(official_code_root=canary.CODE_ROOT, checkpoint_path=canary.CHECKPOINT)
    roles = canary.SPECS[args.case]["objects"]
    human_roles = [role for role in roles if role["class"] == "human_core"]
    point_roles = [
        {**role, "priming_point": canary.SPECS[args.case]["object_protection_points"][0]}
        for role in roles if role["class"] != "human_core"
    ]
    streams = {}
    stream_meta = {}
    try:
        human_streams, human_meta = collect_text_humans(adapter.model, args.frame_root, args.frame_count, human_roles, height, width)
        streams.update(human_streams)
        stream_meta["human_core"] = human_meta
        if human_meta["status"] != "COMPLETE":
            raise RuntimeError("human text initialization HOLD")
        for role in point_roles:
            role_stream, role_meta = collect_point_role(adapter.model, args.frame_root, args.frame_count, role, height, width)
            streams[role["name"]] = role_stream
            stream_meta[role["name"]] = role_meta
            if role_meta["status"] != "COMPLETE":
                raise RuntimeError(f"{role['name']} initialization HOLD")
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        adapter.predictor.shutdown()
    canary_evaluation = evaluate_canary(streams, args.frame_count, protocol)
    canary_path = write_canary_only(args, frame_paths, streams, canary_evaluation)
    (args.output_root / "CANARY12_RESULT.json").write_text(json.dumps(canary_evaluation, indent=2) + "\n")
    if canary_evaluation["status"] != "PASS":
        result = {
            "schema_version": "assisted-bilateral-wearable-temporal-result-v1",
            "status": "HOLD_CANARY12_NO_FULL_ARTIFACT_PROMOTION",
            "consumption_authorized": False,
            "case": args.case,
            "frame_count": args.frame_count,
            "canary": canary_evaluation,
            "stream_initialization": stream_meta,
            "artifacts": {"canary_contact_sheet": str(canary_path)},
            "forbidden_repairs_used": [],
            "claim_limit": "assisted phone-baseline diagnostic HOLD; no full artifact promotion",
        }
    elif args.canary_only:
        result = {
            "schema_version": "assisted-bilateral-wearable-temporal-result-v1",
            "status": "PASS_CANARY12_NO_FULL_ARTIFACT_PROMOTION_BY_REQUEST",
            "consumption_authorized": False,
            "case": args.case,
            "frame_count": args.frame_count,
            "canary": canary_evaluation,
            "stream_initialization": stream_meta,
            "artifacts": {"canary_contact_sheet": str(canary_path)},
            "forbidden_repairs_used": [],
            "claim_limit": "assisted phone-baseline 12-frame diagnostic PASS only; full artifact promotion deliberately disabled",
        }
    else:
        result = write_artifacts(args, frame_paths, streams, stream_meta, protocol)
        result["pre_promotion_canary"] = canary_evaluation
        result["artifacts"]["pre_promotion_canary_contact_sheet"] = str(canary_path)
    result["resource"] = {
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    result["pins"] = {
        "protocol_sha256": sha256(PROTOCOL),
        "anchors_sha256": sha256(ANCHORS),
        "checkpoint_sha256": canary.CHECKPOINT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "build_evidence": build_evidence,
    }
    (args.output_root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], **result["resource"]}))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
