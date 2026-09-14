#!/usr/bin/env python3
"""Run a development-only SAM3.1 single-object video-mask probe.

The source phone videos are explicitly baseline/development material.  This
runner never promotes them to PICO input, ground truth, a formal clean donor,
or a blind test.  Every saved binary mask is one unmodified SAM3.1 instance;
spatial gates are used only to select/audit an instance, never to alter pixels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT = Path("/mnt/workspace/code/chaoyang")
DEFAULT_ROOT = PROJECT / "_run/newtask_baseline_mask_probe_20260901_v1"
CODE_ROOT = PROJECT / "third_party/SAM3"
CHECKPOINT = PROJECT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
ADAPTER_PATH = PROJECT / "pipeline/sam31_compat_adapter_v1.py"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
ADAPTER_SHA256 = "32a4b2db7b3fcfa0e150d5552752dbb2727f78ee013901471a9b93683d4dce50"


@dataclass(frozen=True)
class Case:
    name: str
    source: Path
    source_sha256: str
    frame_count: int
    width: int
    height: int
    anchor_xy: tuple[int, int]
    centroid_window_xyxy: tuple[int, int, int, int]
    area_range: tuple[int, int]
    max_bbox_wh: tuple[int, int]
    max_anchor_distance_px: float
    prompts: tuple[str, ...]
    target_semantics: str


CASES = {
    "puke": Case(
        name="puke",
        source=PROJECT / "data/inputs/baseline_background/puke.mp4",
        source_sha256="d041d3ce3b76b32c45b8ace7437e741567c297e4d6611e54d746b7c33f4013ec",
        frame_count=94,
        width=1280,
        height=720,
        anchor_xy=(858, 208),
        centroid_window_xyxy=(790, 125, 930, 295),
        area_range=(2_500, 15_000),
        max_bbox_wh=(165, 180),
        max_anchor_distance_px=45.0,
        prompts=("a playing card", "playing card", "a blue playing card", "a single playing card"),
        target_semantics="middle playing-card instance on the rack at frame 0",
    ),
    "shupian": Case(
        name="shupian",
        source=PROJECT / "data/inputs/baseline_background/shupian.mp4",
        source_sha256="645340e3e85e3d46b8c3ee58432fc1ab26798b67e6e31082db67a5da315b34b0",
        frame_count=95,
        width=1280,
        height=720,
        anchor_xy=(904, 223),
        centroid_window_xyxy=(835, 150, 970, 295),
        area_range=(650, 6_500),
        max_bbox_wh=(120, 120),
        max_anchor_distance_px=45.0,
        prompts=("a potato chip", "potato chip", "a single potato chip", "a yellow potato chip"),
        target_semantics="upper potato-chip instance at frame 0",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def load_adapter_module() -> Any:
    spec = importlib.util.spec_from_file_location("newtask_baseline_sam31_adapter", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned SAM3.1 compatibility adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize(outputs: dict[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks: Any = outputs.get("out_binary_masks")
    if masks is None:
        mask_array = np.zeros((0, height, width), dtype=bool)
    else:
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        elif isinstance(masks, (list, tuple)):
            masks = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item) for item in masks]
            masks = np.stack(masks) if masks else np.zeros((0, height, width))
        mask_array = np.asarray(masks)
        while mask_array.ndim > 3 and mask_array.shape[1] == 1:
            mask_array = np.squeeze(mask_array, axis=1)
        if mask_array.ndim == 2:
            mask_array = mask_array[None]
        if mask_array.size == 0:
            mask_array = np.zeros((0, height, width), dtype=bool)
        if mask_array.ndim != 3 or mask_array.shape[1:] != (height, width):
            raise RuntimeError(f"unexpected SAM mask shape: {mask_array.shape}")
        mask_array = mask_array.astype(bool)
    scores: Any = outputs.get("out_probs", [])
    ids: Any = outputs.get("out_obj_ids", [])
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    id_array = np.asarray(ids).reshape(-1)
    if len(score_array) != len(mask_array) or len(id_array) != len(mask_array):
        raise RuntimeError("SAM masks/scores/object IDs length mismatch")
    return mask_array, score_array, id_array


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"area_pixels": 0, "bbox_xyxy": None, "bbox_wh": None, "centroid_xy": None}
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "area_pixels": int(len(xs)),
        "bbox_xyxy": bbox,
        "bbox_wh": [bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def candidate_metrics(mask: np.ndarray, score: float, obj_id: Any, prompt: str, prompt_index: int, case: Case) -> dict[str, Any]:
    row = {
        "prompt": prompt,
        "prompt_index": prompt_index,
        "object_id": int(obj_id),
        "score": float(score),
        **mask_geometry(mask),
    }
    if row["area_pixels"] == 0:
        row.update({"anchor_inside": False, "anchor_to_mask_distance_px": None, "selector_pass": False})
        return row
    ax, ay = case.anchor_xy
    ys, xs = np.where(mask)
    inside = bool(mask[ay, ax])
    distance = 0.0 if inside else float(np.sqrt((xs - ax) ** 2 + (ys - ay) ** 2).min())
    cx, cy = row["centroid_xy"]
    x0, y0, x1, y1 = case.centroid_window_xyxy
    bw, bh = row["bbox_wh"]
    selector_pass = (
        case.area_range[0] <= row["area_pixels"] <= case.area_range[1]
        and bw <= case.max_bbox_wh[0]
        and bh <= case.max_bbox_wh[1]
        and x0 <= cx <= x1
        and y0 <= cy <= y1
        and distance <= case.max_anchor_distance_px
    )
    row.update({"anchor_inside": inside, "anchor_to_mask_distance_px": distance, "selector_pass": bool(selector_pass)})
    return row


def rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (not row["anchor_inside"], row["anchor_to_mask_distance_px"], -row["score"], row["prompt_index"])


def tint(raw_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = raw_rgb.copy()
    overlay[mask] = (0.30 * overlay[mask] + 0.70 * np.asarray([255, 0, 180])).astype(np.uint8)
    return overlay


def save_sweep_candidate(raw: np.ndarray, mask: np.ndarray, row: dict[str, Any], path_stem: Path, case: Case) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path_stem.with_suffix(".mask.png"))
    overlay = tint(raw, mask)
    cv2.circle(overlay, case.anchor_xy, 6, (0, 255, 255), 2)
    cv2.rectangle(overlay, (0, 0), (case.width - 1, 48), (0, 0, 0), -1)
    cv2.putText(overlay, f"{row['prompt']} id={row['object_id']} score={row['score']:.3f} area={row['area_pixels']} pass={row['selector_pass']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    Image.fromarray(overlay, mode="RGB").save(path_stem.with_suffix(".overlay.png"))


def adjacent_iou(previous: np.ndarray, current: np.ndarray) -> float | None:
    union = int(np.logical_or(previous, current).sum())
    return float(np.logical_and(previous, current).sum() / union) if union else None


def audit_frame(mask: np.ndarray, case: Case) -> dict[str, Any]:
    row = mask_geometry(mask)
    if row["area_pixels"] == 0:
        row["spatial_gate_pass"] = False
        return row
    cx, cy = row["centroid_xy"]
    x0, y0, x1, y1 = case.centroid_window_xyxy
    bw, bh = row["bbox_wh"]
    row["spatial_gate_pass"] = bool(
        case.area_range[0] <= row["area_pixels"] <= case.area_range[1]
        and bw <= case.max_bbox_wh[0]
        and bh <= case.max_bbox_wh[1]
        and x0 <= cx <= x1
        and y0 <= cy <= y1
    )
    return row


def run_case(adapter: Any, build_evidence: dict[str, Any], case: Case, root: Path) -> dict[str, Any]:
    case_root = root / case.name
    if case_root.exists():
        raise RuntimeError(f"fresh case output already exists: {case_root}")
    frames_root = root / "input_frames" / case.name
    frame_paths = sorted(frames_root.glob("*.png"))
    if len(frame_paths) != case.frame_count:
        raise RuntimeError(f"{case.name}: decoded frame count drift")
    prompt_frame = root / "prompt_frame" / case.name
    prompt_frame.mkdir(parents=True, exist_ok=False)
    os.symlink(frame_paths[0], prompt_frame / frame_paths[0].name)
    case_root.mkdir()
    sweep_root = case_root / "prompt_sweep"
    masks_root = case_root / "raw_single_instance_masks"
    overlays_root = case_root / "mask_overlays"
    review_root = case_root / "review_frames"
    for path in (sweep_root, masks_root, overlays_root, review_root):
        path.mkdir()
    raw0 = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    all_rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(case.prompts):
        session_id = f"newtask-{case.name}-sweep-{prompt_index}"
        adapter.handle_request({"type": "start_session", "session_id": session_id, "resource_path": str(prompt_frame), "offload_video_to_cpu": True, "offload_state_to_cpu": False, "async_loading_frames": False})
        try:
            initial = adapter.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt, "output_prob_thresh": 0.5})
            masks, scores, ids = normalize(initial["outputs"], case.height, case.width)
            for instance_index in range(len(masks)):
                row = candidate_metrics(masks[instance_index], float(scores[instance_index]), ids[instance_index], prompt, prompt_index, case)
                all_rows.append(row)
                save_sweep_candidate(raw0, masks[instance_index], row, sweep_root / f"prompt{prompt_index}_instance{instance_index}", case)
        finally:
            adapter.handle_request({"type": "close_session", "session_id": session_id})
            gc.collect()
            torch.cuda.empty_cache()
    eligible = sorted((row for row in all_rows if row["selector_pass"]), key=rank)
    (case_root / "PROMPT_SWEEP.json").write_text(json.dumps({"case": case.name, "target_semantics": case.target_semantics, "anchor_xy": case.anchor_xy, "candidates": all_rows, "eligible_ranked": eligible}, indent=2) + "\n")
    if not eligible:
        raise RuntimeError(f"{case.name}: no unmodified text instance passed the frozen selector")
    chosen = eligible[0]
    chosen_prompt = str(chosen["prompt"])
    session_id = f"newtask-{case.name}-full-video"
    adapter.handle_request({"type": "start_session", "session_id": session_id, "resource_path": str(frames_root), "offload_video_to_cpu": True, "offload_state_to_cpu": False, "async_loading_frames": False})
    selected_id: int | None = None
    selected_masks: dict[int, np.ndarray] = {}
    propagation_started = time.perf_counter()
    try:
        initial = adapter.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": chosen_prompt, "output_prob_thresh": 0.5})
        initial_masks, initial_scores, initial_ids = normalize(initial["outputs"], case.height, case.width)
        initial_rows = [candidate_metrics(initial_masks[index], float(initial_scores[index]), initial_ids[index], chosen_prompt, int(chosen["prompt_index"]), case) for index in range(len(initial_masks))]
        initial_eligible = sorted((row for row in initial_rows if row["selector_pass"]), key=rank)
        if not initial_eligible:
            raise RuntimeError(f"{case.name}: full-video initial selection did not reproduce")
        selected_id = int(initial_eligible[0]["object_id"])
        matches = np.flatnonzero(initial_ids.astype(np.int64) == selected_id)
        if len(matches) != 1:
            raise RuntimeError("selected initial SAM object ID is not unique")
        selected_masks[0] = initial_masks[int(matches[0])]
        for item in adapter.handle_stream_request({"type": "propagate_in_video", "session_id": session_id, "propagation_direction": "forward", "start_frame_index": 0, "max_frame_num_to_track": case.frame_count, "output_prob_thresh": 0.5}):
            frame_index = int(item["frame_index"])
            masks, _, ids = normalize(item["outputs"], case.height, case.width)
            matches = np.flatnonzero(ids.astype(np.int64) == selected_id)
            if len(matches) == 1:
                selected_masks[frame_index] = masks[int(matches[0])]
            elif len(matches) > 1:
                raise RuntimeError(f"duplicate selected object ID at frame {frame_index}")
    finally:
        adapter.handle_request({"type": "close_session", "session_id": session_id})
    propagation_seconds = time.perf_counter() - propagation_started
    frame_rows = []
    ious = []
    centroid_steps = []
    previous_mask: np.ndarray | None = None
    previous_centroid: np.ndarray | None = None
    for frame_index, raw_path in enumerate(frame_paths):
        raw = np.asarray(Image.open(raw_path).convert("RGB"))
        mask = selected_masks.get(frame_index, np.zeros((case.height, case.width), dtype=bool))
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(masks_root / f"{frame_index:05d}.png")
        audit = audit_frame(mask, case)
        audit.update({"frame_index": frame_index, "present": bool(audit["area_pixels"])})
        frame_rows.append(audit)
        if previous_mask is not None:
            value = adjacent_iou(previous_mask, mask)
            if value is not None:
                ious.append(value)
        if audit["centroid_xy"] is not None:
            centroid = np.asarray(audit["centroid_xy"], dtype=np.float64)
            if previous_centroid is not None:
                centroid_steps.append(float(np.linalg.norm(centroid - previous_centroid)))
            previous_centroid = centroid
        previous_mask = mask
        overlay = tint(raw, mask)
        cv2.rectangle(overlay, (0, 0), (case.width - 1, 48), (0, 0, 0), -1)
        status = "RAW SAM3.1 INSTANCE" if audit["area_pixels"] else "MISSING / NO REPAIR"
        cv2.putText(overlay, f"f{frame_index:05d} | {status} | prompt={chosen_prompt} | id={selected_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255) if audit["area_pixels"] else (0, 0, 255), 2, cv2.LINE_AA)
        Image.fromarray(overlay, mode="RGB").save(overlays_root / f"{frame_index:05d}.png")
        raw_small = cv2.resize(raw, (640, 360), interpolation=cv2.INTER_AREA)
        overlay_small = cv2.resize(overlay, (640, 360), interpolation=cv2.INTER_AREA)
        pair = np.concatenate([raw_small, overlay_small], axis=1)
        cv2.putText(pair, "RAW", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pair, "RAW SINGLE-INSTANCE MASK OVERLAY", (650, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        Image.fromarray(pair, mode="RGB").save(review_root / f"{frame_index:05d}.png")
    review_video = case_root / f"{case.name.upper()}_RAW_VS_SAM31_MASK_OVERLAY.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "30", "-i", str(review_root / "%05d.png"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(review_video)], check=True)
    sample_indices = np.linspace(0, case.frame_count - 1, 10, dtype=int).tolist()
    tiles = [np.asarray(Image.open(review_root / f"{index:05d}.png").convert("RGB").resize((640, 180))) for index in sample_indices]
    sheet = np.zeros((5 * 180, 2 * 640, 3), dtype=np.uint8)
    for slot, tile in enumerate(tiles):
        sheet[(slot // 2) * 180:(slot // 2 + 1) * 180, (slot % 2) * 640:(slot % 2 + 1) * 640] = tile
    sheet_path = case_root / f"{case.name.upper()}_RAW_VS_SAM31_MASK_CONTACT_10FRAMES.png"
    Image.fromarray(sheet, mode="RGB").save(sheet_path)
    present = [row for row in frame_rows if row["present"]]
    present_count = len(present)
    spatial_pass_count = sum(row["spatial_gate_pass"] for row in frame_rows)
    status = "PASS_STATIC_DEVELOPMENT_PROBE" if present_count == case.frame_count and spatial_pass_count >= int(0.9 * case.frame_count) else "FAIL_STATIC_DEVELOPMENT_PROBE_REVIEW_REQUIRED"
    result = {
        "schema_version": "newtask-baseline-sam31-mask-probe-v1",
        "status": status,
        "data_role": "USER_AUTHORIZED_PHONE_BASELINE_DEVELOPMENT_ONLY_NOT_PICO_NOT_FORMAL_NOT_BLIND_TEST",
        "formal_clean_donor_allowed": False,
        "source": {"path": str(case.source), "bytes": case.source.stat().st_size, "sha256": sha256(case.source), "frame_count": case.frame_count, "width": case.width, "height": case.height},
        "target_semantics": case.target_semantics,
        "chosen_prompt": chosen_prompt,
        "selected_object_id": selected_id,
        "selection": chosen,
        "present_frames": present_count,
        "missing_frames": [row["frame_index"] for row in frame_rows if not row["present"]],
        "spatial_gate_pass_frames": spatial_pass_count,
        "spatial_gate_fail_frames": [row["frame_index"] for row in frame_rows if not row["spatial_gate_pass"]],
        "area_pixels_min_median_max": [int(min(row["area_pixels"] for row in present)), float(np.median([row["area_pixels"] for row in present])), int(max(row["area_pixels"] for row in present))] if present else None,
        "adjacent_raw_iou_median": float(np.median(ious)) if ious else None,
        "centroid_step_px_median_max": [float(np.median(centroid_steps)), float(max(centroid_steps))] if centroid_steps else None,
        "propagation_seconds": propagation_seconds,
        "peak_cuda_allocated_bytes_run_so_far": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes_run_so_far": int(torch.cuda.max_memory_reserved()),
        "mask_semantics": "exactly one unmodified propagated raw SAM3.1 instance per present frame; selector never changes mask pixels",
        "coverage_limit": "static/light-camera-motion baseline only; contains no hands, sleeves, wristbands, trackers, object pickup, contact, or occlusion",
        "claim_limit": "algorithm preparation evidence only; cannot promote phone pixels to PICO, formal output, formal clean donor, metric geometry, or blind evaluation",
        "review_video": {"path": str(review_video), "bytes": review_video.stat().st_size, "sha256": sha256(review_video)},
        "contact_sheet": {"path": str(sheet_path), "bytes": sheet_path.stat().st_size, "sha256": sha256(sheet_path)},
        "prompt_sweep": all_rows,
        "frames": frame_rows,
        "build_evidence": build_evidence,
    }
    (case_root / "RESULTS.json").write_text(json.dumps(jsonable(result), indent=2) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=sorted(CASES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.is_dir() or "_run" not in root.parts:
        raise RuntimeError("root must be an existing _run directory")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA drift")
    if sha256(ADAPTER_PATH) != ADAPTER_SHA256:
        raise RuntimeError("adapter SHA drift")
    for name in args.cases:
        case = CASES[name]
        if sha256(case.source) != case.source_sha256:
            raise RuntimeError(f"{name}: source SHA drift")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter_module = load_adapter_module()
    adapter, build_evidence = adapter_module.build_pinned_adapter(official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT)
    results = []
    for name in args.cases:
        result = run_case(adapter, build_evidence, CASES[name], root)
        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "schema_version": "newtask-baseline-sam31-mask-probe-summary-v1",
        "status": "PASS_DEVELOPMENT_PROBES_COMPLETE" if all(result["status"].startswith("PASS") for result in results) else "COMPLETE_WITH_REVIEW_REQUIRED",
        "data_role": "USER_AUTHORIZED_PHONE_BASELINE_DEVELOPMENT_ONLY_NOT_PICO_NOT_FORMAL_NOT_BLIND_TEST",
        "cases": [{key: result[key] for key in ("status", "source", "target_semantics", "chosen_prompt", "selected_object_id", "present_frames", "spatial_gate_pass_frames", "area_pixels_min_median_max", "adjacent_raw_iou_median", "centroid_step_px_median_max", "review_video", "contact_sheet")} for result in results],
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "mask_semantics": "one raw SAM3.1 object instance per case/frame; no morphology, color threshold, fill, union, or pixel repair",
        "next_required_probe": "action-window H/O/U canary on the two test MOV files is needed to evaluate hands, sleeves, wristbands/trackers, contact and occlusion",
        "formal_claim_limit": "none of these phone baseline artifacts is a formal PICO result or formal clean donor",
    }
    (root / "SUMMARY.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    print(json.dumps(jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
