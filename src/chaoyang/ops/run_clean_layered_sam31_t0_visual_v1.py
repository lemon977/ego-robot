#!/usr/bin/env python3
"""Real SAM3.1 T0 masks plus a non-authoritative Clean boundary comparison.

The candidate deliberately reuses frozen Clean pixels only where the new mask
is covered by the predecessor removal.  It visualizes mask/boundary effects;
it is not the later causal layered-donor successor.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
CODE_ROOT = PROJECT / "vendor/SAM3"
CHECKPOINT = PROJECT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
DEFAULT_ATTEMPT = (
    PROJECT
    / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/attempts/attempt_0001"
)
FRAMESET = (
    PROJECT
    / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_layered_sam31_exploration_v1/T0_FROZEN_FRAMESET.json"
)
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
FONT = ImageFont.truetype(str(FONT_PATH), 24)
FONT_SMALL = ImageFont.truetype(str(FONT_PATH), 18)
PROMPTS = {
    "left_human": [
        "the person's left hand and forearm",
        "left human hand and arm",
        "a person's hand and forearm",
    ],
    "right_human": [
        "the person's right hand and forearm",
        "right human hand and arm",
        "a person's hand and forearm",
    ],
}


@dataclass(frozen=True)
class Case:
    label: str
    task: str
    session: str
    manifest: Path
    baseline_clean_frames: Path
    shallow_name: str


CASES = (
    Case(
        label="Poker245",
        task="poker",
        session="play_cards_0903_245",
        manifest=PROJECT
        / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/sessions/play_cards_0903_245/expanded_role_handoff/FRAME_MANIFEST.json",
        baseline_clean_frames=PROJECT
        / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/propainter_v1/play_cards_0903_245/clean_frames",
        shallow_name="Clean_SAM31分层路线_Poker245_全片慢放.mp4",
    ),
    Case(
        label="Chips039",
        task="chips",
        session="get_potato_chips_0902_039",
        manifest=PROJECT
        / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/sessions/get_potato_chips_0902_039/expanded_role_handoff/FRAME_MANIFEST.json",
        baseline_clean_frames=PROJECT
        / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/propainter_v1/get_potato_chips_0902_039/clean_frames",
        shallow_name="Clean_SAM31分层路线_Chips039_全片慢放.mp4",
    ),
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != shape:
        raise RuntimeError(f"mask shape drift: {path}: {mask.shape} != {shape}")
    return mask


def normalize(
    outputs: dict[str, Any], height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks: Any = outputs.get("out_binary_masks")
    if masks is None:
        mask_array = np.zeros((0, height, width), dtype=bool)
    else:
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        elif isinstance(masks, (list, tuple)):
            masks = [
                item.detach().cpu().numpy()
                if isinstance(item, torch.Tensor)
                else np.asarray(item)
                for item in masks
            ]
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


def mask_candidate_rows(
    masks: np.ndarray, scores: np.ndarray, ids: np.ndarray, guide: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    guide_area = int(guide.sum())
    for index, mask in enumerate(masks):
        intersection = int((mask & guide).sum())
        union = int((mask | guide).sum())
        area = int(mask.sum())
        iou = intersection / union if union else 0.0
        guide_coverage = intersection / guide_area if guide_area else 0.0
        spill_ratio = int((mask & ~guide).sum()) / max(area, 1)
        selector_score = iou + 0.35 * guide_coverage - 0.10 * spill_ratio
        rows.append(
            {
                "instance_index": index,
                "object_id": int(ids[index]),
                "model_score": float(scores[index]),
                "area_pixels": area,
                "guide_intersection_pixels": intersection,
                "iou_to_pinned_guide": iou,
                "pinned_guide_coverage": guide_coverage,
                "spill_ratio_to_pinned_guide": spill_ratio,
                "selector_score": selector_score,
            }
        )
    return sorted(rows, key=lambda row: row["selector_score"], reverse=True)


def object_union(row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for key, value in row.items():
        if not key.startswith("physical_object_") or not isinstance(value, dict):
            continue
        if value.get("valid") is False or not value.get("path"):
            continue
        result |= read_mask(Path(value["path"]), shape)
    return result


def choose_anchor(manifest: dict[str, Any], role: str, shape: tuple[int, int]) -> int:
    height, width = shape
    candidates: list[tuple[float, int]] = []
    for row in manifest["frames"]:
        role_entry = row["role_masks"][role]
        mask = read_mask(Path(role_entry["path"]), shape)
        ys, xs = np.where(mask)
        if len(xs) < 1000:
            continue
        margin = min(int(xs.min()), int(ys.min()), width - 1 - int(xs.max()), height - 1 - int(ys.max()))
        obj = object_union(row, shape)
        overlap_ratio = float((mask & obj).sum() / max(mask.sum(), 1))
        score = float(margin) + 0.002 * float(np.sqrt(mask.sum())) - 500.0 * overlap_ratio
        candidates.append((score, int(row["source_frame"])))
    if not candidates:
        raise RuntimeError(f"no valid anchor candidates: {role}")
    return max(candidates)[1]


def start_session(adapter: Any, session_id: str, frames: Path) -> None:
    adapter.handle_request(
        {
            "type": "start_session",
            "session_id": session_id,
            "resource_path": str(frames.resolve()),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": False,
            "async_loading_frames": False,
        }
    )


def prompt_sweep(
    adapter: Any,
    case: Case,
    role: str,
    raw_path: Path,
    guide: np.ndarray,
    shape: tuple[int, int],
    output: Path,
) -> tuple[str, list[dict[str, Any]]]:
    prompt_frame = output / f"prompt_frame_{role}"
    prompt_frame.mkdir()
    os.symlink(raw_path, prompt_frame / "00000.png")
    all_rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(PROMPTS[role]):
        session_id = f"{case.session}-{role}-sweep-{prompt_index}"
        start_session(adapter, session_id, prompt_frame)
        try:
            initial = adapter.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                    "output_prob_thresh": 0.5,
                }
            )
            masks, scores, ids = normalize(initial["outputs"], *shape)
            rows = mask_candidate_rows(masks, scores, ids, guide)
            for row in rows:
                row.update(prompt=prompt, prompt_index=prompt_index)
                all_rows.append(row)
        finally:
            adapter.handle_request({"type": "close_session", "session_id": session_id})
    eligible = [row for row in all_rows if row["guide_intersection_pixels"] > 0]
    if not eligible:
        raise RuntimeError(f"{case.session}/{role}: no prompt instance overlaps guide")
    chosen = max(eligible, key=lambda row: row["selector_score"])
    return str(chosen["prompt"]), all_rows


def track_role(
    adapter: Any,
    case: Case,
    role: str,
    manifest: dict[str, Any],
    frames_root: Path,
    shape: tuple[int, int],
    output: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = int(manifest["frame_count"])
    anchor = choose_anchor(manifest, role, shape)
    guide = read_mask(Path(manifest["frames"][anchor]["role_masks"][role]["path"]), shape)
    prompt, sweep = prompt_sweep(
        adapter,
        case,
        role,
        Path(manifest["frames"][anchor]["source_rgb"]["path"]),
        guide,
        shape,
        output,
    )
    session_id = f"{case.session}-{role}-full"
    start_session(adapter, session_id, frames_root)
    selected = np.zeros((count, *shape), dtype=bool)
    direction_counts: dict[str, int] = {}
    started = time.perf_counter()
    try:
        initial = adapter.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": anchor,
                "text": prompt,
                "output_prob_thresh": 0.5,
            }
        )
        masks, scores, ids = normalize(initial["outputs"], *shape)
        rows = mask_candidate_rows(masks, scores, ids, guide)
        if not rows or rows[0]["guide_intersection_pixels"] <= 0:
            raise RuntimeError(f"{case.session}/{role}: full initial selection failed")
        object_id = int(rows[0]["object_id"])
        selected[anchor] = masks[int(rows[0]["instance_index"])]
        for direction in ("forward", "backward"):
            seen = 0
            maximum = count - anchor if direction == "forward" else anchor + 1
            for item in adapter.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": direction,
                    "start_frame_index": anchor,
                    "max_frame_num_to_track": maximum,
                    "output_prob_thresh": 0.5,
                }
            ):
                frame_index = int(item["frame_index"])
                frame_masks, _, frame_ids = normalize(item["outputs"], *shape)
                matches = np.flatnonzero(frame_ids.astype(np.int64) == object_id)
                if len(matches) == 1:
                    selected[frame_index] = frame_masks[int(matches[0])]
                    seen += 1
                elif len(matches) > 1:
                    raise RuntimeError(
                        f"{case.session}/{role}: duplicate object id at {frame_index}"
                    )
            direction_counts[direction] = seen
    finally:
        adapter.handle_request({"type": "close_session", "session_id": session_id})
    elapsed = time.perf_counter() - started
    coverage = float(np.count_nonzero(selected.reshape(count, -1).any(axis=1)) / count)
    evidence = {
        "role": role,
        "anchor_frame": anchor,
        "selected_prompt": prompt,
        "selected_object_id": object_id,
        "direction_frame_counts": direction_counts,
        "present_frame_coverage": coverage,
        "propagation_seconds": elapsed,
        "prompt_sweep": sweep,
        "selection_guide": ref(Path(manifest["frames"][anchor]["role_masks"][role]["path"])),
    }
    return selected, evidence


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.58) -> np.ndarray:
    result = image.copy()
    if mask.any():
        result[mask] = ((1.0 - alpha) * result[mask] + alpha * np.asarray(color)).astype(np.uint8)
    return result


def panel(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    resized = cv2.resize(image, (512, 384), interpolation=cv2.INTER_AREA)
    canvas = Image.fromarray(resized)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 512, 58), fill=(0, 0, 0))
    draw.text((10, 4), title, font=FONT, fill=(255, 255, 255))
    draw.text((10, 34), subtitle, font=FONT_SMALL, fill=(225, 225, 225))
    return np.asarray(canvas)


def checker_unknown(image: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    result = image.copy()
    yy, xx = np.indices(unknown.shape)
    checker = ((xx // 12 + yy // 12) % 2) == 0
    result[unknown & checker] = (255, 0, 255)
    result[unknown & ~checker] = (40, 40, 40)
    return result


def encode(frames: Path, output: Path, fps: int) -> int:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps), "-i", str(frames / "%06d.jpg"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nk=1:nw=1",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return int(probe.stdout.strip())


def case_frameset(session: str) -> list[int]:
    data = json.loads(FRAMESET.read_text(encoding="utf-8"))
    row = next(item for item in data["sessions"] if item["session_id"] == session)
    return [int(value) for value in row["selected"]]


def run_case(adapter: Any, build_evidence: dict[str, Any], case: Case) -> dict[str, Any]:
    output = ATTEMPT / case.label
    output.mkdir()
    manifest = json.loads(case.manifest.read_text(encoding="utf-8"))
    count = int(manifest["frame_count"])
    if len(manifest["frames"]) != count:
        raise RuntimeError(f"{case.session}: manifest frame count mismatch")
    if [int(row["source_frame"]) for row in manifest["frames"]] != list(range(count)):
        raise RuntimeError(f"{case.session}: non-contiguous frame identity")
    first = read_rgb(Path(manifest["frames"][0]["source_rgb"]["path"]))
    shape = first.shape[:2]
    frames_root = output / "input_frames"
    frames_root.mkdir()
    for index, row in enumerate(manifest["frames"]):
        source = Path(row["source_rgb"]["path"])
        if not source.is_file():
            raise RuntimeError(f"missing raw frame: {source}")
        os.symlink(source, frames_root / f"{index:05d}.png")

    role_evidence = []
    packed_paths = {}
    for role in ("left_human", "right_human"):
        masks, evidence = track_role(
            adapter, case, role, manifest, frames_root, shape, output
        )
        packed_path = output / f"{role}_SAM31_BIDIRECTIONAL_PACKED.npz"
        np.savez_compressed(
            packed_path,
            packed=np.packbits(masks.reshape(count, -1), axis=1),
            frame_count=np.int32(count),
            height=np.int32(shape[0]),
            width=np.int32(shape[1]),
        )
        evidence["packed_masks"] = ref(packed_path)
        role_evidence.append(evidence)
        packed_paths[role] = packed_path
        del masks
        gc.collect()
        torch.cuda.empty_cache()

    packed = {
        role: np.load(path)["packed"] for role, path in packed_paths.items()
    }
    selected = set(case_frameset(case.session))
    review_frames = output / "review_frames_staging"
    selected_frames = output / "selected_review_frames"
    review_frames.mkdir()
    selected_frames.mkdir()
    rows = []
    totals = {
        "old_removal_pixels": 0,
        "new_removal_pixels": 0,
        "restored_raw_pixels": 0,
        "unknown_pixels": 0,
        "visible_object_changed_pixels": 0,
        "sam_side_overlap_pixels": 0,
    }
    height, width = shape
    flat_count = height * width
    for index, row in enumerate(manifest["frames"]):
        raw = read_rgb(Path(row["source_rgb"]["path"]))
        baseline_clean = read_rgb(case.baseline_clean_frames / f"{index:06d}.png")
        old_removal = read_mask(Path(row["clean_removal_object_protected"]["path"]), shape)
        left = np.unpackbits(packed["left_human"][index], count=flat_count).reshape(shape).astype(bool)
        right = np.unpackbits(packed["right_human"][index], count=flat_count).reshape(shape).astype(bool)
        trackers = read_mask(Path(row["role_masks"]["left_tracker"]["path"]), shape)
        trackers |= read_mask(Path(row["role_masks"]["right_tracker"]["path"]), shape)
        obj = object_union(row, shape)
        near = dilate(obj, 20)
        human = left | right
        human_support = (dilate(human, 4) & near) | (dilate(human, 8) & ~near)
        tracker_support = (dilate(trackers, 8) & near) | (dilate(trackers, 20) & ~near)
        new_removal = (human_support | tracker_support) & ~obj
        covered = new_removal & old_removal
        unknown = new_removal & ~old_removal & ~obj
        restored = old_removal & ~new_removal
        candidate = raw.copy()
        candidate[covered] = baseline_clean[covered]
        candidate[obj] = raw[obj]
        object_changed = int((np.any(candidate != raw, axis=2) & obj).sum())

        current_overlay = tint(raw, old_removal, (255, 30, 30))
        current_overlay = tint(current_overlay, obj, (0, 255, 255), 0.70)
        sam_overlay = tint(raw, left, (0, 255, 80))
        sam_overlay = tint(sam_overlay, right, (255, 0, 180))
        sam_overlay = tint(sam_overlay, trackers, (255, 160, 0))
        adaptive_overlay = tint(raw, new_removal, (255, 210, 0))
        adaptive_overlay = tint(adaptive_overlay, obj, (0, 255, 255), 0.70)
        candidate_display = checker_unknown(candidate, unknown)
        provenance = np.zeros_like(raw)
        provenance[:] = (25, 25, 25)
        provenance[restored] = (0, 210, 80)
        provenance[covered] = (150, 60, 220)
        provenance[obj] = (0, 230, 230)
        provenance[unknown] = (255, 0, 255)
        reduction = 1.0 - float(new_removal.sum() / max(old_removal.sum(), 1))
        subtitle = f"帧{index:03d} 原始时间{index / 30.0:.2f}s"
        panels = [
            panel(raw, "原始 RGB", subtitle),
            panel(current_overlay, "当前冻结删除区", "红=删除；青=可见任务物体"),
            panel(sam_overlay, "真实 SAM3.1 双向角色", "绿=左；紫=右；橙=Tracker"),
            panel(baseline_clean, "当前 Clean 基线", "结构B，不代表接触/语义正确"),
            panel(
                candidate_display,
                "SAM3.1 自适应边界候选",
                f"删除面积变化 {reduction * 100:+.1f}%；棋盘=UNKNOWN",
            ),
            panel(provenance, "变化与来源", "绿=恢复Raw；紫=旧Clean；青=物体；粉=未知"),
        ]
        mosaic = np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:])))
        canvas = Image.fromarray(mosaic)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 740, 1536, 768), fill=(0, 0, 0))
        draw.text(
            (10, 743),
            "T0开发可视化：不是新Clean authority；候选仅复用旧Clean覆盖像素，未测试完整因果分层donor",
            font=FONT_SMALL,
            fill=(255, 220, 80),
        )
        mosaic = np.asarray(canvas)
        review_path = review_frames / f"{index:06d}.jpg"
        Image.fromarray(mosaic).save(review_path, quality=91)
        if index in selected:
            Image.fromarray(mosaic).save(selected_frames / f"frame_{index:03d}.jpg", quality=94)

        metrics = {
            "frame": index,
            "old_removal_pixels": int(old_removal.sum()),
            "new_removal_pixels": int(new_removal.sum()),
            "restored_raw_pixels": int(restored.sum()),
            "unknown_new_outside_old_pixels": int(unknown.sum()),
            "visible_object_pixels": int(obj.sum()),
            "visible_object_changed_pixels": object_changed,
            "sam_left_pixels": int(left.sum()),
            "sam_right_pixels": int(right.sum()),
            "sam_left_right_overlap_pixels": int((left & right).sum()),
        }
        rows.append(metrics)
        for key in totals:
            totals[key] += {
                "old_removal_pixels": metrics["old_removal_pixels"],
                "new_removal_pixels": metrics["new_removal_pixels"],
                "restored_raw_pixels": metrics["restored_raw_pixels"],
                "unknown_pixels": metrics["unknown_new_outside_old_pixels"],
                "visible_object_changed_pixels": metrics["visible_object_changed_pixels"],
                "sam_side_overlap_pixels": metrics["sam_left_right_overlap_pixels"],
            }[key]

    review_video = output / f"{case.session}_SAM31_CLEAN_T0_六栏全片慢放.mp4"
    decoded = encode(review_frames, review_video, 15)
    if decoded != count:
        raise RuntimeError(f"review decode mismatch {decoded} != {count}")
    contact_sheet = output / f"{case.session}_SAM31_CLEAN_T0_24帧总览.jpg"
    selected_paths = sorted(selected_frames.glob("*.jpg"))
    thumbs = [
        cv2.resize(read_rgb(path), (384, 192), interpolation=cv2.INTER_AREA)
        for path in selected_paths
    ]
    sheet = np.zeros((6 * 192, 4 * 384, 3), dtype=np.uint8)
    for slot, thumb in enumerate(thumbs):
        y, x = divmod(slot, 4)
        sheet[y * 192 : (y + 1) * 192, x * 384 : (x + 1) * 384] = thumb
    Image.fromarray(sheet).save(contact_sheet, quality=92)
    shutil.rmtree(review_frames)

    shallow = PROJECT / "docs/current_visuals" / case.shallow_name
    if shallow.exists() or shallow.is_symlink():
        if shallow.resolve() != review_video.resolve():
            raise RuntimeError(f"refusing to replace shallow visual: {shallow}")
    else:
        os.symlink(review_video, shallow)

    removal_reduction = 1.0 - totals["new_removal_pixels"] / max(totals["old_removal_pixels"], 1)
    result = {
        "schema_version": "clean-layered-sam31-t0-case-result-v1",
        "status": "PASS_DEVELOPMENT_VISUAL_REVIEW_REQUIRED",
        "case": case.label,
        "task": case.task,
        "session": case.session,
        "frame_count": count,
        "input_fps": 30,
        "review_fps": 15,
        "review_speed": "0.5x relative to the 30 FPS input",
        "mode": "OFFLINE_BIDIRECTIONAL_QA",
        "sam31": role_evidence,
        "totals": totals,
        "new_vs_old_removal_reduction_ratio": removal_reduction,
        "selected_frames": sorted(selected),
        "artifacts": {
            "review_video": ref(review_video),
            "contact_sheet": ref(contact_sheet),
            "shallow_visual": str(shallow),
        },
        "build_evidence": build_evidence,
        "frames": rows,
        "quality_interpretation": {
            "visible_object_byte_exact": totals["visible_object_changed_pixels"] == 0,
            "human_masks_are_real_sam31": True,
            "fresh_propainter_executed": False,
            "causal_layered_donor_executed": False,
            "manual_visual_review_required": True,
        },
        "claim_limit": "Real SAM3.1 offline bidirectional mask and adaptive-boundary T0 visualization. Candidate RGB reuses predecessor Clean pixels only; it is not the full causal layered Clean successor and grants no authority.",
    }
    result_path = output / "RESULT.json"
    write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the real-SAM3.1 T0 Clean comparison visuals."
    )
    parser.add_argument(
        "--attempt-root",
        type=Path,
        default=DEFAULT_ATTEMPT,
        help="Fresh immutable attempt directory (must not already exist).",
    )
    args = parser.parse_args()
    global ATTEMPT
    ATTEMPT = args.attempt_root.resolve()
    if ATTEMPT.exists():
        raise RuntimeError(f"fresh attempt already exists: {ATTEMPT}")
    ATTEMPT.mkdir(parents=True)
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    from chaoyang.pipeline import sam31_compat_adapter_v1 as adapter_module

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT,
        checkpoint_path=CHECKPOINT,
    )
    results = []
    for case in CASES:
        results.append(run_case(adapter, build_evidence, case))
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "schema_version": "clean-layered-sam31-t0-summary-v1",
        "status": "PASS_DEVELOPMENT_VISUALS_REVIEW_REQUIRED",
        "created_at": now(),
        "task_id": "clean_layered_sam31_t0_visual_v1",
        "cases": [
            {
                "case": row["case"],
                "session": row["session"],
                "frame_count": row["frame_count"],
                "new_vs_old_removal_reduction_ratio": row["new_vs_old_removal_reduction_ratio"],
                "visible_object_changed_pixels": row["totals"]["visible_object_changed_pixels"],
                "sam_side_overlap_pixels": row["totals"]["sam_side_overlap_pixels"],
                "artifacts": row["artifacts"],
            }
            for row in results
        ],
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "frozen_frameset": ref(FRAMESET),
        "authority": False,
        "claim_limit": "T0 visual diagnostic only; no fresh ProPainter, causal layered donor, Clean authority, contact truth or Robotized training authority.",
    }
    write_json(ATTEMPT / "RESULT.json", summary)
    print(json.dumps({"status": summary["status"], "cases": summary["cases"], "wall_seconds": summary["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
