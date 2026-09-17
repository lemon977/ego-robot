#!/usr/bin/env python3
"""Probe SAM3.1 text prompts for white gloves and PICO Controllers on one frame.

This is a bounded development diagnostic.  It deliberately does not propagate
masks or grant Mask/Clean authority.  The target anchor is used only to rank
raw SAM3.1 instances; no mask pixels are edited.
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
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "vendor/SAM3"
CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
ADAPTER = ROOT / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"

ROLE_PROMPTS = {
    "left_glove": {
        "anchor_xy": [570, 780],
        "prompts": [
            "a white instrumented glove",
            "a white motion capture glove",
            "a gloved human hand",
            "a white glove and all five fingers",
        ],
    },
    "right_glove": {
        "anchor_xy": [1000, 580],
        "prompts": [
            "a white instrumented glove",
            "a white motion capture glove",
            "a gloved human hand",
            "a white glove and all five fingers",
        ],
    },
    "left_controller": {
        "anchor_xy": [230, 760],
        "prompts": [
            "a PICO hand controller",
            "a VR hand controller",
            "a handheld virtual reality controller",
            "a white virtual reality controller",
        ],
    },
    "right_controller": {
        "anchor_xy": [1090, 560],
        "prompts": [
            "a PICO hand controller",
            "a VR hand controller",
            "a handheld virtual reality controller",
            "a white virtual reality controller",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    scores = outputs.get("out_probs", [])
    ids = outputs.get("out_obj_ids", [])
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    id_array = np.asarray(ids).reshape(-1)
    if len(mask_array) != len(score_array) or len(mask_array) != len(id_array):
        raise RuntimeError("SAM output length mismatch")
    return mask_array, score_array, id_array


def candidate(mask: np.ndarray, score: float, object_id: Any, anchor_xy: list[int]) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if not len(xs):
        return {"object_id": int(object_id), "score": score, "area_pixels": 0,
                "bbox_xyxy": None, "centroid_xy": None, "anchor_inside": False,
                "anchor_distance_px": None}
    ax, ay = anchor_xy
    inside = bool(mask[ay, ax])
    distance = 0.0 if inside else float(np.sqrt((xs - ax) ** 2 + (ys - ay) ** 2).min())
    return {
        "object_id": int(object_id),
        "score": float(score),
        "area_pixels": int(len(xs)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "anchor_inside": inside,
        "anchor_distance_px": distance,
    }


def rank(row: dict[str, Any]) -> tuple[Any, ...]:
    distance = row["anchor_distance_px"] if row["anchor_distance_px"] is not None else float("inf")
    return (not row["anchor_inside"], distance, -row["score"])


def render_tile(raw: np.ndarray, mask: np.ndarray, role: str, prompt: str, row: dict[str, Any] | None,
                anchor_xy: list[int]) -> np.ndarray:
    overlay = raw.copy()
    if mask.any():
        overlay[mask] = (0.30 * overlay[mask] + 0.70 * np.asarray([255, 0, 180])).astype(np.uint8)
    tile = cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_AREA)
    ax, ay = int(anchor_xy[0] / 2), int(anchor_xy[1] / 2)
    cv2.circle(tile, (ax, ay), 6, (0, 255, 255), 2)
    cv2.rectangle(tile, (0, 0), (639, 55), (0, 0, 0), -1)
    if row is None:
        detail = "NO INSTANCE"
    else:
        detail = f"inside={row['anchor_inside']} dist={row['anchor_distance_px']} area={row['area_pixels']} score={row['score']:.3f}"
    cv2.putText(tile, f"{role}: {prompt}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, detail, (7, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (80, 255, 80) if row and row["anchor_inside"] else (255, 180, 80), 1, cv2.LINE_AA)
    return tile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    frame = args.frame.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output required: {output}")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    raw = np.asarray(Image.open(frame).convert("RGB"))
    height, width = raw.shape[:2]
    if (width, height) != (1280, 960):
        raise RuntimeError(f"expected 1280x960 input, got {width}x{height}")
    output.mkdir(parents=True)
    frame_root = output / "input_frame"
    frame_root.mkdir()
    os.symlink(frame, frame_root / "00000.png")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(CODE_ROOT))
    from chaoyang.ops.run_newtask_baseline_sam31_mask_probe import load_adapter_module

    adapter_module = load_adapter_module()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT
    )
    records: list[dict[str, Any]] = []
    tiles: list[np.ndarray] = []
    try:
        for role_index, (role, spec) in enumerate(ROLE_PROMPTS.items()):
            for prompt_index, prompt in enumerate(spec["prompts"]):
                session_id = f"sensor-text-{role_index}-{prompt_index}"
                adapter.handle_request({
                    "type": "start_session", "session_id": session_id,
                    "resource_path": str(frame_root), "offload_video_to_cpu": True,
                    "offload_state_to_cpu": False, "async_loading_frames": False,
                })
                try:
                    response = adapter.handle_request({
                        "type": "add_prompt", "session_id": session_id,
                        "frame_index": 0, "text": prompt, "output_prob_thresh": 0.5,
                    })
                    masks, scores, ids = normalize(response["outputs"], height, width)
                    candidates = [candidate(masks[i], float(scores[i]), ids[i], spec["anchor_xy"])
                                  for i in range(len(masks))]
                    order = sorted(range(len(candidates)), key=lambda i: rank(candidates[i]))
                    selected_index = order[0] if order else None
                    selected = candidates[selected_index] if selected_index is not None else None
                    selected_mask = masks[selected_index] if selected_index is not None else np.zeros((height, width), bool)
                    records.append({
                        "role": role, "prompt": prompt, "anchor_xy": spec["anchor_xy"],
                        "instance_count": len(candidates), "selected": selected,
                        "candidates": candidates,
                    })
                    tiles.append(render_tile(raw, selected_mask, role, prompt, selected, spec["anchor_xy"]))
                finally:
                    adapter.handle_request({"type": "close_session", "session_id": session_id})
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        adapter.predictor.shutdown()

    sheet = np.zeros((4 * 480, 4 * 640, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, 4)
        sheet[row * 480:(row + 1) * 480, column * 640:(column + 1) * 640] = tile
    sheet_path = output / "SAM31_手套手柄文本提示_16项探针.png"
    Image.fromarray(sheet, mode="RGB").save(sheet_path)
    per_role = {}
    for role in ROLE_PROMPTS:
        rows = [row for row in records if row["role"] == role]
        per_role[role] = {
            "prompts_tested": len(rows),
            "prompts_returning_instances": sum(row["instance_count"] > 0 for row in rows),
            "prompts_covering_frozen_anchor": sum(bool(row["selected"] and row["selected"]["anchor_inside"]) for row in rows),
            "best_anchor_distance_px": min(
                (row["selected"]["anchor_distance_px"] for row in rows if row["selected"] and row["selected"]["anchor_distance_px"] is not None),
                default=None,
            ),
        }
    result = {
        "schema_version": "sensor-sam31-text-prompt-sweep-v1",
        "status": "COMPLETE_DEVELOPMENT_TEXT_PROMPT_SWEEP",
        "model": "SAM3.1 multiplex",
        "frame": {"path": str(frame), "bytes": frame.stat().st_size, "sha256": sha256(frame)},
        "checkpoint": {"path": str(CHECKPOINT), "bytes": CHECKPOINT.stat().st_size, "sha256": sha256(CHECKPOINT)},
        "adapter": {"path": str(ADAPTER), "bytes": ADAPTER.stat().st_size, "sha256": sha256(ADAPTER)},
        "build_evidence": build_evidence,
        "per_role": per_role,
        "prompt_records": records,
        "outputs": {"contact_sheet": str(sheet_path), "contact_sheet_sha256": sha256(sheet_path)},
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "claim_limit": "One frozen frame and anchor-proximity diagnostic only. Text recognition here neither grants temporal Mask authority nor proves the glove acquisition scheme feasible/infeasible.",
    }
    result_path = output / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "per_role": per_role,
                      "contact_sheet": str(sheet_path), "result": str(result_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
