#!/usr/bin/env python3
"""One-frame assisted SAM3.1 point canary for bilateral hand/wearable classes.

This runner is deliberately small and fail-closed.  It tests whether explicit
human-selected points can produce independent raw SAM masks for both foreground
hands and their wearables before any temporal propagation is authorized.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
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
CODE_ROOT = PROJECT / "third_party/SAM3"
CHECKPOINT = PROJECT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
SOURCE_ROOT = PROJECT / "_run/newtask_baseline_mask_probe_20260901_v1/action_input_frames"


SPECS = {
    "chips": {
        "frame": SOURCE_ROOT / "chips/00000.png",
        "objects": [
            {"id": 101, "name": "upper_human_core", "class": "human_core", "side": "upper", "positive": [[410, 350]], "negative": [[410, 620], [235, 362]]},
            {"id": 102, "name": "lower_human_core", "class": "human_core", "side": "lower", "positive": [[400, 625]], "negative": [[516, 615], [410, 350], [172, 555]]},
            {"id": 201, "name": "upper_tracker_wearable", "class": "tracker_wearable", "side": "upper", "positive": [[494, 290], [505, 317]], "negative": [[460, 350], [516, 615]]},
            {"id": 202, "name": "lower_tracker_wearable", "class": "tracker_wearable", "side": "lower", "positive": [[516, 615], [516, 661]], "negative": [[455, 620], [494, 290]]},
        ],
        "object_protection_points": [[235, 362], [210, 410], [275, 398]],
    },
    "poker": {
        "frame": SOURCE_ROOT / "poker/00000.png",
        "objects": [
            {"id": 101, "name": "upper_human_core", "class": "human_core", "side": "upper", "positive": [[394, 548]], "negative": [[326, 670], [232, 463]]},
            {"id": 102, "name": "lower_human_core", "class": "human_core", "side": "lower", "positive": [[326, 690], [505, 875]], "negative": [[405, 826], [394, 548], [185, 548]]},
            {"id": 201, "name": "upper_tracker_wearable", "class": "tracker_wearable", "side": "upper", "positive": [[463, 495], [472, 519]], "negative": [[420, 548], [405, 826]]},
            {"id": 202, "name": "lower_tracker_wearable", "class": "tracker_wearable", "side": "lower", "positive": [[376, 838]], "negative": [[405, 826], [326, 690]]},
            {"id": 301, "name": "lower_sleeve_cuff", "class": "sleeve_cuff", "side": "lower", "positive": [[405, 826], [430, 842]], "negative": [[376, 838], [326, 690], [505, 875]]},
        ],
        "object_protection_points": [[232, 463], [207, 499], [185, 548]],
    },
}


COLORS = {
    "human_core": np.asarray([255, 40, 170], dtype=np.float32),
    "tracker_wearable": np.asarray([30, 225, 255], dtype=np.float32),
    "sleeve_cuff": np.asarray([70, 255, 80], dtype=np.float32),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_masks(outputs: dict, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    masks = outputs.get("out_binary_masks")
    ids = outputs.get("out_obj_ids")
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    elif isinstance(masks, (list, tuple)):
        masks = np.stack([item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item) for item in masks])
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    masks = np.asarray(masks if masks is not None else np.zeros((0, height, width), bool))
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = np.squeeze(masks, axis=1)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[1:] != (height, width):
        raise RuntimeError(f"mask shape drift: {masks.shape}, expected (*,{height},{width})")
    return masks.astype(bool), np.asarray(ids if ids is not None else [], dtype=np.int64).reshape(-1)


def geometry(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if not len(xs):
        return {"area_pixels": 0, "bbox_xyxy": None, "centroid_xy": None}
    return {
        "area_pixels": int(mask.sum()),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def tint(raw: np.ndarray, masks: list[tuple[np.ndarray, str]]) -> np.ndarray:
    output = raw.astype(np.float32).copy()
    for mask, class_name in masks:
        color = COLORS[class_name]
        output[mask] = 0.35 * output[mask] + 0.65 * color
    return output.astype(np.uint8)


def run_case(model, case_name: str, spec: dict, root: Path) -> dict:
    frame_path = Path(spec["frame"]).resolve(strict=True)
    raw = np.asarray(Image.open(frame_path).convert("RGB"))
    height, width = raw.shape[:2]
    one_frame = root / "inputs" / case_name
    one_frame.mkdir(parents=True)
    os.symlink(frame_path, one_frame / "00000.png")
    mask_by_name: dict[str, np.ndarray] = {}
    provider_by_name: dict[str, str] = {}
    rows = []

    # The neutral human text prompt reliably emits both foreground people as
    # separate raw instances.  We select the two foreground identities only by
    # their pre-registered skin anchors; no pixels are changed.
    human_state = model.init_state(resource_path=str(one_frame), offload_video_to_cpu=True, async_loading_frames=False)
    try:
        _, human_outputs = model.add_prompt(
            inference_state=human_state,
            frame_idx=0,
            text_str="a person's hand and forearm",
            output_prob_thresh=0.5,
        )
        human_masks, human_ids = normalize_masks(human_outputs, height, width)
        for item in (value for value in spec["objects"] if value["class"] == "human_core"):
            x, y = item["positive"][0]
            candidates = []
            for index, mask in enumerate(human_masks):
                ys, xs = np.where(mask)
                distance = float(np.sqrt((xs - x) ** 2 + (ys - y) ** 2).min()) if len(xs) else float("inf")
                if distance <= 12.0:
                    candidates.append((distance, int(mask.sum()), index))
            if candidates:
                distance, _, chosen = min(candidates)
                mask_by_name[item["name"]] = human_masks[chosen]
                provider_by_name[item["name"]] = f"sam31_text_raw_id_{int(human_ids[chosen])}_anchor_distance_{distance:.3f}px"
            else:
                mask_by_name[item["name"]] = np.zeros((height, width), dtype=bool)
                provider_by_name[item["name"]] = "sam31_text_missing"
    finally:
        human_state.clear()

    # Wearables remain independent point-prompt raw IDs.  Each role receives a
    # separate state because SAM's multiplex non-overlap arbitration can erase
    # the first of two spatially distinct wrist devices in a single state.
    wearable_items = [value for value in spec["objects"] if value["class"] != "human_core"]
    for item in wearable_items:
        wearable_state = model.init_state(resource_path=str(one_frame), offload_video_to_cpu=True, async_loading_frames=False)
        try:
            prime_x, prime_y = spec["object_protection_points"][0]
            priming_id = int(item["id"]) + 10000
            model.add_prompt(
                inference_state=wearable_state,
                frame_idx=0,
                points=torch.tensor([[prime_x / width, prime_y / height]], dtype=torch.float32),
                point_labels=torch.tensor([1], dtype=torch.int32),
                obj_id=priming_id,
                rel_coordinates=True,
                clear_old_points=True,
                output_prob_thresh=0.5,
            )
            points = [*item["positive"], *item["negative"]]
            labels = [1] * len(item["positive"]) + [0] * len(item["negative"])
            points_rel = [[x / width, y / height] for x, y in points]
            _, outputs = model.add_prompt(
                inference_state=wearable_state,
                frame_idx=0,
                points=torch.tensor(points_rel, dtype=torch.float32),
                point_labels=torch.tensor(labels, dtype=torch.int32),
                obj_id=int(item["id"]),
                rel_coordinates=True,
                clear_old_points=True,
                output_prob_thresh=0.5,
            )
            masks, ids = normalize_masks(outputs, height, width)
            mask_by_id = {int(obj_id): masks[index] for index, obj_id in enumerate(ids)}
            mask_by_name[item["name"]] = mask_by_id.get(int(item["id"]), np.zeros((height, width), bool))
            provider_by_name[item["name"]] = f"sam31_point_raw_id_{item['id']}_independent_state_after_discarded_priming_id_{priming_id}"
        finally:
            wearable_state.clear()

    union = np.zeros((height, width), dtype=bool)
    overlays = []
    case_root = root / case_name
    (case_root / "class_masks").mkdir(parents=True)
    for item in spec["objects"]:
        mask = mask_by_name[item["name"]]
        union |= mask
        overlays.append((mask, item["class"]))
        mask_path = case_root / "class_masks" / f"{item['id']}_{item['name']}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        row = {**item, **geometry(mask), "provider": provider_by_name[item["name"]], "mask_path": str(mask_path), "mask_sha256": sha256(mask_path)}
        ys, xs = np.where(mask)
        anchor_x, anchor_y = item["positive"][0]
        row["selection_anchor_distance_px"] = float(np.sqrt((xs - anchor_x) ** 2 + (ys - anchor_y) ** 2).min()) if len(xs) else None
        row["positive_point_coverage"] = [bool(mask[y, x]) for x, y in item["positive"]]
        row["negative_point_exclusion"] = [not bool(mask[y, x]) for x, y in item["negative"]]
        rows.append(row)
    object_points_clear = [not bool(union[y, x]) for x, y in spec["object_protection_points"]]
    Image.fromarray(union.astype(np.uint8) * 255).save(case_root / "BILATERAL_UNION.png")
    overlay = tint(raw, overlays)
    for item in spec["objects"]:
        for x, y in item["positive"]:
            cv2.circle(overlay, (x, y), 5, (255, 255, 255), 2, cv2.LINE_AA)
    for x, y in spec["object_protection_points"]:
        cv2.drawMarker(overlay, (x, y), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 2)
    Image.fromarray(overlay).save(case_root / "BILATERAL_CLASS_OVERLAY.png")
    pair = np.concatenate([raw, overlay], axis=1)
    cv2.rectangle(pair, (0, 0), (pair.shape[1] - 1, 50), (0, 0, 0), -1)
    cv2.putText(pair, f"{case_name.upper()} | RAW | ASSISTED BILATERAL CLASS OVERLAY", (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    Image.fromarray(pair).save(case_root / "RAW_VS_BILATERAL_OVERLAY.png")
    all_present = all(row["area_pixels"] > 0 for row in rows)
    point_contract = all(
        (
            row["selection_anchor_distance_px"] is not None
            and row["selection_anchor_distance_px"] <= 12.0
            and all(row["negative_point_exclusion"])
        )
        if row["class"] == "human_core"
        else (all(row["positive_point_coverage"]) and all(row["negative_point_exclusion"]))
        for row in rows
    )
    status = "PASS_FRAME0_ASSISTED_CANARY" if all_present and point_contract and all(object_points_clear) else "HOLD_FRAME0_ASSISTED_CANARY"
    result = {
        "case": case_name,
        "status": status,
        "source": str(frame_path),
        "source_sha256": sha256(frame_path),
        "width": width,
        "height": height,
        "objects": rows,
        "object_protection_points": spec["object_protection_points"],
        "object_protection_points_clear": object_points_clear,
        "raw_mask_policy": "human_core uses selected unmodified SAM3.1 text raw instances; wearable classes use unmodified point-prompt raw IDs; union is review-only",
        "forbidden_repairs_used": [],
    }
    (case_root / "RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint SHA drift")
    lease = json.loads((PROJECT / "_run/GPU_LEASE.json").read_text())
    if lease.get("status") not in {"ACTIVE", "ACQUIRED"} or lease.get("holder") != "mask-bilateral-tracker-final":
        raise RuntimeError("central GPU lease is not active for this runner")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    root = PROJECT / "_run/gpt_mask_assisted_bilateral_wearable_canary_20260902_v8"
    if root.exists():
        raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(CODE_ROOT))
    from tools import run_newtask_baseline_sam31_mask_probe as baseline
    adapter_module = baseline.load_adapter_module()
    adapter, build_evidence = adapter_module.build_pinned_adapter(
        official_code_root=CODE_ROOT,
        checkpoint_path=CHECKPOINT,
    )
    results = []
    try:
        for case_name, spec in SPECS.items():
            results.append(run_case(adapter.model, case_name, spec, root))
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        adapter.predictor.shutdown()
    summary = {
        "schema_version": "assisted-bilateral-wearable-frame0-canary-v1",
        "status": "PASS_FRAME0_BOTH_CASES" if all(item["status"].startswith("PASS") for item in results) else "HOLD_FRAME0",
        "cases": results,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "build_evidence": build_evidence,
        "runner_sha256": sha256(Path(__file__)),
        "claim_limit": "frame0 assisted point canary only; no temporal or cross-session generalization claim",
    }
    (root / "RESULT.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "wall_seconds", "peak_cuda_reserved_bytes")}))
    return 0 if summary["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
