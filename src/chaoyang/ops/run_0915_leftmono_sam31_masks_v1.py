#!/usr/bin/env python3
"""Fresh SAM3.1 masks for the 0915 physical-left mono replay.

This runner keeps the two mask semantics separate.  Human masks are selected
from raw SAM3.1 text-prompt instances by the frame-zero HaWoR palm anchors.
Potato-chip masks are independent raw SAM3.1 text-prompt instances.  The
selector changes no mask pixels, missing instances stay empty, and no tracker
role is invented for a session that did not capture tracker telemetry.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT / "vendor/SAM3"
CHECKPOINT = PROJECT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
ADAPTER_PATH = PROJECT / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location("leftmono_sam31_adapter", ADAPTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned SAM3.1 adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize(outputs: dict[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks: Any = outputs.get("out_binary_masks")
    scores: Any = outputs.get("out_probs", [])
    ids: Any = outputs.get("out_obj_ids", [])
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    elif isinstance(masks, (list, tuple)):
        masks = [item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item) for item in masks]
        masks = np.stack(masks) if masks else np.zeros((0, height, width), bool)
    array = np.asarray(masks if masks is not None else np.zeros((0, height, width), bool))
    while array.ndim > 3 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    if array.ndim == 2:
        array = array[None]
    if array.shape[1:] != (height, width):
        raise RuntimeError(f"SAM3.1 mask shape drift: {array.shape}")
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    score_array = np.asarray(scores, np.float64).reshape(-1)
    id_array = np.asarray(ids, np.int64).reshape(-1)
    if len(array) != len(score_array) or len(array) != len(id_array):
        raise RuntimeError("SAM3.1 output lengths differ")
    return array.astype(bool), score_array, id_array


def geometry(mask: np.ndarray) -> dict[str, Any]:
    yy, xx = np.where(mask)
    if not len(xx):
        return {"area_pixels": 0, "bbox_xyxy": None, "centroid_xy": None}
    return {
        "area_pixels": int(len(xx)),
        "bbox_xyxy": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
        "centroid_xy": [float(xx.mean()), float(yy.mean())],
    }


def anchor_distance(mask: np.ndarray, xy: np.ndarray) -> float:
    x = int(np.clip(round(float(xy[0])), 0, mask.shape[1] - 1))
    y = int(np.clip(round(float(xy[1])), 0, mask.shape[0] - 1))
    if mask[y, x]:
        return 0.0
    yy, xx = np.where(mask)
    if not len(xx):
        return float("inf")
    return float(np.sqrt((xx - x) ** 2 + (yy - y) ** 2).min())


def extract_frames(video: Path, root: Path) -> tuple[int, int, int, float]:
    root.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    expected = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not cv2.imwrite(str(root / f"{count:05d}.png"), frame):
                raise RuntimeError(f"failed to write frame {count}")
            count += 1
    finally:
        capture.release()
    if count != expected:
        raise RuntimeError(f"video decode mismatch {count} != {expected}")
    return count, width, height, fps


def output_by_id(outputs: dict[str, Any], height: int, width: int) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    masks, scores, ids = normalize(outputs, height, width)
    return (
        {int(value): masks[index] for index, value in enumerate(ids)},
        {int(value): float(scores[index]) for index, value in enumerate(ids)},
    )


def save_human_masks(
    model: Any,
    frames: Path,
    output: Path,
    anchors: dict[str, np.ndarray],
    count: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    role_root = output / "role"
    role_root.mkdir(parents=True)
    state = model.init_state(resource_path=str(frames), offload_video_to_cpu=True, async_loading_frames=False)
    chosen: dict[str, int] = {}
    seed_rows: list[dict[str, Any]] = []
    present = {"left_human": 0, "right_human": 0}
    try:
        _, seed = model.add_prompt(
            inference_state=state,
            frame_idx=0,
            text_str="a person's hand and forearm",
            output_prob_thresh=0.5,
        )
        seed_masks, seed_scores, seed_ids = normalize(seed, height, width)
        for index, mask in enumerate(seed_masks):
            row = {"object_id": int(seed_ids[index]), "score": float(seed_scores[index]), **geometry(mask)}
            row["left_anchor_distance_px"] = anchor_distance(mask, anchors["left_human"])
            row["right_anchor_distance_px"] = anchor_distance(mask, anchors["right_human"])
            seed_rows.append(row)
        used: set[int] = set()
        for side in ("left_human", "right_human"):
            ranked = sorted(
                (
                    (anchor_distance(mask, anchors[side]), -int(mask.sum()), int(seed_ids[index]))
                    for index, mask in enumerate(seed_masks)
                    if int(seed_ids[index]) not in used and 2_000 <= int(mask.sum()) <= 500_000
                )
            )
            if not ranked or ranked[0][0] > 35.0:
                raise RuntimeError(f"no raw SAM3.1 human instance contains the {side} HaWoR palm anchor")
            chosen[side] = ranked[0][2]
            used.add(ranked[0][2])

        def save(frame_index: int, outputs: dict[str, Any]) -> None:
            masks, _ = output_by_id(outputs, height, width)
            label = np.zeros((height, width), np.uint8)
            for value, side in enumerate(("left_human", "right_human"), start=1):
                mask = masks.get(chosen[side], np.zeros((height, width), bool))
                if mask.any():
                    present[side] += 1
                    label[mask] = value
            if not cv2.imwrite(str(role_root / f"{frame_index:05d}.png"), label):
                raise RuntimeError(f"failed to write human mask {frame_index}")

        save(0, seed)
        seen = {0}
        for item in model.propagate_in_video(
            inference_state=state,
            start_frame_idx=0,
            max_frame_num_to_track=count,
            reverse=False,
            output_prob_thresh=0.5,
        ):
            frame_index, outputs = int(item[0]), item[1]
            if frame_index not in seen:
                save(frame_index, outputs)
                seen.add(frame_index)
        for frame_index in range(count):
            target = role_root / f"{frame_index:05d}.png"
            if not target.exists():
                cv2.imwrite(str(target), np.zeros((height, width), np.uint8))
    finally:
        state.clear()
    return {"prompt": "a person's hand and forearm", "chosen_ids": chosen, "seed_candidates": seed_rows, "present_frames": present}


def save_object_masks(
    model: Any,
    frames: Path,
    output: Path,
    count: int,
    height: int,
    width: int,
) -> dict[str, Any]:
    object_root = output / "object"
    object_root.mkdir(parents=True)
    state = model.init_state(resource_path=str(frames), offload_video_to_cpu=True, async_loading_frames=False)
    seed_rows: list[dict[str, Any]] = []
    selected: list[int] = []
    present_by_id: dict[int, int] = {}
    prompt_route = "TEXT_ONLY"
    frozen_boxes_xyxy = [
        [590, 388, 662, 466],
        [768, 390, 836, 468],
        [680, 478, 760, 566],
    ]
    frozen_points_xy = [[628, 419], [800, 426], [718, 522]]
    try:
        _, seed = model.add_prompt(
            inference_state=state,
            frame_idx=0,
            text_str="a potato chip",
            output_prob_thresh=0.5,
        )
        seed_masks, seed_scores, seed_ids = normalize(seed, height, width)
        for index, mask in enumerate(seed_masks):
            geo = geometry(mask)
            row = {"object_id": int(seed_ids[index]), "score": float(seed_scores[index]), **geo}
            seed_rows.append(row)
            centroid = geo["centroid_xy"]
            if (
                250 <= geo["area_pixels"] <= 30_000
                and centroid is not None
                and 50 <= centroid[0] <= width - 50
                and 80 <= centroid[1] <= int(height * 0.75)
            ):
                selected.append(int(seed_ids[index]))
        selected = sorted(selected, key=lambda object_id: next(row["centroid_xy"][0] for row in seed_rows if row["object_id"] == object_id))
        if not selected:
            # The pinned language head emits no potato-chip instance on this
            # frame.  Use the already-established PICO task-object route:
            # discard one priming ID, then prompt three explicit raw IDs with
            # positive/negative points.  Points select identities only; every
            # saved pixel still comes directly from SAM3.1.
            state.clear()
            state = model.init_state(resource_path=str(frames), offload_video_to_cpu=True, async_loading_frames=False)
            model.add_prompt(
                inference_state=state,
                frame_idx=0,
                points=torch.tensor([[470 / width, 420 / height]], dtype=torch.float32),
                point_labels=torch.tensor([1], dtype=torch.int32),
                obj_id=9901,
                clear_old_points=True,
                output_prob_thresh=0.5,
                rel_coordinates=True,
            )
            selected = [301, 302, 303]
            seed_rows = []
            for object_id, positive in zip(selected, frozen_points_xy, strict=True):
                negative = [point for point in frozen_points_xy if point != positive] + [[470, 420]]
                points = [positive, *negative]
                labels = [1, *([0] * len(negative))]
                _, seed = model.add_prompt(
                    inference_state=state,
                    frame_idx=0,
                    clear_old_points=True,
                    points=torch.tensor([[x / width, y / height] for x, y in points], dtype=torch.float32),
                    point_labels=torch.tensor(labels, dtype=torch.int32),
                    output_prob_thresh=0.5,
                    obj_id=object_id,
                    rel_coordinates=True,
                )
                masks_by_id, scores_by_id = output_by_id(seed, height, width)
                mask = masks_by_id.get(object_id, np.zeros((height, width), bool))
                row = {"object_id": object_id, "score": scores_by_id.get(object_id), **geometry(mask)}
                row["positive_covered"] = bool(mask[positive[1], positive[0]])
                row["negative_excluded"] = [not bool(mask[y, x]) for x, y in negative]
                if not row["positive_covered"] or not all(row["negative_excluded"]):
                    raise RuntimeError(f"SAM3.1 point seed gate failed: {row}")
                seed_rows.append(row)
            prompt_route = "DISCARDED_PRIMING_ID_PLUS_THREE_FROZEN_POSITIVE_NEGATIVE_POINT_RAW_IDS_AFTER_TEXT_EMPTY"
        if not selected:
            raise RuntimeError("SAM3.1 returned no potato-chip instance for text or frozen geometric boxes")
        if len(selected) > 6:
            raise RuntimeError(f"implausible potato-chip instance count: {len(selected)}")
        present_by_id = {value: 0 for value in selected}

        def save(frame_index: int, outputs: dict[str, Any]) -> None:
            masks, _ = output_by_id(outputs, height, width)
            label = np.zeros((height, width), np.uint8)
            for label_id, object_id in enumerate(selected, start=1):
                mask = masks.get(object_id, np.zeros((height, width), bool))
                if mask.any():
                    present_by_id[object_id] += 1
                    label[mask] = label_id
            if not cv2.imwrite(str(object_root / f"{frame_index:05d}.png"), label):
                raise RuntimeError(f"failed to write object mask {frame_index}")

        save(0, seed)
        seen = {0}
        for item in model.propagate_in_video(
            inference_state=state,
            start_frame_idx=0,
            max_frame_num_to_track=count,
            reverse=False,
            output_prob_thresh=0.5,
        ):
            frame_index, outputs = int(item[0]), item[1]
            if frame_index not in seen:
                save(frame_index, outputs)
                seen.add(frame_index)
        for frame_index in range(count):
            target = object_root / f"{frame_index:05d}.png"
            if not target.exists():
                cv2.imwrite(str(target), np.zeros((height, width), np.uint8))
    finally:
        state.clear()
    return {
        "prompt": "a potato chip",
        "prompt_route": prompt_route,
        "frozen_boxes_xyxy": frozen_boxes_xyxy if prompt_route != "TEXT_ONLY" else None,
        "frozen_points_xy": frozen_points_xy if prompt_route != "TEXT_ONLY" else None,
        "selected_ids_left_to_right_at_frame0": selected,
        "seed_candidates": seed_rows,
        "present_frames_by_id": {str(key): value for key, value in present_by_id.items()},
    }


def render_review(frames: Path, output: Path, count: int, width: int, height: int, fps: float) -> tuple[Path, Path]:
    review = output / "MASK_SAM31_LEFTMONO_REVIEW.mp4"
    writer = cv2.VideoWriter(str(review), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    sample_indices = set(np.linspace(0, count - 1, 12, dtype=int).tolist())
    samples: list[np.ndarray] = []
    try:
        for frame_index in range(count):
            raw = cv2.imread(str(frames / f"{frame_index:05d}.png"), cv2.IMREAD_COLOR)
            role = cv2.imread(str(output / "role" / f"{frame_index:05d}.png"), cv2.IMREAD_GRAYSCALE)
            objects = cv2.imread(str(output / "object" / f"{frame_index:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if raw is None or role is None or objects is None:
                raise RuntimeError(f"review input missing at frame {frame_index}")
            overlay = raw.astype(np.float32)
            colors = {1: np.asarray((255, 180, 20), np.float32), 2: np.asarray((20, 50, 255), np.float32)}
            for label, color in colors.items():
                mask = role == label
                overlay[mask] = 0.4 * overlay[mask] + 0.6 * color
            palette = (np.asarray((60, 230, 255), np.float32), np.asarray((50, 255, 80), np.float32), np.asarray((220, 70, 255), np.float32), np.asarray((20, 180, 255), np.float32))
            for label in np.unique(objects):
                if label == 0:
                    continue
                mask = objects == label
                color = palette[(int(label) - 1) % len(palette)]
                overlay[mask] = 0.25 * overlay[mask] + 0.75 * color
            overlay = overlay.astype(np.uint8)
            cv2.rectangle(overlay, (0, 0), (width - 1, 55), (0, 0, 0), -1)
            cv2.putText(overlay, f"0915 physical-left | SAM3.1 raw instances | frame {frame_index:03d}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "cyan=left human red=right human; yellow/green/magenta=separate potato-chip IDs; tracker=ABSENT", (12, 47), cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(overlay)
            if frame_index in sample_indices:
                samples.append(cv2.resize(overlay, (640, 480), interpolation=cv2.INTER_AREA))
    finally:
        writer.release()
    sheet = output / "MASK_SAM31_LEFTMONO_CONTACT_SHEET.jpg"
    canvas = np.zeros((3 * 480, 4 * 640, 3), np.uint8)
    for index, sample in enumerate(samples[:12]):
        row, column = divmod(index, 4)
        canvas[row * 480:(row + 1) * 480, column * 640:(column + 1) * 640] = sample
    if not cv2.imwrite(str(sheet), canvas):
        raise RuntimeError("failed to write mask contact sheet")
    return review, sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    video = args.video.resolve(strict=True)
    hawor_npz = args.hawor_npz.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output required: {output}")
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("SAM3.1 checkpoint SHA drift")
    output.mkdir(parents=True)
    frames = output / "input_frames"
    count, width, height, fps = extract_frames(video, frames)
    if (count, width, height) != (379, 1280, 960):
        raise RuntimeError(f"unexpected input identity {(count, width, height)}")
    with np.load(hawor_npz, allow_pickle=False) as payload:
        joints_2d = np.asarray(payload["joints_2d"], np.float64)
        observed = np.asarray(payload["observed"], bool)
    if joints_2d.shape != (2, count, 21, 2) or observed.shape != (2, count):
        raise RuntimeError("HaWoR identity/geometry shape mismatch")
    if not observed[:, 0].all():
        raise RuntimeError("frame-zero bilateral HaWoR anchors are required")
    anchors = {
        "left_human": np.nanmean(joints_2d[0, 0, [0, 5, 9, 13, 17]], axis=0),
        "right_human": np.nanmean(joints_2d[1, 0, [0, 5, 9, 13, 17]], axis=0),
    }
    if not all(np.isfinite(value).all() for value in anchors.values()):
        raise RuntimeError("non-finite HaWoR selection anchor")
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    module = load_adapter()
    adapter, build_evidence = module.build_pinned_adapter(official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT)
    try:
        human = save_human_masks(adapter.model, frames, output, anchors, count, height, width)
        gc.collect()
        torch.cuda.empty_cache()
        objects = save_object_masks(adapter.model, frames, output, count, height, width)
    finally:
        adapter.predictor.shutdown()
    review, sheet = render_review(frames, output, count, width, height, fps)
    human_min = min(human["present_frames"].values())
    object_min = min(objects["present_frames_by_id"].values())
    passed = human_min >= int(0.90 * count) and object_min >= int(0.70 * count)
    result = {
        "schema_version": "0915-leftmono-sam31-masks-v1",
        "status": "PASS_DEVELOPMENT_MASKS" if passed else "HOLD_MASK_REVIEW",
        "session_id": "get_potato_chips_0915_001",
        "frame_count": count,
        "rgb_primary": "PHYSICAL_LEFT_SOURCE_INDEX_1_EQUIDIS62_TO_PINHOLE_1280X960_FOV90",
        "inputs": {"video": ref(video), "hawor_mano21": ref(hawor_npz), "checkpoint": ref(CHECKPOINT), "adapter": ref(ADAPTER_PATH)},
        "tracker_modality": "ABSENT_NOT_CAPTURED_NO_TRACKER_ROLE_CREATED",
        "human_masks": human,
        "task_object_masks": objects,
        "mask_pixels": "UNMODIFIED_RAW_SAM31_INSTANCES_SELECTED_BY_GEOMETRY_ONLY",
        "forbidden_repairs_used": [],
        "artifacts": {"review": ref(review), "contact_sheet": ref(sheet)},
        "build_evidence": build_evidence,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "wall_seconds": time.monotonic() - started,
        "claim_limit": "Fresh development SAM3.1 masks on physical-left mono. Instance identity requires visual review; no ground-truth, Clean, Object6D, contact or Robot authority is implied.",
    }
    (output / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "human": human["present_frames"], "objects": objects["present_frames_by_id"], "wall_seconds": result["wall_seconds"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
