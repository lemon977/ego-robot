#!/usr/bin/env python3
"""Final bounded SAM3.1 mask test for the glove/controller sensor route.

This is deliberately a mask feasibility canary, not a Clean authority run.
White gloves are detected with the pinned SAM3.1 text model.  Controllers are
seeded from their same-frame recorded 6D origins projected through the
session's equidistant camera model.  Visible task objects are independently
protected.  A failed gate stops before ProPainter.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT / "vendor/SAM3"
CHECKPOINT = PROJECT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
DATA_ROOT = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0910/cleaned")
H1_ROOT = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor/h1_hand/sidecars"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
SIDES = ("left", "right")
COLORS = {
    "left": (60, 220, 80), "right": (230, 70, 230),
    "left_controller": (255, 220, 30), "right_controller": (30, 150, 255),
    "object": (30, 230, 255), "removal": (40, 50, 240),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def project_equidis(points: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float64)
    x, y, z = points.T
    radius = np.hypot(x, y)
    theta = np.arctan2(radius, z)
    dx = np.divide(x, radius, out=np.zeros_like(x), where=radius > 1e-12)
    dy = np.divide(y, radius, out=np.zeros_like(y), where=radius > 1e-12)
    theta2 = theta * theta
    radial = np.ones_like(theta)
    power = theta2.copy()
    for coefficient in distortion[:6]:
        radial += coefficient * power
        power *= theta2
    qx, qy = theta * radial * dx, theta * radial * dy
    radius2 = qx * qx + qy * qy
    p1, p2 = distortion[6:]
    sx, sy = qx.copy(), qy.copy()
    qx = sx + 2 * p1 * sx * sy + p2 * (radius2 + 2 * sx * sx)
    qy = sy + p1 * (radius2 + 2 * sy * sy) + 2 * p2 * sx * sy
    return np.column_stack((intrinsics[0, 0] * qx + intrinsics[0, 2], intrinsics[1, 1] * qy + intrinsics[1, 2]))


def normalize(outputs: dict[str, Any], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    masks = outputs.get("out_binary_masks")
    ids = outputs.get("out_obj_ids")
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    elif isinstance(masks, (list, tuple)):
        masks = np.stack([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x) for x in masks])
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().numpy()
    masks = np.asarray(masks if masks is not None else np.zeros((0, height, width), bool))
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = np.squeeze(masks, axis=1)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape[1:] != (height, width):
        raise RuntimeError(f"SAM3.1 mask geometry drift: {masks.shape}")
    return masks.astype(bool), np.asarray(ids if ids is not None else [], np.int64).reshape(-1)


def distance_to_mask(mask: np.ndarray, uv: np.ndarray) -> float:
    if not mask.any() or not np.isfinite(uv).all():
        return float("inf")
    x, y = np.rint(uv).astype(int)
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
        return 0.0
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return float("inf")
    return float(dist[y, x])


def assign_two_instances(masks: np.ndarray, ids: np.ndarray, anchors: dict[str, np.ndarray]) -> tuple[dict[str, int], dict[str, Any]]:
    if len(masks) < 2:
        return {}, {"reason": "fewer_than_two_instances", "instance_count": int(len(masks))}
    candidates = list(range(len(masks)))
    best = None
    for pair in permutations(candidates, 2):
        cost = distance_to_mask(masks[pair[0]], anchors["left"]) + distance_to_mask(masks[pair[1]], anchors["right"])
        if best is None or cost < best[0]:
            best = (cost, pair)
    assert best is not None
    mapping = {"left": int(ids[best[1][0]]), "right": int(ids[best[1][1]])}
    return mapping, {
        "instance_count": int(len(masks)), "total_anchor_distance_px": float(best[0]),
        "mapping": mapping,
        "left_distance_px": distance_to_mask(masks[best[1][0]], anchors["left"]),
        "right_distance_px": distance_to_mask(masks[best[1][1]], anchors["right"]),
    }


def text_stream(model: Any, frame_root: Path, prompt: str, anchor: int, height: int, width: int,
                anchors: dict[str, np.ndarray] | None = None) -> tuple[Any, dict[str, Any]]:
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True, async_loading_frames=False)
    try:
        _, initial = model.add_prompt(inference_state=state, frame_idx=anchor, text_str=prompt, output_prob_thresh=0.5)
        masks, ids = normalize(initial, height, width)
        if anchors is None:
            selected_ids = {int(x) for x in ids}
            selection = {"instance_count": int(len(ids)), "selected_ids": sorted(selected_ids)}
        else:
            mapping, selection = assign_two_instances(masks, ids, anchors)
            if len(mapping) != 2:
                return {side: {} for side in SIDES}, {"status": "FAILED_ID_SELECTION", **selection}
            selected_ids = set(mapping.values())
        raw: dict[str, dict[int, np.ndarray]] | dict[int, np.ndarray]
        if anchors is None:
            raw = {}
            initial_union = np.logical_or.reduce([masks[i] for i, value in enumerate(ids) if int(value) in selected_ids]) if selected_ids else np.zeros((height, width), bool)
            raw[anchor] = initial_union
        else:
            raw = {side: {} for side in SIDES}
            by_id = {int(value): masks[i] for i, value in enumerate(ids)}
            for side in SIDES:
                raw[side][anchor] = by_id[mapping[side]].copy()
        for reverse, maximum in ((False, 24 - anchor), (True, anchor + 1)):
            for frame, outputs in model.propagate_in_video(
                inference_state=state, start_frame_idx=anchor, max_frame_num_to_track=maximum,
                reverse=reverse, output_prob_thresh=0.5,
            ):
                pmasks, pids = normalize(outputs, height, width)
                by_id = {int(value): pmasks[i] for i, value in enumerate(pids)}
                if anchors is None:
                    parts = [by_id[value] for value in selected_ids if value in by_id]
                    raw[int(frame)] = np.logical_or.reduce(parts) if parts else np.zeros((height, width), bool)
                else:
                    for side in SIDES:
                        if mapping[side] in by_id:
                            raw[side][int(frame)] = by_id[mapping[side]].copy()
        return raw, {"status": "COMPLETE", "prompt": prompt, "anchor_slot": anchor, **selection}
    finally:
        state.clear()


def controller_stream(model: Any, frame_root: Path, controller_uv: np.ndarray, height: int, width: int) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    state = model.init_state(resource_path=str(frame_root), offload_video_to_cpu=True, async_loading_frames=False)
    streams = {side: {} for side in SIDES}
    rows = []
    ids = {"left": 201, "right": 202}
    try:
        model.add_prompt(
            inference_state=state, frame_idx=0,
            points=torch.tensor([[0.02, 0.02]], dtype=torch.float32),
            point_labels=torch.tensor([1], dtype=torch.int32), obj_id=9901,
            rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
        )
        for slot in range(24):
            outputs = None
            for side_index, side in enumerate(SIDES):
                own = controller_uv[slot, side_index]
                other = controller_uv[slot, 1 - side_index]
                points = [own]
                labels = [1]
                if np.isfinite(other).all():
                    points.append(other)
                    labels.append(0)
                _, outputs = model.add_prompt(
                    inference_state=state, frame_idx=slot,
                    points=torch.tensor([[float(p[0]) / width, float(p[1]) / height] for p in points], dtype=torch.float32),
                    point_labels=torch.tensor(labels, dtype=torch.int32), obj_id=ids[side],
                    rel_coordinates=True, clear_old_points=True, output_prob_thresh=0.5,
                )
            assert outputs is not None
            masks, raw_ids = normalize(outputs, height, width)
            by_id = {int(value): masks[i] for i, value in enumerate(raw_ids)}
            row = {"slot": slot}
            for side_index, side in enumerate(SIDES):
                mask = by_id.get(ids[side], np.zeros((height, width), bool))
                streams[side][slot] = mask.copy()
                row[side] = {
                    "area_fraction": float(mask.mean()),
                    "seed_distance_px": distance_to_mask(mask, controller_uv[slot, side_index]),
                }
            rows.append(row)
        return streams, {"status": "COMPLETE", "provider": "SAM31_PROJECTED_CONTROLLER6D_POINT", "rows": rows}
    finally:
        state.clear()


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    output = image.copy()
    output[mask] = np.rint((1 - alpha) * output[mask] + alpha * np.asarray(color)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2)
    return output


def label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(FONT_PATH), 22)
    y = 8
    for line in lines:
        box = draw.textbbox((8, y), line, font=font)
        draw.rectangle((4, y - 2, box[2] + 5, box[3] + 2), fill=(0, 0, 0, 190))
        draw.text((8, y), line, font=font, fill=(255, 255, 255))
        y += 27
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def render_case(case_root: Path, session_id: str, selected: list[int], images: list[np.ndarray],
                wrists_uv: np.ndarray, joints_uv: np.ndarray, controllers_uv: np.ndarray,
                gloves: dict[str, dict[int, np.ndarray]], controllers: dict[str, dict[int, np.ndarray]],
                objects: dict[int, np.ndarray], fps: float) -> tuple[Path, Path, list[dict[str, Any]]]:
    review_frames, metrics = [], []
    h, w = images[0].shape[:2]
    for slot, source_frame in enumerate(selected):
        raw = images[slot]
        glove_union = np.zeros((h, w), bool)
        controller_union = np.zeros((h, w), bool)
        glove_tile, controller_tile = raw.copy(), raw.copy()
        row: dict[str, Any] = {"slot": slot, "source_frame": source_frame, "time_s": source_frame / fps}
        for si, side in enumerate(SIDES):
            gm = gloves[side].get(slot, np.zeros((h, w), bool))
            cm = controllers[side].get(slot, np.zeros((h, w), bool))
            glove_union |= gm
            controller_union |= cm
            glove_tile = tint(glove_tile, gm, COLORS[side])
            controller_tile = tint(controller_tile, cm, COLORS[f"{side}_controller"])
            valid_joints = joints_uv[slot, si]
            finite = np.isfinite(valid_joints).all(axis=1)
            in_frame = finite & (valid_joints[:, 0] >= 0) & (valid_joints[:, 0] < w) & (valid_joints[:, 1] >= 0) & (valid_joints[:, 1] < h)
            covered = []
            for point in valid_joints[in_frame]:
                x, y = np.rint(point).astype(int)
                covered.append(bool(gm[y, x]))
            row[side] = {
                "glove_area_fraction": float(gm.mean()),
                "wrist_distance_px": distance_to_mask(gm, wrists_uv[slot, si]),
                "joint_projection_count": int(in_frame.sum()),
                "joint_projection_coverage": float(np.mean(covered)) if covered else None,
                "controller_area_fraction": float(cm.mean()),
                "controller_seed_distance_px": distance_to_mask(cm, controllers_uv[slot, si]),
            }
        object_mask = objects.get(slot, np.zeros((h, w), bool))
        removal = (glove_union | controller_union) & ~object_mask
        row["object_area_fraction"] = float(object_mask.mean())
        row["removal_area_fraction"] = float(removal.mean())
        row["protected_overlap_pixels_after_subtraction"] = int((removal & object_mask).sum())
        metrics.append(row)

        seed_tile = raw.copy()
        for si, side in enumerate(SIDES):
            for p in joints_uv[slot, si]:
                if np.isfinite(p).all(): cv2.circle(seed_tile, tuple(np.rint(p).astype(int)), 3, COLORS[side], -1, cv2.LINE_AA)
            if np.isfinite(controllers_uv[slot, si]).all():
                cv2.drawMarker(seed_tile, tuple(np.rint(controllers_uv[slot, si]).astype(int)), COLORS[f"{side}_controller"], cv2.MARKER_CROSS, 24, 3)
        object_tile = tint(raw, object_mask, COLORS["object"])
        removal_tile = tint(raw, removal, COLORS["removal"])
        checker = raw.copy()
        grid = ((np.indices((h, w)).sum(axis=0) // 24) % 2).astype(bool)
        checker[removal & grid] = (230, 230, 230)
        checker[removal & ~grid] = (80, 80, 80)
        title = f"{session_id} | 原帧 {source_frame}/{selected[-1]} | {source_frame/fps:.2f}s"
        tiles = [
            label(seed_tile, [title, "MANUS21投影: 绿=左 紫=右；×=Controller 6D投影"]),
            label(glove_tile, [title, "SAM3.1 白手套实例"]),
            label(controller_tile, [title, "SAM3.1 + Controller 6D投影点提示"]),
            label(object_tile, [title, "SAM3.1 可见任务物体保护(黄色)"]),
            label(removal_tile, [title, "候选消除区(红)：手套+手柄-物体保护"]),
            label(checker, [title, "仅Mask棋盘预览；不是Clean结果"]),
        ]
        tiles = [cv2.resize(x, (640, 480), interpolation=cv2.INTER_AREA) for x in tiles]
        review_frames.append(np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:]))))

    frames_dir = case_root / "review_frames"
    frames_dir.mkdir(parents=True)
    for index, frame in enumerate(review_frames):
        cv2.imwrite(str(frames_dir / f"{index:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    video = case_root / f"{session_id}_最终测试_手套Controller混合Mask_24帧.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "3",
        "-i", str(frames_dir / "%05d.jpg"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(video)
    ], check=True)
    thumbs = [cv2.resize(frame, (640, 320), interpolation=cv2.INTER_AREA) for frame in review_frames]
    sheet = np.vstack([np.hstack(thumbs[i:i+4]) for i in range(0, 24, 4)])
    contact = case_root / f"{session_id}_最终测试_24帧总览.jpg"
    cv2.imwrite(str(contact), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return video, contact, metrics


def case_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    role_rows = [row[side] for row in rows for side in SIDES]
    glove_ok = [r["glove_area_fraction"] >= 0.002 and r["glove_area_fraction"] <= 0.25 and r["wrist_distance_px"] <= 20 and (r["joint_projection_coverage"] or 0) >= 0.55 for r in role_rows]
    controller_ok = [r["controller_area_fraction"] >= 0.001 and r["controller_area_fraction"] <= 0.20 and r["controller_seed_distance_px"] <= 3 for r in role_rows]
    object_ok = [row["object_area_fraction"] >= 0.0002 for row in rows]
    removal_ok = [row["removal_area_fraction"] <= 0.45 for row in rows]
    metrics = {
        "glove_role_pass_ratio": float(np.mean(glove_ok)),
        "controller_role_pass_ratio": float(np.mean(controller_ok)),
        "object_frame_pass_ratio": float(np.mean(object_ok)),
        "removal_frame_pass_ratio": float(np.mean(removal_ok)),
        "max_removal_area_fraction": float(max(row["removal_area_fraction"] for row in rows)),
        "protected_overlap_pixels_after_subtraction": int(sum(row["protected_overlap_pixels_after_subtraction"] for row in rows)),
    }
    gate = bool(metrics["glove_role_pass_ratio"] >= 0.90 and metrics["controller_role_pass_ratio"] >= 0.90 and metrics["object_frame_pass_ratio"] >= 0.80 and metrics["removal_frame_pass_ratio"] == 1.0 and metrics["protected_overlap_pixels_after_subtraction"] == 0)
    return {"pass": gate, "thresholds": {"glove_role_pass_ratio": 0.90, "controller_role_pass_ratio": 0.90, "object_frame_pass_ratio": 0.80, "max_removal_area_fraction": 0.45}, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--frameset", type=Path, required=True)
    args = parser.parse_args()
    output = args.attempt_root.resolve()
    if output.exists():
        raise RuntimeError(f"fresh/no-clobber attempt required: {output}")
    output.mkdir(parents=True)
    frameset = json.loads(args.frameset.read_text(encoding="utf-8"))
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("pinned SAM3.1 checkpoint SHA mismatch")
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter

    started = time.time()
    adapter, build_evidence = build_pinned_adapter(official_code_root=CODE_ROOT, checkpoint_path=CHECKPOINT)
    case_results = []
    try:
        for spec in frameset["sessions"]:
            task_dir = "playing_cards" if spec["task"] == "poker" else "potato_chips"
            session = DATA_ROOT / task_dir / spec["session_id"]
            selected = [int(x) for x in spec["selected_frames"]]
            case_root = output / spec["session_id"]
            view = case_root / "sam_input"
            view.mkdir(parents=True)
            images, rows = [], []
            for slot, frame in enumerate(selected):
                source = session / "preprocess/all_data" / f"{frame:05d}" / "rgb.png"
                os.symlink(source.resolve(strict=True), view / f"{slot:05d}.png")
                images.append(cv2.imread(str(source), cv2.IMREAD_COLOR))
                rows.append(json.loads((session / "preprocess/all_data" / f"{frame:05d}" / "training_data.json").read_text(encoding="utf-8")))
            h, w = images[0].shape[:2]
            k = np.asarray(rows[0]["metadata"]["k"], np.float64)
            d = np.asarray(rows[0]["metadata"]["d"], np.float64)
            h1 = np.load(H1_ROOT / task_dir / f"{spec['session_id']}.npz")
            wrist_xyz = h1["T_wrist_to_camera"][selected, :, :3, 3]
            joints_xyz = h1["joint_xyz_camera_m"][selected]
            wrist_uv = np.stack([[project_equidis(wrist_xyz[slot, si][None], k, d)[0] for si in range(2)] for slot in range(24)])
            joints_uv = np.stack([[project_equidis(joints_xyz[slot, si], k, d) for si in range(2)] for slot in range(24)])
            controller_xyz = np.stack([[np.asarray(rows[slot]["entities"]["hands"][side]["controller6d"]["T_controller_to_camera"], np.float64)[:3, 3] for side in SIDES] for slot in range(24)])
            controller_uv = np.stack([[project_equidis(controller_xyz[slot, si][None], k, d)[0] for si in range(2)] for slot in range(24)])
            anchor = 12
            gloves, glove_meta = text_stream(adapter.model, view, "a white instrumented glove", anchor, h, w, {side: wrist_uv[anchor, si] for si, side in enumerate(SIDES)})
            controllers, controller_meta = controller_stream(adapter.model, view, controller_uv, h, w)
            object_prompt = "playing cards" if spec["task"] == "poker" else "potato chips"
            objects, object_meta = text_stream(adapter.model, view, object_prompt, anchor, h, w)
            video, contact, per_frame = render_case(case_root, spec["session_id"], selected, images, wrist_uv, joints_uv, controller_uv, gloves, controllers, objects, float(rows[0]["metadata"]["fps"]))
            gate = case_gate(per_frame)
            per_frame_path = case_root / "PER_FRAME_METRICS.json"
            atomic_json(per_frame_path, per_frame)
            case_results.append({
                "task": spec["task"], "session_id": spec["session_id"], "selected_frames": selected,
                "gate": gate, "glove_inference": glove_meta, "controller_inference": controller_meta,
                "object_inference": object_meta,
                "artifacts": {"video": artifact(video), "contact_sheet": artifact(contact), "per_frame_metrics": artifact(per_frame_path)},
            })
    finally:
        adapter.predictor.shutdown()
    automatic_pass = all(case["gate"]["pass"] for case in case_results)
    result = {
        "schema_version": "sensor-glove-controller-final-canary-v1",
        "status": "MASK_GATE_PASSED_VISUAL_REVIEW_REQUIRED" if automatic_pass else "FAILED_QUALITY_C",
        "automatic_gate_pass": automatic_pass,
        "fresh_clean_started": False,
        "fresh_clean_stop_reason": "MASK_GATE_FAILED" if not automatic_pass else "AWAITING_VISUAL_REVIEW",
        "algorithm": {
            "glove": "pinned SAM3.1 text instance: a white instrumented glove",
            "controller": "pinned SAM3.1 point prompt from same-frame Controller 6D projection",
            "object_protection": "pinned SAM3.1 task-object text instance",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "build_evidence": build_evidence,
        },
        "frameset": artifact(args.frameset.resolve(strict=True)),
        "cases": case_results,
        "elapsed_seconds": time.time() - started,
        "claim_limit": "Development feasibility decision for sensor-domain visual removal only; not Clean authority, contact truth, Robot control truth, or a rejection of Controller+MANUS/Tactile sidecars.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "cases": [{"session_id": c["session_id"], **c["gate"]["metrics"]} for c in case_results]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
