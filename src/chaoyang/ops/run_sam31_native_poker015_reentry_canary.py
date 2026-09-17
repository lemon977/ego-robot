#!/usr/bin/env python3
"""Bounded same-session SAM3.1 native route through Poker015 re-entry frames.

Development diagnostic only.  The frozen independent reference supplies frame
selection, not pixel truth; presence or object ID alone never grants authority.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "vendor/SAM3"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as helpers
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops.run_sam31_semantic_full_route_diagnostic import ANNOTATIONS, BOOTSTRAP, CHECKPOINT, collect_selected_object
from chaoyang.ops.run_sam31_state_route_diagnostic import _render


REFERENCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001/MASK_NEXT_EVALUATION_REFERENCE.json"
SESSION_ID = "play_cards_0901_015"
FRAMES = 224  # Ends on the second frozen re-entry boundary; not a full-session claim.


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    annotation = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["annotations"]["poker015"]
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))["tasks"]["poker015"]["object"]
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["tasks"]["poker_object"]
    keyframes = sorted(int(item) for item in reference["diagnostic_frames"])
    if keyframes[-1] != 223:
        raise RuntimeError("frozen Poker re-entry frame set changed")
    source_frame = Path(annotation["source_rgb"]["path"]).resolve(strict=True)
    frame_root = source_frame.parents[1]
    if frame_root.parent.parent.name != SESSION_ID:
        raise RuntimeError("source RGB does not belong to the frozen full session")
    input_dir = output / "inputs/real_prefix_224"
    input_dir.mkdir(parents=True)
    source_paths = []
    for frame in range(FRAMES):
        path = (frame_root / f"{frame:05d}" / "rgb.png").resolve(strict=True)
        os.symlink(path, input_dir / f"{frame:05d}.png")
        source_paths.append(path)
    adapter, build_evidence = sam31_compat_adapter_v1.build_pinned_adapter(
        official_code_root=ROOT / "vendor/SAM3", checkpoint_path=CHECKPOINT,
    )
    model = adapter.model
    selected_id = int(bootstrap["selected_object_id"])
    x0, y0, x1, y1 = annotation["box_xyxy"]
    box = [[x0 / 1280, y0 / 960, (x1 - x0 + 1) / 1280, (y1 - y0 + 1) / 960]]
    try:
        state = model.init_state(resource_path=str(input_dir), offload_video_to_cpu=False, async_loading_frames=False)
        try:
            _, prompt = model.add_prompt(
                inference_state=state, frame_idx=0, text_str="a purple-backed playing card",
                boxes_xywh=box, box_labels=[1], output_prob_thresh=0.5,
            )
            _, _, ids = helpers.normalize(prompt, 960, 1280)
            if selected_id not in [int(item) for item in ids]:
                raise RuntimeError("frozen physical-instance detector ID missing at seed")
            route, action_ids = model.parse_action_history_for_propagation(state)
            if route != "propagation_full":
                raise RuntimeError(f"native route unexpectedly changed: {route}")
            rows, masks = collect_selected_object(
                model, state, accepted_id=selected_id, frame_count=FRAMES, height=960, width=1280,
            )
        finally:
            state.clear()
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    for row in rows:
        row["bbox_xyxy"] = _bbox(masks[row["frame_index"]])
    packed = np.stack([np.packbits(masks[frame].reshape(-1)) for frame in range(FRAMES)])
    packed_path = output / "PACKED_SELECTED_MASKS.npz"
    np.savez_compressed(packed_path, frame_ids=np.arange(FRAMES, dtype=np.int32), packed=packed)
    spec = {
        "frames": FRAMES,
        "source_loader": lambda frame: cv2.imread(str(source_paths[frame]), cv2.IMREAD_COLOR),
    }
    video = _render(output, "Poker015_native_semantic_full_224", spec, [("原生SAM3.1语义Full", {"rows": rows}, masks)])
    # Keep the frozen difficult frames in one lossless montage for instance QA.
    tile_w, tile_h = 640, 480
    tiles = []
    for frame in keyframes:
        raw = cv2.imread(str(source_paths[frame]), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f"source decode failed: {frame}")
        painted = raw.copy()
        mask = masks[frame]
        painted[mask] = (0.45 * painted[mask] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        cell = np.concatenate([cv2.resize(raw, (tile_w, tile_h)), cv2.resize(painted, (tile_w, tile_h))], axis=1)
        cv2.putText(cell, f"frame={frame} area={rows[frame]['area_pixels']} id={selected_id}", (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        tiles.append(cell)
    montage = output / "POKER015_FROZEN_REENTRY_KEYFRAMES.png"
    if not cv2.imwrite(str(montage), np.concatenate(tiles, axis=0)):
        raise RuntimeError("montage encode failed")
    present = sum(row["present"] for row in rows)
    result = {
        "schema_version": "chaoyang-sam31-poker015-reentry-canary-v1",
        "task_id": "research_sam31_native_poker015_reentry_canary",
        "session_id": SESSION_ID,
        "created_at": now_iso(),
        "status": "PASSED_DEVELOPMENT_DIAGNOSTIC",
        "authority_promoted": False,
        "training_eligible": False,
        "input_mode": "OFFLINE_FULL_SEQUENCE_DIAGNOSTIC",
        "frames": FRAMES,
        "present_frames": present,
        "frozen_difficult_frame_ids": keyframes,
        "rows": rows,
        "same_physical_instance_proven": False,
        "pixel_accuracy_proven": False,
        "claim_limit": "Frozen play_cards_0901_015 prefix through frame223; presence is not instance or boundary quality, and full propagation may use future frames.",
        "inputs": {
            "annotation": artifact_ref(ANNOTATIONS),
            "bootstrap": artifact_ref(BOOTSTRAP),
            "independent_frame_reference": artifact_ref(REFERENCE),
            "source_rgb_frame0": artifact_ref(source_frame),
            "source_rgb_frame223": artifact_ref(source_paths[-1]),
            "checkpoint": artifact_ref(CHECKPOINT),
        },
        "build_evidence": build_evidence,
        "outputs": {"video": artifact_ref(video), "montage": artifact_ref(montage), "packed_masks": artifact_ref(packed_path)},
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "present": f"{present}/{FRAMES}", "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
