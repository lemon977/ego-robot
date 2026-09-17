#!/usr/bin/env python3
"""Frozen SAM3.1 semantic-route ablation of ``is_last_batch`` only.

This is a bounded research diagnostic, not Mask authority.  Prompt, source,
checkpoint, accepted object ID and all propagation arguments except the
batch-final flag are identical between the two variants.
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
from pathlib import Path
import socket
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAM_ROOT = ROOT / "vendor/SAM3"
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as sam_helpers
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops.run_sam31_semantic_full_route_diagnostic import (
    ANNOTATIONS, BOOTSTRAP, CHECKPOINT, _prepare_sequences,
)
from chaoyang.ops.run_sam31_state_route_diagnostic import _render


def collect_variant(
    model: Any,
    state: dict[str, Any],
    *,
    accepted_id: int,
    frame_count: int,
    is_last_batch: bool,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Materialize each frame; missing hotstart outputs remain explicit."""
    seen: dict[int, dict[str, Any]] = {}
    masks: dict[int, np.ndarray] = {}
    for frame, output in model.propagate_in_video(
        inference_state=state,
        start_frame_idx=0,
        max_frame_num_to_track=frame_count,
        reverse=False,
        output_prob_thresh=0.5,
        is_last_batch=is_last_batch,
    ):
        frame = int(frame)
        if not 0 <= frame < frame_count or frame in seen:
            raise RuntimeError(f"duplicate or out-of-range propagated frame: {frame}")
        normalized, scores, ids = sam_helpers.normalize(output, 960, 1280)
        id_list = [int(value) for value in ids]
        matching = [index for index, value in enumerate(id_list) if value == accepted_id]
        if len(matching) > 1:
            raise RuntimeError("frozen object ID emitted more than once")
        if matching:
            index = matching[0]
            mask = np.asarray(normalized[index], dtype=bool)
            score = float(scores[index])
        else:
            mask = np.zeros((960, 1280), dtype=bool)
            score = None
        seen[frame] = {
            "frame_index": frame,
            "present": bool(mask.any()),
            "area_pixels": int(mask.sum()),
            "score": score,
            "output_object_ids": id_list,
            "reason": "FROZEN_ID_PRESENT" if mask.any() else "FROZEN_ID_ABSENT",
            "mask_sha256": hashlib.sha256(np.packbits(mask.reshape(-1)).tobytes()).hexdigest(),
        }
        masks[frame] = mask
    rows = []
    for frame in range(frame_count):
        rows.append(seen.get(frame, {
            "frame_index": frame,
            "present": False,
            "area_pixels": 0,
            "score": None,
            "output_object_ids": [],
            "reason": "NO_STREAM_OUTPUT",
            "mask_sha256": hashlib.sha256(np.packbits(np.zeros((960, 1280), dtype=bool).reshape(-1)).tobytes()).hexdigest(),
        }))
        masks.setdefault(frame, np.zeros((960, 1280), dtype=bool))
    return rows, masks


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    frozen = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["annotations"]["poker015"]
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))["tasks"]["poker015"]["object"]
    accepted_id = int(bootstrap["selected_object_id"])
    source_rgb = Path(frozen["source_rgb"]["path"]).resolve(strict=True)
    sequences = _prepare_sequences(output, source_rgb)
    adapter, build_evidence = sam31_compat_adapter_v1.build_pinned_adapter(
        official_code_root=SAM_ROOT,
        checkpoint_path=CHECKPOINT,
    )
    model = adapter.model
    x0, y0, x1, y1 = frozen["box_xyxy"]
    box = [[x0 / 1280, y0 / 960, (x1 - x0 + 1) / 1280, (y1 - y0 + 1) / 960]]
    matrix: dict[str, Any] = {}
    videos = []
    try:
        for scenario in ("repeated_seed_8", "real_prefix_16"):
            spec = sequences[scenario]
            variants = []
            matrix[scenario] = {}
            for flag in (True, False):
                state = model.init_state(
                    resource_path=str(Path(spec["path"]).resolve()),
                    offload_video_to_cpu=False,
                    async_loading_frames=False,
                )
                try:
                    _, prompt_output = model.add_prompt(
                        inference_state=state,
                        frame_idx=0,
                        text_str="a purple-backed playing card",
                        boxes_xywh=box,
                        box_labels=[1],
                        output_prob_thresh=0.5,
                    )
                    seed_masks, _, seed_ids = sam_helpers.normalize(prompt_output, 960, 1280)
                    seed_ids_int = [int(value) for value in seed_ids]
                    if accepted_id not in seed_ids_int:
                        raise RuntimeError(f"frozen detector ID absent: {seed_ids_int}")
                    seed_area = int(np.asarray(seed_masks[seed_ids_int.index(accepted_id)], dtype=bool).sum())
                    route, obj_ids = model.parse_action_history_for_propagation(state)
                    rows, masks = collect_variant(
                        model, state,
                        accepted_id=accepted_id,
                        frame_count=int(spec["frames"]),
                        is_last_batch=flag,
                    )
                    label = f"is_last_batch={flag}"
                    summary = {
                        "route_before_propagation": route,
                        "action_object_ids": obj_ids,
                        "seed_area_pixels": seed_area,
                        "frames": len(rows),
                        "stream_frames": sum(row["reason"] != "NO_STREAM_OUTPUT" for row in rows),
                        "present_frames": sum(row["present"] for row in rows),
                        "rows": rows,
                    }
                    matrix[scenario][label] = summary
                    atomic_json(output / f"{scenario}_is_last_batch_{str(flag).lower()}.json", summary)
                    variants.append((label, summary, masks))
                finally:
                    state.clear()
            true_rows = matrix[scenario]["is_last_batch=True"]["rows"]
            false_rows = matrix[scenario]["is_last_batch=False"]["rows"]
            matrix[scenario]["comparison"] = {
                "same_stream_frames": matrix[scenario]["is_last_batch=True"]["stream_frames"] == matrix[scenario]["is_last_batch=False"]["stream_frames"],
                "same_presence": all(a["present"] == b["present"] for a, b in zip(true_rows, false_rows, strict=True)),
                "same_pixel_masks": all(a["mask_sha256"] == b["mask_sha256"] for a, b in zip(true_rows, false_rows, strict=True)),
                "different_mask_frame_indices": [a["frame_index"] for a, b in zip(true_rows, false_rows, strict=True) if a["mask_sha256"] != b["mask_sha256"]],
            }
            atomic_json(output / f"{scenario}_comparison.json", matrix[scenario]["comparison"])
            videos.append(artifact_ref(_render(output, f"{scenario}_BatchFlag", spec, variants)))
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    result = {
        "schema_version": "chaoyang-sam31-semantic-batchflag-ablation-v1",
        "task_id": "research_sam31_semantic_batchflag_ablation",
        "created_at": now_iso(),
        "status": "PASSED_DIAGNOSTIC",
        "authority_promoted": False,
        "training_eligible": False,
        "claim_limit": "Frozen Poker015 short sequences only; isolates batch-final flag within the semantic route, not a production Mask quality gate.",
        "accepted_detector_object_id": accepted_id,
        "build_evidence": build_evidence,
        "matrix": matrix,
        "videos": videos,
        "inputs": {
            "annotations": artifact_ref(ANNOTATIONS),
            "bootstrap": artifact_ref(BOOTSTRAP),
            "checkpoint": artifact_ref(CHECKPOINT),
            "source_rgb": artifact_ref(source_rgb),
        },
        "host": socket.gethostname(),
        "wall_seconds": time.perf_counter() - start,
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "task_id": result["task_id"], "status": result["status"],
        "created_at": result["created_at"],
        "code": artifact_ref(Path(__file__).resolve()),
    })
    print(json.dumps({"status": result["status"], "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
