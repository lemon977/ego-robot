#!/usr/bin/env python3
"""Diagnose the official SAM3.1 semantic full-propagation route.

This is development evidence only.  It deliberately avoids the project's
point-refinement/stable-ID route and asks the pinned multiplex model to run its
native text+box grounding followed by a complete full-video propagation with
``is_last_batch=True``.  The experiment answers whether the frozen detector
object survives on repeated, translated, and real short sequences.
"""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
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
from chaoyang.ops.run_sam31_state_route_diagnostic import _prepare_sequences, _render


CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
ANNOTATIONS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/PROMPT_ANNOTATIONS_FROZEN.json"
BOOTSTRAP = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/attempts/attempt_0002/PROMPT_CANDIDATES.json"


def collect_selected_object(
    model: Any,
    state: dict[str, Any],
    *,
    accepted_id: int,
    frame_count: int,
    height: int,
    width: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    """Run a complete native full propagation and retain one frozen object ID."""
    rows_by_frame: dict[int, dict[str, Any]] = {}
    masks: dict[int, np.ndarray] = {}
    for frame, output in model.propagate_in_video(
        inference_state=state,
        start_frame_idx=0,
        max_frame_num_to_track=frame_count,
        reverse=False,
        output_prob_thresh=0.5,
        is_last_batch=True,
    ):
        normalized_masks, scores, ids = sam_helpers.normalize(output, height, width)
        ids_int = [int(value) for value in ids]
        matches = [index for index, value in enumerate(ids_int) if value == accepted_id]
        if len(matches) > 1:
            raise RuntimeError("frozen object ID emitted more than once")
        unexpected = sorted(set(ids_int) - {accepted_id})
        if matches:
            index = matches[0]
            mask = np.asarray(normalized_masks[index], dtype=bool)
            score = float(scores[index])
        else:
            mask = np.zeros((height, width), dtype=bool)
            score = None
        rows_by_frame[int(frame)] = {
            "frame_index": int(frame),
            "present": bool(mask.any()),
            "area_pixels": int(mask.sum()),
            "score": score,
            "output_object_ids": ids_int,
            "unexpected_object_ids": unexpected,
            "reason": "SEMANTIC_FULL_ID_PRESENT" if mask.any() else "SEMANTIC_FULL_ID_ABSENT",
        }
        masks[int(frame)] = mask

    rows: list[dict[str, Any]] = []
    for frame in range(frame_count):
        row = rows_by_frame.get(frame)
        if row is None:
            row = {
                "frame_index": frame,
                "present": False,
                "area_pixels": 0,
                "score": None,
                "output_object_ids": [],
                "unexpected_object_ids": [],
                "reason": "NO_STREAM_OUTPUT",
            }
            masks[frame] = np.zeros((height, width), dtype=bool)
        rows.append(row)
    return rows, masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["annotations"]["poker015"]
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))["tasks"]["poker015"]["object"]
    source_rgb = Path(annotations["source_rgb"]["path"]).resolve(strict=True)
    sequences = _prepare_sequences(output, source_rgb)
    adapter, build_evidence = sam31_compat_adapter_v1.build_pinned_adapter(
        official_code_root=SAM_ROOT,
        checkpoint_path=CHECKPOINT,
    )
    model = adapter.model
    accepted_id = int(bootstrap["selected_object_id"])
    # The frozen prompt box is the independent manual action-contract input.
    # ``bootstrap.selected_row.bbox_xyxy`` is a model output and is therefore
    # not a valid replacement for the prompt.  Earlier attempt_0001 also used
    # a nonexistent top-level key and failed before model inference.
    x0, y0, x1, y1 = annotations["box_xyxy"]
    box = [[x0 / 1280, y0 / 960, (x1 - x0 + 1) / 1280, (y1 - y0 + 1) / 960]]
    matrix: dict[str, Any] = {}
    videos = []

    try:
        for scenario, spec in sequences.items():
            state = model.init_state(
                resource_path=str(Path(spec["path"]).resolve()),
                offload_video_to_cpu=False,
                async_loading_frames=False,
            )
            _, seed_output = model.add_prompt(
                inference_state=state,
                frame_idx=0,
                text_str="a purple-backed playing card",
                boxes_xywh=box,
                box_labels=[1],
                output_prob_thresh=0.5,
            )
            seed_masks, seed_scores, seed_ids = sam_helpers.normalize(seed_output, 960, 1280)
            seed_ids_int = [int(value) for value in seed_ids]
            if accepted_id not in seed_ids_int:
                raise RuntimeError(f"frozen detector ID {accepted_id} absent in {scenario}: {seed_ids_int}")
            rows, masks = collect_selected_object(
                model,
                state,
                accepted_id=accepted_id,
                frame_count=int(spec["frames"]),
                height=960,
                width=1280,
            )
            present = sum(bool(row["present"]) for row in rows)
            payload = {
                "route": "PINNED_OFFICIAL_LOW_LEVEL_SEMANTIC_FULL_IS_LAST_BATCH_TRUE",
                "frames": len(rows),
                "present_frames": present,
                "presence_fraction": present / max(len(rows), 1),
                "seed_object_ids": seed_ids_int,
                "seed_scores": [float(value) for value in seed_scores],
                "seed_areas": [int(np.asarray(mask, dtype=bool).sum()) for mask in seed_masks],
                "rows": rows,
            }
            matrix[scenario] = payload
            video = _render(output, f"{scenario}_SemanticFull", spec, [("官方语义Full", payload, masks)])
            videos.append(artifact_ref(video))
            # Preserve a finished scenario even if a later scenario or cleanup fails.
            atomic_json(output / f"{scenario}_SCENARIO_RESULT.json", {
                "status": "PASSED_DIAGNOSTIC_SCENARIO_ONLY",
                "scenario": scenario,
                "payload": payload,
                "video": videos[-1],
                "claim_limit": "Development route diagnostic only; no Mask authority.",
            })
            state.clear()
    finally:
        # The compatibility adapter intentionally exposes only the pinned
        # model surface; some revisions do not implement an explicit close.
        # Cleanup must therefore be optional and must never turn completed
        # inference into a runtime failure.
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    result = {
        "schema_version": "chaoyang-sam31-semantic-full-route-diagnostic-v1",
        "task_id": "research_sam31_semantic_full_route_diagnostic",
        "created_at": now_iso(),
        "status": "PASSED_DIAGNOSTIC",
        "authority_promoted": False,
        "training_eligible": False,
        "claim_limit": "Frozen Poker015 semantic full-route diagnostic only; not Mask accuracy, RC1 retry, or authority.",
        "accepted_detector_object_id": accepted_id,
        "model_runtime": {
            "hotstart_delay": int(model.hotstart_delay),
            "postprocess_batch_size": int(model.postprocess_batch_size),
            "masklet_confirmation_enable": bool(model.masklet_confirmation_enable),
        },
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
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "METRICS.json", {"matrix": matrix})
    atomic_json(output / "RUN_RECEIPT.json", {
        "task_id": result["task_id"],
        "status": result["status"],
        "created_at": result["created_at"],
        "code": artifact_ref(Path(__file__).resolve()),
    })
    atomic_json(output / "NEXT_ACTION.json", {
        "status": "DIAGNOSTIC_COMPLETE",
        "next_task_id": None,
        "claim_limit": result["claim_limit"],
    })
    print(json.dumps({"status": result["status"], "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
