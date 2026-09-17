#!/usr/bin/env python3
"""Localize the frozen Poker015 SAM3.1 seed-stage route divergence."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "vendor/SAM3"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as helpers
from chaoyang.governance.common import artifact_ref, atomic_json, now_iso
from chaoyang.ops.run_sam31_semantic_full_route_diagnostic import ANNOTATIONS, BOOTSTRAP, CHECKPOINT


def _stage(model, state, output, label: str, selected_id: int, destination: Path) -> dict:
    masks, scores, ids = helpers.normalize(output, 960, 1280)
    rows = [
        {"object_id": int(object_id), "area_pixels": int(np.asarray(mask, dtype=bool).sum()), "score": float(score)}
        for mask, score, object_id in zip(masks, scores, ids, strict=True)
    ]
    route, action_ids = model.parse_action_history_for_propagation(state)
    selected = [row for row in rows if row["object_id"] == selected_id]
    payload = {
        "stage": label,
        "selected_object_id": selected_id,
        "selected_area_pixels": selected[0]["area_pixels"] if len(selected) == 1 else None,
        "rows": rows,
        "propagation_route": route,
        "action_object_ids": [int(x) for x in (action_ids or [])],
        "state_top_level_keys": sorted(str(key) for key in state),
    }
    atomic_json(destination, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resource-path", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    resource = args.resource_path.resolve(strict=True)
    frozen = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["annotations"]["poker015"]
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))["tasks"]["poker015"]["object"]
    detector_id = int(bootstrap["selected_object_id"])
    tracking_id = 10001 + detector_id
    x0, y0, x1, y1 = frozen["box_xyxy"]
    box = [[x0 / 1280, y0 / 960, (x1 - x0 + 1) / 1280, (y1 - y0 + 1) / 960]]
    px, py = frozen["positive_point_xy"]
    adapter, build_evidence = sam31_compat_adapter_v1.build_pinned_adapter(
        official_code_root=ROOT / "vendor/SAM3", checkpoint_path=CHECKPOINT,
    )
    model = adapter.model
    stages = []
    try:
        semantic = model.init_state(resource_path=str(resource), offload_video_to_cpu=False, async_loading_frames=False)
        try:
            _, response = model.add_prompt(
                inference_state=semantic, frame_idx=0, text_str="a purple-backed playing card",
                boxes_xywh=box, box_labels=[1], output_prob_thresh=0.5,
            )
            stages.append(_stage(model, semantic, response, "SEMANTIC_TEXT_BOX", detector_id, output / "01_SEMANTIC_TEXT_BOX.json"))
        finally:
            semantic.clear()
        tracking = model.init_state(resource_path=str(resource), offload_video_to_cpu=False, async_loading_frames=False)
        try:
            _, response = model.add_prompt(
                inference_state=tracking, frame_idx=0, text_str=None,
                clear_old_points=True, points=None, point_labels=None,
                boxes_xywh=box, box_labels=[1], clear_old_boxes=True,
                obj_id=tracking_id, rel_coordinates=True, output_prob_thresh=0.5,
            )
            stages.append(_stage(model, tracking, response, "TRACKING_BOX_ONLY", tracking_id, output / "02_TRACKING_BOX_ONLY.json"))
            _, response = model.add_prompt(
                inference_state=tracking, frame_idx=0, text_str=None,
                clear_old_points=True,
                points=torch.tensor([[px / 1280, py / 960]], dtype=torch.float32),
                point_labels=torch.tensor([1], dtype=torch.int32),
                boxes_xywh=None, box_labels=None, clear_old_boxes=False,
                obj_id=tracking_id, rel_coordinates=True, output_prob_thresh=0.5,
            )
            stages.append(_stage(model, tracking, response, "TRACKING_BOX_THEN_POINT", tracking_id, output / "03_TRACKING_BOX_THEN_POINT.json"))
        finally:
            tracking.clear()
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
    result = {
        "schema_version": "chaoyang-sam31-seed-stage-probe-v1",
        "task_id": "research_sam31_seed_stage_probe",
        "session_id": "play_cards_0901_015",
        "created_at": now_iso(),
        "status": "PASSED_DIAGNOSTIC",
        "claim_limit": "Single frozen Poker015 seed frame; identifies first observed stage, not model root cause or Mask accuracy.",
        "authority_promoted": False,
        "training_eligible": False,
        "stages": stages,
        "build_evidence": build_evidence,
        "inputs": {
            "annotations": artifact_ref(ANNOTATIONS),
            "bootstrap": artifact_ref(BOOTSTRAP),
            "checkpoint": artifact_ref(CHECKPOINT),
            "source_rgb": artifact_ref(Path(frozen["source_rgb"]["path"])),
        },
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "stages": [(r["stage"], r["selected_area_pixels"], r["propagation_route"]) for r in stages], "result": artifact_ref(output / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
