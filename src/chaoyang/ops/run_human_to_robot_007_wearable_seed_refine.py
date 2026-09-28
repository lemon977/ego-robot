"""Fixed 007 frame-184 SAM3.1 text+box seed followed by same-ID point refinement."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_007_wearable_sam_canary import (
    CHECKPOINT, CHECKPOINT_SHA, CODE, HEIGHT, PROTECT_POINTS, RAW, SPECS,
    WIDTH, evaluate_instance, normalize_outputs,
)

TASK = "human_to_robot_007_wearable_seed_refine_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/frame184_seed_refine_v1"
PROMPT = "finger-mounted sensor"


def choose_seed(masks: np.ndarray, ids: np.ndarray, spec: tuple) -> int | None:
    """Choose only an actual raw ID containing the pre-frozen positive point."""
    _, _, (x, y, w, h), (px, py), _ = spec
    candidates = []
    for i, raw_id in enumerate(ids):
        mask = masks[i]
        area = int(mask.sum())
        if mask[py, px] and area:
            overlap = int(mask[y:y+h, x:x+w].sum())
            candidates.append((-overlap, area, int(raw_id)))
    return min(candidates)[2] if candidates else None


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    rows = index.get("task_packets", [])
    if len(rows) != 1 or rows[0].get("task_id") != TASK or not rows[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if (lease.get("status") != "ACQUIRED" or lease.get("task_id") != TASK + ":scene"
            or lease.get("gpu_process_pid") != os.getpid()
            or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise RuntimeError("GPU_LEASE_REQUIRED")
    if DEST.exists():
        raise FileExistsError(DEST)
    raw = cv2.imread(str(RAW), cv2.IMREAD_COLOR)
    if raw is None or raw.shape != (HEIGHT, WIDTH, 3):
        raise RuntimeError("RAW_DOMAIN")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    DEST.mkdir(parents=True)
    frames = DEST / "frames"
    masks_dir = DEST / "raw_instances"
    frames.mkdir()
    masks_dir.mkdir()
    if not cv2.imwrite(str(frames / "00000.png"), raw):
        raise RuntimeError("FRAME_WRITE_FAILED")
    sys.path.insert(0, str(CODE))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    adapter, build_evidence = build_pinned_adapter(official_code_root=CODE, checkpoint_path=CHECKPOINT)
    result_rows = []
    accepted = np.zeros((HEIGHT, WIDTH), bool)
    try:
        for spec in SPECS:
            object_id, name, (x, y, w, h), positive, negatives = spec
            state = adapter.model.init_state(resource_path=str(frames), offload_video_to_cpu=True,
                                             async_loading_frames=False)
            try:
                _, seeded = adapter.model.add_prompt(
                    inference_state=state, frame_idx=0, text_str=PROMPT,
                    boxes_xywh=[[x / WIDTH, y / HEIGHT, w / WIDTH, h / HEIGHT]],
                    box_labels=[1], clear_old_boxes=True, output_prob_thresh=0.5,
                )
                seed_masks, seed_ids = normalize_outputs(seeded)
                seed_id = choose_seed(seed_masks, seed_ids, spec)
                seed_mask = (seed_masks[int(np.where(seed_ids == seed_id)[0][0])]
                             if seed_id is not None else np.zeros((HEIGHT, WIDTH), bool))
                seed_path = masks_dir / f"{object_id}_{name}_seed.png"
                if not cv2.imwrite(str(seed_path), seed_mask.astype(np.uint8) * 255):
                    raise RuntimeError("SEED_MASK_WRITE_FAILED")
                refined_ids = np.empty(0, dtype=np.int64)
                refined_mask = np.zeros((HEIGHT, WIDTH), bool)
                if seed_id is not None:
                    points = [positive, *negatives]
                    _, refined = adapter.model.add_prompt(
                        inference_state=state, frame_idx=0, text_str=None,
                        points=torch.tensor([[px / WIDTH, py / HEIGHT] for px, py in points], dtype=torch.float32),
                        point_labels=torch.tensor([1] + [0] * len(negatives), dtype=torch.int32),
                        boxes_xywh=None, box_labels=None, obj_id=seed_id,
                        rel_coordinates=True, clear_old_points=True,
                        clear_old_boxes=False, output_prob_thresh=0.5,
                    )
                    refined_masks, refined_ids = normalize_outputs(refined)
                    matches = np.where(refined_ids == seed_id)[0]
                    if len(matches):
                        refined_mask = refined_masks[int(matches[0])]
                refined_path = masks_dir / f"{object_id}_{name}_refined.png"
                if not cv2.imwrite(str(refined_path), refined_mask.astype(np.uint8) * 255):
                    raise RuntimeError("REFINED_MASK_WRITE_FAILED")
                row = evaluate_instance(refined_mask, spec)
                row.update(seed_raw_ids=seed_ids.astype(int).tolist(), chosen_seed_id=seed_id,
                           seed_area_pixels=int(seed_mask.sum()), seed_mask=artifact_ref(seed_path),
                           refined_raw_ids=refined_ids.astype(int).tolist(),
                           refined_mask=artifact_ref(refined_path))
                result_rows.append(row)
                if row["status"] == "SUPPORTED_SINGLE_FRAME":
                    accepted |= refined_mask
            finally:
                state.clear()
    finally:
        adapter.predictor.shutdown()
    overlay = raw.copy()
    overlay[accepted] = (0.55 * overlay[accepted] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    for row in result_rows:
        px, py = row["positive_xy"]
        cv2.circle(overlay, (px, py), 8,
                   (0, 255, 0) if row["status"] == "SUPPORTED_SINGLE_FRAME" else (0, 0, 255), 2)
        cv2.putText(overlay, row["name"], (px + 8, py - 7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
    review = DEST / "007_FRAME184_SEED_REFINE_RAW_INSTANCES.png"
    if not cv2.imwrite(str(review), np.concatenate([raw, overlay], axis=1)):
        raise RuntimeError("REVIEW_WRITE_FAILED")
    protect_clear = all(not accepted[py, px] for px, py in PROTECT_POINTS)
    supported = [row["name"] for row in result_rows if row["status"] == "SUPPORTED_SINGLE_FRAME"]
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_WEARABLE_SEED_REFINE_V1", "task_id": TASK,
        "session_id": "get_potato_chips_0915_007", "source_frame": 184,
        "execution": "EXECUTED", "structure": "PASS",
        "instance_quality": "PASS" if len(supported) == len(SPECS) and protect_clear else "REJECTED_QUALITY",
        "clean_quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
        "prompt": PROMPT, "raw": artifact_ref(RAW), "checkpoint": artifact_ref(CHECKPOINT),
        "checkpoint_expected_sha256": CHECKPOINT_SHA, "runner": artifact_ref(Path(__file__)),
        "model_build": build_evidence, "lease_fencing_token": lease["fencing_token"],
        "rows": result_rows, "supported": supported,
        "object_protect_points_clear": protect_clear, "review": artifact_ref(review),
        "claim_limit": "Raw one-frame seed/refine instances only; no propagation, Clean or product authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"instance_quality": result["instance_quality"],
                      "supported": supported, "result": str(DEST / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
