"""Pinned SAM3.1 one-frame wearable instance canary for 007 frame 184."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json

TASK = "human_to_robot_007_wearable_sam_canary_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/frame184_sam_v2"
RAW = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw/000184.png"
CODE = REPO_ROOT / "vendor/SAM3"
CHECKPOINT = REPO_ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
WIDTH, HEIGHT = 1280, 960
# Box and positive/negative points were frozen from the raw source image, before any model output.
SPECS = (
    (201, "left_index_device", (165, 483, 85, 96), (209, 539), ((225, 650), (490, 350), (870, 410))),
    (202, "left_middle_device", (235, 530, 120, 105), (276, 576), ((270, 675), (490, 350), (870, 410))),
    (203, "left_thumb_device", (385, 712, 105, 120), (444, 766), ((410, 855), (490, 350), (870, 410))),
    (204, "right_index_device", (630, 299, 58, 73), (657, 328), ((654, 393), (490, 350), (870, 410))),
    (205, "right_middle_device", (679, 298, 66, 78), (703, 329), ((690, 389), (490, 350), (870, 410))),
    (206, "right_ring_device", (719, 356, 78, 76), (749, 389), ((703, 429), (490, 350), (870, 410))),
)
PROTECT_POINTS = ((490, 350), (870, 410))


def normalize_outputs(outputs: dict) -> tuple[np.ndarray, np.ndarray]:
    masks = outputs.get("out_binary_masks")
    ids = outputs.get("out_obj_ids")
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    elif isinstance(masks, (tuple, list)):
        masks = np.stack([x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x) for x in masks])
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().numpy()
    value = np.asarray(masks if masks is not None else np.zeros((0, HEIGHT, WIDTH), bool))
    while value.ndim > 3 and value.shape[1] == 1:
        value = np.squeeze(value, axis=1)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[1:] != (HEIGHT, WIDTH):
        raise ValueError(f"SAM_MASK_DOMAIN:{value.shape}")
    return value.astype(bool), np.asarray(ids if ids is not None else [], dtype=np.int64).reshape(-1)


def evaluate_instance(mask: np.ndarray, spec: tuple) -> dict:
    object_id, name, box, positive, negatives = spec
    if mask.shape != (HEIGHT, WIDTH) or mask.dtype != bool:
        raise ValueError("INSTANCE_DOMAIN")
    x, y, w, h = box
    area = int(mask.sum())
    inside_box = int(mask[y:y + h, x:x + w].sum())
    positive_covered = bool(mask[positive[1], positive[0]])
    negative_excluded = [not bool(mask[ny, nx]) for nx, ny in negatives]
    status = "SUPPORTED_SINGLE_FRAME" if (
        20 <= area <= 15000 and positive_covered and all(negative_excluded)
        and inside_box >= 0.5 * area
    ) else "REJECTED_INSTANCE"
    return {"object_id": object_id, "name": name, "box_xywh": list(box),
            "positive_xy": list(positive), "negative_xy": [list(p) for p in negatives],
            "area_pixels": area, "pixels_inside_frozen_box": inside_box,
            "positive_covered": positive_covered, "negative_excluded": negative_excluded,
            "status": status}


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK or not index["task_packets"][0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") != "ACQUIRED" or lease.get("task_id") != TASK + ":scene" or lease.get("gpu_process_pid") != os.getpid() or str(lease.get("gpu_id")) != os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("GPU_LEASE_REQUIRED")
    if DEST.exists():
        raise FileExistsError(DEST)
    raw = cv2.imread(str(RAW), cv2.IMREAD_COLOR)
    if raw is None or raw.shape != (HEIGHT, WIDTH, 3):
        raise ValueError("RAW_DOMAIN")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    retry_signature = ATTEMPT / "RUNTIME_RETRY_SIGNATURE.json"
    if retry_signature.exists():
        raise FileExistsError(retry_signature)
    atomic_json(retry_signature, {
        "schema_version": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_RUNTIME_RETRY_SIGNATURE_V1",
        "task_id": TASK, "reason": "PINNED_SAM_POINT_PROMPT_FORBIDS_SIMULTANEOUS_BOX",
        "runner": artifact_ref(Path(__file__)), "raw": artifact_ref(RAW),
        "checkpoint": artifact_ref(CHECKPOINT),
        "prompt_spec": [[item[0], item[1], list(item[2]), list(item[3]),
                         [list(point) for point in item[4]]] for item in SPECS],
        "calibration_or_absent": "ABSENT_PIXEL_ONLY",
        "output_schema": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_CANARY_V1",
        "prior_failure": artifact_ref(ATTEMPT / "SAM_GPU_RECEIPT.json"),
    })
    DEST.mkdir(parents=True)
    frames = DEST / "frames"
    masks_root = DEST / "raw_instances"
    frames.mkdir()
    masks_root.mkdir()
    if not cv2.imwrite(str(frames / "00000.png"), raw):
        raise RuntimeError("FRAME_COPY_FAILED")
    sys.path.insert(0, str(CODE))
    from chaoyang.pipeline.sam31_compat_adapter_v1 import build_pinned_adapter
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE")
    adapter, build_evidence = build_pinned_adapter(official_code_root=CODE, checkpoint_path=CHECKPOINT)
    rows = []
    union = np.zeros((HEIGHT, WIDTH), bool)
    try:
        for spec in SPECS:
            object_id, name, box, positive, negatives = spec
            state = adapter.model.init_state(resource_path=str(frames), offload_video_to_cpu=True,
                                             async_loading_frames=False)
            try:
                points = [positive, *negatives]
                point_tensor = torch.tensor([[x / WIDTH, y / HEIGHT] for x, y in points], dtype=torch.float32)
                labels = torch.tensor([1] + [0] * len(negatives), dtype=torch.int32)
                # The pinned multiplex API forbids simultaneous point and box prompts.
                # The frozen box remains an independent acceptance region below.
                _, outputs = adapter.model.add_prompt(
                    inference_state=state, frame_idx=0, text_str=None,
                    points=point_tensor, point_labels=labels, boxes_xywh=None,
                    box_labels=None, obj_id=object_id, rel_coordinates=True,
                    clear_old_points=True, clear_old_boxes=True, output_prob_thresh=0.5,
                )
                masks, ids = normalize_outputs(outputs)
                selected = next((masks[i] for i, value in enumerate(ids) if int(value) == object_id),
                                np.zeros((HEIGHT, WIDTH), bool))
                row = evaluate_instance(selected, spec)
                path = masks_root / f"{object_id}_{name}.png"
                if not cv2.imwrite(str(path), selected.astype(np.uint8) * 255):
                    raise RuntimeError(f"MASK_WRITE_FAILED:{name}")
                row["mask"] = artifact_ref(path)
                row["raw_instance_ids"] = [int(value) for value in ids]
                rows.append(row)
                if row["status"] == "SUPPORTED_SINGLE_FRAME":
                    union |= selected
            finally:
                state.clear()
    finally:
        adapter.predictor.shutdown()
    overlay = raw.copy()
    overlay[union] = (0.55 * overlay[union] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)
    for row in rows:
        x, y = row["positive_xy"]
        cv2.circle(overlay, (x, y), 8, (0, 255, 0) if row["status"] == "SUPPORTED_SINGLE_FRAME" else (0, 0, 255), 2)
        cv2.putText(overlay, row["name"], (x + 8, y - 7), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    review = DEST / "007_FRAME184_SAM_WEARABLE_RAW_INSTANCES.png"
    if not cv2.imwrite(str(review), np.concatenate([raw, overlay], axis=1)):
        raise RuntimeError("REVIEW_WRITE_FAILED")
    all_supported = all(row["status"] == "SUPPORTED_SINGLE_FRAME" for row in rows)
    protect_clear = all(not bool(union[y, x]) for x, y in PROTECT_POINTS)
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_WEARABLE_SAM_CANARY_V1",
        "task_id": TASK, "session_id": "get_potato_chips_0915_007", "source_frame": 184,
        "execution": "EXECUTED", "structure": "PASS", "instance_quality": "PASS" if all_supported and protect_clear else "REJECTED_QUALITY",
        "clean_quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
        "raw": artifact_ref(RAW), "checkpoint": artifact_ref(CHECKPOINT),
        "checkpoint_expected_sha256": CHECKPOINT_SHA, "adapter_build": build_evidence,
        "runtime_signature": artifact_ref(retry_signature),
        "lease_fencing_token": lease["fencing_token"],
        "prompt_authority": "AI_FIXED_VISUAL_PROMPTS_NOT_PIXEL_GROUND_TRUTH",
        "rows": rows, "object_protect_points": [list(p) for p in PROTECT_POINTS],
        "object_protect_points_clear": protect_clear, "review": artifact_ref(review),
        "claim_limit": "Raw single-frame SAM instance test only. No temporal identity, full mask, Clean or product quality is established.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["instance_quality"], "supported": [row["name"] for row in rows if row["status"] == "SUPPORTED_SINGLE_FRAME"],
                      "result": str(DEST / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
