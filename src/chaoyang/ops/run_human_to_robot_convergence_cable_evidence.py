#!/usr/bin/env python3
"""Extract the fixed 007 complaint cable as bounded evidence, not as a removal mask authority."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_baseline_v1_convergence_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/cable_007/wave0"
RAW = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw"
MASK_ROOT = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1"
INDEX = REPO / "tasks/current/INDEX.json"
FRAMES = range(181, 197)


def _write(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    index = load_json(INDEX)
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    masks = OUT / "candidate_masks"; masks.mkdir(parents=True)
    video = OUT / "CABLE_TOP1_EVIDENCE_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1280, 960))
    if not writer.isOpened(): raise RuntimeError("VIDEO_WRITER")
    rows = []
    try:
        for frame in FRAMES:
            image = cv2.imread(str(RAW / f"{frame:06d}.png"), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (960, 1280): raise ValueError(f"RAW_FRAME:{frame}")
            maximum, minimum = image.max(axis=2), image.min(axis=2)
            support = (minimum > 160) & ((maximum - minimum) < 65)
            corridor = np.zeros_like(support); corridor[580:960, 420:720] = True
            support &= corridor
            support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 13)))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
            candidates = [idx for idx in range(1, count) if int(stats[idx, cv2.CC_STAT_HEIGHT]) >= 20]
            selected = max(candidates, key=lambda idx: int(stats[idx, cv2.CC_STAT_HEIGHT]))
            ordered = sorted((int(stats[idx, cv2.CC_STAT_HEIGHT]) for idx in candidates), reverse=True)
            x, y, width, height, area = [int(v) for v in stats[selected]]
            candidate = labels == selected
            unique = height >= 120 and (len(ordered) == 1 or height >= 2 * ordered[1])
            cv2.imwrite(str(masks / f"{frame:06d}.png"), candidate.astype(np.uint8) * 255)
            panel = image.copy(); panel[candidate] = (0.35 * panel[candidate] + 0.65 * np.asarray([0, 0, 255])).astype(np.uint8)
            cv2.rectangle(panel, (420, 580), (719, 959), (255, 200, 0), 2)
            cv2.putText(panel, f"007 frame {frame} | top-1 cable candidate | unique={unique}",
                        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .72, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(panel, "red=evidence candidate; NOT an admitted ProPainter mask", (18, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (0,180,255), 2, cv2.LINE_AA)
            writer.write(panel)
            rows.append({"frame_id": frame, "bbox_xywh": [x,y,width,height], "area_px": area,
                         "height_px": height, "runner_up_height_px": ordered[1] if len(ordered)>1 else 0,
                         "unique_top1_in_frozen_complaint_corridor": bool(unique)})
    finally:
        writer.release()
    decoded = 0; capture = cv2.VideoCapture(str(video))
    while True:
        ok, _ = capture.read()
        if not ok: break
        decoded += 1
    capture.release()
    existing_roles = {}
    for role in ("human_forearm", "capture_device", "task_object"):
        counts=[]
        for frame in FRAMES:
            value=cv2.imread(str(MASK_ROOT/role/f"{frame:06d}.png"),cv2.IMREAD_UNCHANGED)
            counts.append(int(np.count_nonzero(value)) if value is not None else -1)
        existing_roles[role] = {"nonzero_frames": int(sum(v>0 for v in counts)), "pixel_counts": counts}
    all_unique = all(row["unique_top1_in_frozen_complaint_corridor"] for row in rows)
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_CABLE_EVIDENCE_V1", "task_id": TASK,
        "session_id": "get_potato_chips_0915_007",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "INCONCLUSIVE_ASSOCIATION_EVIDENCE",
        "improvement": "VISIBLE_COMPLAINT_CABLE_TOP1_LOCALIZED_16_OF_16" if all_unique else "NO_UNIQUE_TOP1",
        "adoption": "NOT_ADOPTED", "window": [181,196], "rows": rows,
        "top1_unique_frames": int(sum(row["unique_top1_in_frozen_complaint_corridor"] for row in rows)),
        "existing_role_support": existing_roles,
        "propainter_admission": {
            "allowed": False,
            "reason": "EXISTING_HUMAN_DEVICE_ROLE_MASKS_ARE_EMPTY_IN_COMPLAINT_WINDOW; LOCAL_COLOR_SHAPE_PROXY_ALONE_CANNOT_AUTHORIZE_REMOVAL",
        },
        "evidence_authority": "AI_REVIEW_PROXY_FIXED_RAW_COMPLAINT_WINDOW",
        "inputs": {"raw_frame_181": artifact_ref(RAW/"000181.png"),
                   "raw_frame_196": artifact_ref(RAW/"000196.png"),
                   "mask_manifest": artifact_ref(MASK_ROOT/"MASK_MANIFEST.json")},
        "review_video": {**artifact_ref(video), "decoded_frames": decoded},
        "claim_limit": "Bounded visual evidence only. The candidate is not an admitted M_remove/M_write mask and was not sent to ProPainter.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    _write(OUT/"RESULT.json",result)
    print(json.dumps({"status":result["quality"],"top1_unique":result["top1_unique_frames"],
                      "propainter":result["propainter_admission"],"output":str(OUT)},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
