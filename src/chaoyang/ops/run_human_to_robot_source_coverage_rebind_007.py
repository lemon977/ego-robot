"""Rebind a traceable prior 007 human mask to the current cable canary, without inpainting."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_source_coverage_diagnostic import (
    TASK, OUT, LEFT_SCREEN_REGION_MODEL, region_support,
)

PRIOR = REPO_ROOT / "_run/current/human_to_robot_product_first_cleanup_20260923/attempts/attempt_0001"
PREP = PRIOR / "lanes/scene/cable_007/window_v1"
SOURCE = REPO_ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/masks_007_recovered_join_v1"
DEST = OUT / "lanes/scene/rebound_007_window_v2"


def merge_support(current_model: np.ndarray, old_human: np.ndarray, write: np.ndarray,
                  protect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if current_model.shape != (720, 960) or any(x.shape != (960, 1280) for x in (old_human, write, protect)):
        raise ValueError("IMAGE_DOMAIN_MISMATCH")
    if np.any(old_human & protect) or np.any(write & protect):
        raise ValueError("TRUSTED_OBJECT_PROTECTION_CONFLICT")
    new_write = write | old_human
    downsampled_write = cv2.resize(new_write.astype(np.uint8), (960, 720), interpolation=cv2.INTER_NEAREST) > 0
    model = current_model | downsampled_write
    # Cover the complete 1280x960 write permission after a lossy resize round-trip.
    model = cv2.dilate(model.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    roundtrip = cv2.resize(model.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST) > 0
    if np.any(new_write & ~roundtrip):
        raise ValueError("WRITE_NOT_PRESENT_IN_MODEL_CONTEXT")
    return model, new_write


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK or not index["task_packets"][0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    source_manifest = load_json(SOURCE / "MASK_MANIFEST.json")
    if source_manifest.get("session_id") != "get_potato_chips_0915_007" or source_manifest.get("frame_count") != 378 or source_manifest.get("new_inference") is not False:
        raise ValueError("SOURCE_MANIFEST_NOT_REUSABLE")
    if source_manifest.get("domain") != "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY":
        raise ValueError("IMAGE_DOMAIN_MISMATCH")
    if DEST.exists():
        raise FileExistsError(DEST)
    for directory in (DEST / "frames", DEST / "model_masks", DEST / "write", DEST / "protect"):
        directory.mkdir(parents=True)
    rows = []
    selected = []
    for local, frame in enumerate(range(181, 197)):
        old_path = SOURCE / "human" / f"{frame:06d}.png"
        current_path = PREP / "model_masks" / f"{local:06d}.png"
        write_path = PREP / "write" / f"{local:06d}.png"
        protect_path = PREP / "protect" / f"{local:06d}.png"
        raw_path = PREP / "frames" / f"{local:06d}.png"
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        old = cv2.imread(str(old_path), cv2.IMREAD_UNCHANGED)
        current = cv2.imread(str(current_path), cv2.IMREAD_UNCHANGED)
        write = cv2.imread(str(write_path), cv2.IMREAD_UNCHANGED)
        protect = cv2.imread(str(protect_path), cv2.IMREAD_UNCHANGED)
        if any(x is None for x in (raw, old, current, write, protect)):
            raise FileNotFoundError(frame)
        model, new_write = merge_support(current > 0, old > 0, write > 0, protect > 0)
        if not cv2.imwrite(str(DEST / "frames" / f"{local:06d}.png"), raw):
            raise RuntimeError("RAW_COPY_FAILED")
        for name, array in (("model_masks", model), ("write", new_write), ("protect", protect > 0)):
            if not cv2.imwrite(str(DEST / name / f"{local:06d}.png"), array.astype(np.uint8) * 255):
                raise RuntimeError(f"MASK_WRITE_FAILED:{name}:{frame}")
        old_small = cv2.resize((old > 0).astype(np.uint8), (960, 720), interpolation=cv2.INTER_NEAREST) > 0
        rows.append({
            "frame_id": frame, "old_human_source": artifact_ref(old_path),
            "new_left_screen_support_pixels": region_support(model, LEFT_SCREEN_REGION_MODEL),
            "reused_human_left_screen_support_pixels": region_support(old_small, LEFT_SCREEN_REGION_MODEL),
            "new_write_pixels": int(new_write.sum()), "protected_conflict_pixels": int(np.count_nonzero(new_write & (protect > 0))),
            "model_covering_write": True,
        })
        if frame in (181, 184, 190, 196):
            overlay = raw.copy()
            overlay[model] = (0.55 * overlay[model] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.putText(overlay, f"{frame} PROVISIONAL REBOUND MODEL MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            selected.append(overlay)
    if any(row["new_left_screen_support_pixels"] == 0 for row in rows):
        raise ValueError("LEFT_REGION_STILL_UNSUPPORTED")
    preview = DEST / "007_REBOUND_MASK_FIXED_FRAMES.png"
    if not cv2.imwrite(str(preview), np.concatenate(selected, axis=0)):
        raise RuntimeError("PREVIEW_WRITE_FAILED")
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_REBOUND_INPUT_V1", "task_id": TASK,
        "session_id": "get_potato_chips_0915_007", "source_frames": list(range(181, 197)),
        "execution": "EXECUTED", "structure": "PASS", "quality": "PROVISIONAL_INPUT_ONLY",
        "adoption": "NOT_ADOPTED", "model_executed": False,
        "source_manifest": artifact_ref(SOURCE / "MASK_MANIFEST.json"),
        "current_preparation": artifact_ref(PREP / "RESULT.json"),
        "preview": artifact_ref(preview), "rows": rows,
        "claim_limit": "A previously rejected but traceable full-session human mask is reused only as a new 16-frame model-input candidate. No claim that its pixel accuracy or ProPainter output passes quality.",
        "training_eligible": False, "control_ground_truth": False, "physical_deployable": False,
        "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PROVISIONAL_INPUT_READY", "result": str(DEST / "RESULT.json"),
                      "left_region_min_pixels": min(row["new_left_screen_support_pixels"] for row in rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
