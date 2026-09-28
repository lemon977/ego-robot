"""Bind the frozen 007 cable proxy to actual Scene masks for one diagnostic window."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.pipeline.attachment_tracker_v1 import (
    admit_bidirectional_candidate, mask_iou, propagate_one_frame,
)
from chaoyang.pipeline.v5_scene import FrameRoles, build_object_protected_repair_window

ROOT = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_product_first_cleanup_20260923"
OUT = ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/cable_007/window_v1"
OLD_CABLE = ROOT / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001/lanes/scene/cable_007/wave0/candidate_masks"
RAW = ROOT / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/get_potato_chips_0915_007/raw"
ROLES = ROOT / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_007_v1"
OLD_PREP = ROOT / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001/lanes/lane1_scene/clean_candidate_007_wave4/prep"
FRAMES = tuple(range(181, 197))
ENDPOINT_MAX_DISTANCE_PX = 8.0  # existing conservative Scene support margin
TEMPORAL_MIN_IOU = 0.45  # existing bidirectional attachment gate


def _read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def classify_support(
    human: np.ndarray, device: np.ndarray, cable: np.ndarray,
    write: np.ndarray, model: np.ndarray,
) -> dict[str, int | str]:
    """Keep role existence distinct from complaint coverage and model support."""
    arrays = tuple(np.asarray(value, dtype=bool) for value in (human, device, cable, write, model))
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("MASK_DOMAIN_MISMATCH")
    h, d, c, w, m = arrays
    role_px = int((h | d).sum())
    direct_px = int(((h | d) & c).sum())
    write_px = int((w & c).sum())
    model_px = int((m & c).sum())
    reason = (
        "EMPTY_HUMAN_DEVICE_ROLES" if role_px == 0 else
        "ROLES_PRESENT_NO_COMPLAINT_COVERAGE" if direct_px == 0 else
        "ROLE_COVERS_COMPLAINT_BUT_WRITE_OR_MODEL_MISSING" if write_px < int(c.sum()) or model_px < int(c.sum()) else
        "ROLE_AND_PREPARED_MODEL_SUPPORT_PRESENT"
    )
    return {
        "reason": reason, "role_pixels": role_px,
        "role_on_complaint_pixels": direct_px, "write_on_complaint_pixels": write_px,
        "prepared_model_on_complaint_pixels": model_px,
        "complaint_proxy_pixels": int(c.sum()),
    }


def _write_json(path: Path, value: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _png(path: Path, value: np.ndarray) -> None:
    if not cv2.imwrite(str(path), value):
        raise RuntimeError(f"IMAGE_WRITE:{path}")


def _endpoint_distance(mask: np.ndarray, human: np.ndarray) -> tuple[list[int], float]:
    pixels = np.argwhere(mask)
    if len(pixels) == 0 or not human.any():
        return [-1, -1], float("inf")
    top_y = int(pixels[:, 0].min())
    tip = pixels[pixels[:, 0] <= top_y + 5]
    top_x = int(np.median(tip[:, 1]))
    distances = cv2.distanceTransform((~human).astype(np.uint8), cv2.DIST_L2, 5)
    return [top_x, top_y], float(distances[top_y, top_x])


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    for folder in ("frames", "model_masks", "write", "protect", "unknown", "candidate"):
        (OUT / folder).mkdir()

    images: list[np.ndarray] = []
    candidates: list[np.ndarray] = []
    humans: list[np.ndarray] = []
    devices: list[np.ndarray] = []
    objects: list[np.ndarray] = []
    old_write: list[np.ndarray] = []
    old_model: list[np.ndarray] = []
    for frame in FRAMES:
        raw = _read(RAW / f"{frame:06d}.png")
        if raw.shape[:2] != (960, 1280):
            raise ValueError(f"RAW_DOMAIN:{frame}")
        images.append(raw)
        candidates.append(_read(OLD_CABLE / f"{frame:06d}.png") > 0)
        humans.append(_read(ROLES / "human_forearm" / f"{frame:06d}.png") > 0)
        devices.append(_read(ROLES / "capture_device" / f"{frame:06d}.png") > 0)
        objects.append(_read(ROLES / "task_object" / f"{frame:06d}.png") > 0)
        old_write.append(_read(OLD_PREP / "write" / f"{frame:06d}.png") > 0)
        prior_model = _read(OLD_PREP / "model_masks" / f"{frame:06d}.png") > 0
        old_model.append(cv2.resize(prior_model.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST) > 0)

    endpoints = [_endpoint_distance(c, h) for c, h in zip(candidates, humans, strict=True)]
    direct = [distance <= ENDPOINT_MAX_DISTANCE_PX for _, distance in endpoints]
    bound_masks = [value.copy() if allowed else np.zeros_like(value) for value, allowed in zip(candidates, direct, strict=True)]
    temporal_sources = ["DIRECT_ENDPOINT_PROXY" if allowed else "UNKNOWN" for allowed in direct]
    for i in range(1, len(FRAMES) - 1):
        if direct[i] or not (direct[i - 1] and direct[i + 1]):
            continue
        forward = propagate_one_frame(images[i - 1], images[i], candidates[i - 1])
        reverse = propagate_one_frame(images[i + 1], images[i], candidates[i + 1])
        admitted, _ = admit_bidirectional_candidate(
            forward, reverse,
            left_seed_area=int(candidates[i - 1].sum()), right_seed_area=int(candidates[i + 1].sum()),
        )
        if admitted and mask_iou(forward & reverse, candidates[i]) >= TEMPORAL_MIN_IOU:
            bound_masks[i] = forward & reverse & candidates[i]
            temporal_sources[i] = "BIDIRECTIONAL_ONE_FRAME_PROXY"

    roles = [FrameRoles(h, d | c, o) for h, d, c, o in zip(humans, devices, bound_masks, objects, strict=True)]
    rows = []
    for i, frame in enumerate(FRAMES):
        start, stop = max(0, i - 2), min(len(FRAMES), i + 3)
        masks = build_object_protected_repair_window(roles[start:stop], i - start, frame, support_margin=4)
        write = np.asarray(masks["write"], dtype=bool)
        protect = np.asarray(masks["protect"], dtype=bool)
        context = np.asarray(masks["context_exclude"], dtype=bool)
        small = cv2.resize(cv2.dilate(context.astype(np.uint8), np.ones((3, 3), np.uint8)),
                           (960, 720), interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate(small, np.ones((3, 3), np.uint8)) > 0
        roundtrip = cv2.resize(small.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST) > 0
        if np.any(context & ~roundtrip):
            raise ValueError(f"CONTEXT_LOST_AT_MODEL_SCALE:{frame}")
        _png(OUT / "frames" / f"{i:06d}.png", cv2.resize(images[i], (960, 720), interpolation=cv2.INTER_AREA))
        _png(OUT / "model_masks" / f"{i:06d}.png", small.astype(np.uint8) * 255)
        _png(OUT / "write" / f"{i:06d}.png", write.astype(np.uint8) * 255)
        _png(OUT / "protect" / f"{i:06d}.png", protect.astype(np.uint8) * 255)
        _png(OUT / "unknown" / f"{i:06d}.png", np.asarray(masks["unknown"], np.uint8) * 255)
        _png(OUT / "candidate" / f"{i:06d}.png", bound_masks[i].astype(np.uint8) * 255)
        old = classify_support(humans[i], devices[i], candidates[i], old_write[i], old_model[i])
        new = classify_support(humans[i], devices[i] | bound_masks[i], bound_masks[i], write, roundtrip)
        candidate_missing = int((bound_masks[i] & ~write).sum())
        protected_conflict = int((bound_masks[i] & protect).sum())
        rows.append({
            "source_frame_id": frame, "window_frame_id": i,
            "upper_endpoint_xy": endpoints[i][0],
            "endpoint_to_human_px": None if not np.isfinite(endpoints[i][1]) else endpoints[i][1],
            "temporal_source": temporal_sources[i], "old": old, "new_prepared": new,
            "new_mask_pixels": int(bound_masks[i].sum()),
            "candidate_outside_write_px": candidate_missing,
            "candidate_protection_conflict_px": protected_conflict,
        })
    structurally_ready = all(row["new_mask_pixels"] > 0 and
                             row["candidate_outside_write_px"] == 0 and
                             row["candidate_protection_conflict_px"] == 0 for row in rows)
    manifest = {
        "schema_version": "HUMAN_TO_ROBOT_007_PRODUCT_FIRST_CABLE_WINDOW_V1",
        "task_id": TASK, "session_id": "get_potato_chips_0915_007",
        "source_frames": list(FRAMES), "frame_count": len(FRAMES),
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_ENCODED_RESIZE_ONLY",
        "quality": "DIAGNOSTIC_ASSOCIATION_PROXY_NOT_GROUND_TRUTH",
        "old_roles_empty_claim": "FALSE_HUMAN_FOREARM_16_OF_16_DEVICE_7_OF_16",
        "old_model_actual_invocation": "HISTORICAL_R2_ONLY",
        "new_model_actual_invocation": "NOT_RUN_PREP_ONLY",
        "inpaint_candidate_structurally_ready": structurally_ready,
        "rows": rows,
        "source": {
            "old_candidate": artifact_ref(OLD_CABLE.parent / "RESULT.json"),
            "role_manifest": artifact_ref(ROLES / "MASK_MANIFEST.json"),
            "old_prep": artifact_ref(OLD_PREP.parent / "SCENE_PREP_MANIFEST.json"),
        },
        "claim_limit": "Frozen-window cable proxy and actual prepared model masks only; no independent cable GT, new ProPainter call, Clean quality or product adoption.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    _write_json(OUT / "RESULT.json", manifest)
    print(json.dumps({
        "status": manifest["quality"], "old_roles_empty_claim": False,
        "direct_frames": sum(direct), "bridged_frames": temporal_sources.count("BIDIRECTIONAL_ONE_FRAME_PROXY"),
        "ready": structurally_ready, "output": str(OUT / "RESULT.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
