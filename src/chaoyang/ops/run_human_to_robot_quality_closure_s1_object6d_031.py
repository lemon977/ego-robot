#!/usr/bin/env python3
"""Observed-only three-card Object6D fields for S1 session 031."""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref
from chaoyang.pipeline.object6d_planar_observability_v2 import (
    COMPONENT_NAMES,
    DEPTH_REFERENCE,
    DIRECT_VISIBILITY,
    MASK_REGISTRATION_AUTHORITY,
    OBJECT_INSTANCE_IDS,
    PlanarFrameInput,
    build_observability_document,
    estimate_planar_frame,
)

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_quality_closure_s1_20260923"
ATTEMPT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
DEPTH = ATTEMPT / "lanes/geometry_contact/depth_full_v1/play_cards_0915_031"
MASKS = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_031_v1"
RAW = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/play_cards_0915_031/raw"
OUTPUT = ATTEMPT / "lanes/geometry_contact/object6d_visible_031_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded: raise RuntimeError(f"IMMUTABLE_CONFLICT:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8"); os.replace(temporary, path)


def mapping() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((480, 640), dtype=np.float64)
    value = np.stack((2.0 * xx + 0.5, 2.0 * yy + 0.5), axis=-1)
    valid = (value[..., 0] >= 0) & (value[..., 0] < 1280) & (value[..., 1] >= 0) & (value[..., 1] < 960)
    return value, valid


def render_frame(raw: np.ndarray, labels: np.ndarray, depth: np.ndarray, valid: np.ndarray, records: list[dict], frame_id: int) -> np.ndarray:
    left = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
    colours = [(30, 80, 255), (30, 220, 80), (255, 100, 30)]
    small_labels = cv2.resize(labels.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    for label, colour in enumerate(colours, 1):
        mask = small_labels == label
        left[mask] = (0.45 * left[mask] + 0.55 * np.asarray(colour)).astype(np.uint8)
    scalar = np.zeros(depth.shape, np.uint8)
    scalar[valid] = np.asarray(255 * (2.0 - np.clip(depth[valid], 0.2, 2.0)) / 1.8, np.uint8)
    right = cv2.applyColorMap(scalar, cv2.COLORMAP_TURBO); right[~valid] = 0
    panel = np.concatenate((left, right), axis=1)
    cv2.rectangle(panel, (0, 0), (1280, 90), (0, 0, 0), -1)
    cv2.putText(panel, f"frame {frame_id:03d} | cards 00/01/02 observed surfaces | encoded optical-Z", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .60, (255,255,255), 2, cv2.LINE_AA)
    for index, record in enumerate(records):
        flags = "/".join("Y" if record[name]["observability"] != "UNOBSERVABLE" else "-" for name in ("center_xyz", "plane_normal", "inplane_rotation"))
        cv2.putText(panel, f"card{index}: center/plane/axis {flags}", (12 + index * 315, 58), cv2.FONT_HERSHEY_SIMPLEX, .47, colours[index], 1, cv2.LINE_AA)
    cv2.putText(panel, "full extent/thickness/back side: UNKNOWN | external metric authority=false", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, .44, (220,220,220), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    if OUTPUT.exists(): raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{OUTPUT}")
    depth_result = json.loads((DEPTH / "RESULT.json").read_text(encoding="utf-8"))
    if depth_result.get("status") != "PASS_OBJECT6D_SUCCESSOR_AUTHORIZED" or depth_result.get("execution_scope") != "FULL_SESSION":
        raise RuntimeError("FULL_DEPTH_NOT_ADMITTED")
    frame_paths = sorted((DEPTH / "frames").glob("*.npz"))
    if len(frame_paths) != 149: raise RuntimeError(f"DEPTH_DENOMINATOR:{len(frame_paths)}")
    OUTPUT.mkdir(parents=True)
    depth_to_mask, registration_valid = mapping()
    object_frames = {name: [] for name in OBJECT_INSTANCE_IDS}
    writer = cv2.VideoWriter(str(OUTPUT / "OBJECT6D_VISIBLE_REVIEW.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 480))
    if not writer.isOpened(): raise RuntimeError("VIDEO_WRITER")
    try:
        for frame_id, depth_path in enumerate(frame_paths):
            with np.load(depth_path, allow_pickle=False) as archive:
                if int(archive["frame_id"]) != frame_id: raise RuntimeError("DEPTH_FRAME_ID")
                depth = np.asarray(archive["depth_m"], np.float64)
                valid = np.asarray(archive["valid"], bool)
                intrinsics = np.asarray(archive["physical_left_intrinsics"], np.float64)
                if str(archive["depth_reference"]) != DEPTH_REFERENCE: raise RuntimeError("DEPTH_REFERENCE")
            labels = cv2.imread(str(MASKS / "task_object" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
            raw = cv2.imread(str(RAW / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
            if labels is None or raw is None or labels.shape != (960, 1280): raise RuntimeError("MASK_OR_RAW_DOMAIN")
            records = []
            for label, instance_id in enumerate(OBJECT_INSTANCE_IDS, 1):
                mask = labels == label
                state = "tracked" if mask.any() else "unknown"
                record = estimate_planar_frame(PlanarFrameInput(
                    frame_index=frame_id, mask=mask, mask_state=state,
                    visibility_state=DIRECT_VISIBILITY if mask.any() else "UNKNOWN",
                    depth_m=depth, depth_valid=valid, depth_intrinsics=intrinsics,
                    depth_to_mask_xy=depth_to_mask, registration_valid=registration_valid,
                    depth_reference=DEPTH_REFERENCE,
                    registration_authority=MASK_REGISTRATION_AUTHORITY,
                ))
                object_frames[instance_id].append(record); records.append(record)
            writer.write(render_frame(raw, labels, depth, valid, records, frame_id))
    finally:
        writer.release()
    inputs = {
        "depth": artifact_ref(DEPTH / "RESULT.json"),
        "task_object_manifest": artifact_ref(MASKS / "MASK_MANIFEST.json"),
        "depth_to_mask_mapping": {
            "formula": "x_mask=2*x_depth+0.5; y_mask=2*y_depth+0.5",
            "authority": MASK_REGISTRATION_AUTHORITY,
        },
        "mask_identity_authority": "SAM_DIRECTION_LOCAL_NOT_GOLD",
    }
    document = build_observability_document(session_id="play_cards_0915_031", object_frames=object_frames, inputs=inputs)
    write_once(OUTPUT / "OBJECT6D_OBSERVABILITY.json", document)
    summary_objects = []
    for obj in document["objects"]:
        rows = obj["frames"]
        summary_objects.append({
            "instance_id": obj["instance_id"],
            "observable": {name: sum(r[name]["observability"] != "UNOBSERVABLE" for r in rows) for name in COMPONENT_NAMES},
            "mask_states": dict(Counter(r["mask_state"] for r in rows)),
            "plane_reject_reasons": dict(Counter(r["plane_normal"]["reason"] for r in rows if r["plane_normal"]["observability"] == "UNOBSERVABLE")),
        })
    video = OUTPUT / "OBJECT6D_VISIBLE_REVIEW.mp4"
    capture = cv2.VideoCapture(str(video)); decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok: break
        decoded += 1
    capture.release()
    if decoded != 149: raise RuntimeError(f"VIDEO_DECODE:{decoded}")
    result = {
        "schema_version": "S1_OBJECT6D_VISIBLE_PLANAR_031_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "status": "COMPLETED_DEVELOPMENT_OBSERVABILITY",
        "frame_count": 149, "objects": summary_objects,
        "coordinate_domain": DEPTH_REFERENCE,
        "full_object_extent": "UNVERIFIED", "hidden_geometry_inferred": False,
        "external_metric_authority": False, "contact_authority": False,
        "document": artifact_ref(OUTPUT / "OBJECT6D_OBSERVABILITY.json"),
        "review_video": {**artifact_ref(video), "decoded_frames": decoded},
        "training_eligible": False,
    }
    write_once(OUTPUT / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "objects": summary_objects}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
