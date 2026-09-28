"""Audit full-timeline MANO/Stereo surface Z with explicit visible-card UNKNOWN."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_031_surface_depth_diagnostic import (
    SOURCE, DOMAIN, MASK_ROOT, DEPTH_ROOT, DEPTH_RESULT,
    load_faces, rasterize_optical_z, admissible_pixels,
)

TASK = "human_to_robot_031_surface_depth_temporal_audit_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/motion/surface_depth_temporal_v1"
OBJECT_ROOT = REPO_ROOT / ("_run/current/0915_robot15h_sam31_task_object_wave0_recovery_v1/"
                          "attempts/attempt_0001/sessions/playing_cards/play_cards_0915_031")
OBJECT_RESULT = OBJECT_ROOT / "RESULT.json"
OBJECT_MANIFEST = OBJECT_ROOT / "OBJECT_INSTANCE_MANIFEST.json"
CARD_IDS = ("playing_card_00", "playing_card_01", "playing_card_02")


def load_visible_cards(frame: int, packed: list[np.ndarray], states: list[list[dict]]) -> tuple[np.ndarray, bool, list[bool]]:
    admitted = [bool(series[frame]["semantic_admitted"]) for series in states]
    masks = []
    for archive, is_admitted in zip(packed, admitted):
        if is_admitted:
            mask = np.unpackbits(archive[frame], bitorder="big").reshape(960, 1280).astype(np.uint8)
            masks.append(mask)
    union = np.maximum.reduce(masks) if masks else np.zeros((960, 1280), np.uint8)
    # 7 source pixels protect the visible card edge and small registration error.
    guard = cv2.dilate(union, np.ones((15, 15), np.uint8))
    return cv2.resize(guard, (640, 480), interpolation=cv2.INTER_NEAREST) > 0, all(admitted), admitted


def summary(values: np.ndarray) -> dict:
    return {"count": int(values.size),
            "median_m": float(np.median(values)) if values.size else None,
            "p10_m": float(np.percentile(values, 10)) if values.size else None,
            "p90_m": float(np.percentile(values, 90)) if values.size else None}


def draw_timeline(rows: list[dict]) -> np.ndarray:
    canvas = np.full((150, 1280, 3), 250, np.uint8)
    cv2.line(canvas, (50, 75), (1240, 75), (150, 150, 150), 1)
    cv2.putText(canvas, "Stereo Z - MANO visible surface Z [m] | blue=valid all cards | gray=object UNKNOWN",
                (28, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (30, 30, 30), 1)
    cv2.putText(canvas, "-0.20", (2, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)
    cv2.putText(canvas, "+0.20", (2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1)
    for row in rows:
        value = row["strict_object_guard"]["median_m"]
        strict = True
        if value is None:
            value = row["known_positive_guard"]["median_m"]
            strict = False
        if value is None:
            continue
        x = int(50 + row["source_frame_id"] * 1190 / 148)
        y = int(np.clip(75 - value * 300, 29, 141))
        cv2.circle(canvas, (x, y), 2, (185, 75, 25) if strict else (130, 130, 130), -1)
    return canvas


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    domain = load_json(DOMAIN)
    depth_receipt = load_json(DEPTH_RESULT)
    object_result = load_json(OBJECT_RESULT)
    object_manifest = load_json(OBJECT_MANIFEST)
    if (domain.get("frame_count") != 149 or domain.get("source_index") != 1
            or depth_receipt.get("local_stereo_metric_dev") is not True
            or object_result.get("frame_count") != 149
            or object_manifest.get("frame_count") != 149
            or object_manifest.get("image_domain") != "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
            or tuple(row["instance_id"] for row in object_result["instances"]) != CARD_IDS):
        raise RuntimeError("INPUT_DOMAIN_OR_CARD_INSTANCE_DRIFT")
    with np.load(SOURCE, allow_pickle=False) as archive:
        if (archive["vertices_3d_camera"].shape != (2, 149, 778, 3)
                or list(archive["anatomical_side_names"]) != ["left", "right"]
                or str(archive["source_domain"]) != "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY"):
            raise RuntimeError("MODEL_SOURCE_DRIFT")
        vertices = np.asarray(archive["vertices_3d_camera"][1], np.float32)
        predicted = np.asarray(archive["predicted_valid"][1], bool)
        timestamp_ns = np.asarray(archive["timestamp_ns"], np.int64)
        source_k = np.asarray(archive["intrinsics"], np.float64)
    packed, states = [], []
    for instance in object_result["instances"]:
        if instance.get("object_mask_consumer_allowed") is not True or instance.get("status") != "PASS_VISIBLE_INSTANCE_PROXY":
            raise RuntimeError("CARD_MASK_CONSUMER_NOT_ALLOWED")
        mask_path = Path(instance["semantic_archive"]["path"])
        state_path = Path(instance["state_ledger"]["path"])
        if artifact_ref(mask_path) != instance["semantic_archive"] or artifact_ref(state_path) != instance["state_ledger"]:
            raise RuntimeError("CARD_INPUT_SHA_DRIFT")
        with np.load(mask_path, allow_pickle=False) as archive:
            if (archive["packed"].shape != (149, 153600) or int(archive["height"]) != 960
                    or int(archive["width"]) != 1280 or str(archive["bitorder"]) != "big"):
                raise RuntimeError("CARD_MASK_DOMAIN_DRIFT")
            packed.append(np.asarray(archive["packed"], np.uint8))
        series = load_json(state_path)["frames"]
        if [row["frame_id"] for row in series] != list(range(149)):
            raise RuntimeError("CARD_TEMPORAL_LEDGER_DRIFT")
        states.append(series)
    faces = load_faces()
    DEST.mkdir(parents=True)
    video = DEST / "031_SURFACE_Z_OBJECT_GUARD_FULL_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 630))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_NOT_OPEN")
    rows = []
    try:
        for frame in range(149):
            raw_path = Path(domain["frames"][frame]["rgb"])
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw is None or raw.shape != (960, 1280, 3):
                raise RuntimeError(f"RAW_DOMAIN_DRIFT:{frame}")
            left = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            right = left.copy()
            guard, all_cards_admitted, card_admitted = load_visible_cards(frame, packed, states)
            before, known_guard, strict = summary(np.array([], np.float32)), summary(np.array([], np.float32)), summary(np.array([], np.float32))
            status = "NO_ORIGINAL_RIGHT_MODEL_INPUT" if not predicted[frame] else "MODEL_INPUT_PRESENT"
            if predicted[frame]:
                mask_path = MASK_ROOT / f"{frame:06d}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None or mask.shape != (960, 1280) or mask.dtype != np.uint16 or mask.max() > 1:
                    raise RuntimeError(f"HAND_MASK_DOMAIN_DRIFT:{frame}")
                depth_path = DEPTH_ROOT / "frames" / f"{frame:06d}.npz"
                with np.load(depth_path, allow_pickle=False) as archive:
                    stereo = {key: np.asarray(archive[key]) for key in ("depth_m", "valid", "lr_consistent")}
                    k_depth = np.asarray(archive["physical_left_intrinsics"], np.float64)
                if (not np.isfinite(vertices[frame]).all()
                        or not np.allclose(k_depth[:2, :2], source_k[frame, :2, :2] / 2, atol=1e-3)):
                    raise RuntimeError(f"MODEL_OR_INTRINSICS_DRIFT:{frame}")
                model_z = rasterize_optical_z(vertices[frame], faces, k_depth)
                admitted = admissible_pixels(model_z, stereo, mask == 1)
                difference = stereo["depth_m"] - model_z
                before = summary(difference[admitted])
                known = admitted & ~guard
                known_guard = summary(difference[known])
                strict = known_guard if all_cards_admitted else summary(np.array([], np.float32))
                status = "STRICT_CARD_GUARD" if all_cards_admitted else "UNKNOWN_CARD_INSTANCE"
                visual = np.zeros((480, 640, 3), np.uint8)
                visual[known & (difference < -0.03)] = (200, 80, 40)
                visual[known & (np.abs(difference) <= 0.03)] = (40, 180, 40)
                visual[known & (difference > 0.03)] = (40, 80, 200)
                visual[admitted & guard] = (0, 215, 255)
                right[admitted] = cv2.addWeighted(right, 0.4, visual, 0.6, 0)[admitted]
            else:
                mask_path, depth_path = None, None
            row = {"source_frame_id": frame, "timestamp_ns": int(timestamp_ns[frame]),
                   "original_right_model_valid": bool(predicted[frame]),
                   "card_instance_admitted": card_admitted, "all_cards_admitted": all_cards_admitted,
                   "status": status, "before_object_guard": before,
                   "known_positive_guard": known_guard, "strict_object_guard": strict,
                   "raw": artifact_ref(raw_path),
                   "hand_mask": artifact_ref(mask_path) if mask_path else None,
                   "stereo": artifact_ref(depth_path) if depth_path else None}
            rows.append(row)
            cv2.putText(left, f"031 frame {frame:03d} RAW | original right valid={int(predicted[frame])}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
            cv2.putText(right, f"Stereo-MANO Z | n={strict['count']} | {status}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2)
            timeline = draw_timeline(rows)
            writer.write(np.vstack((np.hstack((left, right)), timeline)))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        if image.shape != (630, 1280, 3):
            raise RuntimeError("VIDEO_FRAME_DOMAIN_INVALID")
        decoded += 1
    capture.release()
    if decoded != 149:
        raise RuntimeError(f"VIDEO_DECODE_COUNT_INVALID:{decoded}")
    strict_values = [r["strict_object_guard"]["median_m"] for r in rows if r["strict_object_guard"]["median_m"] is not None]
    before_values = [r["before_object_guard"]["median_m"] for r in rows if r["strict_object_guard"]["median_m"] is not None]
    result = {"schema_version": "HUMAN_TO_ROBOT_031_SURFACE_DEPTH_TEMPORAL_AUDIT_V1",
              "task_id": TASK, "session_id": "play_cards_0915_031",
              "created_at": datetime.now().astimezone().isoformat(), "frame_count": 149,
              "original_right_valid_frames": int(predicted.sum()),
              "strict_object_guard_frames_with_samples": len(strict_values),
              "strict_object_guard_samples": sum(r["strict_object_guard"]["count"] for r in rows),
              "strict_frame_median_of_medians_m": float(np.median(strict_values)) if strict_values else None,
              "before_guard_same_frames_median_of_medians_m": float(np.median(before_values)) if before_values else None,
              "rows": rows,
              "inputs": {"model": artifact_ref(SOURCE), "domain": artifact_ref(DOMAIN),
                         "stereo": artifact_ref(DEPTH_RESULT), "object": artifact_ref(OBJECT_RESULT),
                         "object_manifest": artifact_ref(OBJECT_MANIFEST)},
              "review_video": {**artifact_ref(video), "decoded_frames": decoded},
              "execution": "EXECUTED", "structure": "PASS", "quality": "DEVELOPMENT_DIAGNOSTIC_ONLY",
              "adoption": "NOT_ADOPTED", "training_eligible": False,
              "control_ground_truth": False, "physical_deployable": False,
              "external_metric_authority": False,
              "claim_limit": "Same-pixel model/Stereo surface Z with visible-card mask exclusion; UNKNOWN cards, SAM-as-ROI, possible other occluders and true correspondence remain unresolved. No wrist correction, Contact or product authority."}
    with (DEST / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"task_id": TASK, "strict_frames": len(strict_values),
                      "strict_samples": result["strict_object_guard_samples"],
                      "strict_median_m": result["strict_frame_median_of_medians_m"],
                      "before_same_frames_median_m": result["before_guard_same_frames_median_of_medians_m"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
