#!/usr/bin/env python3
"""S2 H3 fixed 031 A/B/C adapter and visible-surface occlusion evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.build_visible_surface_occlusion_canary_v71 import quality_evidence
from chaoyang.pipeline.occlusion_compositor_v1 import (
    ObjectPixelSource,
    Ownership,
    choose_object_pixels,
    exclude_removed_foreground_depth,
    resolve_ownership,
)
from chaoyang.pipeline.v5_product import ProductRobotRenderer, composite


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
OUT = REPO / f"_run/current/{TASK}/attempts/attempt_0001/lanes/scene/h3_occlusion_031/attempt_0003"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"
S1 = REPO / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001"
V3 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"
START, COUNT = 66, 16


def ref(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def read_mask(root: Path, role: str, frame: int, shape: tuple[int, int]) -> np.ndarray:
    path = root / role / f"{frame:06d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != shape:
        raise ValueError(f"MASK_DECODE:{role}:{frame}")
    return image > 0


def label(image: np.ndarray, text: str, sub: str) -> np.ndarray:
    value = cv2.resize(image, (640, 480))
    cv2.rectangle(value, (0, 0), (640, 54), (18, 18, 18), -1)
    cv2.putText(value, text, (9, 22), cv2.FONT_HERSHEY_SIMPLEX, .56,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(value, sub, (9, 44), cv2.FONT_HERSHEY_SIMPLEX, .39,
                (0, 215, 255), 1, cv2.LINE_AA)
    return value


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{OUT}")
    OUT.mkdir(parents=True)
    motion_path = R2 / "lanes/lane2_motion/recovered_031_wave0/ROBOT_R0_V1.npz"
    motion = load_npz(motion_path)
    domain_path = V3 / "lanes/exact78/prepare_full_v1/play_cards_0915_031/DOMAIN_MANIFEST.json"
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    clean_root = S1 / "lanes/scene_evidence/attachment_clean_canary_v1/play_cards_0915_031"
    clean_result_path = clean_root / "RESULT.json"
    clean_result = json.loads(clean_result_path.read_text(encoding="utf-8"))
    if [int(row["source_frame_id"]) for row in clean_result["rows"]] != list(range(START, START + COUNT)):
        raise ValueError("CLEAN_FIXED_WINDOW_DRIFT")
    depth_root = S1 / "lanes/geometry_contact/depth_full_v1/play_cards_0915_031"
    depth_result_path = depth_root / "RESULT.json"
    depth_result = json.loads(depth_result_path.read_text(encoding="utf-8"))
    object_result_path = S1 / "lanes/geometry_contact/object6d_visible_031_v1/RESULT.json"
    object_result = json.loads(object_result_path.read_text(encoding="utf-8"))
    masks_root = V5 / "lanes/scene/masks_031_v1"
    mask_manifest_path = masks_root / "MASK_MANIFEST.json"
    height, width = int(domain["height"]), int(domain["width"])
    shape = (height, width)
    if shape != (960, 1280):
        raise ValueError("UNEXPECTED_PRODUCT_DOMAIN")
    renderer_old = ProductRobotRenderer(REPO, motion, domain, include_adapter=False)
    renderer_new = ProductRobotRenderer(REPO, motion, domain, include_adapter=True)
    video_path = OUT / "H3_031_ADAPTER_OCCLUSION_ABC_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0,
                             (1280, 960))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER_OPEN_FAILED")
    ownership_rows: list[np.ndarray] = []
    counts: Counter[str] = Counter()
    rows = []
    previous_ownership = previous_decision = None
    known_switches = comparable_pairs = 0
    intrinsics_errors = []
    try:
        for local, frame in enumerate(range(START, START + COUNT)):
            raw = cv2.imread(str(domain["frames"][frame]["rgb"]), cv2.IMREAD_COLOR)
            clean = cv2.imread(str(clean_root / "clean" / f"{local:06d}.png"), cv2.IMREAD_COLOR)
            if raw is None or clean is None or raw.shape[:2] != shape or clean.shape[:2] != shape:
                raise ValueError(f"RGB_DECODE:{frame}")
            old = renderer_old.frame_layers(frame)
            new = renderer_new.frame_layers(frame)
            product_a = composite(clean, old.rgb, old.alpha)
            product_b = composite(clean, new.rgb, new.alpha)
            object_mask = read_mask(masks_root, "task_object", frame, shape)
            removed = np.zeros(shape, dtype=bool)
            for role in ("human_hand", "human_forearm", "capture_device"):
                removed |= read_mask(masks_root, role, frame, shape)
            depth_path = depth_root / "frames" / f"{frame:06d}.npz"
            depth = load_npz(depth_path)
            low_shape = depth["depth_m"].shape
            if low_shape != (480, 640):
                raise ValueError("UNEXPECTED_DEPTH_DOMAIN")
            removed_low = cv2.resize(removed.astype(np.uint8), (640, 480),
                                     interpolation=cv2.INTER_NEAREST) > 0
            removed_low = cv2.dilate(removed_low.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            scene_low, valid_low = exclude_removed_foreground_depth(
                scene_depth_m=depth["depth_m"], scene_depth_valid=depth["valid"].astype(bool),
                human_equipment_mask=removed_low,
            )
            scene_depth = cv2.resize(scene_low.astype(np.float32), (width, height),
                                     interpolation=cv2.INTER_NEAREST)
            scene_valid = cv2.resize(valid_low.astype(np.uint8), (width, height),
                                     interpolation=cv2.INTER_NEAREST) > 0
            disparity = cv2.resize(depth["disparity_physical_left_px"].astype(np.float32),
                                   (width, height), interpolation=cv2.INTER_NEAREST) * 2.0
            k_low = np.asarray(depth["physical_left_intrinsics"], dtype=np.float64)
            k_high = k_low.copy()
            k_high[0, 0] *= 2.0; k_high[1, 1] *= 2.0
            k_high[0, 2] = 2.0 * k_low[0, 2] + 0.5
            k_high[1, 2] = 2.0 * k_low[1, 2] + 0.5
            intrinsics_errors.append(float(np.max(np.abs(k_high - renderer_new.k))))
            evidence = quality_evidence(scene_depth, disparity, scene_valid, object_mask, raw)
            object_rgb, object_source = choose_object_pixels(
                raw_rgb=raw, raw_visible_mask=object_mask,
            )
            decision = ((cv2.dilate(object_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) &
                        (cv2.dilate(new.alpha.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
            qualified = (
                depth_result.get("local_stereo_metric_dev") is True
                and object_result.get("coordinate_domain") == "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
                and intrinsics_errors[-1] <= 0.51
            )
            if qualified:
                result = resolve_ownership(
                    human_mask=np.zeros(shape, dtype=bool),
                    object_amodal_mask=object_mask,
                    object_depth_m=scene_depth,
                    object_depth_valid=scene_valid & object_mask,
                    robot_alpha_mask=new.alpha,
                    robot_depth_m=new.optical_depth_m,
                    robot_depth_valid=new.depth_valid,
                    stereo_depth_valid=scene_valid,
                    depth_quality_evidence=evidence,
                    object_rgb=object_rgb,
                    object_pixel_source=object_source,
                    contact_decision_mask=decision,
                )
                ownership = result.ownership
            else:
                ownership = np.full(shape, int(Ownership.BACKGROUND), dtype=np.uint8)
                ownership[decision] = int(Ownership.TIE_UNKNOWN)
            scene_front = decision & (ownership == int(Ownership.OBJECT_FRONT))
            robot_front = decision & (ownership == int(Ownership.ROBOT_FRONT))
            unknown = decision & (ownership == int(Ownership.TIE_UNKNOWN))
            counts["SCENE_FRONT"] += int(scene_front.sum())
            counts["ROBOT_FRONT"] += int(robot_front.sum())
            counts["UNKNOWN"] += int(unknown.sum())
            counts["NOT_APPLICABLE"] += int((~decision).sum())
            composite_c = clean.copy()
            composite_c[scene_front] = object_rgb[scene_front]
            composite_c[robot_front] = new.rgb[robot_front, ::-1]
            checker = (np.indices(shape).sum(axis=0) // 8) % 2 == 0
            composite_c[unknown & checker] = (255, 0, 255)
            composite_c[unknown & ~checker] = (25, 25, 25)
            evidence_view = raw.copy()
            evidence_view[scene_front] = (0, 220, 0)
            evidence_view[robot_front] = (255, 120, 0)
            evidence_view[unknown] = (255, 0, 255)
            if previous_ownership is not None:
                pair = decision & previous_decision
                known = pair & (ownership != int(Ownership.TIE_UNKNOWN)) & (
                    previous_ownership != int(Ownership.TIE_UNKNOWN))
                comparable_pairs += int(known.sum())
                known_switches += int(np.count_nonzero(known & (ownership != previous_ownership)))
            previous_ownership, previous_decision = ownership.copy(), decision.copy()
            ownership_rows.append(ownership.copy())
            rows.append({
                "frame_id": frame, "geometry_qualified": qualified,
                "decision_pixels": int(decision.sum()), "direct_overlap_pixels": int((object_mask & new.alpha).sum()),
                "scene_front": int(scene_front.sum()), "robot_front": int(robot_front.sum()),
                "unknown": int(unknown.sum()), "not_applicable": int((~decision).sum()),
                "removed_depth_pixels_lowres": int(removed_low.sum()),
                "robot_depth_pixels": int(new.depth_valid.sum()),
                "intrinsics_max_abs_error": intrinsics_errors[-1],
            })
            review = np.vstack([
                np.hstack([label(product_a, f"A OLD | {frame}", "same q/camera/mount/background"),
                           label(product_b, f"B ADAPTER | {frame}", "real CAD; collision UNVERIFIED")]),
                np.hstack([label(composite_c, f"C OCCLUSION | {frame}", "purple=UNKNOWN; visible surface only"),
                           label(evidence_view, f"OWNERSHIP | {frame}", "green=scene blue=robot purple=unknown")]),
            ])
            writer.write(review)
    finally:
        renderer_old.close(); renderer_new.close(); writer.release()
    cap = cv2.VideoCapture(str(video_path)); decoded = 0
    while cap.read()[0]: decoded += 1
    cap.release()
    if decoded != COUNT:
        raise ValueError(f"VIDEO_DECODE:{decoded}/{COUNT}")
    np.savez_compressed(OUT / "OCCLUSION_OWNERSHIP_V1.npz",
                        frame_id=np.arange(START, START + COUNT, dtype=np.int32),
                        ownership=np.asarray(ownership_rows, dtype=np.uint8))
    decision_total = counts["SCENE_FRONT"] + counts["ROBOT_FRONT"] + counts["UNKNOWN"]
    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_H3_OCCLUSION_ABC_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "INCONCLUSIVE", "adoption": "CANDIDATE_ONLY",
        "motion_input_quality": "REJECTED_QUALITY",
        "motion_input_adoption": "NOT_ADOPTED",
        "motion_input_purpose": "FROZEN_CURRENT_R0_OCCLUSION_MECHANISM_DIAGNOSTIC_ONLY",
        "frame_count": COUNT, "decoded_frames": decoded,
        "frozen_variables": ["q", "camera", "mount", "background", "frame_set"],
        "only_change_B": "REAL_ADAPTER_RGB_DEPTH_ID",
        "only_change_C": "VISIBLE_SURFACE_DEPTH_OWNERSHIP_COMPOSITOR",
        "object_geometry_scope": "DIRECT_VISIBLE_TASK_OBJECT_MASK_NOT_AMODAL",
        "occlusion_epsilon_m": 0.003,
        "epsilon_semantics": "ORDERING_TOLERANCE_NOT_ACCURACY_CLAIM",
        "ownership_counts": dict(counts),
        "known_decision_coverage": (counts["SCENE_FRONT"] + counts["ROBOT_FRONT"]) / decision_total if decision_total else 0.0,
        "unknown_decision_ratio": counts["UNKNOWN"] / decision_total if decision_total else 1.0,
        "consecutive_known_ownership_switches": known_switches,
        "consecutive_known_comparable_pixels": comparable_pairs,
        "intrinsics_max_abs_error": max(intrinsics_errors),
        "human_equipment_raw_depth_excluded": True,
        "clean_geometry_consumed": False,
        "rows": rows,
        "inputs": {
            "motion": ref(motion_path), "domain": ref(domain_path),
            "clean": ref(clean_result_path), "depth": ref(depth_result_path),
            "object6d": ref(object_result_path), "mask_manifest": ref(mask_manifest_path),
        },
        "video": ref(video_path), "arrays": ref(OUT / "OCCLUSION_OWNERSHIP_V1.npz"),
        "claim_limit": "Fixed-window visible-surface development ordering only; UNKNOWN is not correct, no hidden geometry, Contact truth, external accuracy, control or deployment authority.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"status": "PASS", "known_coverage": result["known_decision_coverage"],
                      "unknown_ratio": result["unknown_decision_ratio"], "output": str(OUT)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
