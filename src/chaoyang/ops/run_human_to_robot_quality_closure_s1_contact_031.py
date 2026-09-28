#!/usr/bin/env python3
"""Close the 031 Interaction/Contact/R1 branch without inventing observations.

The frozen HaWoR source for this session contains 102 right-hand predictions,
but every one is explicitly ``inferred`` rather than ``observed``.  This
producer verifies the image/intrinsic/timestamp domains, records that funnel,
and renders the inferred projections for diagnosis.  It deliberately does not
sample Stereo at inferred joints or create Contact/R1 authority.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_quality_closure_s1_20260923"
ATTEMPT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
HAWOR = REPO / (
    "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/"
    "ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
)
RAW = REPO / (
    "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/"
    "exact78/prepare_full_v1/play_cards_0915_031/raw"
)
MASKS = REPO / (
    "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/"
    "scene/masks_031_v1"
)
DEPTH = ATTEMPT / "lanes/geometry_contact/depth_full_v1/play_cards_0915_031"
OBJECT6D = ATTEMPT / "lanes/geometry_contact/object6d_visible_031_v1"
OUTPUT = ATTEMPT / "lanes/geometry_contact/interaction_contact_031_v1"
TIPS = (4, 8, 12, 16, 20)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_once(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"IMMUTABLE_CONFLICT:{path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{OUTPUT}")
    depth_result = json.loads((DEPTH / "RESULT.json").read_text(encoding="utf-8"))
    object_result = json.loads((OBJECT6D / "RESULT.json").read_text(encoding="utf-8"))
    if depth_result.get("status") != "PASS_OBJECT6D_SUCCESSOR_AUTHORIZED":
        raise RuntimeError("DEPTH_NOT_ADMITTED")
    if object_result.get("status") != "COMPLETED_DEVELOPMENT_OBSERVABILITY":
        raise RuntimeError("OBJECT6D_NOT_AVAILABLE")

    with np.load(HAWOR, allow_pickle=False) as source:
        hawor = {key: np.asarray(source[key]) for key in source.files}
    if hawor["joints_2d"].shape != (2, 149, 21, 2):
        raise RuntimeError("HAWOR_SHAPE_DRIFT")
    if str(hawor["source_domain"]) != "VST_ENCODED_PHYSICAL_LEFT_SOURCEINDEX1_RESIZE_ONLY":
        raise RuntimeError("HAWOR_IMAGE_DOMAIN_DRIFT")
    if not np.array_equal(hawor["original_frame_indices"], np.arange(149)):
        raise RuntimeError("HAWOR_FRAME_MAPPING_DRIFT")
    if np.any(np.diff(hawor["timestamp_ns"]) <= 0):
        raise RuntimeError("HAWOR_TIMESTAMP_DRIFT")

    with np.load(DEPTH / "frames/000000.npz", allow_pickle=False) as first_depth:
        depth_k = np.asarray(first_depth["physical_left_intrinsics"], np.float64)
        depth_domain = str(first_depth["depth_reference"])
    hawor_k = np.asarray(hawor["intrinsics"][0], np.float64)
    focal_delta = np.abs(depth_k.diagonal()[:2] - hawor_k.diagonal()[:2] / 2.0)
    principal_delta = np.abs(depth_k[:2, 2] - hawor_k[:2, 2] / 2.0)
    domain_pass = bool(np.max(focal_delta) < 1e-3 and np.max(principal_delta) < 0.5)

    sides = [str(value) for value in hawor["anatomical_side_names"].tolist()]
    observed_counts = [int(value) for value in hawor["observed"].sum(axis=1)]
    inferred_counts = [int(value) for value in hawor["inferred"].sum(axis=1)]
    predicted_counts = [int(value) for value in hawor["predicted_valid"].sum(axis=1)]
    if observed_counts != [0, 0] or predicted_counts != [0, 102] or inferred_counts != [0, 102]:
        raise RuntimeError("FROZEN_HAWOR_PROVENANCE_DRIFT")

    OUTPUT.mkdir(parents=True)
    video = OUTPUT / "INTERACTION_CONTACT_BLOCKER_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 960))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER")
    try:
        for frame_id in range(149):
            image = cv2.imread(str(RAW / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
            labels = cv2.imread(
                str(MASKS / "task_object" / f"{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED,
            )
            if image is None or labels is None or image.shape[:2] != (960, 1280):
                raise RuntimeError(f"FRAME_DOMAIN:{frame_id}")
            overlay = image.copy()
            for label, colour in ((1, (30, 80, 255)), (2, (30, 220, 80)), (3, (255, 100, 30))):
                mask = labels == label
                overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray(colour)).astype(np.uint8)
            # Diagnostic only: orange marks are inferred projections and are never
            # consumed as Stereo surface samples or Contact evidence.
            for hand_index, side in enumerate(sides):
                if not bool(hawor["predicted_valid"][hand_index, frame_id]):
                    continue
                for joint_index in TIPS:
                    uv = hawor["joints_2d"][hand_index, frame_id, joint_index]
                    if np.isfinite(uv).all():
                        cv2.circle(overlay, tuple(np.rint(uv).astype(int)), 5, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.rectangle(overlay, (0, 0), (1280, 112), (0, 0, 0), -1)
            cv2.putText(overlay, f"frame {frame_id:03d} | Object6D observed patches + HaWoR inferred projections", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "orange = inferred-only (diagnostic; NOT sampled for Contact)", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, .56, (0,165,255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "direct-observed hands: 0 | Contact/R1: BLOCKED_LOCAL_EVIDENCE", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, .56, (80,80,255), 2, cv2.LINE_AA)
            writer.write(overlay)
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(video)); decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if decoded != 149:
        raise RuntimeError(f"VIDEO_DECODE:{decoded}")

    common_inputs = {
        "hawor": artifact_ref(HAWOR),
        "depth": artifact_ref(DEPTH / "RESULT.json"),
        "object6d": artifact_ref(OBJECT6D / "OBJECT6D_OBSERVABILITY.json"),
        "task_object_masks": artifact_ref(MASKS / "MASK_MANIFEST.json"),
    }
    alignment = {
        "schema_version": "HUMAN_STEREO_ALIGNMENT_CHECK_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "status": "PASS_IMAGE_TIME_DOMAIN_BLOCKED_DIRECT_OBSERVATION",
        "image_domain": "PHYSICAL_LEFT_SOURCEINDEX1_ENCODED_RESIZE_ONLY",
        "depth_domain": depth_domain,
        "frame_mapping_exact": True, "timestamps_strictly_increasing": True,
        "intrinsic_half_scale_check": {
            "pass": domain_pass,
            "focal_abs_delta_px": focal_delta.tolist(),
            "principal_abs_delta_px": principal_delta.tolist(),
            "tolerance_focal_px": 0.001, "tolerance_principal_px": 0.5,
        },
        "ray_depth_scale_offset_fit": "NOT_RUN_NO_DIRECT_OBSERVED_HAND_SURFACE",
        "contact_or_object_fit_used": False, "constant_48mm_bias_subtracted": False,
        "hawor_absolute_z_consumed": False, "metric_wrist_translation_authorized": False,
        "external_metric_authority": False, "inputs": common_inputs,
    }
    interaction = {
        "schema_version": "INTERACTION_EVIDENCE_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "status": "BLOCKED_LOCAL_EVIDENCE",
        "fixed_denominator": {"frames": 149, "hands": sides, "fingers_per_hand": 5, "objects": 3, "pair_rows": 4470},
        "funnel": {
            "hawor_predicted_valid_left_right": predicted_counts,
            "hawor_direct_observed_left_right": observed_counts,
            "hawor_inferred_left_right": inferred_counts,
            "finger_associated_visible_surface_points": 0,
            "finite_patch_metric_rows": 0,
            "strict_contact_rows": 0,
            "continuous_strict_windows": 0,
        },
        "first_blocker": "NO_DIRECT_OBSERVED_HAWOR_HAND_FRAMES",
        "blocked_consumer": "FINGER_ASSOCIATED_VISIBLE_SURFACE_AND_CONTACT",
        "unblock_action": "PRODUCE_DIRECT_OBSERVED_SIDE_RESOLVED_HAND_PROJECTIONS_WITH_SAME_DOMAIN_PROVENANCE",
        "short_gap_inferred_consumed": False, "hawor_absolute_z_consumed": False,
        "removal_envelope_consumed": False, "hidden_object_geometry_inferred": False,
        "inputs": common_inputs,
    }
    contact = {
        "schema_version": "CONTACT_CANDIDATE_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "status": "BLOCKED_LOCAL_EVIDENCE", "strict_distance_gate_m": 0.005,
        "uncertainty_expands_distance_gate": False,
        "finite_visible_patch_required": True, "support_score_calibrated_probability": False,
        "candidate_rows": 0, "admitted_windows": [],
        "first_blocker": "NO_FINGER_ASSOCIATED_VISIBLE_SURFACE_POINT",
        "tactile_consumed": False, "contact_ground_truth": False,
        "inputs": common_inputs,
    }
    r1 = {
        "schema_version": "KAI22_R1_LOCAL_REFINEMENT_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "status": "BLOCKED_LOCAL_EVIDENCE", "execution": "NOT_STARTED",
        "reason": "NO_ADMITTED_CONTACT_WINDOW",
        "r0_preserved": True, "q22_modified": False, "wrist_modified": False,
        "collision_scope": "ROBOT_SELF_PLUS_OBSERVED_OBJECT_PATCH",
        "full_object_collision": "UNVERIFIED",
        "control_ground_truth": False, "physical_deployable": False,
    }
    write_once(OUTPUT / "HUMAN_STEREO_ALIGNMENT_CHECK_V1.json", alignment)
    write_once(OUTPUT / "INTERACTION_EVIDENCE_V1.json", interaction)
    write_once(OUTPUT / "CONTACT_CANDIDATE_V1.json", contact)
    write_once(OUTPUT / "KAI22_R1_LOCAL_REFINEMENT_V1.json", r1)
    result = {
        "schema_version": "S1_INTERACTION_CONTACT_031_RESULT_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "status": "R0_PRESERVED_R1_BLOCKED_LOCAL_EVIDENCE",
        "execution": "EXECUTED", "structure": "PASS", "quality": "INCONCLUSIVE",
        "adoption": "NOT_ADOPTED", "frame_count": 149,
        "alignment": artifact_ref(OUTPUT / "HUMAN_STEREO_ALIGNMENT_CHECK_V1.json"),
        "interaction": artifact_ref(OUTPUT / "INTERACTION_EVIDENCE_V1.json"),
        "contact": artifact_ref(OUTPUT / "CONTACT_CANDIDATE_V1.json"),
        "r1": artifact_ref(OUTPUT / "KAI22_R1_LOCAL_REFINEMENT_V1.json"),
        "review_video": {**artifact_ref(video), "decoded_frames": decoded},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    write_once(OUTPUT / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "funnel": interaction["funnel"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
