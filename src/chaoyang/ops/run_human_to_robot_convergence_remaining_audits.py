#!/usr/bin/env python3
"""Close bounded evidence gaps without rerunning rejected algorithms."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2

from chaoyang.governance.common import artifact_ref, load_json

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_baseline_v1_convergence_20260923"
ROOT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
RAW = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/exact78/prepare_full_v1/play_cards_0915_031/raw"
HAWOR = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
SOURCE_AUDIT = REPO / "_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/motion_product/h3_motion_audit_031/attempt_0002/RESULT.json"
MOUNT = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/shared/cad/interfaces_mount_v2_corrected/MOUNT_CONTRACT.json"
OCCLUSION = REPO / "_run/current/human_to_robot_evidence_unlock_s2_20260923/attempts/attempt_0001/lanes/scene/h3_occlusion_031/attempt_0003/RESULT.json"
INDEX = REPO / "tasks/current/INDEX.json"
SAMPLED = [0, 10, 20, 30, 39, 49, 59, 69, 79, 89, 99, 109, 118, 128, 138, 148]


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{path}")
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def decode_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path)); count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def left_evidence() -> Path:
    out = ROOT / "lanes/motion_product/left_evidence_031/wave0"
    if out.exists():
        raise FileExistsError(f"IMMUTABLE_OUTPUT_EXISTS:{out}")
    out.mkdir(parents=True)
    video = out / "031_LEFT_EVIDENCE_FIXED_SAMPLE_REVIEW.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (1280, 960))
    if not writer.isOpened():
        raise RuntimeError("VIDEO_WRITER")
    try:
        for frame_id in SAMPLED:
            frame = cv2.imread(str(RAW / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
            if frame is None or frame.shape[:2] != (960, 1280):
                raise ValueError(f"RAW_FRAME:{frame_id}")
            cv2.rectangle(frame, (0, 0), (1279, 92), (0, 0, 0), -1)
            cv2.putText(frame, f"031 raw-only independent sample | frame {frame_id}/148", (18, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "AI_REVIEW_PROXY: no reliable independent second-hand evidence in this sample", (18, 66),
                        cv2.FONT_HERSHEY_SIMPLEX, .53, (0, 210, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Sample does NOT prove full-timeline absence; HaWoR/q not read for selection", (18, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 220, 255), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    result = {
        "schema_version": "HUMAN_TO_ROBOT_031_LEFT_EVIDENCE_AUDIT_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "execution": "EXECUTED", "structure": "PASS", "quality": "INCONCLUSIVE_OBSERVABILITY",
        "improvement": "FROZEN_RAW_ONLY_SAMPLE_REVIEWED", "adoption": "NOT_ADOPTED",
        "selection_authority": "RAW_RGB_ONLY_EQUAL_TIME_FIXED_16_FRAME_SAMPLE",
        "sampled_frame_ids": SAMPLED,
        "sample_observation": "NO_RELIABLE_INDEPENDENT_SECOND_HAND_EVIDENCE_IN_SAMPLED_FRAMES",
        "full_timeline_left_conclusion": "NOT_EVALUATED",
        "existing_producer_state": {"left_roi": 0, "left_model_output": 0, "right_inferred_frames": 102,
                                    "frozen_experiment_denominator": "149/102/0_UNCHANGED"},
        "model_invocation": "NOT_STARTED_NO_DETERMINISTIC_INDEPENDENT_LEFT_ROI",
        "inputs": {"raw_frame_000": artifact_ref(RAW / "000000.png"),
                   "raw_frame_148": artifact_ref(RAW / "000148.png"),
                   "existing_hawor": artifact_ref(HAWOR), "source_audit": artifact_ref(SOURCE_AUDIT)},
        "review_video": {**artifact_ref(video), "decoded_frames": decode_count(video)},
        "claim_limit": "AI review proxy over 16 frozen raw frames. It does not establish full-timeline absence or authorize filling, mirroring, holding, or a new model run.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    write(out / "RESULT.json", result)
    return out / "RESULT.json"


def collision_audit() -> Path:
    out = ROOT / "lanes/motion_product/adapter_collision/wave0/RESULT.json"
    contract = load_json(MOUNT)
    mesh = Path(contract["decoded_review_mesh"]["path"])
    result = {
        "schema_version": "HUMAN_TO_ROBOT_ADAPTER_COLLISION_AUDIT_V1",
        "task_id": TASK, "created_at": now(), "execution": "EXECUTED", "structure": "PASS",
        "quality": "NOT_EVALUATED_NO_APPROVED_COLLISION_SEMANTICS", "improvement": "COLLISION_SCOPE_MADE_EXPLICIT",
        "adoption": "NOT_ADOPTED",
        "visual_geometry": {"status": "REAL_STEP_DERIVED_VISUAL_MESH", "mesh": artifact_ref(mesh),
                            "mount_contract": artifact_ref(MOUNT)},
        "collision_geometry": {"status": "NOT_AUTHORIZED", "approved_mesh_or_envelope": None,
                               "mounting_contact_whitelist": None, "queries_executed": 0},
        "preserved_claim": "Existing arm-and-hand collision results retain their original scope only.",
        "forbidden_claim": "The adapter is not included in a complete-machine collision PASS.",
        "unblock_action": "Provide an approved adapter collision mesh or proven conservative envelope plus mounting-contact semantics; then run the existing collision consumer.",
        "claim_limit": contract["claim_limit"],
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    write(out, result)
    return out


def occlusion_audit() -> Path:
    out = ROOT / "lanes/scene/occlusion_same_surface_031/wave0/RESULT.json"
    source = load_json(OCCLUSION)
    rows = source["rows"]
    result = {
        "schema_version": "HUMAN_TO_ROBOT_031_SAME_SURFACE_OCCLUSION_AUDIT_V1",
        "task_id": TASK, "session_id": "play_cards_0915_031", "created_at": now(),
        "execution": "EXECUTED_REUSED_ARRAYS", "structure": "PASS",
        "quality": "NOT_EVALUATED_NO_TEMPORAL_SURFACE_IDENTITY",
        "improvement": "PIXEL_SCREEN_SCOPE_SEPARATED_FROM_SAME_SURFACE_METRIC", "adoption": "NOT_ADOPTED",
        "existing_fixed_window": [int(rows[0]["frame_id"]), int(rows[-1]["frame_id"])],
        "pixel_screen_diagnostic": {
            "frames": len(rows), "decision_pixels": int(sum(r["decision_pixels"] for r in rows)),
            "known_decision_coverage": source["known_decision_coverage"],
            "legacy_consecutive_known_comparable_pixels": source["consecutive_known_comparable_pixels"],
            "legacy_consecutive_known_ownership_switches": source["consecutive_known_ownership_switches"],
            "authority": "PIXEL_SCREEN_DIAGNOSTIC_ONLY_NOT_ACCURACY",
        },
        "same_surface_metric": {
            "correspondence_attempts": 0, "legal_correspondences": 0, "boundary_or_exposure": 0,
            "geometric_uncertainty": 0, "reasonable_crossings": 0, "unexplained_flips": 0,
            "a_to_b_to_a": 0, "status": "NOT_EVALUATED",
            "reason": "Saved ownership arrays contain per-pixel class only; no persistent scene-surface/link correspondence identity exists across frames.",
        },
        "input": artifact_ref(OCCLUSION),
        "unblock_action": "Version a frozen scene-surface correspondence producer with persistent IDs before computing temporal same-surface transitions; do not infer identity from adjacent screen pixels.",
        "claim_limit": "The old 736/3269 counts remain on their original pixel-screen denominator and cannot be called accuracy or subtracted from a future same-surface metric.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    write(out, result)
    return out


def main() -> int:
    index = load_json(INDEX)
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    outputs = [left_evidence(), collision_audit(), occlusion_audit()]
    print(json.dumps({"status": "PASS", "outputs": [str(p) for p in outputs]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
