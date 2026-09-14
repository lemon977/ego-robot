#!/usr/bin/env python3
"""Audit a transactional visual Object6D output and promote it to Grade B baseline.

Promotion authorizes only Clean and Robot *visualization* consumers.  It never
authorizes physical contact, calibrated deployment, or task-success claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RUN_ID = "20260908_two_task_e2e_baseline_v1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular non-symlink artifact required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"no-clobber output exists: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def audit(root: Path, task: str, session: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve(strict=True)
    orchestrated_path = root / "RESULT.json"
    numeric_path = root / "numeric/RESULT.json"
    trajectory_path = root / "numeric/VISUAL_FIXED_INSTANCE_OBJECT6D.npz"
    frame_manifest_path = root / "numeric/FRAME_MANIFEST.json"
    review_manifest_path = root / "REVIEW_MANIFEST.json"
    video_path = root / "OBJECT6D_中文逐帧审查.mp4"
    orchestrated = load(orchestrated_path)
    numeric = load(numeric_path)
    frames = load(frame_manifest_path)
    review = load(review_manifest_path)
    gates = {
        "session_identity": "PASS",
        "numeric_status": "PASS",
        "frame_coverage": "PASS",
        "identity_onset_validity_policy": "PASS",
        "single_visual_instance": "PASS",
        "finite_valid_pose": "PASS",
        "direct_observation_coverage": "PASS",
        "bounded_occlusion_gap": "PASS",
        "bounded_world_pose_rate": "PASS",
        "direct_vs_propagated_depth_semantics": "PASS",
        "forbidden_hand_pinch_robot_inputs": "PASS",
        "video_full_decode": "PASS",
    }
    if session != numeric.get("session_id") or task != numeric.get("task"):
        gates["session_identity"] = "FAIL"
    if not str(numeric.get("status", "")).startswith("PASS"):
        gates["numeric_status"] = "FAIL"
    frame_count = int(numeric.get("frame_count", -1))
    if frame_count <= 0 or len(frames.get("frames", [])) != frame_count:
        gates["frame_coverage"] = "FAIL"
    if numeric.get("instance_count") != 1:
        gates["single_visual_instance"] = "FAIL"
    if numeric.get("forbidden_inputs_absent") is not True:
        gates["forbidden_hand_pinch_robot_inputs"] = "FAIL"
    with np.load(trajectory_path, allow_pickle=False) as values:
        valid = values["valid"].astype(bool)
        if "observed" not in values.files:
            raise RuntimeError("Object6D trajectory lacks explicit observed/propgated semantics")
        observed = values["observed"].astype(bool)
        transforms = values["T_object_to_camera"].astype(np.float64)
        world_transforms = values["T_object_to_world"].astype(np.float64)
        observed_nf = values["observed_near_far_optical_z_m"].astype(np.float64)
        analytic_nf = values["analytic_near_far_optical_z_m"].astype(np.float64)
        visibility = values["visibility"].astype(np.float64)
        frame_ids = values["frame_indices"].astype(np.int64)
        instances = values["physical_instance_id"].astype(np.int64)
    direct_indices = np.flatnonzero(observed)
    active_start = int(direct_indices[0]) if len(direct_indices) else frame_count
    onset_policy = numeric.get("leading_unobserved_policy") == "INVALID_UNTIL_VISUAL_IDENTITY_ONSET"
    keep_invalid = numeric.get("unobserved_pose_policy") == "KEEP_INVALID"
    if keep_invalid:
        validity_ok = np.array_equal(valid, observed) and np.all(instances[~valid] == -1)
    elif onset_policy:
        validity_ok = (
            not valid[:active_start].any()
            and valid[active_start:].all()
            and np.all(instances[:active_start] == -1)
        )
    else:
        validity_ok = valid.all()
    if not validity_ok or int(valid.sum()) != int(numeric.get("valid_frames", -1)):
        gates["identity_onset_validity_policy"] = "FAIL"
    if (
        len(valid) != frame_count
        or not np.isfinite(transforms[valid]).all()
        or not np.isfinite(world_transforms[valid]).all()
        or len(set(frame_ids.tolist())) != frame_count
        or len(set(instances[valid].tolist())) != 1
    ):
        gates["finite_valid_pose"] = "FAIL"
    active_observed = observed[active_start:]
    observed_fraction = (
        float(observed.sum() / max(1, valid.sum()))
        if keep_invalid
        else (float(active_observed.mean()) if len(active_observed) else 0.0)
    )
    propagated = valid & ~observed
    gap_lengths: list[int] = []
    gap = 0
    for present in propagated.tolist() + [False]:
        if present:
            gap += 1
        elif gap:
            gap_lengths.append(gap)
            gap = 0
    max_gap = max(gap_lengths, default=0)
    if observed.shape != (frame_count,) or observed_fraction < 0.70:
        gates["direct_observation_coverage"] = "FAIL"
    if max_gap > 45:
        gates["bounded_occlusion_gap"] = "FAIL"
    translation_steps = []
    rotation_steps: list[float] = []
    for index in range(1, frame_count):
        if not (valid[index - 1] and valid[index]):
            continue
        translation_steps.append(float(np.linalg.norm(world_transforms[index, :3, 3] - world_transforms[index - 1, :3, 3])))
        relative = (
            world_transforms[index - 1, :3, :3].T
            @ world_transforms[index, :3, :3]
        )
        rotation_vector, _ = cv2.Rodrigues(relative)
        rotation_steps.append(float(np.degrees(np.linalg.norm(rotation_vector))))
    dense_translation_max = max(translation_steps, default=0.0)
    dense_rotation_max = max(rotation_steps, default=0.0)
    producer_pose_metrics = numeric.get("pose_rate_metrics", {})
    producer_dense_metrics = numeric.get("dense_pose_rate_metrics", {})
    if (
        float(producer_pose_metrics.get("bounded_translation_step_max_m", np.inf)) > 0.030001
        or float(producer_pose_metrics.get("bounded_rotation_step_max_deg", np.inf)) > 12.001
        or float(producer_dense_metrics.get("translation_step_max_m", np.inf)) > 0.030001
        or float(producer_dense_metrics.get("rotation_step_max_deg", np.inf)) > 12.001
        or dense_translation_max > 0.030001
        or dense_rotation_max > 12.001
    ):
        gates["bounded_world_pose_rate"] = "FAIL"
    if (
        observed_nf.shape != (frame_count, 2)
        or analytic_nf.shape != (frame_count, 2)
        or not np.isfinite(observed_nf[observed]).all()
        or np.isfinite(observed_nf[~observed]).any()
        or not np.isfinite(analytic_nf[valid]).all()
        or np.isfinite(analytic_nf[~valid]).any()
        or not np.all(visibility[~observed] == 0.0)
    ):
        gates["direct_vs_propagated_depth_semantics"] = "FAIL"
    capture = cv2.VideoCapture(str(video_path))
    decoded = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    if decoded != frame_count or (width, height) != (1920, 720):
        gates["video_full_decode"] = "FAIL"
    if orchestrated.get("agent_baseline_review") is not True or review.get("agent_baseline_review") is not True:
        gates["video_full_decode"] = "FAIL"
    failed = [name for name, value in gates.items() if value == "FAIL"]
    grade = "C" if failed else "B"
    authorized = not failed
    artifacts = {
        "orchestration_result": artifact(orchestrated_path),
        "numeric_result": artifact(numeric_path),
        "trajectory": artifact(trajectory_path),
        "frame_manifest": artifact(frame_manifest_path),
        "review_manifest": artifact(review_manifest_path),
        "review_video": artifact(video_path),
    }
    agent_review = {
        "schema_version": "baseline-agent-stage-review-v1",
        "created_at": now(),
        "run_id": RUN_ID,
        "stage": "OBJECT6D",
        "task": task,
        "session": session,
        "grade": grade,
        "downstream_authorized": authorized,
        "hard_gates": gates,
        "soft_defects": [
            "VISUAL_SHAPE_PRIOR_AND_STEREO_SURFACE_WITHOUT_PHYSICAL_POSE_GROUND_TRUTH",
            "OCCLUDED_FRAMES_USE_EXPLICIT_TEMPORAL_WORLD_POSE_PROPAGATION",
            "CONTACT_NOT_AUTHORIZED",
        ],
        "inputs": {"input_spec": orchestrated["input_spec"]},
        "artifacts": artifacts,
        "metrics": {"frame_count": frame_count, "valid_frames": int(valid.sum()), "invalid_pre_identity_frames": int((~valid[:active_start]).sum()), "active_start_local_frame": active_start, "direct_observed_frames": int(observed.sum()), "propagated_frames": int(propagated.sum()), "direct_observed_fraction_active_segment": observed_fraction, "max_propagated_gap_frames": max_gap, "instance_count": numeric.get("instance_count"), "dense_world_translation_step_max_m": dense_translation_max, "dense_world_rotation_step_max_deg": dense_rotation_max},
        "claim_limit": "Grade B is for Clean and Robot visualization only; no contact, physical calibration, deployment, or task-success authority.",
    }
    baseline_result = {
        "schema_version": "visual-object6d-agent-baseline-result-v1",
        "created_at": now(),
        "status": "PASS_VISUAL_OBJECT6D_BASELINE_GRADE_B" if authorized else "C_OBJECT6D_HARD_GATE_FAILED",
        "task": task,
        "session": session,
        "session_id": session,
        "frame_count": frame_count,
        "grade": grade,
        "consumption_authorized": authorized,
        "authorized_scopes": ["CLEAN_BASELINE", "ROBOT_VISUAL_BASELINE"] if authorized else [],
        "robot_contact_authorized": False,
        "observation_summary": {
            "direct_observed_frames": int(observed.sum()),
            "temporally_propagated_frames": int(propagated.sum()),
            "invalid_pre_identity_frames": int((~valid[:active_start]).sum()),
            "active_start_local_frame": active_start,
            "direct_observed_fraction_active_segment": observed_fraction,
            "max_propagated_gap_frames": max_gap,
        },
        "artifacts": artifacts,
        "failed_hard_gates": failed,
        "claim_limit": agent_review["claim_limit"],
    }
    return agent_review, baseline_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    review, result = audit(args.output_root, args.task, args.session)
    atomic_json(args.output_root / "AGENT_REVIEW.json", review)
    atomic_json(args.output_root / "BASELINE_RESULT.json", result)
    print(json.dumps({"grade": review["grade"], "downstream_authorized": review["downstream_authorized"]}))
    return 0 if review["downstream_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
