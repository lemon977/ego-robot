#!/usr/bin/env python3
"""R2 wave 1: conservative Scene repair and Sensor collision/extension."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.run_v5_scene import _mask_files, load as load_json, merge_roles
from chaoyang.ops.run_v5_sensor import _source_rows, hand_motion_arrays, _render_review
from chaoyang.pipeline.huro_hand_frame_v2 import solve_aligned_sequence, temporal_metrics
from chaoyang.pipeline.huro_hand_only_retarget_v1 import load_hand_model
from chaoyang.pipeline.kai22_full_fk_sidecar_v1 import (
    PyBulletNonAdjacentSelfCollisionChecker,
    load_pinned_kaihand_models,
)
from chaoyang.pipeline.v5_scene import build_conservative_repair_window, build_scene_mask_window


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_root_cause_gated_r2_20260923"
ATTEMPT = REPO / "_run/current" / TASK / "attempts/attempt_0001"
V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"
V3_AI1 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return ref(path)


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def summary(values) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"count": int(values.size),
            "p50": float(np.median(values)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None,
            "max": float(values.max()) if values.size else None}


def scene_repair(short: str, source_spec: Path | None = None) -> dict:
    output = ATTEMPT / f"lanes/lane1_scene/conservative_repair_{short}_wave1"
    result_path = output / "RESULT.json"
    if result_path.is_file():
        return load_json(result_path)
    if output.exists():
        raise RuntimeError(f"partial scene repair: {output}")
    output.mkdir(parents=True)
    sample_root = output / "samples"
    sample_root.mkdir()
    spec_path = source_spec or (V5 / f"lanes/scene/DIAG_{short}.json")
    spec = load_json(spec_path)
    domain = load_json(Path(spec["domain_manifest"]))
    mask_paths = spec.get("mask_manifests") or [spec["mask_manifest"]]
    docs = [load_json(Path(path)) for path in mask_paths]
    frame_count = int(domain["frame_count"])
    shape = (int(domain["height"]), int(domain["width"]))
    samples = set(np.linspace(0, frame_count - 1, min(16, frame_count), dtype=int).tolist())
    cache = {}
    old_area, new_area, old_object, new_object, repair_fraction = [], [], [], [], []
    device_nonempty = 0
    for frame in range(frame_count):
        start, stop = max(0, frame - 2), min(frame_count, frame + 3)
        for index in range(start, stop):
            if index not in cache:
                cache[index] = merge_roles(_mask_files(docs, index, shape), shape)
        for index in list(cache):
            if index < start:
                del cache[index]
        window = [cache[index] for index in range(start, stop)]
        center = frame - start
        old = build_scene_mask_window(window, center, frame, False)
        new = build_conservative_repair_window(window, center, frame, support_margin=4)
        role = cache[frame]
        if np.any((role.human | role.device) & ~new["context_exclude"]):
            raise RuntimeError(f"current semantic support lost at frame {frame}")
        if np.any(new["repair"] & ~cv2.dilate((role.human | role.device).astype(np.uint8),
                                               np.ones((17, 17), np.uint8)).astype(bool)):
            raise RuntimeError(f"repair escaped local evidence at frame {frame}")
        oa, na = int(old["write"].sum()), int(new["write"].sum())
        old_area.append(oa); new_area.append(na)
        old_object.append(int((old["write"] & role.object_visible).sum()))
        new_object.append(int((new["write"] & role.object_visible).sum()))
        repair_fraction.append(float(new["repair"].sum() / max(1, na)))
        device_nonempty += int(role.device.any())
        if frame in samples:
            canvas = np.zeros((*shape, 3), dtype=np.uint8)
            canvas[..., 2] = old["write"].astype(np.uint8) * 255
            canvas[..., 1] = new["write"].astype(np.uint8) * 255
            canvas[..., 0] = role.object_visible.astype(np.uint8) * 255
            if not cv2.imwrite(str(sample_root / f"{frame:06d}_Bobject_Gnew_Rold.png"), canvas):
                raise RuntimeError("scene comparison image write failed")
    old_total, new_total = int(sum(old_area)), int(sum(new_area))
    old_overlap, new_overlap = int(sum(old_object)), int(sum(new_object))
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SCENE_CONSERVATIVE_REPAIR_V1",
        "task_id": TASK, "session_id": spec["session_id"],
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED",
        "frame_count": frame_count,
        "baseline_write_px_total": old_total,
        "candidate_write_px_total": new_total,
        "area_reduction_fraction": float(1 - new_total / max(1, old_total)),
        "baseline_object_overlap_px_total": old_overlap,
        "candidate_object_overlap_px_total": new_overlap,
        "object_overlap_reduction_fraction": float(1 - new_overlap / max(1, old_overlap)),
        "repair_contribution_fraction": summary(repair_fraction),
        "device_nonempty_frames": device_nonempty,
        "samples": [ref(path) for path in sorted(sample_root.glob("*.png"))],
        "clean_executed": False,
        "blockers": ["VISIBLE_OBJECT_PROTECTION_UNTRUSTED",
                     "DEVICE_ROLE_INCOMPLETE" if device_nonempty < frame_count else None],
        "claim_limit": "Conservative semantic-base mask candidate only; no inpaint or Clean adoption.",
        "training_eligible": False, "control_ground_truth": False,
        "created_at": now(),
    }
    result["blockers"] = [item for item in result["blockers"] if item]
    write_json(result_path, result)
    return result


def collision_audit(states: dict, checker) -> dict:
    valid = np.asarray(states["valid"], dtype=bool)
    q22 = np.asarray(states["q22"], dtype=float)
    counts = np.zeros(valid.shape, dtype=np.int32)
    penetration = np.full(valid.shape, np.nan)
    known = np.zeros(valid.shape, dtype=bool)
    for frame, side in np.argwhere(valid):
        row = checker.check(int(side), q22[frame, side])
        known[frame, side] = row.known
        counts[frame, side] = row.illegal_contact_count
        penetration[frame, side] = row.max_penetration_m
    return {
        "evaluated_side_frames": int(known.sum()),
        "expected_side_frames": int(valid.sum()),
        "collision_side_frames": int(((counts > 0) & known).sum()),
        "max_penetration_m": summary(penetration),
        "all_valid_frames_evaluated": bool(np.array_equal(known, valid)),
        "pass": bool(np.array_equal(known, valid) and not np.any(counts[known] > 0)),
    }


def sensor_session(row: dict, hands, checker, asset_refs: dict) -> dict:
    sid = row["session_id"]
    output = ATTEMPT / "lanes/lane3_sensor" / f"run_{sid[-3:]}_wave1"
    result_path = output / "RESULT.json"
    if result_path.is_file():
        return load_json(result_path)
    if output.exists():
        raise RuntimeError(f"partial sensor extension: {output}")
    output.mkdir(parents=True)
    motion = hand_motion_arrays(row["nominal"], row["fitted"])
    motion_path = output / "HAND_MOTION_V1.npz"
    np.savez_compressed(motion_path, **motion)
    review = _render_review(row["video"], row["fitted"], output / "SENSOR_REVIEW.mp4")
    frames = np.asarray(motion["frame_id"])
    states = solve_aligned_sequence(
        hands, np.asarray(motion["joints21_root"]).transpose(1, 0, 2, 3),
        np.asarray(motion["manus_hand_valid"]).T, frames, max_evaluations=30)
    times = (np.asarray(motion["timestamp_ns"]) - int(motion["timestamp_ns"][0])) * 1e-9
    temporal = temporal_metrics(states["q22"], states["valid"], times, frames)
    collision = collision_audit(states, checker)
    states.update({"timestamp_ns": np.asarray(motion["timestamp_ns"]), "timestamp_s": times,
                   "motion_source": np.asarray("controller_manus"),
                   "clip_delta": np.where(states["valid"][..., None], 0.0, np.nan), **temporal})
    robot_path = output / "KAI22_COMMON_BACKEND_V1.npz"
    np.savez_compressed(robot_path, **states)
    tier = "KINEMATIC_ONLY" if collision["pass"] else "KINEMATIC_CANDIDATE_REJECTED_COLLISION"
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SENSOR_COMMON_BACKEND_V1",
        "task_id": TASK, "session_id": sid,
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "PASS_KINEMATIC_ONLY" if collision["pass"] else "REJECTED_QUALITY",
        "adoption": "CANDIDATE_ONLY", "motion_source": "controller_manus",
        "frames": int(len(frames)), "solver_valid_side_frames": int(states["valid"].sum()),
        "solver_valid_left_right": states["valid"].sum(axis=0).tolist(),
        "collision": collision, "kinematic_class": tier,
        "development_r0": False, "h50_ready": False,
        "hand_motion": ref(motion_path), "robot_backend": ref(robot_path),
        "review": review, "source_video": ref(row["video"]), "assets": asset_refs,
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "created_at": now(),
    }
    write_json(result_path, result)
    return result


def upgrade_097_collision(checker) -> dict:
    root = ATTEMPT / "lanes/lane3_sensor/run_097_wave0"
    source = npz(root / "KAI22_COMMON_BACKEND_V1.npz")
    audit = collision_audit(source, checker)
    result = {"schema_version": "HUMAN_TO_ROBOT_R2_SENSOR_COLLISION_V1",
              "task_id": TASK, "session_id": "play_cards_0916_097",
              "execution": "EXECUTED", "structure": "PASS",
              "quality": "PASS_KINEMATIC_ONLY" if audit["pass"] else "REJECTED_QUALITY",
              "adoption": "CANDIDATE_ONLY", "collision": audit,
              "kinematic_class": "KINEMATIC_ONLY" if audit["pass"] else "KINEMATIC_CANDIDATE_REJECTED_COLLISION",
              "development_r0": False, "h50_ready": False,
              "source": ref(root / "KAI22_COMMON_BACKEND_V1.npz"),
              "training_eligible": False, "control_ground_truth": False, "created_at": now()}
    write_json(root / "COLLISION_AUDIT.json", result)
    return result


def update_states(scene_rows: list[dict], sensor_rows: list[dict]) -> None:
    scene_state = load_json(ATTEMPT / "lanes/lane1_scene/STATE.json")
    scene_state.update({
        "execution": "EXECUTED", "structure": "PASS", "quality": "REJECTED_QUALITY",
        "adoption": "NOT_ADOPTED", "current_action": "Conservative repair reduced spill; object/device evidence still blocks Clean.",
        "latest_artifacts": [ref(ATTEMPT / f"lanes/lane1_scene/conservative_repair_{short}_wave1/RESULT.json") for short in ("031", "007")],
        "blocker": {"code": "OBJECT_PROTECTION_AND_DEVICE_EVIDENCE_INCOMPLETE",
                    "consumer": "Clean", "owner": "lane1_scene",
                    "unblock_action": "validate local device observations and direct visible-object protection on frozen windows",
                    "unaffected": ["Motion", "Sensor", "numeric comparison"]},
        "writer": {"pid": os.getpid(), "proc_start_ticks": None}, "updated_at": now()})
    write_json(ATTEMPT / "lanes/lane1_scene/STATE.json", scene_state)
    sensor_state = load_json(ATTEMPT / "lanes/lane3_sensor/STATE.json")
    collision_pass = sum(row.get("kinematic_class") == "KINEMATIC_ONLY" for row in sensor_rows)
    sensor_state.update({
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "PASS_KINEMATIC_ONLY" if collision_pass == 3 else "REJECTED_QUALITY",
        "adoption": "CANDIDATE_ONLY",
        "current_action": "097/098/101 actual common backend, FK, collision and replay completed.",
        "latest_artifacts": [ref(path) for path in [
            ATTEMPT / "lanes/lane3_sensor/run_097_wave0/COLLISION_AUDIT.json",
            ATTEMPT / "lanes/lane3_sensor/run_098_wave1/RESULT.json",
            ATTEMPT / "lanes/lane3_sensor/run_101_wave1/RESULT.json"]],
        "blocker": None if collision_pass == 3 else {"code": "SENSOR_KAI22_COLLISION_REJECTION",
                    "consumer": "KINEMATIC_ONLY", "owner": "lane3_sensor",
                    "unblock_action": "inspect fixed collision examples; do not loosen limits",
                    "unaffected": ["native MANUS replay", "HandMotion arrays"]},
        "writer": {"pid": os.getpid(), "proc_start_ticks": None}, "updated_at": now()})
    write_json(ATTEMPT / "lanes/lane3_sensor/STATE.json", sensor_state)


def main() -> int:
    scene = [scene_repair(short) for short in ("031", "007")]
    models, asset_refs = load_pinned_kaihand_models(REPO)
    hands = tuple(load_hand_model(model.path, side)
                  for model, side in zip(models, ("left", "right"), strict=True))
    checker = PyBulletNonAdjacentSelfCollisionChecker(models)
    try:
        sensor = [upgrade_097_collision(checker)]
        rows = {row["session_id"]: row for row in _source_rows(V3_AI1)}
        for sid in ("play_cards_0916_098", "play_cards_0916_101"):
            sensor.append(sensor_session(rows[sid], hands, checker, asset_refs))
    finally:
        checker.close()
    update_states(scene, sensor)
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1", "task_id": TASK,
        "stage": "WAVE1_CONSERVATIVE_SCENE_AND_SENSOR_EXTENSION", "observed_at": now(),
        "scene": [{"session_id": row["session_id"],
                   "area_reduction_fraction": row["area_reduction_fraction"],
                   "object_overlap_reduction_fraction": row["object_overlap_reduction_fraction"],
                   "quality": row["quality"]} for row in scene],
        "sensor": [{"session_id": row["session_id"], "kinematic_class": row["kinematic_class"],
                    "collision": row["collision"]} for row in sensor],
        "next": ["031 independent side observability/ROI", "Scene evidence gate before inpaint",
                 "renderer four-way test", "per-session product only when own Clean is structurally legal"],
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE1.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
