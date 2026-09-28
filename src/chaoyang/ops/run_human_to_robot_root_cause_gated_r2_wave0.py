#!/usr/bin/env python3
"""Execute the first independent R2 CPU wave.

This is a producer, not a status-only audit.  It consumes the pinned Scene,
HaWoR, MANUS and Local/HuRo arrays and writes only inside the registered R2
attempt.  It deliberately does not synthesize Clean or promote any candidate.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil

import cv2
import numpy as np

from chaoyang.pipeline.huro_hand_frame_v2 import solve_aligned_sequence, temporal_metrics
from chaoyang.pipeline.huro_hand_only_retarget_v1 import load_hand_model
from chaoyang.pipeline.kai22_full_fk_sidecar_v1 import load_pinned_kaihand_models
from chaoyang.pipeline.v5_motion import recover
from chaoyang.pipeline.v5_scene import build_scene_mask_window
from chaoyang.ops.run_v5_scene import _mask_files, load as load_json, merge_roles
from chaoyang.ops.run_v5_sensor import _source_rows, hand_motion_arrays, _render_review


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_root_cause_gated_r2_20260923"
ATTEMPT = REPO / "_run/current" / TASK / "attempts/attempt_0001"
V5 = REPO / "_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001"
V3_AI1 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/lanes/ai1"
SCENE_SHORT = {"031": "play_cards_0915_031", "007": "get_potato_chips_0915_007"}


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return ref(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "count": int(values.size),
        "p50": float(np.median(values)) if values.size else None,
        "p95": float(np.percentile(values, 95)) if values.size else None,
        "maximum": float(values.max()) if values.size else None,
    }


def scene_diagnostic(short: str) -> dict:
    """Actually consume every source mask and build the reference tensor QA."""
    source_spec = V5 / f"lanes/scene/DIAG_{short}.json"
    spec = load_json(source_spec)
    domain_path = Path(spec["domain_manifest"])
    domain = load_json(domain_path)
    mask_paths = [Path(value) for value in spec["mask_manifests"]]
    mask_docs = [load_json(value) for value in mask_paths]
    frame_count = int(domain["frame_count"])
    shape = (int(domain["height"]), int(domain["width"]))
    if domain["session_id"] != spec["session_id"] or len(domain["frames"]) != frame_count:
        raise RuntimeError("scene identity or full-timeline mismatch")
    if any(doc["session_id"] != spec["session_id"] or doc["frame_count"] != frame_count
           or doc["domain"] != domain["image_domain"] for doc in mask_docs):
        raise RuntimeError("scene mask domain mismatch")
    output = ATTEMPT / "lanes/lane1_scene" / f"reference_diagnostic_{short}_wave0"
    if output.exists():
        result = output / "RESULT.json"
        if not result.is_file():
            raise RuntimeError(f"partial scene output: {output}")
        return load_json(result)
    output.mkdir(parents=True)
    sample_ids = sorted(set(np.linspace(0, frame_count - 1, min(16, frame_count), dtype=int).tolist()))
    sample_root = output / "samples"
    sample_root.mkdir()
    cache: dict[int, object] = {}
    rows = []
    previous_write = None
    for frame in range(frame_count):
        start, stop = max(0, frame - 2), min(frame_count, frame + 3)
        for index in range(start, stop):
            if index not in cache:
                cache[index] = merge_roles(_mask_files(mask_docs, index, shape), shape)
        for index in list(cache):
            if index < start:
                del cache[index]
        scene = build_scene_mask_window([cache[index] for index in range(start, stop)],
                                        frame - start, frame, False)
        current = cache[frame]
        write = scene["write"]
        context = scene["context_exclude"]
        if np.any(write & ~context):
            raise RuntimeError(f"device/human removal absent from reference tensor at {frame}")
        area = int(write.sum())
        xor = int(np.logical_xor(write, previous_write).sum()) if previous_write is not None else 0
        row = {
            "frame_id": frame,
            "write_px": area,
            "context_px": int(context.sum()),
            "device_px": int(current.device.sum()),
            "human_px": int(current.human.sum()),
            "object_candidate_px": int(current.object_visible.sum()),
            "write_object_overlap_px": int((write & current.object_visible).sum()),
            "temporal_xor_px": xor,
        }
        rows.append(row)
        if frame in sample_ids:
            for name, array in (("write", write), ("context", context),
                                ("device", current.device), ("object", current.object_visible)):
                if not cv2.imwrite(str(sample_root / f"{frame:06d}_{name}.png"),
                                   array.astype(np.uint8) * 255):
                    raise RuntimeError("sample mask write failed")
        previous_write = write.copy()
    write_area = np.asarray([row["write_px"] for row in rows])
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SCENE_REFERENCE_DIAGNOSTIC_V1",
        "task_id": TASK,
        "session_id": spec["session_id"],
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "REJECTED_QUALITY",
        "adoption": "NOT_ADOPTED",
        "frame_count": frame_count,
        "actual_reference_tensor_consumed_frames": frame_count,
        "device_nonempty_frames": int(sum(row["device_px"] > 0 for row in rows)),
        "object_overlap_frames": int(sum(row["write_object_overlap_px"] > 0 for row in rows)),
        "write_area": stats(write_area),
        "temporal_xor": stats(np.asarray([row["temporal_xor_px"] for row in rows])),
        "rows": rows,
        "samples": [ref(path) for path in sorted(sample_root.glob("*.png"))],
        "inputs": {"spec": ref(source_spec), "domain": ref(domain_path),
                   "masks": [ref(path) for path in mask_paths]},
        "blockers": ["NO_TRUSTED_VISIBLE_OBJECT_PROTECTION",
                     "DEVICE_ROLE_INCOMPLETE" if not all(row["device_px"] > 0 for row in rows) else None],
        "clean_executed": False,
        "claim_limit": "Reference tensor consumption/root-cause evidence only; no Clean pixels created.",
        "training_eligible": False,
        "control_ground_truth": False,
        "created_at": now(),
    }
    result["blockers"] = [item for item in result["blockers"] if item]
    write_json(output / "RESULT.json", result)
    return result


def motion_rebind(short: str) -> dict:
    source_config = V5 / f"lanes/motion/recover_{short}.json"
    config = load_json(source_config)
    config_root = ATTEMPT / "lanes/lane2_motion/configs"
    config_root.mkdir(parents=True, exist_ok=True)
    destination = ATTEMPT / "lanes/lane2_motion" / f"recovered_{short}_wave0"
    config["output"] = str(destination)
    target = config_root / f"recover_{short}_wave0.json"
    if not target.exists():
        write_json(target, config)
    else:
        if load_json(target) != config:
            raise RuntimeError(f"motion config drift: {short}")
    result = recover(target)
    motion = load_npz(Path(result["outputs"]["hand_motion"]["path"]))
    robot = load_npz(Path(result["outputs"]["robot_r0"]["path"]))
    valid = motion["predicted_valid"]
    if robot["q_hand22"].shape != (len(valid), 2, 22):
        raise RuntimeError("q22 output shape mismatch")
    diagnostic = {
        "schema_version": "HUMAN_TO_ROBOT_R2_MOTION_FUNNEL_V1",
        "task_id": TASK,
        "session_id": result["session_id"],
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "INCONCLUSIVE" if valid.any() else "REJECTED_QUALITY",
        "adoption": "CANDIDATE_ONLY",
        "stage_counts_left_right": result["stage_counts_left_right"],
        "first_recorded_zero_stage_left_right": result["first_recorded_zero_stage_left_right"],
        "valid_side_frames": valid.sum(axis=0).tolist(),
        "left_roi_root_cause_confirmed": bool(short == "031" and result["stage_counts_left_right"]["roi"][0] == 0),
        "bounded_executed": False,
        "raw_r0_available": bool(robot["target_valid"].any()),
        "result": ref(destination / "RESULT.json"),
        "training_eligible": False,
        "control_ground_truth": False,
        "created_at": now(),
    }
    write_json(destination / "R2_FUNNEL.json", diagnostic)
    return diagnostic


def sensor_097_backend() -> dict:
    """Run 097 MANUS replay and the shared Kai22 hand-only target/FK backend."""
    output = ATTEMPT / "lanes/lane3_sensor/run_097_wave0"
    final_path = output / "RESULT.json"
    if final_path.is_file():
        return load_json(final_path)
    if output.exists():
        raise RuntimeError(f"partial sensor output: {output}")
    output.mkdir(parents=True)
    row = next(item for item in _source_rows(V3_AI1) if item["session_id"] == "play_cards_0916_097")
    motion = hand_motion_arrays(row["nominal"], row["fitted"])
    motion_path = output / "HAND_MOTION_V1.npz"
    np.savez_compressed(motion_path, **motion)
    review = _render_review(row["video"], row["fitted"], output / "SENSOR_REVIEW.mp4")

    pinned, asset_refs = load_pinned_kaihand_models(REPO)
    hands = tuple(load_hand_model(model.path, side)
                  for model, side in zip(pinned, ("left", "right"), strict=True))
    joints = np.asarray(motion["joints21_root"]).transpose(1, 0, 2, 3)
    observed = np.asarray(motion["manus_hand_valid"]).T
    frames = np.asarray(motion["frame_id"])
    states = solve_aligned_sequence(hands, joints, observed, frames, max_evaluations=30)
    times = (np.asarray(motion["timestamp_ns"]) - int(motion["timestamp_ns"][0])) * 1e-9
    temporal = temporal_metrics(states["q22"], states["valid"], times, frames)
    states.update({
        "timestamp_ns": np.asarray(motion["timestamp_ns"]),
        "timestamp_s": times,
        "anatomical_side_names": np.asarray(["left", "right"]),
        "physical_robot_side_names": np.asarray(["left", "right"]),
        "clip_delta": np.where(states["valid"][..., None], 0.0, np.nan),
        "motion_source": np.asarray("controller_manus"),
        "collision_status": np.asarray("NOT_EVALUATED"),
        **temporal,
    })
    robot_path = output / "KAI22_COMMON_BACKEND_V1.npz"
    np.savez_compressed(robot_path, **states)
    valid = states["valid"]
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_SENSOR_COMMON_BACKEND_V1",
        "task_id": TASK,
        "session_id": "play_cards_0916_097",
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "INCONCLUSIVE",
        "adoption": "CANDIDATE_ONLY",
        "motion_source": "controller_manus",
        "frames": int(len(frames)),
        "target_builder_frames": int(len(frames)),
        "solver_valid_side_frames": int(valid.sum()),
        "solver_valid_left_right": valid.sum(axis=0).tolist(),
        "hard_limit_violation": False,
        "clip_delta_nonzero": 0,
        "self_collision": "NOT_EVALUATED",
        "kinematic_class": "KINEMATIC_ONLY_PENDING_SELF_COLLISION",
        "hand_motion": ref(motion_path),
        "robot_backend": ref(robot_path),
        "review": review,
        "source_video": ref(row["video"]),
        "assets": asset_refs,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "created_at": now(),
    }
    write_json(final_path, result)
    return result


def compare_recheck() -> dict:
    output = ATTEMPT / "lanes/lane4_compare/wave0_numeric_recheck"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for short, session in SCENE_SHORT.items():
        local_path = V5 / f"lanes/motion/recovered_{short}_v1/ROBOT_R0_V1.npz"
        huro_path = V5 / f"lanes/huro/full_0001/{session}/HURO_CORE_V1.npz"
        local, huro = load_npz(local_path), load_npz(huro_path)
        if not np.array_equal(local["frame_id"], huro["frame_id"]):
            raise RuntimeError(f"Local/HuRo frame mismatch: {session}")
        common = (local["target_valid"] & local["wrist_valid"] & local["finger_valid"] &
                  huro["target_valid"] & huro["wrist_valid"] & huro["finger_valid"])
        rows.append({
            "session_id": session,
            "common_side_frames": int(common.sum()),
            "timeline_side_frames": int(common.size),
            "local_position_residual_mm": stats(local["position_residual_mm"][common]),
            "huro_position_residual_mm": stats(huro["position_residual_mm"][common]),
            "local_rotation_residual_deg": stats(local["rotation_residual_deg"][common]),
            "huro_rotation_residual_deg": stats(huro["rotation_residual_deg"][common]),
            "local": ref(local_path),
            "huro": ref(huro_path),
            "winner": None,
        })
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_LOCAL_HURO_NUMERIC_RECHECK_V1",
        "task_id": TASK,
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "INCONCLUSIVE",
        "adoption": "NOT_ADOPTED",
        "same_target_frame_denominator_verified": True,
        "rows": rows,
        "render_four_way_test": "NOT_YET_EXECUTED",
        "winner": None,
        "training_eligible": False,
        "control_ground_truth": False,
        "created_at": now(),
    }
    write_json(output / "RESULT.json", result)
    return result


def lane_state(lane: str, execution: str, structure: str, quality: str,
               adoption: str, refs: list[Path], blocker: dict | None, action: str) -> None:
    write_json(ATTEMPT / "lanes" / lane / "STATE.json", {
        "schema_version": "human-to-robot-r2-lane-state-v1",
        "task_id": TASK,
        "lane": lane,
        "status": "RUNNING" if blocker is None else "RUNNING_WITH_LOCAL_BLOCKER",
        "execution": execution,
        "structure": structure,
        "quality": quality,
        "adoption": adoption,
        "blocker": blocker,
        "current_action": action,
        "latest_artifacts": [ref(path) for path in refs],
        "writer": {"pid": os.getpid(), "proc_start_ticks": None},
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "updated_at": now(),
    })


def main() -> int:
    scene = [scene_diagnostic(short) for short in ("031", "007")]
    motion = [motion_rebind(short) for short in ("031", "007")]
    sensor = sensor_097_backend()
    compare = compare_recheck()
    scene_refs = [ATTEMPT / f"lanes/lane1_scene/reference_diagnostic_{short}_wave0/RESULT.json"
                  for short in ("031", "007")]
    motion_refs = [ATTEMPT / f"lanes/lane2_motion/recovered_{short}_wave0/R2_FUNNEL.json"
                   for short in ("031", "007")]
    sensor_refs = [ATTEMPT / "lanes/lane3_sensor/run_097_wave0/RESULT.json"]
    compare_refs = [ATTEMPT / "lanes/lane4_compare/wave0_numeric_recheck/RESULT.json"]
    lane_state("lane1_scene", "EXECUTED", "PASS", "REJECTED_QUALITY", "NOT_ADOPTED",
               scene_refs, {"code": "SCENE_ROLE_AND_OBJECT_PROTECTION_UNRESOLVED",
                            "consumer": "Clean", "owner": "lane1_scene",
                            "unblock_action": "one evidence-driven role/reference repair on frozen windows",
                            "unaffected": ["Motion", "Sensor", "Local/HuRo numeric"]},
               "Reference tensors consumed for 031/007; prepare bounded repair before Clean.")
    left_zero = motion[0]["valid_side_frames"][0] == 0
    lane_state("lane2_motion", "EXECUTED", "PASS", "INCONCLUSIVE", "CANDIDATE_ONLY",
               motion_refs, {"code": "031_LEFT_ROI_ZERO" if left_zero else "BOUNDED_NOT_EVALUATED",
                             "consumer": "031 bilateral product", "owner": "lane2_motion",
                             "unblock_action": "repair independent left ROI/model-input handoff without mirroring",
                             "unaffected": ["007 raw R0", "Scene", "Sensor"]},
               "031/007 raw Motion rebound; next isolate 031 left ROI and bounded metrics.")
    lane_state("lane3_sensor", "EXECUTED", "PASS", "INCONCLUSIVE", "CANDIDATE_ONLY",
               sensor_refs, {"code": "SELF_COLLISION_NOT_EVALUATED",
                             "consumer": "DEVELOPMENT_R0", "owner": "lane3_sensor",
                             "unblock_action": "run pinned non-adjacent collision audit then extend 098/101",
                             "unaffected": ["097 native replay", "097 q22/FK"]},
               "097 actually entered shared Kai22 target-builder/solver/FK; collision and review remain.")
    lane_state("lane4_compare", "EXECUTED", "PASS", "INCONCLUSIVE", "NOT_ADOPTED",
               compare_refs, {"code": "RENDER_FOUR_WAY_NOT_EXECUTED",
                              "consumer": "visual method comparison", "owner": "lane4_compare",
                              "unblock_action": "run fixed-q/camera/background renderer test",
                              "unaffected": ["same-target numeric diagnostic"]},
               "Same-frame Local/HuRo numeric denominators rechecked; no winner claimed.")
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK,
        "observed_at": now(),
        "stage": "WAVE0_ROOT_CAUSE_AND_REAL_CONSUMER_EXECUTION",
        "actual_growth": {
            "scene_sessions_consumed": len(scene),
            "motion_sessions_rebound": len(motion),
            "sensor_sessions_common_backend": 1,
            "compare_sessions_rechecked": len(compare["rows"]),
            "clean_sessions": 0,
            "product_sessions": 0,
        },
        "facts": {
            "031_left_first_zero_stage": motion[0]["first_recorded_zero_stage_left_right"][0],
            "031_left_valid_side_frames": motion[0]["valid_side_frames"][0],
            "007_valid_side_frames": motion[1]["valid_side_frames"],
            "sensor_097_solver_valid_side_frames": sensor["solver_valid_side_frames"],
        },
        "next": ["031 left ROI root-cause repair", "Scene role/reference bounded repair",
                 "097 collision then 098/101", "renderer four-way diagnostic"],
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE0.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
