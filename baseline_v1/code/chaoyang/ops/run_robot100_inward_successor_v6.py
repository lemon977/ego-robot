"""Project the 0916 Robot100 display motion onto the requested inward-forearm posture.

V5 remains the immutable exact-tracking predecessor.  V6 changes only the
offline display trajectory: whenever a wrist is not medial to its elbow, the
smallest joint-space interpolation toward one fixed inward reference is used.
The original target, raw IK, residuals and every inference mask are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_inward_successor_v5 as v5
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_inward_v3 import motion_derivatives
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets

OUTPUT = v5.OUTPUT.parent.parent / "inward_forearm_posture_v6" / "successor_sessions"
POSTURE_REFERENCE_Q = np.asarray([
    [1.76172, -0.32844, -1.07304, -2.23461, -0.24156, 0.42408, 0.68220],
    [-1.76172, -0.32844, 1.07304, -2.23461, 0.24156, 0.42396, -0.68220],
], dtype=np.float64)
MIN_INWARD_M = 0.01
BISECTION_STEPS = 16


def stable_artifact(ref: dict, allowed_root: Path) -> Path:
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_relative_to(allowed_root.resolve(strict=True)):
        raise RuntimeError(f"PREDECESSOR_SCOPE_ESCAPE:{path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise RuntimeError(f"UNSTABLE_PREDECESSOR:{path}")
    if before.st_size != ref["bytes"] or digest.hexdigest() != ref["sha256"]:
        raise RuntimeError(f"PREDECESSOR_REFERENCE_DRIFT:{path}")
    return path


def inward_margin_m(assets, mounts: np.ndarray, neutral: np.ndarray, side: int, q: np.ndarray) -> float:
    joint_map = {
        name: float(neutral[physical_side, index])
        for physical_side in range(2)
        for index, name in enumerate(ARM_JOINT_NAMES[physical_side])
    }
    joint_map.update({
        name: float(value)
        for name, value in zip(ARM_JOINT_NAMES[side], q, strict=True)
    })
    fk = forward_kinematics(assets.tianji, joint_map)
    suffix = ("L", "R")[side]
    sign = 1.0 if side == 0 else -1.0
    elbow = sign * float(fk[f"Link3_{suffix}"][1, 3])
    wrist = sign * float((fk[f"flange_{suffix}"] @ mounts[side])[1, 3])
    return elbow - wrist


def project_one(assets, mounts, neutral, side: int, q: np.ndarray) -> tuple[np.ndarray, float]:
    q = np.asarray(q, dtype=np.float64)
    if inward_margin_m(assets, mounts, neutral, side, q) >= MIN_INWARD_M:
        return q.copy(), 0.0
    reference = POSTURE_REFERENCE_Q[side]
    if inward_margin_m(assets, mounts, neutral, side, reference) < MIN_INWARD_M:
        raise RuntimeError("POSTURE_REFERENCE_DOES_NOT_SATISFY_GATE")
    low, high = 0.0, 1.0
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        candidate = (1.0 - middle) * q + middle * reference
        if inward_margin_m(assets, mounts, neutral, side, candidate) >= MIN_INWARD_M:
            high = middle
        else:
            low = middle
    return (1.0 - high) * q + high * reference, high


def apply_posture_projection(arrays: dict[str, np.ndarray], assets, mounts: np.ndarray) -> dict:
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    source_q = np.asarray(arrays["q_arm"], dtype=np.float64).copy()
    display_q = source_q.copy()
    valid = np.asarray(arrays["wrist_valid"], dtype=bool)
    alpha = np.zeros(valid.shape, dtype=np.float64)
    for frame, side in zip(*np.where(valid)):
        display_q[frame, side], alpha[frame, side] = project_one(
            assets, mounts, neutral, int(side), source_q[frame, side]
        )
    if np.any(display_q[valid] < lower[np.where(valid)[1]] - 1e-9) or np.any(
        display_q[valid] > upper[np.where(valid)[1]] + 1e-9
    ):
        raise RuntimeError("POSTURE_PROJECTION_OUT_OF_LIMITS")

    result = dict(arrays)
    result["q_arm_v5_display"] = source_q
    result["T_actual_root_v5_display"] = np.asarray(arrays["T_actual_root"]).copy()
    result["position_residual_mm_v5_display"] = np.asarray(arrays["position_residual_mm"]).copy()
    result["rotation_residual_deg_v5_display"] = np.asarray(arrays["rotation_residual_deg"]).copy()
    result["display_tolerance_pass_v5"] = np.asarray(arrays["display_tolerance_pass"]).copy()
    result["tolerance_pass_v5_display"] = np.asarray(arrays["tolerance_pass"]).copy()
    result["q_arm"] = display_q
    result["posture_projection_alpha"] = alpha
    result["posture_projected"] = alpha > 0.0
    result["posture_min_inward_m"] = np.asarray(MIN_INWARD_M)
    result["posture_reference_q_arm"] = POSTURE_REFERENCE_Q.copy()

    actual = np.asarray(arrays["T_actual_root"]).copy()
    position = np.asarray(arrays["position_residual_mm"]).copy()
    rotation = np.asarray(arrays["rotation_residual_deg"]).copy()
    for frame, side in zip(*np.where(valid)):
        actual_root = arm._tool_fk(assets, int(side), display_q[frame, side]) @ arrays["T_tool_hand"][side]
        actual[frame, side] = actual_root
        delta = np.linalg.inv(arrays["T_target_root"][frame, side]) @ actual_root
        position[frame, side] = np.linalg.norm(delta[:3, 3]) * 1000.0
        rotation[frame, side] = np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))
    result["T_actual_root"] = actual
    result["position_residual_mm"] = position
    result["rotation_residual_deg"] = rotation
    display_tolerance = (
        valid & arrays["solver_success"] & (position <= 20.0) & (rotation <= 15.0)
    )
    result["display_tolerance_pass"] = display_tolerance
    # The shared renderer reads tolerance_pass for the Chinese per-frame label.
    # Keep raw tracking in tolerance_pass_raw and make the visible label honest.
    result["tolerance_pass"] = display_tolerance.copy()
    result["quality_pass"] = display_tolerance.copy()
    times = (arrays["timestamp_ns"] - arrays["timestamp_ns"][0]).astype(float) * 1e-9
    derivatives = motion_derivatives(display_q, valid, times, arrays["time_edge_valid"])
    for name, values in derivatives.items():
        result[f"arm_{name}"] = values
    return result


def run(row: dict, assets, mounts: np.ndarray) -> dict:
    folder = OUTPUT / row["session_id"]
    result_path = folder / "RESULT.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("session_id") != row["session_id"]:
            raise RuntimeError("RESULT_IDENTITY_DRIFT")
        return result
    if folder.exists():
        raise RuntimeError(f"PARTIAL_OUTPUT:{folder}")

    source_result_path = v5.OUTPUT / row["session_id"] / "RESULT.json"
    source_result = load_json(source_result_path)
    if source_result.get("schema_version") != "CHAOYANG_ROBOT100_INWARD_SUCCESSOR_SESSION_V5":
        raise RuntimeError("V5_SCHEMA_DRIFT")
    motion_path = stable_artifact(source_result["motion"], v5.OUTPUT)
    with np.load(motion_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}

    folder.mkdir(parents=True)
    started = time.monotonic()
    projected = apply_posture_projection(arrays, assets, mounts)
    motion = folder / "ROBOT_MOTION_INWARD_POSTURE_V6.npz"
    np.savez_compressed(motion, **projected)
    video = folder / f"{row['session_id']}_上臂向外_小臂向内_全轨迹约束_Robot_V6.mp4"
    review = baseline._render(row, projected, video)

    valid = projected["wrist_valid"]
    inferred = projected["posture_projected"]
    edge = projected["time_edge_valid"][1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
    step = np.abs(np.diff(projected["q_arm"], axis=0))
    max_step = float(step[edge.repeat(7, axis=2)].max()) if edge.any() else 0.0
    result = {
        "schema_version": "CHAOYANG_ROBOT100_INWARD_POSTURE_SESSION_V6",
        "session_id": row["session_id"],
        "status": "COMPLETE_PENDING_HUMAN_REVIEW",
        "predecessor_v5": artifact_ref(source_result_path),
        "predecessor_selection_row": row,
        "posture_policy": {
            "kind": "MINIMUM_JOINT_INTERPOLATION_TO_FIXED_REFERENCE",
            "minimum_elbow_outside_wrist_m": MIN_INWARD_M,
            "bisection_steps": BISECTION_STEPS,
            "reference_q_arm": POSTURE_REFERENCE_Q.tolist(),
            "projected_frames_left_right": inferred.sum(axis=0).astype(int).tolist(),
            "max_projection_alpha_left_right": projected["posture_projection_alpha"].max(axis=0).tolist(),
        },
        "motion": artifact_ref(motion),
        "video": artifact_ref(video),
        "review": review,
        "max_display_joint_step_rad": max_step,
        "raw_arm_tolerance_pass_physical_left_right": projected["tolerance_pass_raw"].sum(axis=0).astype(int).tolist(),
        "wrist_valid_physical_left_right": valid.sum(axis=0).astype(int).tolist(),
        "elapsed_seconds": time.monotonic() - started,
        "algorithm_quality_adopted": False,
        "claim_limit": (
            "0916 offline Robot posture visualization. V5 raw IK remains the sensor-tracking record; "
            "posture-projected frames are inferred display motion and may not track the wrist target. "
            "No physical, control, calibration or training authority."
        ),
    }
    atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=2)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count <= 4:
        raise ValueError("WORKER_PARTITION")
    selection = load_json(v5.SELECTION)
    if selection.get("count") != 100:
        raise RuntimeError("PREDECESSOR_SELECTION_DRIFT")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    completed, failures = [], []
    for row in selection["rows"][args.worker_index::args.worker_count]:
        try:
            result = run(row, assets, mounts)
            completed.append({"session_id": row["session_id"], "video": result["video"]})
            print(json.dumps({"session": row["session_id"], "status": "COMPLETE", "count": len(completed)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failures.append({"session_id": row["session_id"], "type": type(exc).__name__, "error": str(exc)})
            print(json.dumps({"session": row["session_id"], "status": "FAILED", "error": str(exc)}, ensure_ascii=False), flush=True)
    atomic_json(
        OUTPUT.parent / f"WORKER_{args.worker_index}.json",
        {
            "schema_version": "CHAOYANG_ROBOT100_INWARD_POSTURE_WORKER_V6",
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "completed": completed,
            "failures": failures,
            "selection": artifact_ref(v5.SELECTION),
        },
    )
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
