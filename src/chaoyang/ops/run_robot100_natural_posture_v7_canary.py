"""Render fixed V7 canaries with an added vertical arm-shape constraint.

V6 remains the complete 100-session baseline.  V7 is a non-adopted canary that
starts from V6 display motion and minimally interpolates only frames whose elbow
is more than 20 mm above its shoulder (or whose wrist is more than 50 mm above
the shoulder) toward the same fixed bilateral reference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_inward_successor_v6 as v6
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_inward_v3 import motion_derivatives
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets

OUTPUT = v6.OUTPUT.parent.parent / "inward_natural_posture_v7" / "canaries"
DEFAULT_SESSIONS = (
    "play_cards_0916_025",
    "play_cards_0916_019",
    "play_cards_0916_009",
    "get_potato_chips_0916_035",
)
MIN_UPPER_OUTWARD_M = 0.05
MIN_FOREARM_INWARD_M = 0.01
MIN_ELBOW_DROP_M = -0.02
MIN_WRIST_DROP_M = -0.05
BISECTION_STEPS = 18


def arm_shape_metrics(assets, mounts: np.ndarray, neutral: np.ndarray, side: int, q: np.ndarray) -> np.ndarray:
    joint_map = {
        name: float(neutral[physical_side, index])
        for physical_side in range(2)
        for index, name in enumerate(ARM_JOINT_NAMES[physical_side])
    }
    joint_map.update({name: float(value) for name, value in zip(ARM_JOINT_NAMES[side], q, strict=True)})
    fk = forward_kinematics(assets.tianji, joint_map)
    suffix = ("L", "R")[side]
    sign = 1.0 if side == 0 else -1.0
    shoulder = fk[f"Link2_{suffix}"][:3, 3]
    elbow = fk[f"Link3_{suffix}"][:3, 3]
    wrist = (fk[f"flange_{suffix}"] @ mounts[side])[:3, 3]
    return np.asarray(
        [
            sign * (elbow[1] - shoulder[1]),
            sign * (elbow[1] - wrist[1]),
            shoulder[2] - elbow[2],
            shoulder[2] - wrist[2],
        ],
        dtype=np.float64,
    )


def shape_pass(metrics: np.ndarray) -> bool:
    return bool(
        metrics[0] >= MIN_UPPER_OUTWARD_M
        and metrics[1] >= MIN_FOREARM_INWARD_M
        and metrics[2] >= MIN_ELBOW_DROP_M
        and metrics[3] >= MIN_WRIST_DROP_M
    )


def project_one(assets, mounts, neutral, side: int, q: np.ndarray) -> tuple[np.ndarray, float]:
    q = np.asarray(q, dtype=np.float64)
    if shape_pass(arm_shape_metrics(assets, mounts, neutral, side, q)):
        return q.copy(), 0.0
    reference = v6.POSTURE_REFERENCE_Q[side]
    if not shape_pass(arm_shape_metrics(assets, mounts, neutral, side, reference)):
        raise RuntimeError("NATURAL_REFERENCE_DOES_NOT_SATISFY_GATE")
    low, high = 0.0, 1.0
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        candidate = (1.0 - middle) * q + middle * reference
        if shape_pass(arm_shape_metrics(assets, mounts, neutral, side, candidate)):
            high = middle
        else:
            low = middle
    return (1.0 - high) * q + high * reference, high


def apply_projection(arrays: dict[str, np.ndarray], assets, mounts: np.ndarray) -> dict[str, np.ndarray]:
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    source_q = np.asarray(arrays["q_arm"], dtype=np.float64).copy()
    display_q = source_q.copy()
    valid = np.asarray(arrays["wrist_valid"], dtype=bool)
    alpha = np.zeros(valid.shape, dtype=np.float64)
    metrics = np.full((*valid.shape, 4), np.nan, dtype=np.float64)
    for frame, side in zip(*np.where(valid)):
        display_q[frame, side], alpha[frame, side] = project_one(
            assets, mounts, neutral, int(side), source_q[frame, side]
        )
        metrics[frame, side] = arm_shape_metrics(assets, mounts, neutral, int(side), display_q[frame, side])
        if not shape_pass(metrics[frame, side]):
            raise RuntimeError(f"NATURAL_PROJECTION_GATE:{frame}:{side}")
    if np.any(display_q[valid] < lower[np.where(valid)[1]] - 1e-9) or np.any(
        display_q[valid] > upper[np.where(valid)[1]] + 1e-9
    ):
        raise RuntimeError("NATURAL_PROJECTION_OUT_OF_LIMITS")
    result = dict(arrays)
    result["q_arm_v6_display"] = source_q
    result["T_actual_root_v6_display"] = np.asarray(arrays["T_actual_root"]).copy()
    result["position_residual_mm_v6_display"] = np.asarray(arrays["position_residual_mm"]).copy()
    result["rotation_residual_deg_v6_display"] = np.asarray(arrays["rotation_residual_deg"]).copy()
    result["display_tolerance_pass_v6"] = np.asarray(arrays["display_tolerance_pass"]).copy()
    result["q_arm"] = display_q
    result["natural_projection_alpha"] = alpha
    result["natural_projected"] = alpha > 0.0
    result["arm_shape_metrics_m"] = metrics
    result["arm_shape_minimums_m"] = np.asarray(
        [MIN_UPPER_OUTWARD_M, MIN_FOREARM_INWARD_M, MIN_ELBOW_DROP_M, MIN_WRIST_DROP_M]
    )
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
    display_tolerance = valid & arrays["solver_success"] & (position <= 20.0) & (rotation <= 15.0)
    result["display_tolerance_pass"] = display_tolerance
    result["tolerance_pass"] = display_tolerance.copy()
    result["quality_pass"] = display_tolerance.copy()
    times = (arrays["timestamp_ns"] - arrays["timestamp_ns"][0]).astype(float) * 1e-9
    for name, values in motion_derivatives(display_q, valid, times, arrays["time_edge_valid"]).items():
        result[f"arm_{name}"] = values
    return result


def run(session: str, row: dict, assets, mounts: np.ndarray) -> dict:
    folder = OUTPUT / session
    if folder.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{folder}")
    predecessor_result_path = v6.OUTPUT / session / "RESULT.json"
    predecessor = load_json(predecessor_result_path)
    motion_path = v6.stable_artifact(predecessor["motion"], v6.OUTPUT)
    with np.load(motion_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    folder.mkdir(parents=True)
    started = time.monotonic()
    projected = apply_projection(arrays, assets, mounts)
    motion = folder / "ROBOT_MOTION_NATURAL_POSTURE_V7.npz"
    np.savez_compressed(motion, **projected)
    video = folder / f"{session}_上臂外展_小臂内收_肘不过肩_Robot_V7_CANARY.mp4"
    review = baseline._render(row, projected, video)
    valid = projected["wrist_valid"].astype(bool)
    edge = projected["time_edge_valid"][1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
    delta = np.abs(np.diff(projected["q_arm"], axis=0))
    max_step = float(delta[edge.repeat(7, axis=2)].max()) if edge.any() else 0.0
    result = {
        "schema_version": "CHAOYANG_ROBOT100_NATURAL_POSTURE_CANARY_SESSION_V7",
        "session_id": session,
        "status": "CANARY_COMPLETE_PENDING_HUMAN_REVIEW",
        "predecessor_v6": artifact_ref(predecessor_result_path),
        "motion": artifact_ref(motion),
        "video": artifact_ref(video),
        "review": review,
        "natural_projected_left_right": projected["natural_projected"].sum(axis=0).astype(int).tolist(),
        "maximum_projection_alpha_left_right": projected["natural_projection_alpha"].max(axis=0).tolist(),
        "maximum_display_joint_step_rad": max_step,
        "maximum_display_position_residual_mm_left_right": np.nanmax(projected["position_residual_mm"], axis=0).tolist(),
        "maximum_display_rotation_residual_deg_left_right": np.nanmax(projected["rotation_residual_deg"], axis=0).tolist(),
        "algorithm_quality_adopted": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(folder / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="*", default=list(DEFAULT_SESSIONS))
    args = parser.parse_args()
    selection = load_json(v6.v5.SELECTION)
    rows = {row["session_id"]: row for row in selection["rows"]}
    unknown = sorted(set(args.sessions) - rows.keys())
    if unknown:
        raise RuntimeError(f"UNKNOWN_SESSION:{unknown}")
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [run(session, rows[session], assets, mounts) for session in args.sessions]
    atomic_json(OUTPUT.parent / "RESULT.json", {
        "schema_version": "CHAOYANG_ROBOT100_NATURAL_POSTURE_CANARY_RESULT_V7",
        "status": "CANARIES_COMPLETE_PENDING_HUMAN_REVIEW",
        "count": len(results),
        "sessions": [result["session_id"] for result in results],
        "results": [artifact_ref(OUTPUT / result["session_id"] / "RESULT.json") for result in results],
        "algorithm_quality_adopted": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
