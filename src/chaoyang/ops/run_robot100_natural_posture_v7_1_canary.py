"""Render V7.1 canaries with a temporal majorant over required projection."""
from __future__ import annotations

import argparse
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_inward_successor_v6 as v6
from chaoyang.ops import run_robot100_natural_posture_v7_canary as v7
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_inward_v3 import motion_derivatives
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

OUTPUT = v6.OUTPUT.parent.parent / "inward_natural_posture_v7_1" / "canaries"
ALPHA_MAX_CHANGE_PER_FRAME = 0.03


def temporal_majorant(required: np.ndarray, valid: np.ndarray, time_edge_valid: np.ndarray, slope: float) -> np.ndarray:
    if required.shape != valid.shape or required.ndim != 2:
        raise ValueError("PROJECTION_SHAPE")
    if not 0.0 < slope <= 1.0:
        raise ValueError("PROJECTION_SLOPE")
    result = np.asarray(required, dtype=np.float64).copy()
    frames, sides = result.shape
    for side in range(sides):
        start = 0
        while start < frames:
            while start < frames and not valid[start, side]:
                start += 1
            if start >= frames:
                break
            end = start + 1
            while end < frames and valid[end, side] and bool(time_edge_valid[end]):
                end += 1
            segment = result[start:end, side]
            for index in range(1, len(segment)):
                segment[index] = max(segment[index], segment[index - 1] - slope)
            for index in range(len(segment) - 2, -1, -1):
                segment[index] = max(segment[index], segment[index + 1] - slope)
            result[start:end, side] = segment
            start = end
    result[~valid] = 0.0
    return np.clip(result, 0.0, 1.0)


def apply_projection(arrays: dict[str, np.ndarray], assets, mounts: np.ndarray) -> dict[str, np.ndarray]:
    base = v7.apply_projection(arrays, assets, mounts)
    valid = np.asarray(arrays["wrist_valid"], dtype=bool)
    required = np.asarray(base["natural_projection_alpha"], dtype=np.float64)
    alpha = temporal_majorant(required, valid, arrays["time_edge_valid"], ALPHA_MAX_CHANGE_PER_FRAME)
    source_q = np.asarray(base["q_arm_v6_display"], dtype=np.float64)
    reference = v6.POSTURE_REFERENCE_Q[None, :, :]
    display_q = (1.0 - alpha[:, :, None]) * source_q + alpha[:, :, None] * reference
    result = dict(base)
    result["q_arm"] = display_q
    result["natural_projection_alpha_required"] = required
    result["natural_projection_alpha"] = alpha
    result["natural_projected"] = alpha > 0.0
    result["natural_temporal_expanded"] = alpha > required + 1e-12
    lower, upper = arm._arm_limits(assets)
    neutral = 0.5 * (lower + upper)
    metrics = np.full((*valid.shape, 4), np.nan, dtype=np.float64)
    actual = np.asarray(arrays["T_actual_root"]).copy()
    position = np.asarray(arrays["position_residual_mm"]).copy()
    rotation = np.asarray(arrays["rotation_residual_deg"]).copy()
    for frame, side in zip(*np.where(valid)):
        metrics[frame, side] = v7.arm_shape_metrics(assets, mounts, neutral, int(side), display_q[frame, side])
        if not v7.shape_pass(metrics[frame, side]):
            raise RuntimeError(f"TEMPORAL_PROJECTION_GATE:{frame}:{side}")
        actual_root = arm._tool_fk(assets, int(side), display_q[frame, side]) @ arrays["T_tool_hand"][side]
        actual[frame, side] = actual_root
        delta = np.linalg.inv(arrays["T_target_root"][frame, side]) @ actual_root
        position[frame, side] = np.linalg.norm(delta[:3, 3]) * 1000.0
        rotation[frame, side] = np.degrees(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))
    result["arm_shape_metrics_m"] = metrics
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
    predecessor_path = v6.OUTPUT / session / "RESULT.json"
    predecessor = load_json(predecessor_path)
    motion_path = v6.stable_artifact(predecessor["motion"], v6.OUTPUT)
    with np.load(motion_path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    folder.mkdir(parents=True)
    started = time.monotonic()
    projected = apply_projection(arrays, assets, mounts)
    motion = folder / "ROBOT_MOTION_NATURAL_POSTURE_V7_1.npz"
    np.savez_compressed(motion, **projected)
    video = folder / f"{session}_自然内收_时间连续_Robot_V7_1_CANARY.mp4"
    review = baseline._render(row, projected, video)
    valid = projected["wrist_valid"].astype(bool)
    edge = projected["time_edge_valid"][1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
    delta = np.abs(np.diff(projected["q_arm"], axis=0))
    max_step = float(delta[edge.repeat(7, axis=2)].max()) if edge.any() else 0.0
    result = {
        "schema_version": "CHAOYANG_ROBOT100_NATURAL_POSTURE_CANARY_SESSION_V7_1",
        "session_id": session,
        "status": "CANARY_COMPLETE_PENDING_HUMAN_REVIEW",
        "predecessor_v6": artifact_ref(predecessor_path),
        "motion": artifact_ref(motion),
        "video": artifact_ref(video),
        "review": review,
        "required_projected_left_right": (projected["natural_projection_alpha_required"] > 0).sum(axis=0).astype(int).tolist(),
        "display_projected_left_right": projected["natural_projected"].sum(axis=0).astype(int).tolist(),
        "temporal_expanded_left_right": projected["natural_temporal_expanded"].sum(axis=0).astype(int).tolist(),
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
    parser.add_argument("--sessions", nargs="*", default=list(v7.DEFAULT_SESSIONS))
    args = parser.parse_args()
    selection = load_json(v6.v5.SELECTION)
    rows = {row["session_id"]: row for row in selection["rows"]}
    if not set(args.sessions).issubset(rows):
        raise RuntimeError("UNKNOWN_SESSION")
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = [run(session, rows[session], assets, mounts) for session in args.sessions]
    atomic_json(OUTPUT.parent / "RESULT.json", {
        "schema_version": "CHAOYANG_ROBOT100_NATURAL_POSTURE_CANARY_RESULT_V7_1",
        "status": "CANARIES_COMPLETE_PENDING_HUMAN_REVIEW",
        "count": len(results),
        "results": [artifact_ref(OUTPUT / result["session_id"] / "RESULT.json") for result in results],
        "algorithm_quality_adopted": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
