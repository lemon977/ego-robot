"""Generate the final 0916 Robot100 inward-pose development successor.

The policy is fixed for the whole frozen cohort: one reachable inward reference,
0.75 translation retargeting for robot morphology, and an offline bridge only
where raw IK switches branches by more than 0.75 rad in one acquisition step.
Raw IK arrays and bridge provenance are retained beside display arrays.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as predecessor
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_inward_v3 import motion_derivatives, solve_full_chain
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

SELECTION = REPO_ROOT / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001/SELECTION.json"
OUTPUT = REPO_ROOT / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001/corrections/inward_forearm_reference_v4/successor_sessions"
REFERENCE_Q = np.asarray([
    [1.101075, -0.205275, -0.670650, -1.6747875, -0.150975, 0.265050, 0.426375],
    [-1.101075, -0.205275, 0.670650, -1.6747875, 0.150975, 0.264975, -0.426375],
], dtype=np.float64)
TRANSLATION_SCALE = 0.75
BRANCH_JUMP_RAD = 0.75
BRIDGE_RADIUS_FRAMES = 8


def scaled_motion(arrays: dict) -> dict:
    result = dict(arrays)
    relative = np.asarray(arrays["relative_wrist_T"], dtype=float).copy()
    relative[..., :3, 3] *= TRANSLATION_SCALE
    result["relative_wrist_T"] = relative
    return result


def bridge_branch_jumps(q: np.ndarray, valid: np.ndarray, edge: np.ndarray):
    raw = np.asarray(q, dtype=float)
    display = raw.copy()
    inferred = np.zeros(raw.shape[:2], dtype=bool)
    windows: list[dict] = []
    count = len(raw)
    for side in range(2):
        step = np.max(np.abs(np.diff(raw[:, side], axis=0)), axis=1)
        points = np.flatnonzero(
            (step > BRANCH_JUMP_RAD)
            & edge[1:]
            & valid[1:, side]
            & valid[:-1, side]
        ) + 1
        candidates: list[tuple[int, int]] = []
        for point in points:
            segment_start = int(point - 1)
            while (
                segment_start > 0
                and edge[segment_start]
                and valid[segment_start - 1, side]
            ):
                segment_start -= 1
            segment_end = int(point)
            while (
                segment_end + 1 < count
                and edge[segment_end + 1]
                and valid[segment_end + 1, side]
            ):
                segment_end += 1
            start = max(segment_start, int(point) - BRIDGE_RADIUS_FRAMES)
            end = min(segment_end, int(point) + BRIDGE_RADIUS_FRAMES)
            if end - start >= 2:
                if candidates and start <= candidates[-1][1]:
                    candidates[-1] = (candidates[-1][0], max(candidates[-1][1], end))
                else:
                    candidates.append((start, end))
        for start, end in candidates:
            span = end - start
            for frame in range(start, end + 1):
                alpha = (frame - start) / span
                display[frame, side] = raw[start, side] + alpha * (raw[end, side] - raw[start, side])
            inferred[start + 1:end, side] = True
            windows.append({"physical_side": side, "start_frame": start, "end_frame": end})
    return display, inferred, windows


def apply_display_motion(solved: dict, assets, mounts) -> list[dict]:
    raw_q = solved["q_arm"].copy()
    display_q, inferred, windows = bridge_branch_jumps(
        raw_q, solved["wrist_valid"], solved["time_edge_valid"]
    )
    solved["q_arm_raw"] = raw_q
    solved["T_actual_root_raw"] = solved["T_actual_root"].copy()
    solved["position_residual_mm_raw"] = solved["position_residual_mm"].copy()
    solved["rotation_residual_deg_raw"] = solved["rotation_residual_deg"].copy()
    solved["tolerance_pass_raw"] = solved["tolerance_pass"].copy()
    for name in ("velocity", "acceleration", "jerk", "edge_valid"):
        solved[f"arm_raw_{name}"] = solved[f"arm_{name}"].copy()
    solved["q_arm"] = display_q
    solved["display_inferred"] = inferred
    solved["motion_translation_scale"] = np.asarray(TRANSLATION_SCALE)
    for frame, side in zip(*np.where(inferred)):
        actual = arm._tool_fk(assets, side, display_q[frame, side]) @ solved["T_tool_hand"][side]
        solved["T_actual_root"][frame, side] = actual
        delta = np.linalg.inv(solved["T_target_root"][frame, side]) @ actual
        solved["position_residual_mm"][frame, side] = np.linalg.norm(delta[:3, 3]) * 1000.0
        solved["rotation_residual_deg"][frame, side] = np.degrees(
            np.linalg.norm(arm._rotation_vector(delta[:3, :3]))
        )
    display_tolerance = (
        solved["wrist_valid"]
        & solved["solver_success"]
        & (solved["position_residual_mm"] <= 20.0)
        & (solved["rotation_residual_deg"] <= 15.0)
    )
    solved["display_tolerance_pass"] = display_tolerance
    times = (solved["timestamp_ns"] - solved["timestamp_ns"][0]).astype(float) * 1e-9
    derivatives = motion_derivatives(
        display_q, solved["wrist_valid"], times, solved["time_edge_valid"]
    )
    for name, values in derivatives.items():
        solved[f"arm_{name}"] = values
    return windows


def run(row: dict, assets, mounts, hand_eval: int, arm_eval: int) -> dict:
    folder = OUTPUT / row["session_id"]
    result_path = folder / "RESULT.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("session_id") != row["session_id"]:
            raise RuntimeError("RESULT_IDENTITY_DRIFT")
        return result
    if folder.exists():
        raise RuntimeError(f"PARTIAL_OUTPUT:{folder}")
    folder.mkdir(parents=True)
    started = time.monotonic()
    original, sensor = predecessor._load_motion(row, hand_eval)
    arrays = scaled_motion(original)
    solved = solve_full_chain(
        arrays,
        assets,
        mounts,
        max_nfev=arm_eval,
        continuity_weight=.5,
        reference_q_arm=REFERENCE_Q,
    )
    windows = apply_display_motion(solved, assets, mounts)
    solved["source_video_frame_idx"] = original["source_video_frame_idx"].copy()
    motion = folder / "ROBOT_MOTION_INWARD_V4.npz"
    np.savez_compressed(motion, **solved)
    video = folder / f"{row['session_id']}_上臂向外_小臂向内_Robot_V4.mp4"
    review = predecessor._render(row, solved, video)
    valid = solved["wrist_valid"]
    raw_tolerance = solved["tolerance_pass_raw"]
    display_step = np.abs(np.diff(solved["q_arm"], axis=0))
    eligible = (
        solved["time_edge_valid"][1:, None, None]
        & valid[1:, :, None]
        & valid[:-1, :, None]
    )
    max_display_step = float(display_step[eligible.repeat(7, axis=2)].max()) if eligible.any() else 0.0
    result = {
        "schema_version": "CHAOYANG_ROBOT100_INWARD_SUCCESSOR_SESSION_V4",
        "session_id": row["session_id"],
        "status": "COMPLETE_PENDING_HUMAN_REVIEW",
        "predecessor_selection_row": row,
        "reference_q_arm": REFERENCE_Q.tolist(),
        "reference_policy": "FIXED_75_PERCENT_INWARD_FROM_OLD_MIDPOINT_FOR_ALL_SESSIONS",
        "translation_retarget_scale": TRANSLATION_SCALE,
        "branch_bridge_policy": {
            "jump_threshold_rad": BRANCH_JUMP_RAD,
            "radius_frames": BRIDGE_RADIUS_FRAMES,
            "windows": windows,
            "display_inferred_left_right": solved["display_inferred"].sum(axis=0).astype(int).tolist(),
        },
        "motion": artifact_ref(motion),
        "video": artifact_ref(video),
        "review": review,
        "sensor_metrics": sensor,
        "wrist_valid_physical_left_right": valid.sum(axis=0).astype(int).tolist(),
        "raw_arm_tolerance_pass_physical_left_right": raw_tolerance.sum(axis=0).astype(int).tolist(),
        "max_display_joint_step_rad": max_display_step,
        "elapsed_seconds": time.monotonic() - started,
        "algorithm_quality_adopted": False,
        "claim_limit": (
            "0916 sensor-driven fixed-policy offline virtual Robot visualization. "
            "Bridge frames are inferred display motion, not exact sensor tracking; "
            "no physical/control/training authority."
        ),
    }
    atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--max-hand-evaluations", type=int, default=24)
    parser.add_argument("--max-arm-evaluations", type=int, default=60)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count <= 2:
        raise ValueError("WORKER_PARTITION")
    selection = load_json(SELECTION)
    if selection.get("count") != 100:
        raise RuntimeError("PREDECESSOR_SELECTION_DRIFT")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = predecessor._mounts()
    completed = []
    failures = []
    for row in selection["rows"][args.worker_index::args.worker_count]:
        try:
            result = run(row, assets, mounts, args.max_hand_evaluations, args.max_arm_evaluations)
            completed.append({"session_id": row["session_id"], "video": result["video"]})
            print(json.dumps({"session": row["session_id"], "status": "COMPLETE", "count": len(completed)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failures.append({"session_id": row["session_id"], "type": type(exc).__name__, "error": str(exc)})
            print(json.dumps({"session": row["session_id"], "status": "FAILED", "error": str(exc)}, ensure_ascii=False), flush=True)
    atomic_json(
        OUTPUT.parent / f"WORKER_{args.worker_index}.json",
        {
            "schema_version": "CHAOYANG_ROBOT100_INWARD_SUCCESSOR_WORKER_V4",
            "worker_index": args.worker_index,
            "worker_count": args.worker_count,
            "completed": completed,
            "failures": failures,
            "selection": artifact_ref(SELECTION),
        },
    )
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
