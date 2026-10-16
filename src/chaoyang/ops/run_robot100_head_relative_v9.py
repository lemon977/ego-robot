"""0916 Robot100 head-relative shared-axis trajectory correction canary.

V8.1 fixed anatomical left/right, but still expressed displacement through
each wrist's initial local axes and ignored the tracked HMD pose.  This runner
uses T_head_wrist = inv(T_world_head) @ T_world_wrist, then maps the shared
PICO head axes (X-forward/Y-left/Z-up) directly into the robot base axes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as baseline
from chaoyang.ops import run_robot100_anatomical_identity_v8 as v8
from chaoyang.ops import run_robot100_inward_successor_v4 as v4
from chaoyang.ops import run_robot100_inward_successor_v6 as v6
from chaoyang.pipeline.full_robot_review_inward_v3 import solve_full_chain, validate_se3
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


ROOT = (
    REPO_ROOT
    / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001"
    / "corrections/head_relative_shared_axes_v9"
)
OUTPUT = ROOT / "canaries"
TRANSLATION_SCALE = 0.75
ROBOT_FROM_HEAD_R = np.eye(3, dtype=np.float64)


def load_head_relative_motion(row: dict, hand_eval: int):
    original, sensor_metrics = baseline._load_motion(row, hand_eval)
    arrays = v8.restore_anatomical_identity(original)
    hdf5_path = baseline._stable_ref(Path(row["hdf5"]["path"]), row["hdf5"])
    with h5py.File(hdf5_path, "r") as source:
        head_world = baseline._pose_matrix(np.asarray(source["head_pose"], dtype=np.float64))
        anatomical_world = np.stack(
            [
                baseline._pose_matrix(np.asarray(source["left_wrist_pose"], dtype=np.float64)),
                baseline._pose_matrix(np.asarray(source["right_wrist_pose"], dtype=np.float64)),
            ],
            axis=1,
        )
    if len(head_world) != len(arrays["frame_id"]) or anatomical_world.shape[:2] != arrays["wrist_valid"].shape:
        raise RuntimeError("HEAD_WRIST_TIME_OR_SIDE_SHAPE_DRIFT")
    validate_se3(head_world)
    validate_se3(anatomical_world[arrays["wrist_valid"]])
    head_relative = np.linalg.inv(head_world)[:, None] @ anatomical_world
    validate_se3(head_relative[arrays["wrist_valid"]])
    arrays["head_world_T"] = head_world
    arrays["head_relative_wrist_T"] = head_relative
    sensor_metrics = dict(sensor_metrics)
    sensor_metrics.update(
        {
            "trajectory_frame": "PICO_HEAD_RELATIVE",
            "source_axes": "right-handed X-forward Y-left Z-up",
            "robot_axes": "right-handed X-forward Y-left Z-up",
            "translation_scale": TRANSLATION_SCALE,
        }
    )
    return arrays, sensor_metrics


def path_mm(transforms: np.ndarray, valid: np.ndarray) -> list[float]:
    values = []
    for side in range(2):
        indices = np.flatnonzero(valid[:, side])
        xyz = transforms[indices, side, :3, 3]
        values.append(float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() * 1000.0))
    return values


def displacement_mm(transforms: np.ndarray, valid: np.ndarray) -> list[list[float]]:
    values = []
    for side in range(2):
        indices = np.flatnonzero(valid[:, side])
        delta = (transforms[indices[-1], side, :3, 3] - transforms[indices[0], side, :3, 3]) * 1000.0
        values.append(delta.astype(float).tolist())
    return values


def run(row: dict, assets, mounts: np.ndarray, hand_eval: int, arm_eval: int) -> dict:
    folder = OUTPUT / row["session_id"]
    result_path = folder / "RESULT.json"
    if result_path.is_file():
        return load_json(result_path)
    if folder.exists():
        raise RuntimeError(f"PARTIAL_OUTPUT:{folder}")

    arrays, sensor_metrics = load_head_relative_motion(row, hand_eval)
    started = time.monotonic()
    solved = solve_full_chain(
        arrays,
        assets,
        mounts,
        max_nfev=arm_eval,
        continuity_weight=0.5,
        reference_q_arm=v4.REFERENCE_Q,
        expected_human_to_physical=v8.IDENTITY_MAPPING,
        target_mode="HEAD_RELATIVE_SHARED_AXES",
        robot_from_head_R=ROBOT_FROM_HEAD_R,
        translation_scale=TRANSLATION_SCALE,
    )
    raw_source_map = np.asarray(arrays["source_video_frame_idx"], dtype=np.int64).copy()
    solved["source_video_frame_idx_raw"] = raw_source_map
    solved["source_video_frame_idx"] = np.clip(raw_source_map, 0, int(row["frames"]) - 1)
    solved["head_world_T"] = arrays["head_world_T"].copy()
    solved["head_relative_wrist_T"] = arrays["head_relative_wrist_T"].copy()

    valid = solved["wrist_valid"]
    posture_fail = [0, 0]
    for frame, side in zip(*np.where(valid)):
        if v6.inward_margin_m(assets, mounts, solved["neutral_q_arm"], int(side), solved["q_arm"][frame, side]) < v6.MIN_INWARD_M:
            posture_fail[int(side)] += 1
    folder.mkdir(parents=True)
    motion = folder / "ROBOT_MOTION_HEAD_RELATIVE_V9.npz"
    np.savez_compressed(motion, **solved)
    video = folder / f"{row['session_id']}_头显相对共享坐标_Robot_V9_CANARY.mp4"
    review = baseline._render(
        row,
        solved,
        video,
        camera_yaw_degrees=-140.0,
        view_description="佩戴者同视向后视角；头显相对共享坐标轨迹",
    )

    edge = solved["time_edge_valid"][1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
    step = np.abs(np.diff(solved["q_arm"], axis=0))
    max_step = float(step[edge.repeat(7, axis=2)].max()) if edge.any() else 0.0
    result = {
        "schema_version": "CHAOYANG_ROBOT100_HEAD_RELATIVE_SESSION_V9",
        "session_id": row["session_id"],
        "status": "CANARY_COMPLETE_PENDING_HUMAN_TRAJECTORY_REVIEW",
        "predecessor_v8_1": artifact_ref(
            v8.OUTPUT / row["session_id"] / "RESULT.json"
        ),
        "trajectory_contract": {
            "formula": "T_head_wrist = inv(T_world_head) @ T_world_wrist",
            "translation": "p_robot = p_reference + 0.75 * R_robot_head @ (p_head_wrist - p0_head_wrist)",
            "orientation": "R_robot = R_robot_head @ (R_head_wrist @ R0_head_wrist.T) @ R_robot_head.T @ R_reference",
            "robot_from_head_R": ROBOT_FROM_HEAD_R.tolist(),
            "posthoc_posture_projection": False,
            "branch_bridge": False,
        },
        "head_relative_path_mm_left_right": path_mm(arrays["head_relative_wrist_T"], valid),
        "robot_target_path_mm_left_right": path_mm(solved["T_target_root"], valid),
        "head_relative_first_last_mm_left_right": displacement_mm(arrays["head_relative_wrist_T"], valid),
        "robot_target_first_last_mm_left_right": displacement_mm(solved["T_target_root"], valid),
        "raw_tracking_pass_left_right": solved["tolerance_pass"].sum(axis=0).astype(int).tolist(),
        "wrist_valid_left_right": valid.sum(axis=0).astype(int).tolist(),
        "posture_gate_fail_left_right": posture_fail,
        "maximum_joint_step_rad": max_step,
        "sensor_metrics": sensor_metrics,
        "motion": artifact_ref(motion),
        "video": artifact_ref(video),
        "review": review,
        "elapsed_seconds": time.monotonic() - started,
        "baseline_selected": False,
        "algorithm_quality_adopted": False,
        "claim_limit": "Single-session head-relative trajectory correction canary; visual trajectory review required.",
    }
    atomic_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=["play_cards_0916_025"])
    parser.add_argument("--max-hand-evaluations", type=int, default=24)
    parser.add_argument("--max-arm-evaluations", type=int, default=80)
    args = parser.parse_args()
    selection = load_json(v4.SELECTION)
    rows = {row["session_id"]: row for row in selection["rows"]}
    if any(name not in rows for name in args.sessions):
        raise RuntimeError("SESSION_SELECTION_INVALID")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assets = load_pinned_robot_assets(REPO_ROOT)
    mounts, _ = baseline._mounts()
    results = [run(rows[name], assets, mounts, args.max_hand_evaluations, args.max_arm_evaluations) for name in args.sessions]
    atomic_json(
        ROOT / "RESULT.json",
        {
            "schema_version": "CHAOYANG_ROBOT100_HEAD_RELATIVE_CANARY_RESULT_V9",
            "status": "CANARIES_COMPLETE_PENDING_HUMAN_TRAJECTORY_REVIEW",
            "sessions": args.sessions,
            "results": [artifact_ref(OUTPUT / name / "RESULT.json") for name in args.sessions],
            "baseline_selected": False,
        },
    )
    print(json.dumps({"status": "COMPLETE", "count": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
