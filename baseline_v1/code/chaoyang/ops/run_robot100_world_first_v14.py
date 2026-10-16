"""0916 Robot100 world-first unit-gain canary with corrected anatomical sides.

The target construction borrows only the proven motion semantics from the
0902 Exact78 baseline.  It never reads its images, motion arrays, placement, or
results: 0916 PICO/MANUS remains the sole input.
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
from chaoyang.pipeline.exact78_style_inward_solver_v1 import solve_inward_multiseed
from chaoyang.pipeline.full_robot_review_inward_v3 import solve_full_chain, validate_se3
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


ROOT = (
    REPO_ROOT
    / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001"
    / "corrections/world_first_unit_gain_v14"
)
OUTPUT = ROOT / "canaries"
ROBOT_FROM_WORLD_R = np.eye(3, dtype=np.float64)


def load_world_first_motion(row: dict, hand_eval: int):
    original, sensor_metrics = baseline._load_motion(row, hand_eval)
    arrays = v8.restore_anatomical_identity(original)
    hdf5_path = baseline._stable_ref(Path(row["hdf5"]["path"]), row["hdf5"])
    with h5py.File(hdf5_path, "r") as source:
        raw_controller_world = np.stack(
            [
                baseline._pose_matrix(np.asarray(source["left_wrist_pose"], dtype=np.float64)),
                baseline._pose_matrix(np.asarray(source["right_wrist_pose"], dtype=np.float64)),
            ],
            axis=1,
        )
    anatomical_world = raw_controller_world[:, [1, 0]].copy()
    if anatomical_world.shape[:2] != arrays["wrist_valid"].shape:
        raise RuntimeError("WORLD_WRIST_TIME_OR_SIDE_SHAPE_DRIFT")
    validate_se3(anatomical_world[arrays["wrist_valid"]])
    arrays["world_wrist_T"] = anatomical_world
    sensor_metrics = dict(sensor_metrics)
    sensor_metrics.update(
        {
            "trajectory_frame": "PICO_WORLD_FIRST",
            "source_axes": "right-handed X-forward Y-left Z-up",
            "robot_axes": "right-handed X-forward Y-left Z-up",
            "translation_gain": 1.0,
            "borrowed_method_only": "0902 Exact78 world-first target semantics",
            "borrowed_data": False,
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

    arrays, sensor_metrics = load_world_first_motion(row, hand_eval)
    started = time.monotonic()
    solved = solve_full_chain(
        arrays,
        assets,
        mounts,
        max_nfev=arm_eval,
        continuity_weight=0.01,
        reference_q_arm=v6.POSTURE_REFERENCE_Q,
        expected_human_to_physical=v8.IDENTITY_MAPPING,
        target_mode="WORLD_FIRST_UNIT_GAIN",
        robot_from_head_R=ROBOT_FROM_WORLD_R,
        translation_scale=1.0,
    )
    solved = solve_inward_multiseed(
        solved,
        assets,
        mounts,
        v6.inward_margin_m,
        minimum_inward_m=v6.MIN_INWARD_M,
        max_nfev=arm_eval,
        continuity_weight=0.01,
    )
    raw_source_map = np.asarray(arrays["source_video_frame_idx"], dtype=np.int64).copy()
    solved["source_video_frame_idx_raw"] = raw_source_map
    solved["source_video_frame_idx"] = np.clip(raw_source_map, 0, int(row["frames"]) - 1)
    solved["world_wrist_T"] = arrays["world_wrist_T"].copy()

    valid = solved["wrist_valid"]
    posture_fail = [0, 0]
    for frame, side in zip(*np.where(valid)):
        if v6.inward_margin_m(assets, mounts, solved["neutral_q_arm"], int(side), solved["q_arm"][frame, side]) < v6.MIN_INWARD_M:
            posture_fail[int(side)] += 1
    folder.mkdir(parents=True)
    motion = folder / "ROBOT_MOTION_WORLD_FIRST_UNIT_GAIN_V14.npz"
    np.savez_compressed(motion, **solved)
    video = folder / f"{row['session_id']}_世界坐标优先单位增益_Robot_V14_CANARY.mp4"
    review = baseline._render(
        row,
        solved,
        video,
        camera_yaw_degrees=-140.0,
        view_description="佩戴者同视向后视角；世界坐标优先、单位增益轨迹",
    )

    edge = solved["time_edge_valid"][1:, None, None] & valid[1:, :, None] & valid[:-1, :, None]
    step = np.abs(np.diff(solved["q_arm"], axis=0))
    max_step = float(step[edge.repeat(7, axis=2)].max()) if edge.any() else 0.0
    result = {
        "schema_version": "CHAOYANG_ROBOT100_WORLD_FIRST_SESSION_V14",
        "session_id": row["session_id"],
        "status": "CANARY_COMPLETE_PENDING_HUMAN_TRAJECTORY_REVIEW",
        "predecessor_v13": artifact_ref(REPO_ROOT / "_run/current/robot100_world_first_unit_gain_v13_20260929/attempts/attempt_0001/canaries" / row["session_id"] / "RESULT.json"),
        "trajectory_contract": {
            "source": "0916 PICO calibrated wrist pose only; legacy controller slots restored to anatomical L/R",
            "borrowed_data_from_0902": False,
            "fixed_robot_base": True,
            "anchor": "first valid wrist observation independently per anatomical side",
            "translation": "p_robot = p_reference + R_robot_world @ (p_world_wrist - p0_world_wrist)",
            "translation_gain": 1.0,
            "orientation": "R_robot = R_robot_world @ (R_world_wrist @ R0_world_wrist.T) @ R_robot_world.T @ R_reference",
            "posthoc_posture_projection": False,
            "branch_bridge": False,
        },
        "world_wrist_path_mm_left_right": path_mm(arrays["world_wrist_T"], valid),
        "robot_target_path_mm_left_right": path_mm(solved["T_target_root"], valid),
        "world_wrist_first_last_mm_left_right": displacement_mm(arrays["world_wrist_T"], valid),
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
        "claim_limit": "Single-session 0916 world-first canary; visual trajectory review required.",
    }
    atomic_json(result_path, result)
    return result


def main() -> int:
    global ROOT, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", default=["play_cards_0916_025"])
    parser.add_argument("--max-hand-evaluations", type=int, default=24)
    parser.add_argument("--max-arm-evaluations", type=int, default=500)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.output_root.resolve()
    OUTPUT = ROOT / "canaries"
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
            "schema_version": "CHAOYANG_ROBOT100_WORLD_FIRST_CANARY_RESULT_V14",
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
