"""One predeclared constrained-continuity Poker arm candidate; no parameter sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_v2 import time_edges
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import ARM_JOINT_NAMES, load_pinned_robot_assets

TASK = "human_to_robot_result_breakthrough_20260924"
INPUT = REPO_ROOT / ("_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
                     "lanes/ai2/camera_exact_mount_v2/play_cards_0902_042/CAMERA_ROBOT_MOTION_V3.npz")
OUT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/robot"
WINDOW = range(96, 112)
POSITION_M = .020
ROTATION_RAD = np.deg2rad(15.)
MAXITER = 80
INTERNAL_POSITION_MARGIN_M = 1e-5
INTERNAL_ROTATION_MARGIN_RAD = np.deg2rad(.01)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _mounts(assets, old):
    neutral = old["neutral_q_arm"]
    fk = forward_kinematics(assets.tianji, {
        name: float(neutral[side, j]) for side in range(2)
        for j, name in enumerate(ARM_JOINT_NAMES[side])})
    flange = old["T_flange_hand"]
    return np.stack([np.linalg.inv(fk["left_tool"]) @ fk["flange_L"] @ flange[0],
                     np.linalg.inv(fk["right_tool"]) @ fk["flange_R"] @ flange[1]])


def pose_errors(assets, side: int, q: np.ndarray, target_root: np.ndarray, tool_mount: np.ndarray):
    # The public 20 mm gate is at the hand root, not the upstream tool link.
    delta = np.linalg.inv(target_root) @ arm._tool_fk(assets, side, q) @ tool_mount
    return float(np.linalg.norm(delta[:3, 3])), float(np.linalg.norm(arm._rotation_vector(delta[:3, :3])))


def solve_one(assets, side, target_root, tool_mount, previous, initial, lower, upper):
    """Minimize normalized step with hard pose constraints and one fixed start."""
    span = upper - lower
    if (span <= 0).any() or not np.isfinite(initial).all() or not np.isfinite(previous).all():
        raise ValueError("INVALID_FROZEN_START_OR_LIMITS")
    def objective(q):
        return float(np.sum(((q - previous) / span) ** 2))
    def translation_margin(q):
        p, _ = pose_errors(assets, side, q, target_root, tool_mount)
        return POSITION_M - INTERNAL_POSITION_MARGIN_M - p
    def rotation_margin(q):
        _, r = pose_errors(assets, side, q, target_root, tool_mount)
        return ROTATION_RAD - INTERNAL_ROTATION_MARGIN_RAD - r
    result = minimize(objective, np.clip(initial, lower, upper), method="SLSQP",
                      bounds=list(zip(lower, upper, strict=True)),
                      constraints=[{"type": "ineq", "fun": translation_margin},
                                   {"type": "ineq", "fun": rotation_margin}],
                      options={"maxiter": MAXITER, "ftol": 1e-9, "disp": False})
    q = np.asarray(result.x, dtype=float)
    p, r = pose_errors(assets, side, q, target_root, tool_mount)
    feasible = bool(result.success and np.isfinite(q).all() and np.all(q >= lower - 1e-8)
                    and np.all(q <= upper + 1e-8)
                    and p <= POSITION_M and r <= ROTATION_RAD)
    return q, {"success": bool(result.success), "status": int(result.status),
               "message": str(result.message), "nit": int(result.nit),
               "nfev": int(result.nfev), "position_mm": p * 1000,
               "rotation_deg": float(np.rad2deg(r)), "feasible": feasible,
               "joint_max_step_rad": float(np.max(np.abs(q - previous)))}


def _run_range(frames: range, stem: str, *, require_window: bool):
    packet = load_json(REPO_ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    if packet["task_id"] != TASK or OUT.joinpath(stem + ".npz").exists():
        raise RuntimeError("TASK_NOT_REGISTERED_OR_CANDIDATE_EXISTS")
    if require_window:
        gate = load_json(OUT / "POKER_096_111_CONSTRAINED_V3.json")
        if not gate["window_expansion_gate"]:
            raise RuntimeError("WINDOW_GATE_NOT_MET")
    old = _load(INPUT)
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    mounts = _mounts(assets, old)
    edge, _ = time_edges(old["timestamp_ns"], old["frame_id"])
    q_new = old["q_arm"].copy()
    attempted = np.zeros((len(q_new), 2), bool)
    feasible = np.zeros_like(attempted)
    rows = []
    for side in range(2):
        prior = old["q_arm"][frames.start - 1, side].copy() if frames.start else old["neutral_q_arm"][side].copy()
        prior_feasible = bool(frames.start and old["wrist_valid"][frames.start - 1, side]
                              and np.isfinite(prior).all())
        for frame in frames:
            if not old["wrist_valid"][frame, side] or not old["target_valid"][frame, side]:
                prior_feasible = False
                rows.append({"frame": frame, "side": side, "status": "INVALID_INPUT"})
                continue
            if not edge[frame] or not prior_feasible:
                prior = old["q_arm"][frame - 1, side].copy() if edge[frame] else old["neutral_q_arm"][side].copy()
            target_root = old["T_target_root_base"][frame, side]
            # The only initial rule: current frame's frozen original R0 solution.
            initial = old["q_arm"][frame, side]
            q, diag = solve_one(assets, side, target_root, mounts[side], prior, initial,
                                lower[side], upper[side])
            attempted[frame, side] = True
            feasible[frame, side] = diag["feasible"]
            q_new[frame, side] = q
            prior, prior_feasible = q.copy(), diag["feasible"]
            rows.append({"frame": frame, "side": side, **diag,
                         "old_position_mm": float(old["position_residual_mm"][frame, side]),
                         "old_rotation_deg": float(old["rotation_residual_deg"][frame, side]),
                         "old_joint_max_step_rad": (float(np.max(np.abs(old["q_arm"][frame, side]
                                                                     - old["q_arm"][frame-1, side]))) if frame else None),
                         "q_source": "ONE_CONSTRAINED_SLSQP_CANDIDATE"})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (stem + ".npz")
    np.savez_compressed(path, q_arm=q_new, q22=old["q22"], frame_id=old["frame_id"],
                        timestamp_ns=old["timestamp_ns"], attempted=attempted, feasible=feasible,
                        T_target_root_base=old["T_target_root_base"], T_flange_hand=old["T_flange_hand"],
                        human_to_physical=old["human_to_physical"])
    with np.load(path, allow_pickle=False) as check:
        if not np.array_equal(check["frame_id"], old["frame_id"]):
            raise RuntimeError("RELOAD_FRAME_MAP_MISMATCH")
    result = {"schema_version": "POKER_CONSTRAINED_CONTINUITY_V3_HAND_ROOT_FIX",
              "task_id": TASK, "source": artifact_ref(INPUT), "candidate": artifact_ref(path),
              "processed_frames": [frames.start, frames.stop - 1], "rows": rows,
              "attempted": int(attempted.sum()), "pose_feasible": int(feasible.sum()),
              "window_expansion_gate": (bool(np.all(feasible[frames][attempted[frames]])
                                             and np.all(attempted[frames])) if not require_window else None),
              "finger_104_105": "UNCHANGED_AND_SEPARATELY_UNRESOLVED",
              "quality": "NOT_PRODUCT_PASS", "new_full_session": require_window,
              "fixed_recipe": {"method": "SLSQP", "maxiter": MAXITER, "ftol": 1e-9,
                               "initial": "same_frame_original_R0", "objective": "squared_range_normalized_step",
                               "position_m": POSITION_M, "rotation_deg": 15.0,
                               "pose_point": "actual_hand_root_after_T_flange_hand",
                               "internal_position_margin_m": INTERNAL_POSITION_MARGIN_M,
                               "internal_rotation_margin_deg": .01}}
    atomic_json(OUT / (stem + ".json"), result)
    return {"pose_feasible": result["pose_feasible"], "attempted": result["attempted"],
            "expansion_gate": result["window_expansion_gate"], "receipt": str(OUT / (stem + ".json"))}


def run_window():
    return _run_range(WINDOW, "POKER_096_111_CONSTRAINED_V3", require_window=False)


def run_full():
    old = _load(INPUT)
    return _run_range(range(len(old["frame_id"])), "POKER_171_CONSTRAINED_V3", require_window=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["window", "full"])
    args = parser.parse_args()
    result = run_window() if args.stage == "window" else run_full()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
