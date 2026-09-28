"""One bounded joint-trajectory candidate for the shared-hand delivery task.

This is deliberately not a forward/reverse per-frame chooser.  Each rolling
problem jointly varies a 16-frame output block plus at most eight future
frames.  Up to eight already-fixed frames provide the left boundary.  The
latest immutable V3 candidate is the only initial path.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.optimize import minimize

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
from chaoyang.pipeline import robot_scene_state_cpu as arm
from chaoyang.pipeline.full_robot_review_v2 import motion_derivatives, time_edges
from chaoyang.pipeline.robot_renderer_cycles import forward_kinematics
from chaoyang.pipeline.robot_renderer_eevee_fullchain import (
    ARM_JOINT_NAMES,
    load_pinned_robot_assets,
)

TASK = "human_to_robot_shared_hand_delivery_20260924"
BASELINE_TASK = "human_to_robot_result_breakthrough_20260924"
SOURCE = REPO_ROOT / (
    "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001/"
    "lanes/ai2/camera_exact_mount_v2/play_cards_0902_042/CAMERA_ROBOT_MOTION_V3.npz"
)
SEED = REPO_ROOT / (
    f"_run/current/{BASELINE_TASK}/attempts/attempt_0001/lanes/robot/"
    "POKER_171_CONSTRAINED_V3.npz"
)
OUT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001/lanes/robot"

OUTPUT_BLOCK = 16
PAST_CONTEXT = 8
FUTURE_CONTEXT = 8
MAXITER = 80
POSITION_M = 0.020
ROTATION_RAD = np.deg2rad(15.0)
POSE_NUMERIC_MARGIN = 1e-8
LEXICOGRAPHIC_TOL = 1e-7
FINITE_DIFF_EPS = 2e-6
WINDOW_START = 80
WINDOW_STOP = 112


@dataclass(frozen=True)
class RollingBlock:
    output: np.ndarray
    past: np.ndarray
    future: np.ndarray

    @property
    def variables(self) -> np.ndarray:
        return np.concatenate((self.output, self.future))


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def contiguous_segments(mask: np.ndarray, edge: np.ndarray) -> list[np.ndarray]:
    """Return ordered valid segments without ever crossing a real time gap."""
    mask = np.asarray(mask, dtype=bool)
    edge = np.asarray(edge, dtype=bool)
    if mask.ndim != 1 or edge.shape != mask.shape:
        raise ValueError("SEGMENT_MASK_SHAPE")
    segments: list[np.ndarray] = []
    current: list[int] = []
    for frame in range(len(mask)):
        if not mask[frame]:
            if current:
                segments.append(np.asarray(current, dtype=np.int64))
                current = []
            continue
        if current and not edge[frame]:
            segments.append(np.asarray(current, dtype=np.int64))
            current = []
        current.append(frame)
    if current:
        segments.append(np.asarray(current, dtype=np.int64))
    return segments


def rolling_blocks(segment: np.ndarray, selected: np.ndarray) -> list[RollingBlock]:
    """Make deterministic 16+8 blocks; every selected frame is output once."""
    segment = np.asarray(segment, dtype=np.int64)
    selected_set = set(np.asarray(selected, dtype=np.int64).tolist())
    positions = [i for i, frame in enumerate(segment) if int(frame) in selected_set]
    if not positions:
        return []
    # A selected interval is split separately if the caller supplied holes.
    runs: list[list[int]] = []
    for pos in positions:
        if not runs or pos != runs[-1][-1] + 1:
            runs.append([pos])
        else:
            runs[-1].append(pos)
    blocks: list[RollingBlock] = []
    for run in runs:
        for offset in range(0, len(run), OUTPUT_BLOCK):
            output_pos = np.asarray(run[offset:offset + OUTPUT_BLOCK], dtype=np.int64)
            first, last = int(output_pos[0]), int(output_pos[-1])
            past_pos = np.arange(max(0, first - PAST_CONTEXT), first, dtype=np.int64)
            future_pos = np.arange(last + 1, min(len(segment), last + 1 + FUTURE_CONTEXT), dtype=np.int64)
            blocks.append(RollingBlock(segment[output_pos], segment[past_pos], segment[future_pos]))
    emitted = np.concatenate([block.output for block in blocks])
    expected = np.asarray([frame for frame in segment if int(frame) in selected_set], dtype=np.int64)
    if not np.array_equal(emitted, expected) or len(np.unique(emitted)) != len(emitted):
        raise RuntimeError("ROLLING_OUTPUT_OWNERSHIP_FAILURE")
    return blocks


def _velocity(q: np.ndarray, times: np.ndarray, span: np.ndarray) -> np.ndarray:
    if len(q) < 2:
        return np.empty((0, q.shape[1]), dtype=float)
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("NONPOSITIVE_DT")
    return np.diff(q, axis=0) / (dt[:, None] * span[None])


def _acceleration(q: np.ndarray, times: np.ndarray, span: np.ndarray) -> np.ndarray:
    velocity = _velocity(q, times, span)
    if len(velocity) < 2:
        return np.empty((0, q.shape[1]), dtype=float)
    edge_dt = np.diff(times)
    center_dt = 0.5 * (edge_dt[1:] + edge_dt[:-1])
    return np.diff(velocity, axis=0) / center_dt[:, None]


def _peak(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _sequence_values(xq: np.ndarray, fixed_q: np.ndarray) -> np.ndarray:
    return np.concatenate((fixed_q, xq), axis=0) if len(fixed_q) else xq


def _pose_constraint(
    pose_errors: Callable[[int, np.ndarray], tuple[float, float]],
    frames: np.ndarray,
    joint_count: int,
    extra_count: int,
):
    """Return pose margins and a block finite-difference Jacobian.

    Supplying this Jacobian avoids the quadratic finite-difference pattern of
    treating every FK margin as dependent on every frame's seven joints.
    """
    def fun(x: np.ndarray) -> np.ndarray:
        q = x[:len(frames) * joint_count].reshape(len(frames), joint_count)
        margins = np.empty((len(frames), 2), dtype=float)
        for i, frame in enumerate(frames):
            position, rotation = pose_errors(int(frame), q[i])
            # Unit-normalized hard margins keep SLSQP from comparing metre
            # scale translation constraints directly with radian and
            # range-normalized temporal constraints.  This changes numerical
            # conditioning only; the exact post-solve gate below is unchanged.
            margins[i] = ((POSITION_M - POSE_NUMERIC_MARGIN - position) / POSITION_M,
                          (ROTATION_RAD - POSE_NUMERIC_MARGIN - rotation) / ROTATION_RAD)
        return margins.ravel()

    def jac(x: np.ndarray) -> np.ndarray:
        width = len(frames) * joint_count + extra_count
        value = np.zeros((2 * len(frames), width), dtype=float)
        q = x[:len(frames) * joint_count].reshape(len(frames), joint_count)
        for i, frame in enumerate(frames):
            for joint in range(joint_count):
                plus, minus = q[i].copy(), q[i].copy()
                plus[joint] += FINITE_DIFF_EPS
                minus[joint] -= FINITE_DIFF_EPS
                p_plus, r_plus = pose_errors(int(frame), plus)
                p_minus, r_minus = pose_errors(int(frame), minus)
                column = i * joint_count + joint
                value[2 * i, column] = (-(p_plus - p_minus) /
                                         (2 * FINITE_DIFF_EPS * POSITION_M))
                value[2 * i + 1, column] = (-(r_plus - r_minus) /
                                             (2 * FINITE_DIFF_EPS * ROTATION_RAD))
        return value
    return fun, jac


def _epigraph_constraint(
    measure: Callable[[np.ndarray], np.ndarray],
    q_size: int,
    peak_index: int,
):
    """Two-sided epigraph constraint; Jacobian is finite-differenced once.

    Velocity and acceleration are linear in q.  The compact finite difference
    here is deterministic and does not call FK; pose FK has its own block
    Jacobian above.
    """
    def fun(x: np.ndarray) -> np.ndarray:
        values = measure(x[:q_size]).ravel()
        peak = x[peak_index]
        return np.concatenate((peak - values, peak + values))

    def jac(x: np.ndarray) -> np.ndarray:
        baseline = measure(x[:q_size]).ravel()
        derivative = np.empty((len(baseline), q_size), dtype=float)
        for column in range(q_size):
            plus, minus = x[:q_size].copy(), x[:q_size].copy()
            plus[column] += FINITE_DIFF_EPS
            minus[column] -= FINITE_DIFF_EPS
            derivative[:, column] = ((measure(plus) - measure(minus)) /
                                      (2 * FINITE_DIFF_EPS)).ravel()
        result = np.zeros((2 * len(baseline), len(x)), dtype=float)
        result[:len(baseline), :q_size] = -derivative
        result[len(baseline):, :q_size] = derivative
        result[:, peak_index] = 1.0
        return result
    return fun, jac


def optimize_block(
    initial_q: np.ndarray,
    fixed_q: np.ndarray,
    variable_frames: np.ndarray,
    fixed_times: np.ndarray,
    variable_times: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    pose_errors: Callable[[int, np.ndarray], tuple[float, float]],
) -> tuple[np.ndarray, dict]:
    """Lexicographically minimize normalized velocity then acceleration."""
    initial_q = np.asarray(initial_q, dtype=float)
    fixed_q = np.asarray(fixed_q, dtype=float)
    variable_frames = np.asarray(variable_frames, dtype=np.int64)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if initial_q.shape != (len(variable_frames), len(lower)):
        raise ValueError("INITIAL_PATH_SHAPE")
    if fixed_q.ndim != 2 or fixed_q.shape[1] != len(lower):
        raise ValueError("FIXED_PATH_SHAPE")
    if not np.isfinite(initial_q).all() or not np.isfinite(fixed_q).all():
        raise ValueError("NONFINITE_PATH")
    span = upper - lower
    if np.any(span <= 0):
        raise ValueError("INVALID_LIMIT_SPAN")
    times = np.concatenate((fixed_times, variable_times))
    q_count = initial_q.size

    def unpack(flat: np.ndarray) -> np.ndarray:
        return flat.reshape(initial_q.shape)

    def velocity_measure(flat: np.ndarray) -> np.ndarray:
        return _velocity(_sequence_values(unpack(flat), fixed_q), times, span)

    initial_velocity = _peak(velocity_measure(initial_q.ravel()))
    x1 = np.concatenate((initial_q.ravel(), [initial_velocity + 1e-8]))
    pose1, pose1_jac = _pose_constraint(pose_errors, variable_frames, len(lower), 1)
    speed1, speed1_jac = _epigraph_constraint(velocity_measure, q_count, q_count)
    objective_jac = np.zeros_like(x1); objective_jac[-1] = 1.0
    result1 = minimize(
        lambda x: float(x[-1]), x1, jac=lambda _x: objective_jac,
        method="SLSQP",
        bounds=[*[(float(lo), float(hi)) for _ in variable_frames
                  for lo, hi in zip(lower, upper, strict=True)], (0.0, None)],
        constraints=[{"type": "ineq", "fun": pose1, "jac": pose1_jac},
                     {"type": "ineq", "fun": speed1, "jac": speed1_jac}],
        options={"maxiter": MAXITER, "ftol": 1e-9, "disp": False},
    )
    q1 = unpack(np.asarray(result1.x[:q_count], float))
    velocity_peak = _peak(velocity_measure(q1.ravel()))

    def acceleration_measure(flat: np.ndarray) -> np.ndarray:
        return _acceleration(_sequence_values(unpack(flat), fixed_q), times, span)

    initial_acceleration = _peak(acceleration_measure(q1.ravel()))
    x2 = np.concatenate((q1.ravel(), [initial_acceleration + 1e-8]))
    pose2, pose2_jac = _pose_constraint(pose_errors, variable_frames, len(lower), 1)
    accel2, accel2_jac = _epigraph_constraint(acceleration_measure, q_count, q_count)

    def velocity_bound(x: np.ndarray) -> np.ndarray:
        values = velocity_measure(x[:q_count]).ravel()
        bound = velocity_peak + LEXICOGRAPHIC_TOL
        return np.concatenate((bound - values, bound + values))

    result2 = minimize(
        lambda x: float(x[-1]), x2,
        jac=lambda _x: objective_jac,
        method="SLSQP",
        bounds=[*[(float(lo), float(hi)) for _ in variable_frames
                  for lo, hi in zip(lower, upper, strict=True)], (0.0, None)],
        constraints=[{"type": "ineq", "fun": pose2, "jac": pose2_jac},
                     {"type": "ineq", "fun": velocity_bound},
                     {"type": "ineq", "fun": accel2, "jac": accel2_jac}],
        options={"maxiter": MAXITER, "ftol": 1e-9, "disp": False},
    )
    q2 = unpack(np.asarray(result2.x[:q_count], float))

    def exact_gate(q: np.ndarray) -> tuple[bool, list[dict]]:
        rows = []
        for frame, value in zip(variable_frames, q, strict=True):
            position, rotation = pose_errors(int(frame), value)
            rows.append({"frame": int(frame), "position_mm": 1000.0 * position,
                         "rotation_deg": float(np.rad2deg(rotation)),
                         "pose_pass": bool(position <= POSITION_M and rotation <= ROTATION_RAD),
                         "limit_pass": bool(np.all(value >= lower - 1e-9) and
                                            np.all(value <= upper + 1e-9))})
        return bool(all(row["pose_pass"] and row["limit_pass"] for row in rows)), rows

    stage1_gate, stage1_rows = exact_gate(q1)
    stage2_gate, stage2_rows = exact_gate(q2)
    stage1_ok = bool(result1.success and stage1_gate)
    stage2_velocity = _peak(velocity_measure(q2.ravel()))
    stage2_ok = bool(result2.success and stage2_gate and
                     stage2_velocity <= velocity_peak + LEXICOGRAPHIC_TOL)
    selected = q2 if stage2_ok else q1
    selected_stage = "ACCELERATION" if stage2_ok else ("VELOCITY" if stage1_ok else "NONE")
    diagnostics = {
        "initial_velocity_peak_range_per_s": initial_velocity,
        "stage1_velocity_peak_range_per_s": velocity_peak,
        "initial_stage2_acceleration_peak_range_per_s2": initial_acceleration,
        "stage2_acceleration_peak_range_per_s2": _peak(acceleration_measure(q2.ravel())),
        "stage2_velocity_peak_range_per_s": stage2_velocity,
        "stage1": {"success": bool(result1.success), "status": int(result1.status),
                   "message": str(result1.message), "nit": int(result1.nit),
                   "nfev": int(result1.nfev), "hard_gate": stage1_gate,
                   "rows": stage1_rows},
        "stage2": {"success": bool(result2.success), "status": int(result2.status),
                   "message": str(result2.message), "nit": int(result2.nit),
                   "nfev": int(result2.nfev), "hard_gate": stage2_gate,
                   "rows": stage2_rows},
        "selected_stage": selected_stage,
        "usable": bool(stage2_ok or stage1_ok),
    }
    return selected, diagnostics


def _mounts(assets, source: dict[str, np.ndarray]) -> np.ndarray:
    neutral = source["neutral_q_arm"]
    fk = forward_kinematics(assets.tianji, {
        name: float(neutral[side, joint])
        for side in range(2)
        for joint, name in enumerate(ARM_JOINT_NAMES[side])
    })
    return np.stack([
        np.linalg.inv(fk["left_tool"]) @ fk["flange_L"] @ source["T_flange_hand"][0],
        np.linalg.inv(fk["right_tool"]) @ fk["flange_R"] @ source["T_flange_hand"][1],
    ])


def _pose_errors(assets, side: int, target: np.ndarray, mount: np.ndarray, q: np.ndarray):
    delta = np.linalg.inv(target) @ arm._tool_fk(assets, side, q) @ mount
    return (float(np.linalg.norm(delta[:3, 3])),
            float(np.linalg.norm(arm._rotation_vector(delta[:3, :3]))))


def _stats(q: np.ndarray, valid: np.ndarray, edge: np.ndarray, times: np.ndarray) -> dict:
    step = np.max(np.abs(np.diff(q, axis=0)), axis=-1)
    eligible = valid[1:] & valid[:-1] & edge[1:, None]
    derivatives = motion_derivatives(q, valid, times, edge)
    jerk = np.max(np.abs(derivatives["jerk"]), axis=-1)
    result = {}
    for side in range(2):
        values = step[:, side][eligible[:, side]]
        jerk_values = jerk[:, side][np.isfinite(jerk[:, side])]
        result[str(side)] = {
            "edges": int(len(values)),
            "step_max_rad": float(np.max(values)) if len(values) else None,
            "step_p99_rad": float(np.percentile(values, 99)) if len(values) else None,
            "jerk_p95_rad_s3": float(np.percentile(jerk_values, 95)) if len(jerk_values) else None,
        }
    return result


def _run(selected_frames: Iterable[int], stem: str, *, require_window_gate: bool) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    output_npz, output_json = OUT / f"{stem}.npz", OUT / f"{stem}.json"
    if output_npz.exists() or output_json.exists():
        raise FileExistsError("IMMUTABLE_OUTPUT_EXISTS")
    if require_window_gate:
        gate = json.loads((OUT / "POKER_080_111_JOINT_TRAJECTORY_FIX1.json").read_text())
        if not gate["expansion_gate"]:
            raise RuntimeError("WINDOW_GATE_NOT_MET")
    source, seed = _load(SOURCE), _load(SEED)
    if not np.array_equal(source["frame_id"], seed["frame_id"]):
        raise RuntimeError("SEED_SOURCE_FRAME_MAP_MISMATCH")
    if not np.array_equal(source["q22"], seed["q22"], equal_nan=True):
        raise RuntimeError("SEED_CHANGED_FROZEN_HAND_Q")
    selected = np.zeros(len(source["frame_id"]), dtype=bool)
    selected[np.asarray(list(selected_frames), dtype=np.int64)] = True
    assets = load_pinned_robot_assets(REPO_ROOT)
    lower, upper = arm._arm_limits(assets)
    mounts = _mounts(assets, source)
    edge, times = time_edges(source["timestamp_ns"], source["frame_id"])
    valid = source["wrist_valid"] & source["target_valid"]
    q_new = seed["q_arm"].copy()
    attempted = np.zeros(valid.shape, dtype=bool)
    feasible = np.zeros(valid.shape, dtype=bool)
    emitted = np.zeros(valid.shape, dtype=np.int16)
    block_rows = []
    for side in range(2):
        for segment in contiguous_segments(valid[:, side], edge):
            for block in rolling_blocks(segment, np.flatnonzero(selected & valid[:, side])):
                variables = block.variables
                initial = seed["q_arm"][variables, side]
                fixed = q_new[block.past, side]
                pose = lambda frame, q, side=side: _pose_errors(
                    assets, side, source["T_target_root_base"][frame, side], mounts[side], q)
                solved, diag = optimize_block(
                    initial, fixed, variables, times[block.past], times[variables],
                    lower[side], upper[side], pose,
                )
                output_count = len(block.output)
                attempted[block.output, side] = True
                emitted[block.output, side] += 1
                if diag["usable"]:
                    q_new[block.output, side] = solved[:output_count]
                for local, frame in enumerate(block.output):
                    position, rotation = pose(int(frame), q_new[frame, side])
                    feasible[frame, side] = bool(
                        diag["usable"] and position <= POSITION_M and rotation <= ROTATION_RAD and
                        np.all(q_new[frame, side] >= lower[side] - 1e-9) and
                        np.all(q_new[frame, side] <= upper[side] + 1e-9)
                    )
                block_rows.append({
                    "side": side,
                    "output_frames": block.output.tolist(),
                    "past_boundary_frames": block.past.tolist(),
                    "lookahead_frames": block.future.tolist(),
                    **diag,
                })
    expected = selected[:, None] & valid
    ownership_pass = bool(np.all(emitted[expected] == 1) and np.all(emitted[~expected] == 0))
    pose_pass = bool(np.all(feasible[expected]))
    no_unselected_change = bool(np.array_equal(q_new[~selected], seed["q_arm"][~selected]))
    if not no_unselected_change:
        raise RuntimeError("UNSELECTED_FRAME_CHANGED")
    before = _stats(seed["q_arm"], valid, edge, times)
    after = _stats(q_new, valid, edge, times)
    window_no_regression = all(
        after[str(side)][key] <= before[str(side)][key] + 1e-9
        for side in range(2) for key in ("step_max_rad", "step_p99_rad", "jerk_p95_rad_s3")
    )
    expansion_gate = bool(ownership_pass and pose_pass and window_no_regression and
                          all(row["usable"] for row in block_rows))
    np.savez_compressed(
        output_npz,
        q_arm=q_new, q22=seed["q22"], frame_id=seed["frame_id"],
        timestamp_ns=seed["timestamp_ns"], attempted=attempted, feasible=feasible,
        emitted_count=emitted, T_target_root_base=seed["T_target_root_base"],
        T_flange_hand=seed["T_flange_hand"], human_to_physical=seed["human_to_physical"],
    )
    with np.load(output_npz, allow_pickle=False) as check:
        if not np.array_equal(check["frame_id"], source["frame_id"]):
            raise RuntimeError("RELOAD_FRAME_MAP_MISMATCH")
        if not np.array_equal(check["q_arm"], q_new):
            raise RuntimeError("RELOAD_Q_MISMATCH")
    receipt = {
        "schema_version": "POKER_JOINT_TRAJECTORY_V1",
        "task_id": TASK,
        "source": artifact_ref(SOURCE),
        "seed_v3": artifact_ref(SEED),
        "candidate": artifact_ref(output_npz),
        "selected_frame_range": [int(np.flatnonzero(selected)[0]), int(np.flatnonzero(selected)[-1])],
        "selected_frames": int(selected.sum()),
        "attempted_side_frames": int(attempted.sum()),
        "feasible_side_frames": int(feasible.sum()),
        "output_ownership_pass": ownership_pass,
        "unselected_byte_identity": no_unselected_change,
        "pose_hard_gate": pose_pass,
        "motion_before": before,
        "motion_after": after,
        "motion_no_regression": window_no_regression,
        "collision": "PENDING_EXISTING_DECLARED_SCOPE_ACCEPTANCE;NOT_IN_OPTIMIZER_NO_DISTANCE_INTERFACE",
        "expansion_gate": expansion_gate if not require_window_gate else None,
        "blocks": block_rows,
        "recipe": {
            "output_block": OUTPUT_BLOCK,
            "past_fixed_boundary_max": PAST_CONTEXT,
            "future_joint_variables_max": FUTURE_CONTEXT,
            "initial_path": "IMMUTABLE_POKER_171_CONSTRAINED_V3_ONLY",
            "stage1": "MINIMIZE_RANGE_NORMALIZED_ACTUAL_DT_VELOCITY_PEAK",
            "stage2": "MINIMIZE_RANGE_NORMALIZED_ACTUAL_DT_ACCELERATION_PEAK_WITH_STAGE1_BOUND",
            "maxiter_each_stage": MAXITER,
            "position_m_max": POSITION_M,
            "rotation_deg_max": 15.0,
            "joint_limits": "PINNED_ASSET_ORIGINAL",
        },
        "quality": "WINDOW_NUMERIC_PASS_PENDING_COLLISION" if expansion_gate else "REJECTED_OR_INCONCLUSIVE",
        "adoption": "NOT_ADOPTED",
        "claim_limit": "Offline numeric Robot trajectory candidate; no collision, product, control or deployment authority.",
    }
    atomic_json(output_json, receipt)
    return receipt


def run_window() -> dict:
    return _run(range(WINDOW_START, WINDOW_STOP), "POKER_080_111_JOINT_TRAJECTORY_FIX1",
                require_window_gate=False)


def run_full() -> dict:
    return _run(range(171), "POKER_171_JOINT_TRAJECTORY_FIX1", require_window_gate=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("window", "full"))
    args = parser.parse_args()
    receipt = run_window() if args.stage == "window" else run_full()
    print(json.dumps({
        "candidate": receipt["candidate"],
        "pose_hard_gate": receipt["pose_hard_gate"],
        "motion_no_regression": receipt["motion_no_regression"],
        "expansion_gate": receipt["expansion_gate"],
        "quality": receipt["quality"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
