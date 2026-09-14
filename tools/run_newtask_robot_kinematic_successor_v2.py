#!/usr/bin/env python3
"""Frozen shared Robot successor v2 with whole-window trajectory projection.

The first frame is a static IK/retarget initialization.  Motion limits are
applied only to the observed transitions that follow it.  A first pass obtains
the unconstrained shared IK solution; a second pass renders and gates the
whole-window projection through the unchanged generic canary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, minimize  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "tools/run_newtask_robot_kinematic_canary.py"
PROTOCOL = PROJECT / "tasks/chips/runs/robot/20260903_shared_adapter_successor_v1/PROTOCOL_V2.json"
EXPECTED_THUMB_SHA256 = "89d121a8ca869e9c3e37318e9e10e4a1eb3ef731a996554e00f32f7e73b01132"
VELOCITY_LIMIT_RAD_PER_FRAME = 0.12
ACCELERATION_LIMIT_RAD_PER_FRAME2 = 0.06
NUMERIC_MARGIN = 1e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def difference_matrices(delta_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return velocity, acceleration and jerk operators in true seconds."""
    count = int(delta_t.size + 1)
    d1 = np.zeros((count - 1, count), dtype=np.float64)
    for row, dt in enumerate(delta_t):
        d1[row, row] = -1.0 / dt
        d1[row, row + 1] = 1.0 / dt
    if count < 3:
        return d1, np.empty((0, count)), np.empty((0, count))
    acceleration_dt = 0.5 * (delta_t[:-1] + delta_t[1:])
    d2 = np.zeros((count - 2, count), dtype=np.float64)
    for row, dt in enumerate(acceleration_dt):
        d2[row] = (d1[row + 1] - d1[row]) / dt
    if count < 4:
        return d1, d2, np.empty((0, count))
    jerk_dt = 0.5 * (acceleration_dt[:-1] + acceleration_dt[1:])
    d3 = np.zeros((count - 3, count), dtype=np.float64)
    for row, dt in enumerate(jerk_dt):
        d3[row] = (d2[row + 1] - d2[row]) / dt
    return d1, d2, d3


def project_joint(
    raw: np.ndarray,
    lower: float,
    upper: float,
    d1: np.ndarray,
    d2: np.ndarray,
    d3: np.ndarray,
    velocity_limit: float,
    acceleration_limit: float,
    jerk_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    count = int(raw.size)
    equality = np.zeros((1, count), dtype=np.float64)
    equality[0, 0] = 1.0
    operators = [equality, d1, d2]
    lows = [np.asarray([raw[0]]), np.full(d1.shape[0], -velocity_limit), np.full(d2.shape[0], -acceleration_limit)]
    highs = [np.asarray([raw[0]]), np.full(d1.shape[0], velocity_limit), np.full(d2.shape[0], acceleration_limit)]
    if d3.shape[0]:
        operators.append(d3)
        lows.append(np.full(d3.shape[0], -jerk_limit))
        highs.append(np.full(d3.shape[0], jerk_limit))
    matrix = np.vstack(operators)
    constraint = LinearConstraint(matrix, np.concatenate(lows), np.concatenate(highs))

    # Jerk is regularized in frame coordinates so the tracking term remains
    # well-scaled; the physical-seconds jerk bound above remains authoritative.
    d3_frame = np.diff(np.eye(count), n=3, axis=0)
    regularization = 0.02
    hessian = np.eye(count) + regularization * (d3_frame.T @ d3_frame)
    linear = np.asarray(raw, dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        delta = candidate - linear
        return 0.5 * float(delta @ delta) + 0.5 * regularization * float(np.linalg.norm(d3_frame @ candidate) ** 2)

    def jacobian(candidate: np.ndarray) -> np.ndarray:
        return hessian @ candidate - linear

    initial = np.full(count, float(raw[0]), dtype=np.float64)
    solved = minimize(
        objective,
        initial,
        jac=jacobian,
        method="SLSQP",
        bounds=Bounds(np.full(count, lower), np.full(count, upper)),
        constraints=(constraint,),
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    projected = np.asarray(solved.x, dtype=np.float64)
    velocity = d1 @ projected
    acceleration = d2 @ projected
    jerk = d3 @ projected if d3.shape[0] else np.empty(0)
    violation = max(
        abs(float(projected[0] - raw[0])),
        max(0.0, float(np.max(np.abs(velocity))) - velocity_limit),
        max(0.0, float(np.max(np.abs(acceleration))) - acceleration_limit),
        max(0.0, float(np.max(np.abs(jerk))) - jerk_limit) if jerk.size else 0.0,
        max(0.0, lower - float(np.min(projected))),
        max(0.0, float(np.max(projected)) - upper),
    )
    if not solved.success or violation > 2e-6:
        raise RuntimeError(f"trajectory projection failed: success={solved.success} message={solved.message} violation={violation}")
    return projected, {
        "iterations": int(solved.nit),
        "objective": float(solved.fun),
        "max_constraint_violation": float(violation),
    }


def trajectory_metrics(values: np.ndarray, d1: np.ndarray, d2: np.ndarray, d3: np.ndarray) -> dict[str, float]:
    def maximum(operator: np.ndarray) -> float:
        if not operator.shape[0]:
            return 0.0
        return float(np.max(np.abs(np.einsum("ij,jpk->ipk", operator, values))))
    return {
        "velocity_max_rad_per_second": maximum(d1),
        "acceleration_max_rad_per_second2": maximum(d2),
        "jerk_max_rad_per_second3": maximum(d3),
    }


def patch_thumb(base: Any, rotations: np.ndarray) -> None:
    original_directions = base.hand_finger_directions

    def corrected_directions(model, q, finger, hand_names, basis, official_module, adapter, prefix):
        directions = original_directions(model, q, finger, hand_names, basis, official_module, adapter, prefix)
        if finger == "thumb":
            physical = 0 if prefix == "hand_l" else 1
            directions = directions @ rotations[physical].T
        return directions

    base.hand_finger_directions = corrected_directions


def run_base(base: Any, args: argparse.Namespace, output_root: Path) -> int:
    sys.argv = [
        str(BASE),
        "--task-id", args.task_id,
        "--session-id", args.session_id,
        "--raw-root", str(args.raw_root),
        "--hawor-npz", str(args.hawor_npz),
        "--hawor-result", str(args.hawor_result),
        "--source-admission", str(args.source_admission),
        "--output-root", str(output_root),
    ]
    return int(base.main())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=("chips", "poker"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--hawor-npz", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--source-admission", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thumb-adapter", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_V2_FIT":
        raise RuntimeError("shared successor v2 protocol drift")
    thumb_path = args.thumb_adapter.resolve(strict=True)
    if sha256(thumb_path) != EXPECTED_THUMB_SHA256:
        raise RuntimeError("frozen shared thumb adapter SHA drift")
    with np.load(thumb_path, allow_pickle=False) as archive:
        rotations = np.asarray(archive["rotations"], dtype=np.float64)
    if rotations.shape != (2, 3, 3):
        raise RuntimeError("thumb adapter rotation shape mismatch")

    output = args.output_root.resolve()
    raw_output = output.with_name(output.name + "_RAW_UNPROJECTED")
    if output.exists():
        raise RuntimeError(f"fresh output root required: {output}")

    sys.path.insert(0, str(PROJECT))
    base_raw = load(BASE, "newtask_robot_kinematic_successor_v2_raw")
    patch_thumb(base_raw, rotations)
    if not (raw_output / "RESULT.json").exists():
        if raw_output.exists():
            raise RuntimeError(f"incomplete raw solver evidence exists: {raw_output}")
        run_base(base_raw, args, raw_output)

    raw_result = json.loads((raw_output / "RESULT.json").read_text())
    raw_scene = Path(raw_result["outputs"]["scene"]["path"])
    with np.load(raw_scene, allow_pickle=False) as archive:
        q_arm_raw = np.asarray(archive["q_arm"], dtype=np.float64)
        q_hand_raw = np.asarray(archive["q_hand"], dtype=np.float64)
        source_frames = np.asarray(archive["source_frames"], dtype=np.int64)
    with np.load(args.hawor_npz.resolve(strict=True), allow_pickle=False) as archive:
        fps = float(np.asarray(archive["fps"]).item())
    if q_arm_raw.shape[:2] != (24, 2) or q_hand_raw.shape[:2] != (24, 2):
        raise RuntimeError("unexpected raw trajectory shape")
    delta_t = np.diff(source_frames).astype(np.float64) / fps
    if delta_t.shape != (23,) or np.any(delta_t <= 0):
        raise RuntimeError("source frames do not define 23 positive true delta-times")
    d1, d2, d3 = difference_matrices(delta_t)
    velocity_limit = VELOCITY_LIMIT_RAD_PER_FRAME * fps - NUMERIC_MARGIN
    acceleration_limit = ACCELERATION_LIMIT_RAD_PER_FRAME2 * fps * fps - NUMERIC_MARGIN
    jerk_limit = 2.0 * acceleration_limit / float(np.min(delta_t))

    from pipeline import robot_scene_state_cpu as official  # noqa: PLC0415
    from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets  # noqa: PLC0415

    assets = load_pinned_robot_assets(PROJECT)
    arm_lower, arm_upper = official._arm_limits(assets)
    hand_lower = []
    hand_upper = []
    for model in (assets.left_hand, assets.right_hand):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        hand_lower.append(np.asarray([joint.lower for joint in moving], dtype=np.float64))
        hand_upper.append(np.asarray([joint.upper for joint in moving], dtype=np.float64))
    hand_lower = np.asarray(hand_lower)
    hand_upper = np.asarray(hand_upper)

    q_arm = np.empty_like(q_arm_raw)
    q_hand = np.empty_like(q_hand_raw)
    solver_rows = []
    for family, raw, projected, lower, upper in (
        ("arm", q_arm_raw, q_arm, arm_lower, arm_upper),
        ("hand", q_hand_raw, q_hand, hand_lower, hand_upper),
    ):
        for physical in range(2):
            for joint in range(raw.shape[2]):
                values, solver = project_joint(
                    raw[:, physical, joint],
                    float(lower[physical, joint]),
                    float(upper[physical, joint]),
                    d1, d2, d3,
                    velocity_limit, acceleration_limit, jerk_limit,
                )
                projected[:, physical, joint] = values
                solver_rows.append({"family": family, "physical_side": ("left", "right")[physical], "joint_index": joint, **solver})

    output.parent.mkdir(parents=True, exist_ok=True)
    projection_path = output.parent / f"{output.name}_TRAJECTORY_PROJECTION.npz"
    np.savez_compressed(
        projection_path,
        q_arm_raw=q_arm_raw,
        q_hand_raw=q_hand_raw,
        q_arm_projected=q_arm,
        q_hand_projected=q_hand,
        source_frames=source_frames,
        delta_t_seconds=delta_t,
        velocity_operator=d1,
        acceleration_operator=d2,
        jerk_operator=d3,
    )
    projection_report = {
        "schema_version": "newtask-robot-whole-window-trajectory-projection-v2",
        "status": "PASS_NUMERIC_PROJECTION",
        "first_frame_policy": "STATIC_IK_NO_NEUTRAL_TO_FRAME0_MOTION_CLAIM",
        "source_frames": source_frames.tolist(),
        "fps": fps,
        "delta_t_seconds": delta_t.tolist(),
        "limits": {
            "velocity_rad_per_second": velocity_limit,
            "acceleration_rad_per_second2": acceleration_limit,
            "jerk_rad_per_second3_implied": jerk_limit,
            "unchanged_velocity_rad_per_frame": VELOCITY_LIMIT_RAD_PER_FRAME,
            "unchanged_acceleration_rad_per_frame2": ACCELERATION_LIMIT_RAD_PER_FRAME2,
        },
        "raw": {"arm": trajectory_metrics(q_arm_raw, d1, d2, d3), "hand": trajectory_metrics(q_hand_raw, d1, d2, d3)},
        "projected": {"arm": trajectory_metrics(q_arm, d1, d2, d3), "hand": trajectory_metrics(q_hand, d1, d2, d3)},
        "first_frame_exactly_preserved": bool(np.array_equal(q_arm[0], q_arm_raw[0]) and np.array_equal(q_hand[0], q_hand_raw[0])),
        "joint_projection_solver_rows": solver_rows,
        "projection": evidence(projection_path),
        "raw_solver_result": evidence(raw_output / "RESULT.json"),
    }
    projection_report_path = output.parent / f"{output.name}_TRAJECTORY_PROJECTION.json"
    atomic_json(projection_report_path, projection_report)

    base_final = load(BASE, "newtask_robot_kinematic_successor_v2_final")
    patch_thumb(base_final, rotations)
    arm_position_calls = 0
    hand_solver_calls = 0

    def fixed_arm_position(assets_arg, *, side, base, target_tool, initial_q, lower, upper):
        nonlocal arm_position_calls
        del assets_arg, base, target_tool, initial_q, lower, upper
        slot = arm_position_calls // 2
        expected_side = slot % 2
        frame = slot // 2
        if side != expected_side or frame >= 24:
            raise RuntimeError("arm fixed-trajectory call order drift")
        arm_position_calls += 1
        return q_arm[frame, side].copy(), 0

    def fixed_arm_pose(assets_arg, *, side, base, target_tool, initial_q, lower, upper):
        del assets_arg, side, base, target_tool, lower, upper
        return np.asarray(initial_q, dtype=np.float64).copy(), 0

    def fixed_hand(fun, x0, *positionals, **keywords):
        nonlocal hand_solver_calls
        del fun, x0, positionals, keywords
        instance = hand_solver_calls // 10
        within = hand_solver_calls % 10
        frame = instance // 2
        physical = instance % 2
        finger = within // 2
        if frame >= 24:
            raise RuntimeError("hand fixed-trajectory call order drift")
        group = base_final.HAND_GROUPS[base_final.FINGERS[finger]]
        hand_solver_calls += 1
        return SimpleNamespace(x=q_hand[frame, physical, group].copy(), nfev=0)

    official._solve_one_arm_position_only = fixed_arm_position
    official._solve_one_arm = fixed_arm_pose
    base_final.least_squares = fixed_hand
    exit_code = run_base(base_final, args, output)
    if arm_position_calls != 96 or hand_solver_calls != 480:
        raise RuntimeError(f"fixed-trajectory call count drift: arm={arm_position_calls} hand={hand_solver_calls}")

    result_path = output / "RESULT.json"
    result = json.loads(result_path.read_text())
    first_frame_id = int(source_frames[0])
    static_pose = [row for row in result.get("pose_rows", []) if row.get("source_frame") == first_frame_id]
    static_hand = [row for row in result.get("hand_rows", []) if row.get("source_frame") == first_frame_id]
    static_sat = [row for row in result.get("self_sat_rows", []) if row.get("source_frame") == first_frame_id]
    result["shared_successor_v2"] = {
        "protocol": evidence(PROTOCOL),
        "runner": evidence(Path(__file__)),
        "base_runner": evidence(BASE),
        "thumb_adapter": evidence(thumb_path),
        "trajectory_projection": evidence(projection_report_path),
        "raw_solver_evidence": evidence(raw_output / "RESULT.json"),
        "session_specific_adjustment": False,
        "true_delta_t_seconds": delta_t.tolist(),
        "static_frame_zero": {
            "source_frame": first_frame_id,
            "pose_rows": static_pose,
            "hand_rows": static_hand,
            "exact_visual_triangle_sat_rows": static_sat,
            "all_limits_pass": bool(all(row.get("arm_limits_pass", False) for row in static_pose) and all(row.get("limits_pass", False) for row in static_hand)),
            "pose_pass": bool(all(row.get("pose_10mm_5deg_pass", False) for row in static_pose)),
            "shape_pass": bool(all(row.get("shape_gate_pass", False) for row in static_hand)),
            "sat_pass": bool(static_sat and all(row.get("pass", False) for row in static_sat)),
        },
    }
    atomic_json(result_path, result)
    print(json.dumps({
        "event": "SUCCESSOR_V2_COMPLETE",
        "task": args.task_id,
        "session": args.session_id,
        "visual_kinematic_candidate": result.get("visual_kinematic_candidate"),
        "gates": result.get("gates"),
        "static_frame_zero": result["shared_successor_v2"]["static_frame_zero"],
    }), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
