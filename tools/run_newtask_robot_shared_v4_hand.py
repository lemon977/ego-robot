#!/usr/bin/env python3
"""Bounded shared KaiHand trajectory retarget audit for chips/poker.

This runner deliberately makes only one task-global FIT decision.  It uses the
official KaiHand FK and joint limits, matches wrist-relative normalized MCP
positions, normalized bone directions and wrist-to-tip directions, projects
the whole 24-frame trajectory through the frozen velocity/acceleration limits,
and applies exact distinct-finger triangle SAT as a hard candidate-admission
test.  Object contact is never claimed because no Object6D authority exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
CANARY = PROJECT / "tools/run_newtask_robot_kinematic_canary.py"
V2 = PROJECT / "tools/run_newtask_robot_kinematic_successor_v2.py"
MOUNT_PIN = PROJECT / "tasks/chips/runs/robot/20260903_shared_adapter_successor_v1/PINNED_KAIHAND_MOUNT_AUTHORITY_V1.json"
THUMB = PROJECT / "tasks/chips/runs/robot/20260903_shared_adapter_successor_v1/calibration_v1/SHARED_THUMB_AXIS_ADAPTER.npz"
SIDE_NAMES = ("left", "right")
PHYSICAL_TO_HUMAN = (1, 0)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
GROUPS = {
    "thumb": np.arange(0, 6),
    "index": np.arange(6, 10),
    "middle": np.arange(10, 14),
    "ring": np.arange(14, 18),
    "pinky": np.arange(18, 22),
}
TERMINALS = {"thumb": 6, "index": 4, "middle": 4, "ring": 4, "pinky": 4}
MANO_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
MANO_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}
FIT_GRID = (
    {"name": "balanced", "mcp": 1.0, "bone": 1.0, "tip": 1.0, "prior": 0.015},
    {"name": "direction_priority", "mcp": 0.75, "bone": 1.5, "tip": 1.5, "prior": 0.010},
    {"name": "endpoint_priority", "mcp": 1.5, "bone": 0.75, "tip": 2.0, "prior": 0.020},
)
NEUTRAL_HOMOTOPY = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0)
MAX_NFEV = 60
SHAPE_MAX_DEG = 45.0
MCP_NORMALIZED_MAX = 0.50
VELOCITY_RAD_PER_FRAME = 0.12
ACCELERATION_RAD_PER_FRAME2 = 0.06


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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(length <= 1e-9):
        raise RuntimeError("degenerate feature vector")
    return value / length


def angle_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(unit(first) * unit(second), axis=-1), -1.0, 1.0)))


def pinned_mounts() -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(MOUNT_PIN.read_text())
    if payload.get("status") != "PINNED_DEVELOPMENT_ONLY_NOT_FORMAL_CALIBRATION_AUTHORITY":
        raise RuntimeError("task-local mount authority status drift")
    if payload.get("development_only") is not True or payload.get("formal_consumer_allowed") is not False:
        raise RuntimeError("task-local mount authority claim drift")
    matrices = np.asarray([payload["T_tool_hand"][side] for side in SIDE_NAMES], dtype=np.float64)
    if matrices.shape != (2, 4, 4):
        raise RuntimeError("task-local mount matrix shape drift")
    return matrices, evidence(MOUNT_PIN)


def model_contract(official: Any, adapter: Any, assets: Any):
    models = (assets.left_hand, assets.right_hand)
    rows = []
    for physical, model in enumerate(models):
        moving = tuple(joint for joint in model.joints if joint.joint_type != "fixed")
        names = tuple(joint.name for joint in moving)
        lower = np.asarray([joint.lower for joint in moving], dtype=np.float64)
        upper = np.asarray([joint.upper for joint in moving], dtype=np.float64)
        if len(names) != 22 or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise RuntimeError("KaiHand must expose exactly 22 finite-limit DoF")
        neutral = 0.5 * (lower + upper)
        fk = official.forward_kinematics(model, dict(zip(names, neutral, strict=True)))
        prefix = ("hand_l", "hand_r")[physical]
        dummy = np.zeros((21, 3), dtype=np.float64)
        for index, finger in ((2, "thumb"), (5, "index"), (9, "middle"), (13, "ring"), (17, "pinky")):
            dummy[index] = fk[f"{prefix}_{finger}_link1"][:3, 3]
        basis = adapter.final_v3_mano_palm_basis(dummy, handedness=SIDE_NAMES[physical])
        robot_mcp = np.asarray([fk[f"{prefix}_{finger}_link1"][:3, 3] @ basis for finger in FINGERS])
        scale = float(np.mean(np.linalg.norm(robot_mcp[1:], axis=1)))
        rows.append({
            "model": model,
            "names": names,
            "lower": lower,
            "upper": upper,
            "neutral": neutral,
            "basis": basis,
            "prefix": prefix,
            "mcp_scale": scale,
        })
    return rows


def robot_finger_features(
    official: Any,
    adapter: Any,
    contract: dict[str, Any],
    q: np.ndarray,
    finger: str,
    thumb_rotation: np.ndarray,
) -> dict[str, np.ndarray]:
    fk = official.forward_kinematics(contract["model"], dict(zip(contract["names"], q, strict=True)))
    points = np.asarray(
        [np.zeros(3)]
        + [fk[f"{contract['prefix']}_{finger}_link{number}"][:3, 3] for number in range(1, TERMINALS[finger] + 1)],
        dtype=np.float64,
    )
    local = points @ contract["basis"]
    bones = adapter.resample_polyline_unit_directions(points, bone_count=4) @ contract["basis"]
    tip = unit(local[-1])
    if finger == "thumb":
        bones = bones @ thumb_rotation.T
        tip = tip @ thumb_rotation.T
    return {
        "mcp": local[1] / contract["mcp_scale"],
        "bones": bones,
        "tip": tip,
    }


def human_features(adapter: Any, points: np.ndarray, handedness: str) -> dict[str, dict[str, np.ndarray]]:
    basis = adapter.final_v3_mano_palm_basis(points, handedness=handedness)
    wrist = points[0]
    mcp_rows = np.asarray([(points[MANO_MCP[finger]] - wrist) @ basis for finger in FINGERS])
    scale = float(np.mean(np.linalg.norm(mcp_rows[1:], axis=1)))
    bones = adapter.final_v3_mano_unit_bones_local(points, handedness=handedness)
    result = {}
    for finger in FINGERS:
        chain = MANO_CHAINS[finger]
        result[finger] = {
            "mcp": ((points[MANO_MCP[finger]] - wrist) @ basis) / scale,
            "bones": bones[finger],
            "tip": unit((points[chain[-1]] - wrist) @ basis),
        }
    return result


def feature_row(actual: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, Any]:
    bone = angle_deg(actual["bones"], target["bones"])
    return {
        "mcp_normalized_l2": float(np.linalg.norm(actual["mcp"] - target["mcp"])),
        "tip_direction_error_deg": float(angle_deg(actual["tip"], target["tip"])),
        "bone_error_deg_mean": float(np.mean(bone)),
        "bone_error_deg_max": float(np.max(bone)),
    }


def solve_frame(
    official: Any,
    adapter: Any,
    contracts: list[dict[str, Any]],
    targets: list[list[dict[str, dict[str, np.ndarray]]]],
    thumb_rotations: np.ndarray,
    initial: np.ndarray,
    v2_seed: np.ndarray,
    frame: int,
    weights: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.empty((2, 22), dtype=np.float64)
    rows = []
    for physical in range(2):
        contract = contracts[physical]
        candidate = np.clip(initial[physical], contract["lower"], contract["upper"])
        for finger in FINGERS:
            group = GROUPS[finger]
            target = targets[frame][physical][finger]

            def residual(local: np.ndarray, prior: np.ndarray) -> np.ndarray:
                whole = candidate.copy()
                whole[group] = local
                actual = robot_finger_features(official, adapter, contract, whole, finger, thumb_rotations[physical])
                return np.concatenate((
                    weights["mcp"] * (actual["mcp"] - target["mcp"]),
                    weights["bone"] * (actual["bones"] - target["bones"]).ravel(),
                    weights["tip"] * (actual["tip"] - target["tip"]),
                    weights["prior"] * (local - prior),
                ))

            if frame == 0:
                seeds = (
                    ("v2", np.clip(v2_seed[physical, group], contract["lower"][group], contract["upper"][group])),
                    ("midpoint", contract["neutral"][group]),
                    ("zero", np.clip(np.zeros(len(group)), contract["lower"][group], contract["upper"][group])),
                )
            else:
                seeds = (
                    ("warm_start", candidate[group]),
                    ("v2", np.clip(v2_seed[physical, group], contract["lower"][group], contract["upper"][group])),
                )
            started = time.perf_counter()
            attempts = []
            for seed_name, seed in seeds:
                solved = least_squares(
                    lambda value, prior=seed: residual(value, prior),
                    seed,
                    bounds=(contract["lower"][group], contract["upper"][group]),
                    max_nfev=MAX_NFEV,
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                )
                attempts.append((float(np.linalg.norm(residual(solved.x, seed))), seed_name, np.asarray(solved.x), int(solved.nfev)))
            score, seed_name, selected, nfev = min(attempts, key=lambda row: (row[0], row[1]))
            candidate[group] = selected
            rows.append({
                "frame_slot": frame,
                "physical_side": SIDE_NAMES[physical],
                "finger": finger,
                "seed": seed_name,
                "residual_norm": score,
                "nfev": nfev,
                "attempts": len(attempts),
                "runtime_seconds": time.perf_counter() - started,
            })
        output[physical] = candidate
    return output, rows


def temporal_project(v2: Any, raw: np.ndarray, contracts: list[dict[str, Any]], source_frames: np.ndarray, fps: float):
    raw = np.asarray(raw, dtype=np.float64)
    source_frames = np.asarray(source_frames, dtype=np.int64)
    if raw.ndim != 3 or raw.shape[1] != 2:
        raise ValueError(f"expected raw [frames,2,joints], got {raw.shape}")
    if raw.shape[0] == 0:
        raise ValueError("temporal projection requires at least one frame")
    if source_frames.shape != (raw.shape[0],):
        raise ValueError(
            f"source frame shape mismatch: expected {(raw.shape[0],)}, got {source_frames.shape}"
        )
    source_delta = np.diff(source_frames)
    if source_delta.size and np.any(source_delta <= 0):
        raise ValueError("source frames must be strictly increasing within one observed segment")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")

    delta_t = source_delta.astype(np.float64) / fps
    d1, d2, d3 = v2.difference_matrices(delta_t)
    velocity_limit = VELOCITY_RAD_PER_FRAME * fps - 1e-7
    acceleration_limit = ACCELERATION_RAD_PER_FRAME2 * fps * fps - 1e-7
    jerk_limit = 2.0 * acceleration_limit / float(np.min(delta_t)) if delta_t.size else 0.0
    projected = np.empty_like(raw)
    solvers = []

    def project_one(values: np.ndarray, lower: float, upper: float):
        count = len(values)
        if count == 1:
            candidate = np.asarray(values, dtype=np.float64).copy()
            violation = max(
                max(0.0, lower - float(candidate[0])),
                max(0.0, float(candidate[0]) - upper),
            )
            if not np.isfinite(candidate).all() or violation > 2e-6:
                raise RuntimeError(
                    "V4 single-frame trajectory projection infeasible: "
                    f"value={candidate[0]} bounds=[{lower}, {upper}] violation={violation}"
                )
            return candidate, {
                "iterations": 0,
                "objective": 0.0,
                "optimizer_success": True,
                "optimizer_message": "SINGLE_FRAME_NO_TEMPORAL_CONSTRAINTS",
                "feasible_termination": True,
                "max_constraint_violation": float(violation),
            }

        equality = np.zeros((1, count), dtype=np.float64)
        equality[0, 0] = 1.0
        operators = [equality, d1, d2]
        lows = [np.asarray([values[0]]), np.full(d1.shape[0], -velocity_limit), np.full(d2.shape[0], -acceleration_limit)]
        highs = [np.asarray([values[0]]), np.full(d1.shape[0], velocity_limit), np.full(d2.shape[0], acceleration_limit)]
        if d3.shape[0]:
            operators.append(d3)
            lows.append(np.full(d3.shape[0], -jerk_limit))
            highs.append(np.full(d3.shape[0], jerk_limit))
        matrix = np.vstack(operators)
        constraint = LinearConstraint(matrix, np.concatenate(lows), np.concatenate(highs))
        d3_frame = np.diff(np.eye(count), n=3, axis=0)
        regularization = 0.02
        hessian = np.eye(count) + regularization * (d3_frame.T @ d3_frame)

        def objective(candidate: np.ndarray) -> float:
            delta = candidate - values
            return 0.5 * float(delta @ delta) + 0.5 * regularization * float(np.linalg.norm(d3_frame @ candidate) ** 2)

        def jacobian(candidate: np.ndarray) -> np.ndarray:
            return hessian @ candidate - values

        solved = minimize(
            objective,
            np.full(count, float(values[0])),
            jac=jacobian,
            method="SLSQP",
            bounds=Bounds(np.full(count, lower), np.full(count, upper)),
            constraints=(constraint,),
            options={"ftol": 1e-12, "maxiter": 500, "disp": False},
        )
        candidate = np.asarray(solved.x, dtype=np.float64)
        velocity = d1 @ candidate
        acceleration = d2 @ candidate
        jerk = d3 @ candidate if d3.shape[0] else np.empty(0)

        def max_abs(values: np.ndarray) -> float:
            return float(np.max(np.abs(values))) if values.size else 0.0

        violation = max(
            abs(float(candidate[0] - values[0])),
            max(0.0, max_abs(velocity) - velocity_limit),
            max(0.0, max_abs(acceleration) - acceleration_limit),
            max(0.0, max_abs(jerk) - jerk_limit),
            max(0.0, lower - float(np.min(candidate))),
            max(0.0, float(np.max(candidate)) - upper),
        )
        if not np.isfinite(candidate).all() or violation > 2e-6:
            raise RuntimeError(f"V4 trajectory projection infeasible: status={solved.message} violation={violation}")
        return candidate, {
            "iterations": int(solved.nit),
            "objective": float(solved.fun),
            "optimizer_success": bool(solved.success),
            "optimizer_message": str(solved.message),
            "feasible_termination": True,
            "max_constraint_violation": float(violation),
        }

    for physical in range(2):
        for joint in range(raw.shape[2]):
            projected[:, physical, joint], row = project_one(
                raw[:, physical, joint],
                float(contracts[physical]["lower"][joint]),
                float(contracts[physical]["upper"][joint]),
            )
            solvers.append({"physical_side": SIDE_NAMES[physical], "joint_index": joint, **row})
    metrics = v2.trajectory_metrics(projected, d1, d2, d3)
    metrics.update({
        "velocity_max_rad_per_frame": metrics["velocity_max_rad_per_second"] / fps,
        "acceleration_max_rad_per_frame2": metrics["acceleration_max_rad_per_second2"] / (fps * fps),
        "fps": fps,
        "delta_t_seconds": delta_t.tolist(),
        "velocity_limit_rad_per_frame": VELOCITY_RAD_PER_FRAME,
        "acceleration_limit_rad_per_frame2": ACCELERATION_RAD_PER_FRAME2,
        "pass": bool(
            metrics["velocity_max_rad_per_second"] <= VELOCITY_RAD_PER_FRAME * fps + 2e-6
            and metrics["acceleration_max_rad_per_second2"] <= ACCELERATION_RAD_PER_FRAME2 * fps * fps + 2e-6
        ),
    })
    return projected, metrics, solvers


def trajectory_feature_metrics(
    official: Any,
    adapter: Any,
    contracts: list[dict[str, Any]],
    targets: list[list[dict[str, dict[str, np.ndarray]]]],
    thumb_rotations: np.ndarray,
    q: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for frame in range(24):
        for physical in range(2):
            for finger in FINGERS:
                actual = robot_finger_features(official, adapter, contracts[physical], q[frame, physical], finger, thumb_rotations[physical])
                row = feature_row(actual, targets[frame][physical][finger])
                row.update({"frame_slot": frame, "physical_side": SIDE_NAMES[physical], "finger": finger})
                row["pass"] = bool(
                    row["bone_error_deg_max"] <= SHAPE_MAX_DEG
                    and row["tip_direction_error_deg"] <= SHAPE_MAX_DEG
                    and row["mcp_normalized_l2"] <= MCP_NORMALIZED_MAX
                )
                rows.append(row)
    return rows, {
        "rows": len(rows),
        "pass_rows": sum(row["pass"] for row in rows),
        "all_pass": bool(all(row["pass"] for row in rows)),
        "bone_error_deg_max": float(max(row["bone_error_deg_max"] for row in rows)),
        "bone_error_deg_p95": float(np.percentile([row["bone_error_deg_max"] for row in rows], 95)),
        "tip_direction_error_deg_max": float(max(row["tip_direction_error_deg"] for row in rows)),
        "mcp_normalized_l2_max": float(max(row["mcp_normalized_l2"] for row in rows)),
        "thresholds": {"bone_and_tip_max_deg": SHAPE_MAX_DEG, "mcp_normalized_l2_max": MCP_NORMALIZED_MAX},
    }


def exact_sat(
    canary: Any,
    official: Any,
    contracts: list[dict[str, Any]],
    cache: dict[Any, Any],
    sat: Any,
    q: np.ndarray,
) -> dict[str, Any]:
    rows = []
    for frame in range(24):
        for physical in range(2):
            row = canary.distinct_finger_self_sat(
                contracts[physical]["model"], q[frame, physical], contracts[physical]["names"], official, cache, sat,
            )
            row.update({"frame_slot": frame, "physical_side": SIDE_NAMES[physical]})
            rows.append(row)
            if not row["pass"]:
                return {"pass": False, "evaluated_rows": len(rows), "required_rows": 48, "first_failure": row, "rows": rows}
    return {"pass": True, "evaluated_rows": 48, "required_rows": 48, "first_failure": None, "rows": rows}


def arm_candidate(
    official: Any,
    v2: Any,
    canary: Any,
    assets: Any,
    mounts: np.ndarray,
    q_initial: np.ndarray,
    base_initial: np.ndarray,
    wrist_targets: np.ndarray,
    source_frames: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lower, upper = official._arm_limits(assets)
    tool_targets = np.asarray([[wrist_targets[frame, side] @ np.linalg.inv(mounts[side]) for side in range(2)] for frame in range(24)])
    base, raw, nfev = official._joint_base_refinement(
        assets, base=base_initial, q_arm=q_initial, targets=tool_targets,
        valid=np.ones((24, 2), dtype=bool), lower=lower, upper=upper,
    )
    contract = [{"lower": lower[side], "upper": upper[side]} for side in range(2)]
    projected, temporal, solvers = temporal_project(v2, raw, contract, source_frames, fps)
    rows = []
    for frame in range(24):
        for side in range(2):
            actual = base @ official._tool_fk(assets, side, projected[frame, side]) @ mounts[side]
            position_mm, rotation_deg = canary.pose_error(actual, wrist_targets[frame, side], official)
            rows.append({
                "frame_slot": frame,
                "physical_side": SIDE_NAMES[side],
                "position_mm": position_mm,
                "rotation_deg": rotation_deg,
                "pass": bool(position_mm <= 10.0 and rotation_deg <= 5.0),
            })
    return base, projected, {
        "method": "ONE_SHARED_SE3_OFFICIAL_JOINT_BASE_REFINEMENT_THEN_TRUE_DT_TRAJECTORY_PROJECTION",
        "shared_refinement_nfev": nfev,
        "pose": {
            "rows": rows,
            "pass_rows": sum(row["pass"] for row in rows),
            "required_rows": 48,
            "all_pass": bool(all(row["pass"] for row in rows)),
            "position_mm_max": float(max(row["position_mm"] for row in rows)),
            "rotation_deg_max": float(max(row["rotation_deg"] for row in rows)),
        },
        "temporal": temporal,
        "projection_solvers": solvers,
        "arm_specific_mount": False,
    }


def run_weight_candidate(
    official: Any,
    adapter: Any,
    v2: Any,
    canary: Any,
    contracts: list[dict[str, Any]],
    targets: list[list[dict[str, dict[str, np.ndarray]]]],
    thumb_rotations: np.ndarray,
    q_v2: np.ndarray,
    source_frames: np.ndarray,
    fps: float,
    weights: dict[str, Any],
    cache: dict[Any, Any],
    sat: Any,
) -> dict[str, Any]:
    q_raw = np.empty_like(q_v2)
    solve_rows = []
    previous = np.asarray([contract["neutral"] for contract in contracts])
    for frame in range(24):
        started = time.perf_counter()
        q_raw[frame], rows = solve_frame(
            official, adapter, contracts, targets, thumb_rotations, previous, q_v2[frame], frame, weights,
        )
        previous = q_raw[frame]
        solve_rows.extend(rows)
        print(json.dumps({
            "event": "V4_HAND_FRAME_SOLVED",
            "weights": weights["name"],
            "frame_slot": frame,
            "runtime_seconds": time.perf_counter() - started,
        }), flush=True)
    q_projected, temporal, projection_solvers = temporal_project(v2, q_raw, contracts, source_frames, fps)
    neutral = np.asarray([contract["neutral"] for contract in contracts])[None]
    homotopy_rows = []
    selected = None
    for alpha in NEUTRAL_HOMOTOPY:
        candidate = (1.0 - alpha) * q_projected + alpha * neutral
        feature_rows, feature_summary = trajectory_feature_metrics(
            official, adapter, contracts, targets, thumb_rotations, candidate,
        )
        sat_result = exact_sat(canary, official, contracts, cache, sat, candidate)
        row = {
            "alpha": alpha,
            "feature_summary": feature_summary,
            "exact_distinct_finger_sat": sat_result,
        }
        homotopy_rows.append(row)
        if sat_result["pass"]:
            selected = (candidate, feature_rows, feature_summary, sat_result, alpha)
            break
    if selected is None:
        candidate = q_projected
        feature_rows, feature_summary = trajectory_feature_metrics(
            official, adapter, contracts, targets, thumb_rotations, candidate,
        )
        sat_result = homotopy_rows[0]["exact_distinct_finger_sat"]
        selected = (candidate, feature_rows, feature_summary, sat_result, 0.0)
    q_selected, feature_rows, feature_summary, sat_result, alpha = selected
    runtime_by_frame = []
    for frame in range(24):
        frame_rows = [row for row in solve_rows if row["frame_slot"] == frame]
        runtime_by_frame.append({
            "frame_slot": frame,
            "solver_runtime_seconds": float(sum(row["runtime_seconds"] for row in frame_rows)),
            "nfev_selected_total": int(sum(row["nfev"] for row in frame_rows)),
        })
    return {
        "weights": weights,
        "q_raw": q_raw,
        "q_selected": q_selected,
        "solve_rows": solve_rows,
        "runtime_by_frame": runtime_by_frame,
        "temporal": temporal,
        "projection_solvers": projection_solvers,
        "feature_rows": feature_rows,
        "feature_summary": feature_summary,
        "exact_distinct_finger_sat": sat_result,
        "neutral_homotopy_selected_alpha": alpha,
        "neutral_homotopy_search": homotopy_rows,
    }


def protocol_payload(weights: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "newtask-robot-shared-v4-hand-protocol-v1",
        "status": "FROZEN_AFTER_CHIPS005_FIT_BEFORE_POKER001_EVAL",
        "fit": {"task_id": "chips", "session_id": "get_potato_chips_0901_005"},
        "eval": {"task_id": "poker", "session_id": "play_cards_0901_001"},
        "selected_weights": weights,
        "fit_grid": [dict(row) for row in FIT_GRID],
        "first_frame_seeds": ["v2", "midpoint", "zero"],
        "later_frame_seeds": ["warm_start", "v2"],
        "max_nfev_per_seed_per_finger": MAX_NFEV,
        "neutral_homotopy": list(NEUTRAL_HOMOTOPY),
        "gates": {
            "bone_and_tip_max_deg": SHAPE_MAX_DEG,
            "mcp_normalized_l2_max": MCP_NORMALIZED_MAX,
            "velocity_rad_per_frame": VELOCITY_RAD_PER_FRAME,
            "acceleration_rad_per_frame2": ACCELERATION_RAD_PER_FRAME2,
            "exact_distinct_finger_sat_required_rows": 48,
        },
        "runner": evidence(Path(__file__)),
        "official_fk": evidence(PROJECT / "pipeline/robot_scene_state_cpu.py"),
        "adapter": evidence(PROJECT / "pipeline/robot_wrist_kai_adapter.py"),
        "mount_numeric_authority": evidence(MOUNT_PIN),
        "thumb_adapter": evidence(THUMB),
        "session_specific_adjustment": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fit", "eval"), required=True)
    parser.add_argument("--task-id", choices=("chips", "poker"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--v2-result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    if args.output_root.exists():
        raise RuntimeError(f"fresh output root required: {args.output_root}")
    if args.mode == "fit" and (args.task_id, args.session_id) != ("chips", "get_potato_chips_0901_005"):
        raise RuntimeError("FIT identity must be chips005")
    if args.mode == "eval" and (args.task_id, args.session_id) != ("poker", "play_cards_0901_001"):
        raise RuntimeError("held-out EVAL identity must be poker001")

    from pipeline import robot_scene_state_cpu as official
    from pipeline import robot_wrist_kai_adapter as adapter
    from pipeline.robot_contact_geometry import triangle_triangle_intersects_sat
    from pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

    canary = load(CANARY, "newtask_robot_shared_v4_canary")
    v2 = load(V2, "newtask_robot_shared_v4_v2")
    assets = load_pinned_robot_assets(PROJECT)
    mounts, mount_authority = pinned_mounts()
    contracts = model_contract(official, adapter, assets)
    with np.load(THUMB, allow_pickle=False) as archive:
        thumb_rotations = np.asarray(archive["rotations"], dtype=np.float64)

    v2_result_path = args.v2_result.resolve(strict=True)
    v2_result = json.loads(v2_result_path.read_text())
    with np.load(Path(v2_result["outputs"]["scene"]["path"]), allow_pickle=False) as archive:
        q_arm_v2 = np.asarray(archive["q_arm"], dtype=np.float64)
        q_hand_v2 = np.asarray(archive["q_hand"], dtype=np.float64)
        base_v2 = np.asarray(archive["T_world_rig"], dtype=np.float64)
        local_frames = np.asarray(archive["local_frames"], dtype=np.int64)
        source_frames = np.asarray(archive["source_frames"], dtype=np.int64)
    hawor_path = Path(v2_result["upstream"]["hawor_npz"]["path"]).resolve(strict=True)
    with np.load(hawor_path, allow_pickle=False) as archive:
        world = np.asarray(archive["joints_3d_world"], dtype=np.float64)
        fps = float(np.asarray(archive["fps"]).item())
        names = tuple(str(value) for value in archive["mano_joint_names"].tolist())
        sides = tuple(str(value) for value in archive["anatomical_side_names"].tolist())
    if names != canary.MANO_NAMES or sides != SIDE_NAMES or q_hand_v2.shape != (24, 2, 22):
        raise RuntimeError("MANO/Kai identity or trajectory shape drift")

    targets = []
    wrist_targets = np.zeros((24, 2, 4, 4), dtype=np.float64)
    for slot, local_frame in enumerate(local_frames):
        frame_targets = []
        for physical in range(2):
            human = PHYSICAL_TO_HUMAN[physical]
            points = world[human, local_frame]
            frame_targets.append(human_features(adapter, points, SIDE_NAMES[human]))
            wrist_targets[slot, physical] = np.eye(4)
            wrist_targets[slot, physical, :3, :3] = (
                adapter.final_v3_mano_palm_basis(points, handedness=SIDE_NAMES[human])
                @ contracts[physical]["basis"].T
            )
            wrist_targets[slot, physical, :3, 3] = points[0]
        targets.append(frame_targets)

    base_arm, q_arm, arm = arm_candidate(
        official, v2, canary, assets, mounts, q_arm_v2, base_v2, wrist_targets, source_frames, fps,
    )
    mesh_cache = canary.mesh_cache_for_models((contracts[0]["model"], contracts[1]["model"]))
    if args.mode == "fit":
        if args.protocol.exists():
            raise RuntimeError("FIT protocol path must be fresh")
        candidates = []
        for weights in FIT_GRID:
            candidate = run_weight_candidate(
                official, adapter, v2, canary, contracts, targets, thumb_rotations,
                q_hand_v2, source_frames, fps, weights, mesh_cache, triangle_triangle_intersects_sat,
            )
            candidates.append(candidate)
        candidates.sort(key=lambda row: (
            not row["exact_distinct_finger_sat"]["pass"],
            -row["feature_summary"]["pass_rows"],
            row["feature_summary"]["bone_error_deg_p95"],
            row["weights"]["name"],
        ))
        selected = candidates[0]
        atomic_json(args.protocol, protocol_payload(selected["weights"]))
        fit_candidates = [{
            "weights": row["weights"],
            "feature_summary": row["feature_summary"],
            "exact_distinct_finger_sat": row["exact_distinct_finger_sat"],
            "neutral_homotopy_selected_alpha": row["neutral_homotopy_selected_alpha"],
        } for row in candidates]
    else:
        protocol = json.loads(args.protocol.resolve(strict=True).read_text())
        expected = protocol_payload(protocol["selected_weights"])
        for field in ("schema_version", "status", "fit", "eval", "selected_weights", "first_frame_seeds", "later_frame_seeds", "max_nfev_per_seed_per_finger", "neutral_homotopy", "gates"):
            if protocol.get(field) != expected[field]:
                raise RuntimeError(f"frozen V4 protocol drift: {field}")
        for field in ("runner", "official_fk", "adapter", "mount_numeric_authority", "thumb_adapter"):
            if protocol.get(field, {}).get("sha256") != expected[field]["sha256"]:
                raise RuntimeError(f"frozen V4 authority drift: {field}")
        selected = run_weight_candidate(
            official, adapter, v2, canary, contracts, targets, thumb_rotations,
            q_hand_v2, source_frames, fps, protocol["selected_weights"], mesh_cache, triangle_triangle_intersects_sat,
        )
        fit_candidates = None

    args.output_root.mkdir(parents=True)
    trajectory_path = args.output_root / "SHARED_V4_TRAJECTORY24.npz"
    np.savez_compressed(
        trajectory_path,
        q_arm=q_arm,
        q_hand=selected["q_selected"],
        T_world_rig=base_arm,
        T_tool_hand=mounts,
        local_frames=local_frames,
        source_frames=source_frames,
        fps=np.asarray(fps),
    )
    gates = {
        "G0_source_terminal": v2_result.get("gates", {}).get("G0_source_terminal", "UNKNOWN"),
        "G1_mano21_identity_provenance": "PASS",
        "G2_shared_fit_eval_contract": "PASS" if args.mode == "eval" else "PASS_FIT_FROZEN",
        "G3_one_shared_se3_official_joint_base_refinement": "PASS" if arm["pose"]["all_pass"] else "HOLD_POSE_AFTER_TEMPORAL_PROJECTION",
        "G4_arm_true_dt_limits": "PASS" if arm["temporal"]["pass"] else "HOLD_TEMPORAL_LIMIT",
        "G5_kai22_normalized_mcp_bone_tip": "PASS" if selected["feature_summary"]["all_pass"] else "HOLD_HAND_MORPHOLOGY_OR_RETARGET",
        "G6_hand_true_dt_limits": "PASS" if selected["temporal"]["pass"] else "HOLD_TEMPORAL_LIMIT",
        "G7_exact_distinct_finger_sat_all24_bilateral": "PASS" if selected["exact_distinct_finger_sat"]["pass"] else "HOLD_SELF_COLLISION",
        "G8_object6d_contact_nonpenetration": "NOT_EVALUATED_NO_OBJECT6D",
        "G9_mask_clean_authority": "NOT_EVALUATED_MASK_AND_CLEAN_PENDING",
    }
    numeric_pass = all(gates[key] == "PASS" for key in ("G3_one_shared_se3_official_joint_base_refinement", "G4_arm_true_dt_limits", "G5_kai22_normalized_mcp_bone_tip", "G6_hand_true_dt_limits", "G7_exact_distinct_finger_sat_all24_bilateral"))
    result = {
        "schema_version": "newtask-robot-shared-v4-hand-result-v1",
        "status": "HOLD_NO_OBJECT6D_CONTACT_MASK_CLEAN_AUTHORITY" if numeric_pass else "HOLD_SHARED_KINEMATIC_GATE",
        "task_id": args.task_id,
        "session_id": args.session_id,
        "mode": args.mode,
        "formal_robot_ready": False,
        "visual_kinematic_candidate": numeric_pass,
        "gates": gates,
        "arm": arm,
        "hand": {
            "weights": selected["weights"],
            "runtime_by_frame": selected["runtime_by_frame"],
            "solve_rows": selected["solve_rows"],
            "temporal": selected["temporal"],
            "feature_summary": selected["feature_summary"],
            "feature_rows": selected["feature_rows"],
            "exact_distinct_finger_sat": selected["exact_distinct_finger_sat"],
            "neutral_homotopy_selected_alpha": selected["neutral_homotopy_selected_alpha"],
            "neutral_homotopy_search": selected["neutral_homotopy_search"],
            "fit_candidates": fit_candidates,
        },
        "authority": {
            "runner": evidence(Path(__file__)),
            "protocol": evidence(args.protocol),
            "v2_result": evidence(v2_result_path),
            "hawor": evidence(hawor_path),
            "mount_numeric_authority": mount_authority,
            "thumb_adapter": evidence(THUMB),
            "trajectory": evidence(trajectory_path),
        },
        "claim_limit": "CPU Kai official-FK kinematic audit only. NO_OBJECT6D; NO CONTACT CLAIM; CLEAN PENDING; development-only mount; no RobotRGB or training admission.",
        "next_interface_if_hold": {
            "input": "MANO21 wrist-relative normalized MCP positions, 20 unit-bone directions, 5 unit fingertip directions, physical side, previous q and true delta-t",
            "output": "22DoF bounded Kai q plus confidence and collision margin",
            "recommended": "Amortized shared retarget regressor trained on offline constrained-solver trajectories; retain exact FK/limits/v/a/SAT gates at inference.",
        },
        "gpu_calls": 0,
        "wall_seconds": time.time() - started,
    }
    atomic_json(args.output_root / "RESULT.json", result)
    print(json.dumps({
        "event": "SHARED_V4_HAND_COMPLETE",
        "task": args.task_id,
        "session": args.session_id,
        "mode": args.mode,
        "status": result["status"],
        "gates": gates,
        "hand_feature_summary": selected["feature_summary"],
        "hand_sat": {key: selected["exact_distinct_finger_sat"][key] for key in ("pass", "evaluated_rows", "required_rows", "first_failure")},
        "wall_seconds": result["wall_seconds"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
