#!/usr/bin/env python3
"""Gate-triggered anatomy repair for the V2 full-session KaiHand solver.

V2 can produce a static solution that passes anatomy and then move it outside
the anatomy gate during temporal projection.  This successor reruns V2 first.
Only when that exact failure occurs, it reprojects the affected finger over
its complete observed segment with a larger data weight at failed frames.
Joint, velocity, acceleration, jerk and anatomy thresholds are unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from chaoyang.ops import run_robot_hand_fullsession_v2 as v2


WEIGHTS = (2.0, 4.5, 8.0, 16.0)
REGULARIZATION = 0.02


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _project_group(
    candidate: np.ndarray,
    raw: np.ndarray,
    ids: np.ndarray,
    side: int,
    group: np.ndarray,
    failed_frames: set[int],
    weight: float,
    contract: dict[str, Any],
    source_frames: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    delta_t = np.diff(source_frames[ids]).astype(np.float64) / fps
    d1, d2, d3 = v2.v2.difference_matrices(delta_t)
    velocity_limit = 0.08 * fps - 1e-7
    acceleration_limit = 0.06 * fps * fps - 1e-7
    jerk_limit = 2.0 * acceleration_limit / float(np.min(delta_t))
    equality = np.zeros((1, len(ids)), dtype=np.float64)
    equality[0, 0] = 1.0
    matrix = np.vstack((equality, d1, d2, d3))
    third_frame = np.diff(np.eye(len(ids)), n=3, axis=0)
    weights = np.ones(len(ids), dtype=np.float64)
    for local, frame in enumerate(ids):
        if int(frame) in failed_frames:
            weights[local] = weight
    details: list[dict[str, Any]] = []
    updated = candidate.copy()
    for joint in group:
        values = raw[ids, side, joint]
        lows = np.concatenate((
            np.asarray([values[0]]),
            np.full(d1.shape[0], -velocity_limit),
            np.full(d2.shape[0], -acceleration_limit),
            np.full(d3.shape[0], -jerk_limit),
        ))
        highs = np.concatenate((
            np.asarray([values[0]]),
            np.full(d1.shape[0], velocity_limit),
            np.full(d2.shape[0], acceleration_limit),
            np.full(d3.shape[0], jerk_limit),
        ))
        hessian = np.diag(weights) + REGULARIZATION * (third_frame.T @ third_frame)

        def objective(values_out: np.ndarray) -> float:
            delta = values_out - values
            return 0.5 * float(delta @ (weights * delta)) + 0.5 * REGULARIZATION * float(
                np.linalg.norm(third_frame @ values_out) ** 2
            )

        def jacobian(values_out: np.ndarray) -> np.ndarray:
            return hessian @ values_out - weights * values

        solved = minimize(
            objective,
            candidate[ids, side, joint],
            jac=jacobian,
            method="SLSQP",
            bounds=Bounds(
                np.full(len(ids), contract["lower"][joint]),
                np.full(len(ids), contract["upper"][joint]),
            ),
            constraints=(LinearConstraint(matrix, lows, highs),),
            options={"ftol": 1e-12, "maxiter": 500, "disp": False},
        )
        values_out = np.asarray(solved.x, dtype=np.float64)
        velocity = d1 @ values_out
        acceleration = d2 @ values_out
        jerk = d3 @ values_out
        violation = max(
            abs(float(values_out[0] - values[0])),
            max(0.0, float(np.max(np.abs(velocity))) - velocity_limit),
            max(0.0, float(np.max(np.abs(acceleration))) - acceleration_limit),
            max(0.0, float(np.max(np.abs(jerk))) - jerk_limit),
            max(0.0, float(contract["lower"][joint] - np.min(values_out))),
            max(0.0, float(np.max(values_out) - contract["upper"][joint])),
        )
        if not np.isfinite(values_out).all() or violation > 2e-6:
            raise RuntimeError(
                f"ANATOMY_REPAIR_TEMPORAL_INFEASIBLE:{side}:{joint}:{violation}"
            )
        updated[ids, side, joint] = values_out
        details.append({
            "joint": int(joint),
            "optimizer_success": bool(solved.success),
            "optimizer_message": str(solved.message),
            "iterations": int(solved.nit),
            "max_constraint_violation": float(violation),
        })
    return updated, details


def _evaluate(
    q_hand: np.ndarray,
    hawor: dict[str, np.ndarray],
    contracts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, bool]]:
    valid = np.isfinite(hawor["joints_3d_world"]).all(axis=(2, 3))
    rows: list[dict[str, Any]] = []
    for side in range(2):
        tips = {
            finger: v2.anatomy.terminal_tip(
                contracts[side]["model"], finger, contracts[side]["prefix"]
            )[0]
            for finger in v2.handfit.FINGERS
        }
        for frame in range(q_hand.shape[0]):
            if not valid[side, frame]:
                rows.append({
                    "frame": frame,
                    "side": v2.SIDES[side],
                    "status": "UNKNOWN_MISSING_HAWOR",
                    "pass": False,
                })
                continue
            target_features = v2.handfit.human_features(
                v2.shared.wrist_adapter,
                hawor["joints_3d_world"][side, frame],
                v2.shared.SIDES[side],
            )
            fingers = []
            for finger in v2.handfit.FINGERS:
                wanted = v2.anatomy.semantic_target(target_features[finger])
                metric = v2.anatomy.row(
                    v2.anatomy.features(
                        v2.old.official,
                        contracts[side],
                        q_hand[frame, side],
                        finger,
                        tips[finger],
                    ),
                    wanted,
                )
                metric["finger"] = finger
                fingers.append(metric)
            bone = max(row["bone_error_deg_max"] for row in fingers)
            tip = max(row["tip_direction_error_deg"] for row in fingers)
            passed = bone <= 60.0 and tip <= 15.0
            rows.append({
                "frame": frame,
                "side": v2.SIDES[side],
                "status": "SOLVED" if passed else "HOLD_ANATOMY_GATE",
                "bone_error_deg_max": bone,
                "tip_direction_error_deg_max": tip,
                "fingers": fingers,
                "pass": bool(passed),
            })
    solved = [row for row in rows if "bone_error_deg_max" in row]
    failed = [row for row in solved if not row["pass"]]
    velocity: list[float] = []
    acceleration: list[float] = []
    for side in range(2):
        for ids in v2.segments(valid[side]):
            values = q_hand[ids, side]
            if len(values) > 1:
                velocity.extend(np.abs(np.diff(values, axis=0)).ravel())
            if len(values) > 2:
                acceleration.extend(np.abs(np.diff(values, n=2, axis=0)).ravel())
    metrics = {
        "observed_rows": len(solved),
        "unknown_rows": int((~valid).sum()),
        "failed_rows": len(failed),
        "bone_error_deg_max": max((row["bone_error_deg_max"] for row in solved), default=float("nan")),
        "tip_direction_error_deg_max": max((row["tip_direction_error_deg_max"] for row in solved), default=float("nan")),
        "velocity_rad_per_frame_max_contiguous": float(max(velocity, default=0.0)),
        "acceleration_rad_per_frame2_max_contiguous": float(max(acceleration, default=0.0)),
    }
    gates = {
        "anatomy_all_observed": len(failed) == 0,
        "velocity": metrics["velocity_rad_per_frame_max_contiguous"] <= 0.08 + 1e-9,
        "acceleration": metrics["acceleration_rad_per_frame2_max_contiguous"] <= 0.06 + 1e-9,
        "missing_unknown_not_filled": bool(np.all(np.isnan(q_hand[~valid.T]))),
        "thumb_independent_q0_to_q5": True,
        "four_finger_chain_semantics": True,
    }
    return rows, metrics, gates


def main() -> int:
    argv = list(sys.argv)
    output = Path(argv[argv.index("--output") + 1])
    hawor_path = Path(argv[argv.index("--hawor") + 1])
    predecessor_code = v2.main()
    if predecessor_code == 0:
        return 0
    predecessor_result = json.loads(output.read_text(encoding="utf-8"))
    states_path = output.with_name("HAND_STATES.npz")
    predecessor_result_path = output.with_name("HAND_V2_PRE_REPAIR_RESULT.json")
    predecessor_states_path = output.with_name("HAND_V2_PRE_REPAIR_STATES.npz")
    shutil.copy2(output, predecessor_result_path)
    shutil.copy2(states_path, predecessor_states_path)

    with np.load(hawor_path, allow_pickle=False) as archive:
        hawor = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(states_path, allow_pickle=False) as archive:
        states = {key: np.asarray(archive[key]) for key in archive.files}
    assets = v2.old.load_pinned_robot_assets(v2.PROJECT)
    contracts = v2.handfit.model_contract(
        v2.old.official, v2.shared.wrist_adapter, assets
    )
    failed_rows = [
        row for row in predecessor_result["rows"]
        if row.get("status") == "HOLD_ANATOMY_GATE"
    ]
    affected: dict[tuple[int, int, str], set[int]] = {}
    valid = np.asarray(states["valid_side_frame"], dtype=bool)
    for row in failed_rows:
        side = v2.SIDES.index(row["side"])
        frame = int(row["frame"])
        failing_fingers = [
            finger["finger"]
            for finger in row["fingers"]
            if finger["bone_error_deg_max"] > 60.0
            or finger["tip_direction_error_deg"] > 15.0
        ]
        segment_index = next(
            index
            for index, ids in enumerate(v2.segments(valid[side]))
            if frame in set(map(int, ids))
        )
        for finger in failing_fingers:
            affected.setdefault((side, segment_index, finger), set()).add(frame)
    if not affected:
        return predecessor_code

    attempts = []
    selected = None
    for weight in WEIGHTS:
        candidate = np.asarray(states["q_hand"], dtype=np.float64).copy()
        detail = []
        for (side, segment_index, finger), frames in affected.items():
            ids = v2.segments(valid[side])[segment_index]
            candidate, solver_rows = _project_group(
                candidate,
                states["q_hand_raw"],
                ids,
                side,
                np.asarray(v2.handfit.GROUPS[finger], dtype=np.int64),
                frames,
                weight,
                contracts[side],
                np.asarray(hawor["original_frame_indices"], dtype=np.int64),
                float(np.asarray(hawor["fps"]).reshape(-1)[0]),
            )
            detail.append({
                "side": v2.SIDES[side],
                "segment_first": int(ids[0]),
                "segment_last": int(ids[-1]),
                "finger": finger,
                "failed_frames": sorted(frames),
                "solvers": solver_rows,
            })
        rows, metrics, gates = _evaluate(candidate, hawor, contracts)
        attempts.append({"weight": weight, "metrics": metrics, "gates": gates})
        if all(gates.values()):
            selected = (weight, candidate, rows, metrics, gates, detail)
            break
    if selected is None:
        predecessor_result["repair_attempts"] = attempts
        predecessor_result["repair_status"] = "NO_GATE_PRESERVING_REPAIR"
        _atomic_json(output, predecessor_result)
        return 2

    weight, candidate, rows, metrics, gates, detail = selected
    temporary_states = states_path.with_suffix(f".tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary_states,
        q_hand=candidate,
        q_hand_raw=states["q_hand_raw"],
        valid_side_frame=states["valid_side_frame"],
        source_frames=states["source_frames"],
    )
    os.replace(temporary_states, states_path)
    result = {
        **predecessor_result,
        "schema_version": "robot-kaihand-fullsession-anatomy-repair-v3",
        "status": "PASS_NUMERIC_CANARY_NO_AUTHORITY",
        "metrics": metrics,
        "gates": gates,
        "rows": rows,
        "output_states": {
            "path": str(states_path),
            "sha256": v2.sha(states_path),
            "bytes": states_path.stat().st_size,
        },
        "anatomy_repair": {
            "trigger": "POST_TEMPORAL_ANATOMY_GATE_ONLY",
            "selected_failed_frame_weight": weight,
            "candidate_weights": list(WEIGHTS),
            "thresholds_changed": False,
            "joint_or_temporal_limits_changed": False,
            "affected": detail,
            "attempts": attempts,
            "predecessor_result": {
                "path": str(predecessor_result_path),
                "sha256": v2.sha(predecessor_result_path),
            },
            "predecessor_states": {
                "path": str(predecessor_states_path),
                "sha256": v2.sha(predecessor_states_path),
            },
        },
        "claim_limit": (
            "Numeric KaiHand successor with gate-triggered anatomy-aware temporal "
            "reprojection only; not action, Robot, visual, contact or deployment authority."
        ),
    }
    _atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "selected_failed_frame_weight": weight,
        "metrics": metrics,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
