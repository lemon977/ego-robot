#!/usr/bin/env python3
"""Causal bounded-IK visual trajectory for the corrected P1 mount.

Independent static IK solutions may lie on different branches.  Clipping
their joint vectors can therefore move the rendered hand far from the target.
This successor resolves every visual frame inside the unchanged velocity and
acceleration bounds, and reserves enough hard-limit distance to brake on the
following frame.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np

from chaoyang.governance.common import atomic_json
from chaoyang.ops import run_exact78_session_base_arm_v1 as base
from chaoyang.ops import run_exact78_session_base_arm_v6 as previous
from chaoyang.pipeline.robot_hand_visual_alignment import kai_palmar_axis_local


def _output_root() -> Path:
    return Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve(strict=True)


def _hawor_path() -> Path:
    return Path(sys.argv[sys.argv.index("--hawor") + 1]).resolve(strict=True)


def braking_bounds(lower: np.ndarray, upper: np.ndarray, previous_q: np.ndarray,
                   older: np.ndarray | None, step_limit: float) -> tuple[np.ndarray, np.ndarray]:
    acceleration = float(base.arm.temporal.ACCELERATION_LIMIT)
    previous_q = np.asarray(previous_q, dtype=np.float64)
    lo = np.maximum(lower, previous_q - float(step_limit))
    hi = np.minimum(upper, previous_q + float(step_limit))
    # Preserve one-frame deceleration feasibility at the hard limits.  These
    # bounds are derived from, rather than changes to, the 0.06 acceleration.
    lo = np.maximum(lo, 0.5 * (lower + previous_q - acceleration))
    hi = np.minimum(hi, 0.5 * (upper + previous_q + acceleration))
    if older is not None:
        predicted = 2.0 * previous_q - np.asarray(older, dtype=np.float64)
        lo = np.maximum(lo, predicted - acceleration)
        hi = np.minimum(hi, predicted + acceleration)
    if np.any(lo > hi + 1e-10):
        raise RuntimeError("EMPTY_BRAKING_BOUNDED_JOINT_INTERVAL")
    midpoint = 0.5 * (lo + hi)
    collapsed = hi - lo <= 1e-12
    lo[collapsed] = midpoint[collapsed]
    hi[collapsed] = midpoint[collapsed]
    return lo, hi


def causal_resolve() -> None:
    output = _output_root()
    state_path = output / "ARM_SESSION_BASE_STATES.npz"
    result_path = output / "RESULT.json"
    with np.load(state_path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]).copy() for key in archive.files}
    q_static = np.asarray(state["q_arm_static_pose"], dtype=np.float64)
    q = q_static.copy()
    target = np.asarray(state["T_target_hand_root_world"], dtype=np.float64)
    valid = np.asarray(state["valid_side_frame"], dtype=bool)
    world_base = np.asarray(state["T_world_base"], dtype=np.float64)
    base_world = np.linalg.inv(world_base)
    mounts = np.asarray(state["T_tool_hand_root"], dtype=np.float64)
    assets = base.arm.old.load_pinned_robot_assets(base.REPO_ROOT)
    lower, upper = base.arm.taskfit.arm_limits(assets)
    actual = np.asarray(state["T_actual_hand_root_world"], dtype=np.float64).copy()
    status = np.full(valid.shape, "NEUTRAL_NO_MOTION", dtype="<U40")
    passed = np.zeros(valid.shape, dtype=bool)
    failures = Counter()
    velocities: list[float] = []
    accelerations: list[float] = []

    for side in (0, 1):
        prior = None
        older = None
        segment: list[np.ndarray] = []
        for frame in range(len(q)):
            if not valid[side, frame]:
                prior = older = None
                segment = []
                continue
            desired = q_static[frame, side]
            target_tool = base_world @ target[frame, side] @ np.linalg.inv(mounts[side])
            if prior is None:
                selected = desired.copy()
            else:
                lo, hi = braking_bounds(
                    lower[side], upper[side], prior, older,
                    base.arm.temporal.ARM_SOLVER_STEP_LIMIT,
                )
                attempt = base.arm.solve_one(
                    assets, side, target_tool, lower[side], upper[side],
                    [prior, np.clip(desired, lo, hi), 0.5 * (lo + hi)],
                    previous=prior, lo=lo, hi=hi,
                )
                *_, selected, _, _, _, _ = attempt
            q[frame, side] = selected
            actual[frame, side] = (
                world_base @ base.arm.old.official._tool_fk(assets, side, selected)
                @ mounts[side]
            )
            position, rotation = base.arm.error(assets, side, selected, target_tool)
            margin = float(np.min(base.arm.margins(assets, side, selected)))
            reason = base.classify(float(position), float(rotation), margin)
            passed[side, frame] = reason == "PASS"
            status[side, frame] = "PASS" if reason == "PASS" else f"BOUNDED_IK_{reason}"
            if reason != "PASS":
                failures[status[side, frame]] += 1
            if segment:
                velocities.extend(np.abs(selected - segment[-1]).tolist())
            if len(segment) >= 2:
                accelerations.extend(
                    np.abs(selected - 2.0 * segment[-1] + segment[-2]).tolist()
                )
            segment.append(selected.copy())
            older, prior = prior, selected

    state["q_arm"] = q
    state["T_actual_hand_root_world"] = actual
    state["per_frame_status"] = status
    state["causal_bounded_ik_pass"] = passed
    temporary = state_path.with_name(f".{state_path.stem}.v7-{os.getpid()}.npz")
    np.savez_compressed(temporary, **state)
    os.replace(temporary, state_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = {(int(row["frame"]), row["side"]): row for row in result["rows"]}
    for side, side_name in enumerate(base.SIDES):
        for frame in range(len(q)):
            row = rows[(frame, side_name)]
            row["visual_trajectory_status"] = str(status[side, frame])
            row["visual_trajectory_pass"] = bool(passed[side, frame])
    visual_rows = int(valid.sum())
    metric = np.asarray(state["metric_observed_side_frame"], dtype=bool)
    result["schema_version"] = "exact78-session-base-arm-v7"
    result["metrics"]["visual_rows"] = visual_rows
    result["metrics"]["visual_passed_rows"] = int((passed & valid).sum())
    result["metrics"]["visual_pass_fraction"] = float((passed & valid).sum() / visual_rows)
    result["metrics"]["metric_observed_visual_passed_rows"] = int((passed & metric).sum())
    result["metrics"]["trajectory_failure_clusters"] = dict(failures)
    result["metrics"]["velocity_rad_per_frame_max_contiguous"] = float(max(velocities, default=0.0))
    result["metrics"]["acceleration_rad_per_frame2_max_contiguous"] = float(max(accelerations, default=0.0))

    with np.load(_hawor_path(), allow_pickle=False) as archive:
        c2w = np.asarray(archive["c2w"], dtype=np.float64)
    palms = {"left": [], "right": []}
    for frame in range(len(q)):
        camera_world = np.linalg.inv(c2w[frame])
        for side, side_name in enumerate(base.SIDES):
            root_camera = camera_world @ actual[frame, side]
            palms[side_name].append(float(
                (root_camera[:3, :3] @ kai_palmar_axis_local(side_name))[1]
            ))
    result["metrics"]["palmar_normal_camera_y"] = {
        name: {"minimum": float(np.min(values)), "mean": float(np.mean(values)),
               "positive_fraction": float(np.mean(np.asarray(values) > 0.0))}
        for name, values in palms.items()
    }
    result["runtime_correction"] = {
        "changed": ["flange_to_tool_mount_frame_conversion", "causal_bounded_ik",
                    "hard_limit_braking_feasibility"],
        "numeric_thresholds_changed": False,
        "model_changed": False,
        "ik_branch_selected_inside_temporal_bounds": True,
    }
    atomic_json(result_path, result)


def main() -> int:
    code = previous.main()
    if code == 0:
        causal_resolve()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
