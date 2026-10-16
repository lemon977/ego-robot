#!/usr/bin/env python3
"""Per-session Exact78 base solve with reachable anchors and smooth visual q.

This is a narrow successor to v2.  It keeps the 10 mm / 5 degree pose gates
and the existing joint velocity/acceleration limits unchanged, while fixing
two state errors exposed by P1:

* candidate scoring is anchored to the actually reachable same-session IK
  orientation instead of the often-unreachable neutral orientation;
* pose reachability is solved independently from visual trajectory smoothing,
  so a prior HOLD cannot make all later statically reachable frames fail.

The metric result remains OBSERVED-only.  The emitted ``q_arm`` is a separate
temporally bounded visual trajectory and every temporal miss is reported.
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json
from chaoyang.ops import run_exact78_session_base_arm_v1 as base


def reachable_anchor_candidate(original, spec, hawor, human, visual, sample,
                               anchors, assets, mount, lower, upper, neutral):
    row = original(
        spec, hawor, human, visual, sample, anchors,
        assets, mount, lower, upper, neutral,
    )
    world_base = row["T_world_base"]
    base_world = np.linalg.inv(world_base)
    anchor_rotations = []
    for side in (0, 1):
        actual = (
            world_base
            @ base.arm.old.official._tool_fk(assets, side, row["anchor_q"][side])
            @ mount[side]
        )
        anchor_rotations.append(actual[:3, :3])
    row["anchor_rotations"] = np.asarray(anchor_rotations)

    checks = []
    for frame in sample:
        for side in (0, 1):
            if not visual[side, frame]:
                continue
            root = base.proper(
                human[side, frame, 0] + row["offsets"][side],
                base.target_rotation(
                    human, side, int(frame), int(anchors[side]),
                    row["anchor_rotations"][side],
                ),
            )
            attempt = base.solve_global(
                assets,
                side,
                base_world @ root @ np.linalg.inv(mount[side]),
                lower[side],
                upper[side],
                [
                    row["anchor_q"][side],
                    neutral[side],
                    .25 * lower[side] + .75 * upper[side],
                    .75 * lower[side] + .25 * upper[side],
                ],
            )
            *_, _, _, position, rotation, margins = attempt
            passed = (
                position <= base.POSITION_GATE_MM
                and rotation <= base.ROTATION_GATE_DEG
                and float(np.min(margins)) >= -1e-8
            )
            checks.append(
                (bool(passed), float(position), float(rotation),
                 float(np.min(margins)))
            )
    passes = sum(item[0] for item in checks)
    row["sample_rows"] = len(checks)
    row["sample_passed"] = passes
    row["score"] = (
        -passes,
        sum(not item[0] for item in checks),
        float(np.median([item[1] for item in checks])) if checks else 1e9,
        float(np.median([item[2] for item in checks])) if checks else 1e9,
        -float(min((item[3] for item in checks), default=-1e9)),
    )
    return row


def _output_root() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()
    except (ValueError, IndexError) as error:
        raise RuntimeError("OUTPUT_ROOT_ARGUMENT_REQUIRED") from error


def _smooth_visual_trajectory(output: Path, original_bounded_limits) -> None:
    state_path = output / "ARM_SESSION_BASE_STATES.npz"
    with np.load(state_path, allow_pickle=False) as archive:
        state = {key: np.asarray(archive[key]).copy() for key in archive.files}
    result_path = output / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    q_static = np.asarray(state["q_arm"], dtype=np.float64)
    q_visual = q_static.copy()
    valid = np.asarray(state["valid_side_frame"], dtype=bool)
    target = np.asarray(state["T_target_hand_root_world"], dtype=np.float64)
    world_base = np.asarray(state["T_world_base"], dtype=np.float64)
    base_world = np.linalg.inv(world_base)
    mounts = np.asarray(state["T_tool_hand_root"], dtype=np.float64)
    frame_count = q_static.shape[0]
    assets = base.arm.old.load_pinned_robot_assets(base.REPO_ROOT)
    lower, upper = base.arm.taskfit.arm_limits(assets)

    visual_status = np.full((2, frame_count), "NEUTRAL_NO_MOTION", dtype="<U32")
    visual_pass = np.zeros((2, frame_count), dtype=bool)
    actual = np.asarray(state["T_actual_hand_root_world"], dtype=np.float64).copy()
    trajectory_failures = Counter()
    velocity_values = []
    acceleration_values = []

    for side in (0, 1):
        previous = None
        older = None
        segment_q = []
        for frame in range(frame_count):
            if not valid[side, frame]:
                previous = None
                older = None
                segment_q = []
                continue
            desired = q_static[frame, side]
            if previous is None:
                selected = desired.copy()
            else:
                try:
                    lo, hi = original_bounded_limits(
                        lower[side], upper[side], previous, older,
                        base.arm.temporal.ARM_SOLVER_STEP_LIMIT,
                    )
                    selected = np.clip(desired, lo, hi)
                except base.arm.temporal.TemporalReviewError:
                    selected = previous.copy()
            q_visual[frame, side] = selected
            actual[frame, side] = (
                world_base
                @ base.arm.old.official._tool_fk(assets, side, selected)
                @ mounts[side]
            )
            target_base = base_world @ target[frame, side] @ np.linalg.inv(mounts[side])
            position, rotation = base.arm.error(assets, side, selected, target_base)
            margin = float(np.min(base.arm.margins(assets, side, selected)))
            reason = base.classify(float(position), float(rotation), margin)
            passed = reason == "PASS"
            visual_pass[side, frame] = passed
            visual_status[side, frame] = "PASS" if passed else f"HOLD_TEMPORAL_{reason}"
            if not passed:
                trajectory_failures[visual_status[side, frame]] += 1
            if segment_q:
                velocity_values.extend(np.abs(selected - segment_q[-1]).tolist())
            if len(segment_q) >= 2:
                acceleration_values.extend(
                    np.abs(selected - 2 * segment_q[-1] + segment_q[-2]).tolist()
                )
            segment_q.append(selected.copy())
            older, previous = previous, selected

    old_status = np.asarray(state.get("per_frame_status"))
    state["q_arm_static_pose"] = q_static
    state["q_arm"] = q_visual
    state["T_actual_hand_root_world"] = actual
    state["static_pose_status"] = old_status
    state["per_frame_status"] = visual_status
    temporary = state_path.with_name(f".{state_path.stem}.v3-{os.getpid()}.npz")
    np.savez_compressed(temporary, **state)
    os.replace(temporary, state_path)

    by_key = {(int(item["frame"]), item["side"]): item for item in result["rows"]}
    for side, side_name in enumerate(base.SIDES):
        for frame in range(frame_count):
            item = by_key[(frame, side_name)]
            item["static_pose_status"] = item["status"]
            item["static_pose_pass"] = bool(item["pass"])
            item["visual_trajectory_status"] = str(visual_status[side, frame])
            item["visual_trajectory_pass"] = bool(visual_pass[side, frame])

    metric = np.asarray(state["metric_observed_side_frame"], dtype=bool)
    visual_rows = int(valid.sum())
    result["schema_version"] = "exact78-session-base-arm-v3"
    result["metrics"]["visual_rows"] = visual_rows
    result["metrics"]["visual_passed_rows"] = int((visual_pass & valid).sum())
    result["metrics"]["visual_pass_fraction"] = (
        float((visual_pass & valid).sum() / visual_rows) if visual_rows else None
    )
    result["metrics"]["metric_observed_visual_passed_rows"] = int(
        (visual_pass & metric).sum()
    )
    result["metrics"]["trajectory_failure_clusters"] = dict(trajectory_failures)
    result["metrics"]["velocity_rad_per_frame_max_contiguous"] = float(
        max(velocity_values, default=0.0)
    )
    result["metrics"]["acceleration_rad_per_frame2_max_contiguous"] = float(
        max(acceleration_values, default=0.0)
    )
    result["outputs"]["states"] = artifact_ref(state_path)
    result["separated_contract"] = {
        "metric_pose_reachability": "OBSERVED rows, independent full-limit IK, unchanged 10 mm / 5 degree / branch gates",
        "visual_trajectory": "causally clipped to existing velocity and acceleration bounds; temporal misses remain explicit",
        "numeric_thresholds_changed": False,
    }
    result["claim_limit"] = (
        "Per-session fixed-base offline visual IK. Static OBSERVED pose reachability "
        "and temporally bounded visual q are reported separately; neither is physical "
        "control authority or visual-quality approval."
    )
    atomic_json(result_path, result)


def main() -> int:
    output = _output_root()
    original_candidate = base.evaluate_candidate
    original_bounded_limits = base.arm.temporal.bounded_limits

    def candidate(*args, **kwargs):
        return reachable_anchor_candidate(original_candidate, *args, **kwargs)

    def independent_pose_limits(lower, upper, previous, previous_previous, step_limit):
        del previous, previous_previous, step_limit
        return lower.copy(), upper.copy()

    base.evaluate_candidate = candidate
    base.arm.temporal.bounded_limits = independent_pose_limits
    try:
        code = base.main()
    finally:
        base.evaluate_candidate = original_candidate
        base.arm.temporal.bounded_limits = original_bounded_limits
    if code != 0:
        return code
    _smooth_visual_trajectory(output, original_bounded_limits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
