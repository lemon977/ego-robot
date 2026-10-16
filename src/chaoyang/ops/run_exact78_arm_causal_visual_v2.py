#!/usr/bin/env python3
"""Fast causal Arm postprocess from one immutable terminal P1 solve."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
from chaoyang.ops import run_exact78_session_base_arm_v1 as base
from chaoyang.ops.run_exact78_session_base_arm_v7 import braking_bounds
from chaoyang.pipeline.robot_hand_visual_alignment import kai_palmar_axis_local


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(strict=True), allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in archive.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--source-states", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--hawor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"FRESH_OUTPUT_REQUIRED:{output}")
    output.mkdir(parents=True)
    source_states = args.source_states.resolve(strict=True)
    source_result = args.source_result.resolve(strict=True)
    hawor_path = args.hawor.resolve(strict=True)
    if args.session_id not in str(source_states) or args.session_id not in str(hawor_path):
        raise RuntimeError("SAME_SESSION_INPUTS_REQUIRED")
    state = arrays(source_states)
    hawor = arrays(hawor_path)
    source = json.loads(source_result.read_text(encoding="utf-8"))
    if source.get("status") != "COMPLETE_PER_FRAME_STATUS":
        raise RuntimeError("TERMINAL_ARM_RESULT_REQUIRED")
    if "q_arm_static_pose" not in state:
        raise RuntimeError("STATIC_POSE_TRAJECTORY_REQUIRED")

    q_static = np.asarray(state["q_arm_static_pose"], dtype=np.float64)
    q = q_static.copy()
    target = np.asarray(state["T_target_hand_root_world"], dtype=np.float64)
    valid = np.asarray(state["valid_side_frame"], dtype=bool)
    metric = np.asarray(state["metric_observed_side_frame"], dtype=bool)
    world_base = np.asarray(state["T_world_base"], dtype=np.float64)
    base_world = np.linalg.inv(world_base)
    mounts = np.asarray(state["T_tool_hand_root"], dtype=np.float64)
    assets = base.arm.old.load_pinned_robot_assets(REPO_ROOT)
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
            position, rotation = base.arm.arm.error(assets, side, selected, target_tool)
            margin = float(np.min(base.arm.arm.margins(assets, side, selected)))
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
    output_states = output / "ARM_CAUSAL_VISUAL_STATES.npz"
    temporary = output / f".ARM_CAUSAL_VISUAL_STATES-{os.getpid()}.npz"
    np.savez_compressed(temporary, **state)
    os.replace(temporary, output_states)

    c2w = np.asarray(hawor["c2w"], dtype=np.float64)
    intrinsics = np.asarray(hawor["intrinsics"], dtype=np.float64)
    palms = {"left": [], "right": []}
    root_steps = {"left": [], "right": []}
    previous_uv = [None, None]
    for frame in range(len(q)):
        camera_world = np.linalg.inv(c2w[frame])
        for side, side_name in enumerate(base.SIDES):
            root_camera = camera_world @ actual[frame, side]
            palms[side_name].append(float(
                (root_camera[:3, :3] @ kai_palmar_axis_local(side_name))[1]
            ))
            point = root_camera[:3, 3]
            uv = intrinsics[frame, :2, :2] @ (point[:2] / point[2]) + intrinsics[frame, :2, 2]
            if previous_uv[side] is not None:
                root_steps[side_name].append(float(np.linalg.norm(uv - previous_uv[side])))
            previous_uv[side] = uv

    visual_rows = int(valid.sum())
    result = {
        "schema_version": "exact78-arm-causal-visual-v2",
        "status": "COMPLETE_PER_FRAME_STATUS",
        "session_id": args.session_id,
        "inputs": {
            "terminal_source_states": artifact_ref(source_states),
            "terminal_source_result": artifact_ref(source_result),
            "hawor": artifact_ref(hawor_path),
        },
        "thresholds": {
            "position_mm_max": base.POSITION_GATE_MM,
            "rotation_deg_max": base.ROTATION_GATE_DEG,
            "velocity_rad_per_frame_max": float(base.arm.temporal.ARM_SOLVER_STEP_LIMIT),
            "acceleration_rad_per_frame2_max": float(base.arm.temporal.ACCELERATION_LIMIT),
            "changed": False,
        },
        "metrics": {
            "visual_rows": visual_rows,
            "visual_passed_rows": int((passed & valid).sum()),
            "visual_pass_fraction": float((passed & valid).sum() / visual_rows),
            "metric_observed_rows": int(metric.sum()),
            "metric_observed_visual_passed_rows": int((passed & metric).sum()),
            "trajectory_failure_clusters": dict(failures),
            "velocity_rad_per_frame_max_contiguous": float(max(velocities, default=0.0)),
            "acceleration_rad_per_frame2_max_contiguous": float(max(accelerations, default=0.0)),
            "hand_root_pixel_step": {
                name: {"max": float(max(values, default=0.0)),
                       "p95": float(np.percentile(values, 95)) if values else 0.0}
                for name, values in root_steps.items()
            },
            "palmar_normal_camera_y": {
                name: {"minimum": float(np.min(values)), "mean": float(np.mean(values)),
                       "positive_fraction": float(np.mean(np.asarray(values) > 0.0))}
                for name, values in palms.items()
            },
            "wall_seconds": time.monotonic() - started,
        },
        "outputs": {"states": artifact_ref(output_states)},
        "runtime_contract": {
            "input_static_solve_reused_by_sha": True,
            "ik_branch_selected_inside_temporal_bounds": True,
            "hard_limit_braking_feasibility": True,
            "model_changed": False,
            "numeric_thresholds_changed": False,
        },
        "visual_qualified": False,
        "training_eligible": False,
        "control_ground_truth": False,
        "claim_limit": "Fast postprocess of one immutable terminal Arm solve; no model, threshold, training, control or physical authority.",
    }
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
