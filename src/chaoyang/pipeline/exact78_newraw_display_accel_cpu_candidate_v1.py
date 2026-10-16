"""Cards001 CPU shadow with acceleration measured on the displayed trajectory.

The sealed velocity-tube implementation supplies the unchanged base search,
IK bounds and source checks. This candidate reruns all 201 frames from sealed
HaWoR; no frozen Robot or prior candidate states are motion inputs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
from chaoyang.pipeline import exact78_newraw_velocity_tube_cpu_candidate_v1 as tube


SESSION = tube.SESSION
FRAMES = tube.FRAMES
FPS = tube.FPS
BASELINE_ARM_PASS = tube.BASELINE_ARM_PASS


def displayed_acceleration(candidate: np.ndarray, previous_display: np.ndarray,
                           previous_previous_display: np.ndarray) -> float:
    """Max joint second difference of the candidate and two displayed frames."""
    return float(np.max(np.abs(candidate - 2 * previous_display
                               + previous_previous_display)))


def solve_display_accel(joints: np.ndarray, valid: np.ndarray, fps: float) -> tuple[
        dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if joints.shape != (2, FRAMES, 21, 3) or valid.shape != (2, FRAMES) or fps != FPS:
        raise stage.FreshRobotError("DISPLAY_ACCEL_INPUT_SHAPE_OR_FPS_DRIFT")
    shadow, base_report, shadow_rows, captured = tube._shadow_and_base(joints, valid, fps)
    for side in stage.SIDES:
        count = sum(row["arm"]["pass_numeric"] for row in shadow_rows if row["side"] == side)
        if count != BASELINE_ARM_PASS[side]:
            raise stage.FreshRobotError("DISPLAY_ACCEL_SHADOW_BASELINE_COUNT_DRIFT")
    base, _, offsets, anchor_q, neutral_roots = captured
    assets = stage.load_pinned_robot_assets(stage.ROOT)
    mounts, _ = stage._mounts()
    lower, upper = stage._arm_limits(assets)
    neutral = .5 * (lower + upper)
    inverse = np.linalg.inv(base)
    display = shadow["display_joints_3d_camera"]
    drawable = shadow["display_drawable"]
    sources = shadow["motion_source"]
    anchors = np.asarray(base_report["anchor_frames"], dtype=int)
    origin_basis: list[np.ndarray | None] = []
    for side, anchor in enumerate(anchors):
        try:
            origin_basis.append(stage._palm_basis(display[side, anchor], side)
                                if anchor >= 0 else None)
        except Exception:
            origin_basis.append(None)
    q_arm = np.tile(neutral[None], (FRAMES, 1, 1))
    rows: list[dict[str, Any]] = []
    for side in range(2):
        previous = None
        previous_frame = None
        for frame in range(FRAMES):
            previous_display = q_arm[frame - 1, side].copy() if frame else None
            previous_previous_display = q_arm[frame - 2, side].copy() if frame >= 2 else None
            item: dict[str, Any] = {
                "frame": frame, "side": stage.SIDES[side],
                "motion_source": str(sources[side, frame]), "candidate_q_rad": None,
                "previous_display_q_rad": (None if previous_display is None
                                           else previous_display.tolist()),
            }
            if not drawable[side, frame] or origin_basis[side] is None:
                status = ("DIAGNOSTIC_NEUTRAL_NO_ANCHOR" if not drawable[side, frame]
                          else "HOLD_BASIS_UNVERIFIED")
                item.update(status=status, pass_numeric=False)
            else:
                try:
                    basis = stage._palm_basis(display[side, frame], side)
                    rotation = basis @ origin_basis[side].T @ neutral_roots[side, :3, :3]
                    target = stage._pose(display[side, frame, 0] + offsets[side], rotation)
                    target_base = inverse @ target @ np.linalg.inv(mounts[side])
                    seeds = ([anchor_q[side], neutral[side]] if previous is None
                             else [previous, anchor_q[side]])
                    narrowed_lower, narrowed_upper = tube.velocity_bounds(
                        lower[side], upper[side], previous_display)
                    solved, position, orientation, margin = stage._arm_attempt(
                        assets, side, target_base, narrowed_lower, narrowed_upper,
                        seeds, previous=previous)
                    status = stage._arm_status(position, orientation, margin)
                    candidate_dq = (None if previous_display is None else
                                    float(np.max(np.abs(solved - previous_display))))
                    candidate_ddq = (None if previous_previous_display is None else
                                     displayed_acceleration(solved, previous_display,
                                                            previous_previous_display))
                    if status.startswith("PASS"):
                        if (previous is not None and previous_frame == frame - 1 and
                                float(np.max(np.abs(solved - previous))) >
                                stage.ARM_VELOCITY_GATE_RAD + 1e-9):
                            status = "HOLD_ARM_VELOCITY"
                        elif (candidate_ddq is not None and
                              candidate_ddq > stage.ARM_ACCELERATION_GATE_RAD + 1e-9):
                            status = "HOLD_ARM_ACCELERATION"
                    if status.startswith("PASS"):
                        q_arm[frame, side] = solved
                        previous = solved.copy()
                    else:
                        q_arm[frame, side] = previous if previous is not None else neutral[side]
                    previous_frame = frame
                    item.update(status=status, pass_numeric=status.startswith("PASS"),
                                candidate_q_rad=solved.tolist(),
                                candidate_dq_max_rad=candidate_dq,
                                candidate_ddq_max_rad=candidate_ddq,
                                position_mm=position, rotation_deg=orientation,
                                branch_margin_m_min=margin,
                                solve_bounds_lower_rad=narrowed_lower.tolist(),
                                solve_bounds_upper_rad=narrowed_upper.tolist())
                except stage.FreshRobotError:
                    raise
                except Exception as exc:
                    q_arm[frame, side] = previous if previous is not None else neutral[side]
                    item.update(status="HOLD_ARM_SOLVER", pass_numeric=False,
                                error_type=type(exc).__name__)
            item["display_q_rad"] = q_arm[frame, side].tolist()
            item["display_dq_max_rad"] = (None if previous_display is None else
                                           float(np.max(np.abs(q_arm[frame, side]
                                                               - previous_display))))
            rows.append(item)
    rows.sort(key=lambda row: (row["frame"], row["side"]))
    candidate = {**shadow, "q_arm": q_arm, "q_arm_shadow": shadow["q_arm"]}
    return candidate, base_report, rows, shadow_rows


def run_cpu_candidate(*, hawor_npz: Path, frame_provenance: Path,
                      input_audit: Path, raw_video: Path, output_root: Path) -> dict[str, Any]:
    values, provenance, audit = stage.load_fresh_inputs(
        hawor_npz, frame_provenance, input_audit, raw_video, SESSION)
    if (audit.get("source_kind") != "tracker"
            or len(provenance.get("frame_rows", [])) != FRAMES
            or output_root.exists()):
        raise stage.FreshRobotError("DISPLAY_ACCEL_SOURCE_OR_FRESH_OUTPUT_DRIFT")
    states, base, rows, shadow = solve_display_accel(
        values["joints_3d_camera"], values["prediction_valid"],
        float(values["fps_nominal"]))
    if (len(rows) != 2 * FRAMES or len(shadow) != 2 * FRAMES
            or any((row["frame"], row["side"]) != (i // 2, stage.SIDES[i % 2])
                   for i, row in enumerate(rows))):
        raise stage.FreshRobotError("DISPLAY_ACCEL_INCOMPLETE_GRID")
    output_root.mkdir(parents=True)
    tube._write(output_root / "BASE_SEARCH.json", base)
    tube._write(output_root / "FRAME_NUMERIC_DIAGNOSTIC.json", {
        "schema_version": "exact78-newraw-display-accel-cpu-frames-v1",
        "session_id": SESSION, "rows": rows})
    tube._write(output_root / "SHADOW_BASELINE.json", {
        "schema_version": "exact78-newraw-display-accel-cpu-shadow-v1",
        "session_id": SESSION, "rows": shadow})
    attempted = np.full((FRAMES, 2, states["q_arm"].shape[-1]), np.nan)
    for row in rows:
        if row["candidate_q_rad"] is not None:
            attempted[row["frame"], stage.SIDES.index(row["side"])] = row["candidate_q_rad"]
    np.savez_compressed(output_root / "NUMERIC_STATES.npz",
                        q_arm_attempt=attempted, q_arm_display=states["q_arm"],
                        q_arm_shadow=states["q_arm_shadow"], q_hand=states["q_hand"],
                        T_camera_base=states["T_camera_base"],
                        T_tool_hand_root=states["T_tool_hand_root"],
                        display_joints_3d_camera=states["display_joints_3d_camera"],
                        prediction_valid=states["prediction_valid"],
                        display_drawable=states["display_drawable"])
    counts = {side: sum(row["pass_numeric"] for row in rows if row["side"] == side)
              for side in stage.SIDES}
    continuity_ok = all(row["display_dq_max_rad"] is None or
                        row["display_dq_max_rad"] <= stage.ARM_VELOCITY_GATE_RAD + 1e-9
                        for row in rows)
    acceleration_ok = all(
        not row["pass_numeric"] or row["candidate_ddq_max_rad"] is None or
        row["candidate_ddq_max_rad"] <= stage.ARM_ACCELERATION_GATE_RAD + 1e-9
        for row in rows)
    result = {
        "schema_version": "exact78-newraw-display-accel-cpu-result-v1",
        "status": "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": SESSION, "frame_count": FRAMES, "fps_nominal": FPS,
        "single_change": "ARM_ACCELERATION_USES_TWO_PREVIOUS_ACTUALLY_DISPLAYED_Q",
        "original_numeric_gates": {
            "position_mm": stage.POSITION_GATE_MM,
            "orientation_deg": stage.ROTATION_GATE_DEG,
            "arm_velocity_rad_per_frame": stage.ARM_VELOCITY_GATE_RAD,
            "arm_acceleration_rad_per_frame2": stage.ARM_ACCELERATION_GATE_RAD},
        "baseline_arm_pass": BASELINE_ARM_PASS, "candidate_arm_pass": counts,
        "display_velocity_within_existing_gate": continuity_ok,
        "accepted_acceleration_within_existing_gate": acceleration_ok,
        "numeric_hypothesis_pass": (counts["right"] > BASELINE_ARM_PASS["right"]
                                    and counts["left"] >= BASELINE_ARM_PASS["left"]
                                    and continuity_ok and acceleration_ok),
        "status_counts": {side: dict(Counter(row["status"] for row in rows
                                             if row["side"] == side)) for side in stage.SIDES},
        "model_calls": 0, "render_calls": 0, "gpu_count": 0,
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "user_review": "PENDING",
        "inputs": {"hawor_npz": file_ref(hawor_npz),
                   "frame_provenance": file_ref(frame_provenance),
                   "input_audit": file_ref(input_audit), "raw_video": file_ref(raw_video)},
        "outputs": {name: file_ref(output_root / name) for name in (
            "BASE_SEARCH.json", "FRAME_NUMERIC_DIAGNOSTIC.json",
            "SHADOW_BASELINE.json", "NUMERIC_STATES.npz")},
    }
    tube._write(output_root / "RESULT.json", result)
    return result
