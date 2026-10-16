"""Cards001 CPU-only IK shadow: bound each later solve by displayed q ±0.12.

Fresh HaWoR is the sole motion source. The original stage computes the shadow
and the same automatic base; this module changes only the arm solver bounds.
No model, rendering, threshold change, old Robot q, or hand-side change.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage


SESSION = "play_cards_0902_001"
FRAMES = 201
FPS = 30.0
BASELINE_ARM_PASS = {"left": 44, "right": 2}


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")


def velocity_bounds(lower: np.ndarray, upper: np.ndarray,
                    displayed_q: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Use the existing velocity gate as a solve-time feasible interval."""
    if displayed_q is None:
        return lower.copy(), upper.copy()
    radius = stage.ARM_VELOCITY_GATE_RAD
    constrained_lower = np.maximum(lower, displayed_q - radius)
    constrained_upper = np.minimum(upper, displayed_q + radius)
    if (not np.isfinite(constrained_lower).all()
            or not np.isfinite(constrained_upper).all()
            or np.any(constrained_lower > constrained_upper)):
        raise stage.FreshRobotError("VELOCITY_TUBE_BOUNDS_INVALID")
    return constrained_lower, constrained_upper


def _shadow_and_base(joints: np.ndarray, valid: np.ndarray, fps: float) -> tuple[
        dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], tuple[Any, ...]]:
    original = stage._base_search
    captured: list[tuple[Any, ...]] = []
    def capture(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    stage._base_search = capture
    try:
        states, report, rows = stage.solve_camera_diagnostic(joints, valid, fps)
    finally:
        stage._base_search = original
    if len(captured) != 1 or captured[0][1] != report:
        raise stage.FreshRobotError("AUTOMATIC_BASE_CAPTURE_DRIFT")
    return states, report, rows, captured[0]


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for side in stage.SIDES:
        out[side] = dict(Counter(row["status"] for row in rows if row["side"] == side))
    return out


def solve_velocity_tube(joints: np.ndarray, valid: np.ndarray, fps: float) -> tuple[
        dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if joints.shape != (2, FRAMES, 21, 3) or valid.shape != (2, FRAMES) or fps != FPS:
        raise stage.FreshRobotError("VELOCITY_TUBE_INPUT_SHAPE_OR_FPS_DRIFT")
    shadow, base_report, shadow_rows, captured = _shadow_and_base(joints, valid, fps)
    for side in stage.SIDES:
        count = sum(row["arm"]["pass_numeric"] for row in shadow_rows if row["side"] == side)
        if count != BASELINE_ARM_PASS[side]:
            raise stage.FreshRobotError("SHADOW_BASELINE_ARM_COUNT_DRIFT")
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
        previous = previous_previous = None
        previous_frame = None
        for frame in range(FRAMES):
            previous_display = q_arm[frame - 1, side].copy() if frame else None
            item: dict[str, Any] = {"frame": frame, "side": stage.SIDES[side],
                                    "motion_source": str(sources[side, frame]),
                                    "candidate_q_rad": None,
                                    "previous_display_q_rad": None if previous_display is None
                                    else previous_display.tolist()}
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
                    narrowed_lower, narrowed_upper = velocity_bounds(
                        lower[side], upper[side], previous_display)
                    solved, position, orientation, margin = stage._arm_attempt(
                        assets, side, target_base, narrowed_lower, narrowed_upper,
                        seeds, previous=previous)
                    status = stage._arm_status(position, orientation, margin)
                    candidate_dq = (None if previous_display is None else
                                    float(np.max(np.abs(solved - previous_display))))
                    candidate_ddq = None
                    if previous is not None and previous_frame == frame - 1:
                        if previous_previous is not None:
                            candidate_ddq = float(np.max(np.abs(
                                solved - 2 * previous + previous_previous)))
                        if status.startswith("PASS"):
                            if float(np.max(np.abs(solved - previous))) > stage.ARM_VELOCITY_GATE_RAD + 1e-9:
                                status = "HOLD_ARM_VELOCITY"
                            elif (candidate_ddq is not None and
                                  candidate_ddq > stage.ARM_ACCELERATION_GATE_RAD + 1e-9):
                                status = "HOLD_ARM_ACCELERATION"
                    if status.startswith("PASS"):
                        q_arm[frame, side] = solved
                        previous_previous = (previous.copy() if previous is not None
                                             and previous_frame == frame - 1 else None)
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
        raise stage.FreshRobotError("VELOCITY_TUBE_SOURCE_OR_FRESH_OUTPUT_DRIFT")
    states, base, rows, shadow = solve_velocity_tube(
        values["joints_3d_camera"], values["prediction_valid"],
        float(values["fps_nominal"]))
    if (len(rows) != 2 * FRAMES or len(shadow) != 2 * FRAMES
            or any((row["frame"], row["side"]) != (i // 2, stage.SIDES[i % 2])
                   for i, row in enumerate(rows))):
        raise stage.FreshRobotError("VELOCITY_TUBE_INCOMPLETE_GRID")
    output_root.mkdir(parents=True)
    _write(output_root / "BASE_SEARCH.json", base)
    _write(output_root / "FRAME_NUMERIC_DIAGNOSTIC.json", {
        "schema_version": "exact78-newraw-velocity-tube-cpu-frames-v1",
        "session_id": SESSION, "rows": rows})
    _write(output_root / "SHADOW_BASELINE.json", {
        "schema_version": "exact78-newraw-velocity-tube-cpu-shadow-v1",
        "session_id": SESSION, "rows": shadow})
    attempted = np.full((FRAMES, 2, states["q_arm"].shape[-1]), np.nan)
    for row in rows:
        if row["candidate_q_rad"] is not None:
            attempted[row["frame"], stage.SIDES.index(row["side"])] = row["candidate_q_rad"]
    np.savez_compressed(output_root / "NUMERIC_STATES.npz",
                        q_arm_attempt=attempted, q_arm_display=states["q_arm"],
                        q_arm_shadow=states["q_arm_shadow"],
                        q_hand=states["q_hand"], T_camera_base=states["T_camera_base"],
                        T_tool_hand_root=states["T_tool_hand_root"],
                        display_joints_3d_camera=states["display_joints_3d_camera"],
                        prediction_valid=states["prediction_valid"],
                        display_drawable=states["display_drawable"])
    counts = {side: sum(row["pass_numeric"] for row in rows if row["side"] == side)
              for side in stage.SIDES}
    continuity_ok = all(row["display_dq_max_rad"] is None or
                        row["display_dq_max_rad"] <= stage.ARM_VELOCITY_GATE_RAD + 1e-9
                        for row in rows)
    result = {
        "schema_version": "exact78-newraw-velocity-tube-cpu-result-v1",
        "status": "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": SESSION, "frame_count": FRAMES, "fps_nominal": FPS,
        "single_change": "LEAST_SQUARES_BOUNDS_INTERSECT_PREVIOUS_DISPLAY_Q_PLUS_MINUS_EXISTING_VELOCITY_GATE",
        "original_numeric_gates": {"position_mm": stage.POSITION_GATE_MM,
            "orientation_deg": stage.ROTATION_GATE_DEG,
            "arm_velocity_rad_per_frame": stage.ARM_VELOCITY_GATE_RAD,
            "arm_acceleration_rad_per_frame2": stage.ARM_ACCELERATION_GATE_RAD},
        "baseline_arm_pass": BASELINE_ARM_PASS, "candidate_arm_pass": counts,
        "display_velocity_within_existing_gate": continuity_ok,
        "numeric_hypothesis_pass": (counts["right"] > BASELINE_ARM_PASS["right"]
                                     and counts["left"] >= BASELINE_ARM_PASS["left"]
                                     and continuity_ok),
        "status_counts": _status_counts(rows),
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
    _write(output_root / "RESULT.json", result)
    return result
