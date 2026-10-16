"""Cards001 two-seed CPU diagnostic; no model, render, or old Robot input.

The frozen stage computes both IK seeds but discards the unchosen solution.
This candidate captures those solutions without changing least_squares, proves
the original score/default selection, then tests only temporal-gate fallback.
It is not a maintained operation and must run behind a fenced Task Packet.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage


SESSION = "play_cards_0902_001"
FRAMES = 201
FPS = 30.0


def _write(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _candidate_metrics(assets: Any, side: int, target: np.ndarray,
                       q: np.ndarray) -> dict[str, Any]:
    actual = stage._tool_fk(assets, side, q)
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]) * 1000.0)
    rotation = float(np.degrees(np.linalg.norm(Rotation.from_matrix(
        target[:3, :3].T @ actual[:3, :3]).as_rotvec())))
    margin = float(np.min(stage._arm_margins(assets, side, q)))
    failed = not (position <= stage.POSITION_GATE_MM
                  and rotation <= stage.ROTATION_GATE_DEG and margin >= -1e-8)
    score = (failed, max(0.0, -margin),
             max(0.0, position - stage.POSITION_GATE_MM)
             + max(0.0, rotation - stage.ROTATION_GATE_DEG),
             position + rotation)
    return {"q": q, "position_mm": position, "rotation_deg": rotation,
            "branch_margin_m_min": margin, "score": score}


def _capture_attempt(assets: Any, side: int, target: np.ndarray,
                     lower: np.ndarray, upper: np.ndarray, seeds: list[np.ndarray],
                     previous: np.ndarray | None) -> tuple[int, list[dict[str, Any]]]:
    """Observe the frozen two solves; fail if its default differs from score."""
    if len(seeds) != 2:
        raise stage.FreshRobotError("TWO_SEED_COUNT_DRIFT")
    original = stage.least_squares
    solutions: list[np.ndarray] = []
    free = upper - lower > 1e-12

    def capture(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        solutions.append(np.asarray(result.x, dtype=np.float64).copy())
        return result

    stage.least_squares = capture
    try:
        chosen_q, position, rotation, margin = stage._arm_attempt(
            assets, side, target, lower, upper, seeds, previous=previous)
    finally:
        stage.least_squares = original
    if len(solutions) != (2 if np.any(free) else 0):
        raise stage.FreshRobotError("TWO_SEED_SOLVER_CALL_COUNT_DRIFT")
    candidates = []
    for index, seed in enumerate(seeds):
        q = np.clip(np.asarray(seed, dtype=np.float64), lower, upper).copy()
        if np.any(free):
            q[free] = solutions[index]
        candidates.append(_candidate_metrics(assets, side, target, q))
    default = min(range(2), key=lambda index: candidates[index]["score"])
    selected = candidates[default]
    if (not np.allclose(chosen_q, selected["q"], rtol=0, atol=1e-10)
            or not np.allclose([position, rotation, margin],
                               [selected["position_mm"], selected["rotation_deg"],
                                selected["branch_margin_m_min"]], rtol=0, atol=1e-8)):
        raise stage.FreshRobotError("TWO_SEED_DEFAULT_SCORE_REPRODUCTION_FAILED")
    return default, candidates


def _temporal_status(candidate: dict[str, Any], previous: np.ndarray | None,
                     previous_previous: np.ndarray | None,
                     previous_frame: int | None, frame: int) -> tuple[str, float | None, float | None]:
    status = stage._arm_status(candidate["position_mm"], candidate["rotation_deg"],
                               candidate["branch_margin_m_min"])
    dq = ddq = None
    if previous is not None and previous_frame == frame - 1:
        dq = float(np.max(np.abs(candidate["q"] - previous)))
        if previous_previous is not None:
            ddq = float(np.max(np.abs(candidate["q"] - 2 * previous + previous_previous)))
        if status.startswith("PASS"):
            if dq > stage.ARM_VELOCITY_GATE_RAD + 1e-9:
                status = "HOLD_ARM_VELOCITY"
            elif ddq is not None and ddq > stage.ARM_ACCELERATION_GATE_RAD + 1e-9:
                status = "HOLD_ARM_ACCELERATION"
    return status, dq, ddq


def choose_seed_on_temporal_rejection(
        default: int, statuses: list[tuple[str, float | None, float | None]]) -> int:
    """The experiment's only changed decision; no structural/gate relaxation."""
    if default not in (0, 1) or len(statuses) != 2:
        raise stage.FreshRobotError("TWO_SEED_CHOICE_SHAPE_DRIFT")
    if (statuses[default][0] in {"HOLD_ARM_VELOCITY", "HOLD_ARM_ACCELERATION"}
            and statuses[1 - default][0].startswith("PASS")):
        return 1 - default
    return default


def _row(frame: int, side: int, default: int, candidates: list[dict[str, Any]],
         statuses: list[tuple[str, float | None, float | None]], chosen: int,
         display_q: np.ndarray, previous_display: np.ndarray | None) -> dict[str, Any]:
    out: dict[str, Any] = {"frame": frame, "side": stage.SIDES[side],
                           "default_seed_index": default, "selected_seed_index": chosen,
                           "alternate_selected": chosen != default,
                           "display_q_rad": display_q.tolist(),
                           "display_dq_max_rad": None if previous_display is None else
                           float(np.max(np.abs(display_q - previous_display))),
                           "candidates": []}
    for index, candidate in enumerate(candidates):
        status, dq, ddq = statuses[index]
        out["candidates"].append({
            "seed_index": index, "q_rad": candidate["q"].tolist(),
            "score": [bool(candidate["score"][0]), *map(float, candidate["score"][1:])],
            "position_mm": candidate["position_mm"],
            "rotation_deg": candidate["rotation_deg"],
            "branch_margin_m_min": candidate["branch_margin_m_min"],
            "status": status, "candidate_dq_max_rad": dq,
            "candidate_ddq_max_rad": ddq})
    out["status"] = statuses[chosen][0]
    out["pass_numeric"] = out["status"].startswith("PASS")
    return out


def solve_two_seed(joints: np.ndarray, valid: np.ndarray, fps: float) -> tuple[
        dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return new states, same automatically selected base, rows, shadow rows."""
    if joints.shape != (2, FRAMES, 21, 3) or valid.shape != (2, FRAMES) or fps != FPS:
        raise stage.FreshRobotError("TWO_SEED_INPUT_SHAPE_OR_FPS_DRIFT")
    original_base_search = stage._base_search
    captured: list[tuple[Any, ...]] = []

    def capture_base(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        result = original_base_search(*args, **kwargs)
        captured.append(result)
        return result

    stage._base_search = capture_base
    try:
        shadow_states, base_report, shadow_rows = stage.solve_camera_diagnostic(joints, valid, fps)
    finally:
        stage._base_search = original_base_search
    if len(captured) != 1 or captured[0][1] != base_report:
        raise stage.FreshRobotError("TWO_SEED_BASE_CAPTURE_DRIFT")
    base, _, offsets, anchor_q, neutral_roots = captured[0]
    assets = stage.load_pinned_robot_assets(stage.ROOT)
    mounts, _ = stage._mounts()
    lower, upper = stage._arm_limits(assets)
    neutral = .5 * (lower + upper)
    inverse = np.linalg.inv(base)
    display = shadow_states["display_joints_3d_camera"]
    drawable = shadow_states["display_drawable"]
    source = shadow_states["motion_source"]
    anchors = np.asarray(base_report["anchor_frames"], dtype=int)
    origin_basis = []
    for side, anchor in enumerate(anchors):
        try:
            origin_basis.append(stage._palm_basis(display[side, anchor], side) if anchor >= 0 else None)
        except Exception:
            origin_basis.append(None)
    q_arm = np.tile(neutral[None], (FRAMES, 1, 1))
    rows = []
    for side in range(2):
        previous = previous_previous = None
        previous_frame = None
        for frame in range(FRAMES):
            previous_display = q_arm[frame - 1, side].copy() if frame else None
            if not drawable[side, frame] or origin_basis[side] is None:
                status = ("DIAGNOSTIC_NEUTRAL_NO_ANCHOR" if not drawable[side, frame]
                          else "HOLD_BASIS_UNVERIFIED")
                rows.append({"frame": frame, "side": stage.SIDES[side], "motion_source":
                             str(source[side, frame]), "status": status, "pass_numeric": False,
                             "default_seed_index": None, "selected_seed_index": None,
                             "alternate_selected": False, "candidates": [],
                             "display_q_rad": q_arm[frame, side].tolist(),
                             "display_dq_max_rad": 0.0 if frame else None})
                continue
            try:
                basis = stage._palm_basis(display[side, frame], side)
                rotation = basis @ origin_basis[side].T @ neutral_roots[side, :3, :3]
                target = stage._pose(display[side, frame, 0] + offsets[side], rotation)
                target_base = inverse @ target @ np.linalg.inv(mounts[side])
                seeds = ([anchor_q[side], neutral[side]] if previous is None
                         else [previous, anchor_q[side]])
                default, candidates = _capture_attempt(assets, side, target_base,
                                                        lower[side], upper[side], seeds, previous)
                statuses = [_temporal_status(c, previous, previous_previous,
                                             previous_frame, frame) for c in candidates]
                chosen = choose_seed_on_temporal_rejection(default, statuses)
                status = statuses[chosen][0]
                if status.startswith("PASS"):
                    q_arm[frame, side] = candidates[chosen]["q"]
                    previous_previous = (previous.copy() if previous is not None
                                         and previous_frame == frame - 1 else None)
                    previous = candidates[chosen]["q"].copy()
                else:
                    q_arm[frame, side] = previous if previous is not None else neutral[side]
                previous_frame = frame
                item = _row(frame, side, default, candidates, statuses, chosen,
                            q_arm[frame, side], previous_display)
                item["motion_source"] = str(source[side, frame])
                rows.append(item)
            except stage.FreshRobotError:
                raise
            except Exception as exc:
                q_arm[frame, side] = previous if previous is not None else neutral[side]
                rows.append({"frame": frame, "side": stage.SIDES[side], "motion_source":
                             str(source[side, frame]), "status": "HOLD_ARM_SOLVER",
                             "pass_numeric": False, "error_type": type(exc).__name__,
                             "default_seed_index": None, "selected_seed_index": None,
                             "alternate_selected": False, "candidates": [],
                             "display_q_rad": q_arm[frame, side].tolist(),
                             "display_dq_max_rad": None if previous_display is None else
                             float(np.max(np.abs(q_arm[frame, side] - previous_display)))})
    rows.sort(key=lambda row: (row["frame"], row["side"]))
    new_states = {**shadow_states, "q_arm": q_arm,
                  "q_arm_shadow": shadow_states["q_arm"]}
    return new_states, base_report, rows, shadow_rows


def run_cpu_candidate(*, hawor_npz: Path, frame_provenance: Path,
                      input_audit: Path, raw_video: Path, output_root: Path) -> dict[str, Any]:
    values, provenance, audit = stage.load_fresh_inputs(
        hawor_npz, frame_provenance, input_audit, raw_video, SESSION)
    if (audit.get("source_kind") != "tracker" or len(provenance.get("frame_rows", [])) != FRAMES
            or output_root.exists()):
        raise stage.FreshRobotError("TWO_SEED_SOURCE_OR_FRESH_OUTPUT_DRIFT")
    states, base, rows, shadow = solve_two_seed(
        values["joints_3d_camera"], values["prediction_valid"], float(values["fps_nominal"]))
    if (len(rows) != 2 * FRAMES or len(shadow) != 2 * FRAMES
            or any((row["frame"], row["side"]) != (i // 2, stage.SIDES[i % 2])
                   for i, row in enumerate(rows))):
        raise stage.FreshRobotError("TWO_SEED_INCOMPLETE_GRID")
    output_root.mkdir(parents=True)
    _write(output_root / "BASE_SEARCH.json", base)
    _write(output_root / "TWO_SEED_FRAMES.json", {"schema_version":
           "exact78-newraw-two-seed-cpu-frames-v1", "session_id": SESSION, "rows": rows})
    _write(output_root / "SHADOW_BASELINE.json", {"schema_version":
           "exact78-newraw-two-seed-shadow-v1", "session_id": SESSION,
           "rows": [{"frame": row["frame"], "side": row["side"],
                     "arm": row["arm"], "hand": row["hand"]} for row in shadow]})
    np.savez_compressed(output_root / "NUMERIC_STATES.npz", q_arm=states["q_arm"],
                        q_arm_shadow=states["q_arm_shadow"], q_hand=states["q_hand"],
                        T_camera_base=states["T_camera_base"])
    counts = {side: Counter(row["status"] for row in rows if row["side"] == side)
              for side in stage.SIDES}
    result = {"schema_version": "exact78-newraw-two-seed-cpu-result-v1",
              "status": "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS",
              "session_id": SESSION, "frame_count": FRAMES, "fps_nominal": FPS,
              "single_change": "ALTERNATE_EXISTING_SEED_ONLY_ON_TEMPORAL_REJECTION",
              "source_role": "DEVELOPMENT_ONLY", "model_calls": 0, "render_calls": 0,
              "numeric_gates_changed": False, "base_selection_changed": False,
              "old_robot_result_input": False, "visual_qualified": False,
              "product_qualified": False, "training_eligible": False,
              "control_ground_truth": False,
              "user_review": "PENDING",
              "inputs": {"hawor_npz": file_ref(hawor_npz),
                         "frame_provenance": file_ref(frame_provenance),
                         "input_audit": file_ref(input_audit),
                         "raw_video": file_ref(raw_video)},
              "outputs": {name: file_ref(output_root / name) for name in (
                  "BASE_SEARCH.json", "TWO_SEED_FRAMES.json", "SHADOW_BASELINE.json",
                  "NUMERIC_STATES.npz")},
              "arm_status_counts": {side: dict(counts[side]) for side in stage.SIDES},
              "alternate_selected_count": sum(row["alternate_selected"] for row in rows)}
    _write(output_root / "RESULT.json", result)
    return result
