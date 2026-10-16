"""CPU-prepared, development-only first-anchor continuity successor.

Not a registered operation. A future fenced task must pin this file and an exact
fresh output root. The only algorithmic delta from the pinned flange successor is
applying its existing arm velocity gate to a first accepted anchor after frame 0,
against the previously displayed q (neutral until an accepted anchor exists).
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import threading
from typing import Any, Iterator

import numpy as np

from chaoyang.governance.exact78_cold_start_guard_v1 import file_ref
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
from chaoyang.pipeline import exact78_fresh_robot_flange_successor_v1 as flange


ROOT = Path(__file__).resolve().parents[3]
_LOCK = threading.Lock()
_PINS = {
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py":
        "9018d49be9030ec84c48e470f5b600ac828665bfe7e974b64b5f0736375b196c",
    "src/chaoyang/pipeline/exact78_fresh_robot_flange_successor_v1.py":
        "0df9c81e3ee2226212d673e3186cd6169fac2d315b1fc410ec3bfbf85556188c",
}


def _check_pins() -> None:
    for relative, expected in _PINS.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise stage.FreshRobotError(f"FIRST_ANCHOR_CODE_DRIFT:{relative}")
    if stage.ARM_VELOCITY_GATE_RAD != 0.12:
        raise stage.FreshRobotError("FIRST_ANCHOR_EXISTING_GATE_DRIFT")


def first_anchor_admission(status: str, *, frame: int, previous_accepted: np.ndarray | None,
                           candidate_q: np.ndarray, previous_display_q: np.ndarray | None
                           ) -> tuple[str, bool, float | None]:
    """Check only the first post-F0 PASS; return status, applied, max |dq|."""
    if not status.startswith("PASS") or frame == 0 or previous_accepted is not None:
        return status, False, None
    if frame < 0 or previous_display_q is None:
        raise stage.FreshRobotError("FIRST_ANCHOR_PREVIOUS_DISPLAY_REQUIRED")
    candidate = np.asarray(candidate_q, dtype=float)
    displayed = np.asarray(previous_display_q, dtype=float)
    if candidate.shape != (7,) or displayed.shape != (7,) or not np.isfinite(candidate).all() or not np.isfinite(displayed).all():
        raise stage.FreshRobotError("FIRST_ANCHOR_Q_INVALID")
    dq = float(np.max(np.abs(candidate - displayed)))
    if dq > stage.ARM_VELOCITY_GATE_RAD + 1e-9:
        return "HOLD_ARM_VELOCITY", True, dq
    return status, True, dq


def _solve_with_first_anchor(joints: np.ndarray, prediction_valid: np.ndarray, fps: float,
                             diagnostics: list[dict[str, Any]]
                             ) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Pinned v1 solve structure with one additional first-anchor admission check."""
    display, sources, short, edge_kind = stage.display_tracks(joints, prediction_valid, fps)
    drawable = np.isfinite(display).all(axis=(2, 3))
    assets = stage.load_pinned_robot_assets(ROOT)
    mounts, _ = stage._mounts()
    lower, upper = stage._arm_limits(assets)
    neutral = .5 * (lower + upper)
    base, base_report, offsets, anchor_q, neutral_roots = stage._base_search(
        display, prediction_valid, assets, mounts, lower, upper, neutral)
    inverse = np.linalg.inv(base)
    anchors = np.asarray(base_report["anchor_frames"], dtype=int)
    origin_basis = []
    for side, anchor in enumerate(anchors):
        try:
            origin_basis.append(stage._palm_basis(display[side, anchor], side) if anchor >= 0 else None)
        except Exception:
            origin_basis.append(None)
    count = joints.shape[1]
    q_arm = np.tile(neutral[None], (count, 1, 1))
    arm_rows = []
    for side in range(2):
        previous = None
        previous_previous = None
        previous_frame = None
        for frame in range(count):
            source = str(sources[side, frame])
            prior_display = q_arm[frame - 1, side].copy() if frame else None
            candidate = None
            position = orientation = margin = None
            checked = False
            candidate_dq = None
            if not drawable[side, frame]:
                arm_rows.append({"frame": frame, "side": stage.SIDES[side], "motion_source": source,
                                 "status": "DIAGNOSTIC_NEUTRAL_NO_ANCHOR", "pass_numeric": False})
            elif origin_basis[side] is None:
                arm_rows.append({"frame": frame, "side": stage.SIDES[side], "motion_source": source,
                                 "status": "HOLD_BASIS_UNVERIFIED", "pass_numeric": False})
            else:
                try:
                    basis = stage._palm_basis(display[side, frame], side)
                    rotation = basis @ origin_basis[side].T @ neutral_roots[side, :3, :3]
                    target = stage._pose(display[side, frame, 0] + offsets[side], rotation)
                    seeds = [anchor_q[side], neutral[side]] if previous is None else [previous, anchor_q[side]]
                    solved, position, orientation, margin = stage._arm_attempt(
                        assets, side, inverse @ target @ np.linalg.inv(mounts[side]),
                        lower[side], upper[side], seeds, previous=previous)
                    candidate = np.asarray(solved, dtype=float).copy()
                    status = stage._arm_status(position, orientation, margin)
                    if status.startswith("PASS") and previous is not None and previous_frame == frame - 1:
                        if np.max(np.abs(solved - previous)) > stage.ARM_VELOCITY_GATE_RAD + 1e-9:
                            status = "HOLD_ARM_VELOCITY"
                        elif (previous_previous is not None and
                              np.max(np.abs(solved - 2 * previous + previous_previous))
                              > stage.ARM_ACCELERATION_GATE_RAD + 1e-9):
                            status = "HOLD_ARM_ACCELERATION"
                    status, checked, candidate_dq = first_anchor_admission(
                        status, frame=frame, previous_accepted=previous,
                        candidate_q=candidate, previous_display_q=prior_display)
                    if status.startswith("PASS"):
                        q_arm[frame, side] = solved
                        previous_previous = previous.copy() if previous is not None and previous_frame == frame - 1 else None
                        previous = solved
                    else:
                        q_arm[frame, side] = previous if previous is not None else neutral[side]
                    previous_frame = frame
                    arm_rows.append({"frame": frame, "side": stage.SIDES[side], "motion_source": source,
                                     "status": status, "position_mm": position,
                                     "rotation_deg": orientation, "branch_margin_m_min": margin,
                                     "pass_numeric": status.startswith("PASS")})
                except Exception as exc:
                    q_arm[frame, side] = previous if previous is not None else neutral[side]
                    arm_rows.append({"frame": frame, "side": stage.SIDES[side], "motion_source": source,
                                     "status": "HOLD_ARM_SOLVER", "pass_numeric": False,
                                     "error_type": type(exc).__name__})
            row = arm_rows[-1]
            diagnostics.append({
                "frame": frame, "side": stage.SIDES[side],
                "candidate_q_rad": candidate.tolist() if candidate is not None else None,
                "previous_display_q_rad": prior_display.tolist() if prior_display is not None else None,
                "display_q_rad": q_arm[frame, side].tolist(),
                "candidate_minus_previous_display_q_rad":
                    (candidate - prior_display).tolist() if candidate is not None and prior_display is not None else None,
                "candidate_max_abs_dq_rad":
                    float(np.max(np.abs(candidate - prior_display))) if candidate is not None and prior_display is not None else None,
                "display_minus_previous_display_q_rad":
                    (q_arm[frame, side] - prior_display).tolist() if prior_display is not None else None,
                "branch_margin_m_min": margin, "position_mm": position, "rotation_deg": orientation,
                "first_anchor_gate_applied": checked, "first_anchor_checked_max_abs_dq_rad": candidate_dq,
                "status": row["status"], "pass_numeric": row["pass_numeric"],
            })
    q_hand, hand_rows = stage._solve_hands(display, drawable, assets)
    states = {"q_arm": q_arm, "q_hand": q_hand, "T_camera_base": base,
              "T_tool_hand_root": mounts, "display_joints_3d_camera": display,
              "prediction_valid": prediction_valid, "display_drawable": drawable,
              "short_gap": short, "motion_source": sources, "edge_gap_kind": edge_kind}
    rows = []
    arm_map = {(row["frame"], row["side"]): row for row in arm_rows}
    hand_map = {(row["frame"], row["side"]): row for row in hand_rows}
    for frame in range(count):
        for side in stage.SIDES:
            arm_row, hand_row = arm_map[frame, side], hand_map[frame, side]
            rows.append({"frame": frame, "side": side, "motion_source": arm_row["motion_source"],
                         "edge_gap_kind": str(edge_kind[stage.SIDES.index(side), frame]),
                         "source_coordinate_frame": "CAMERA", "world_pose_verified": False,
                         "robot_drawable": True, "motion_evidence_eligible": False,
                         "training_eligible": False, "arm": arm_row, "hand": hand_row,
                         "display_policy": "DIAGNOSTIC_CAMERA_FRAME_NO_OCCLUSION_PROOF"})
    return states, base_report, rows


@contextmanager
def _first_anchor_override(diagnostics: list[dict[str, Any]]) -> Iterator[None]:
    _check_pins()
    with _LOCK:
        original = stage.solve_camera_diagnostic
        stage.solve_camera_diagnostic = lambda joints, valid, fps: _solve_with_first_anchor(
            joints, valid, fps, diagnostics)
        try:
            yield
        finally:
            stage.solve_camera_diagnostic = original


def run_first_anchor_stage(*, hawor_npz: Path, frame_provenance: Path,
                           input_audit: Path, raw_video: Path, source_kind: str,
                           session_id: str, output_root: Path) -> dict[str, Any]:
    """Future governed caller only; no model is run and no history is consumed."""
    _check_pins()
    flange.validate_sealed_development(
        hawor_npz=hawor_npz, frame_provenance=frame_provenance,
        input_audit=input_audit, raw_video=raw_video,
        source_kind=source_kind, session_id=session_id)
    output = output_root.resolve()
    if not output.is_relative_to(ROOT / "_run") or output.exists():
        raise stage.FreshRobotError("FIRST_ANCHOR_FRESH_PROJECT_OUTPUT_REQUIRED")
    diagnostics: list[dict[str, Any]] = []
    with _first_anchor_override(diagnostics):
        child = flange.run_flange_robot_stage(
            hawor_npz=hawor_npz, frame_provenance=frame_provenance,
            input_audit=input_audit, raw_video=raw_video,
            source_kind=source_kind, session_id=session_id,
            output_root=output / "flange_first_anchor_stage")
    expected = child["frame_count"]
    if len(diagnostics) != 2 * expected or child["status"] != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS":
        raise stage.FreshRobotError("FIRST_ANCHOR_DIAGNOSTIC_COUNT_DRIFT")
    audit_path = output / "FIRST_ANCHOR_DIAGNOSTIC.json"
    output.mkdir(parents=True, exist_ok=True)
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "exact78-first-anchor-continuity-audit-v1",
                   "session_id": session_id, "frame_count": expected,
                   "existing_velocity_gate_rad": stage.ARM_VELOCITY_GATE_RAD,
                   "rows": diagnostics}, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    result = {"schema_version": "exact78-first-anchor-robot-successor-result-v1",
              "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
              "session_id": session_id, "frame_count": expected,
              "source_role": "DEVELOPMENT_ONLY",
              "sealed_hawor_result": child["sealed_hawor_result"],
              "child_result": file_ref(output / "flange_first_anchor_stage" / "RESULT.json"),
              "first_anchor_diagnostic": file_ref(audit_path),
              "verified_videos": child["verified_videos"],
              "single_change": "FIRST_ACCEPTED_POST_F0_ANCHOR_CHECKED_AGAINST_PREVIOUS_DISPLAY_Q",
              "numeric_gates_changed": False, "base_search_rule_changed": False,
              "ik_objective_changed": False, "flange_parent_changed_from_child": False,
              "renderer_camera_base_rule_changed": False,
              "visual_qualified": False, "product_qualified": False,
              "training_eligible": False, "control_ground_truth": False}
    with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result
