"""CPU-only candidate: rank a frozen-grid shortlist by chronological arm feasibility.

This is not a registered operation. A future fenced successor may invoke it on
sealed development HaWoR only. Candidate generation, calibration, IK, flange,
hand solver and numerical gates remain those of the pinned parent modules.
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
from chaoyang.pipeline import exact78_fresh_robot_first_anchor_continuity_v1 as first_anchor


ROOT = Path(__file__).resolve().parents[3]
MAX_SHORTLIST = 4
_LOCK = threading.Lock()
_PINS = {
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py":
        "9018d49be9030ec84c48e470f5b600ac828665bfe7e974b64b5f0736375b196c",
    "src/chaoyang/pipeline/exact78_fresh_robot_flange_successor_v1.py":
        "0df9c81e3ee2226212d673e3186cd6169fac2d315b1fc410ec3bfbf85556188c",
    "src/chaoyang/pipeline/exact78_fresh_robot_first_anchor_continuity_v1.py":
        "e026d0bfef05f44bf0d62e57340592b1032fd630db503438cdcaaafa841fd954",
}


def _check_pins() -> None:
    for relative, expected in _PINS.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise stage.FreshRobotError(f"CONTINUITY_BASE_CODE_DRIFT:{relative}")
    if stage.ARM_VELOCITY_GATE_RAD != .12 or stage.ARM_ACCELERATION_GATE_RAD != .06:
        raise stage.FreshRobotError("CONTINUITY_BASE_NUMERIC_GATE_DRIFT")


def shortlist_indices(report: dict[str, Any]) -> list[int]:
    """Stable original-score top four; no session hand tuning or new grid points."""
    candidates = report.get("candidates", [])
    if len(candidates) not in (1, 32):
        raise stage.FreshRobotError("CONTINUITY_BASE_GRID_DRIFT")
    if report.get("selected_index") not in range(len(candidates)):
        raise stage.FreshRobotError("CONTINUITY_BASE_ORIGINAL_SELECTION_DRIFT")
    for row in candidates:
        spec, score = row.get("spec"), row.get("score")
        if (not isinstance(spec, list) or len(spec) != 4 or
                not isinstance(score, list) or len(score) != 2 or
                not np.isfinite(spec).all() or not np.isfinite(score).all()):
            raise stage.FreshRobotError("CONTINUITY_BASE_CANDIDATE_INVALID")
    ranked = sorted(range(len(candidates)), key=lambda i: (tuple(candidates[i]["score"]), i))
    chosen = ranked[:MAX_SHORTLIST]
    if report["selected_index"] not in chosen:
        raise stage.FreshRobotError("CONTINUITY_BASE_ORIGINAL_TOP4_DRIFT")
    return chosen


def _calibrate_spec(joints: np.ndarray, valid: np.ndarray, assets: Any,
                    mounts: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                    neutral: np.ndarray, spec: list[float]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Exact anchor/offset construction from pinned v1 evaluate(spec)."""
    anchors = np.asarray([int(np.flatnonzero(valid[side])[0]) if valid[side].any() else -1
                          for side in range(2)], dtype=int)
    points = [joints[side, anchor, 0] for side, anchor in enumerate(anchors) if anchor >= 0]
    midpoint = np.mean(points, axis=0) if points else np.asarray((0.0, 0.0, 0.8))
    base = stage._candidate_base(midpoint, tuple(spec))
    inverse = np.linalg.inv(base)
    offsets = np.zeros((2, 3))
    q_anchor = neutral.copy()
    neutral_roots = np.asarray([base @ stage._tool_fk(assets, side, neutral[side]) @ mounts[side]
                                for side in range(2)])
    origin_basis: list[np.ndarray | None] = [None, None]
    for side, anchor in enumerate(anchors):
        if anchor < 0:
            continue
        try:
            origin_basis[side] = stage._palm_basis(joints[side, anchor], side)
            wanted = stage._pose(joints[side, anchor, 0], neutral_roots[side, :3, :3])
            solved, _, _, _ = stage._arm_attempt(
                assets, side, inverse @ wanted @ np.linalg.inv(mounts[side]),
                lower[side], upper[side], [neutral[side]])
            q_anchor[side] = solved
            actual = base @ stage._tool_fk(assets, side, solved) @ mounts[side]
            offsets[side] = actual[:3, 3] - joints[side, anchor, 0]
        except Exception:
            origin_basis[side] = None
    return base, offsets, q_anchor, neutral_roots, origin_basis


def _chronological_score(joints: np.ndarray, assets: Any, mounts: np.ndarray,
                         lower: np.ndarray, upper: np.ndarray, neutral: np.ndarray,
                         calibrated: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                           list[np.ndarray | None]]) -> dict[str, Any]:
    """Simulate the same arm admission over every displayed frame, no rendering."""
    base, offsets, q_anchor, neutral_roots, origin_basis = calibrated
    inverse = np.linalg.inv(base)
    count = joints.shape[1]
    passes = [0, 0]
    longest = [0, 0]
    holds: list[dict[str, int]] = [{}, {}]
    evaluated = [0, 0]
    for side in range(2):
        previous = None
        previous_previous = None
        previous_frame = None
        displayed = neutral[side].copy()
        run = 0
        for frame in range(count):
            prior_display = displayed.copy() if frame else None
            if not np.isfinite(joints[side, frame]).all():
                displayed = neutral[side].copy()
                run = 0
                continue
            if origin_basis[side] is None:
                displayed = neutral[side].copy()
                holds[side]["HOLD_BASIS_UNVERIFIED"] = holds[side].get("HOLD_BASIS_UNVERIFIED", 0) + 1
                run = 0
                continue
            evaluated[side] += 1
            try:
                basis = stage._palm_basis(joints[side, frame], side)
                rotation = basis @ origin_basis[side].T @ neutral_roots[side, :3, :3]
                target = stage._pose(joints[side, frame, 0] + offsets[side], rotation)
                seeds = [q_anchor[side], neutral[side]] if previous is None else [previous, q_anchor[side]]
                solved, pos, orient, margin = stage._arm_attempt(
                    assets, side, inverse @ target @ np.linalg.inv(mounts[side]),
                    lower[side], upper[side], seeds, previous=previous)
                status = stage._arm_status(pos, orient, margin)
                if status.startswith("PASS") and previous is not None and previous_frame == frame - 1:
                    if np.max(np.abs(solved - previous)) > stage.ARM_VELOCITY_GATE_RAD + 1e-9:
                        status = "HOLD_ARM_VELOCITY"
                    elif (previous_previous is not None and
                          np.max(np.abs(solved - 2 * previous + previous_previous))
                          > stage.ARM_ACCELERATION_GATE_RAD + 1e-9):
                        status = "HOLD_ARM_ACCELERATION"
                status, _, _ = first_anchor.first_anchor_admission(
                    status, frame=frame, previous_accepted=previous,
                    candidate_q=solved, previous_display_q=prior_display)
                if status.startswith("PASS"):
                    displayed = solved.copy()
                    previous_previous = previous.copy() if previous is not None and previous_frame == frame - 1 else None
                    previous = solved
                    passes[side] += 1
                    run += 1
                    longest[side] = max(longest[side], run)
                else:
                    displayed = previous.copy() if previous is not None else neutral[side].copy()
                    holds[side][status] = holds[side].get(status, 0) + 1
                    run = 0
                previous_frame = frame
            except Exception:
                displayed = previous.copy() if previous is not None else neutral[side].copy()
                holds[side]["HOLD_ARM_SOLVER"] = holds[side].get("HOLD_ARM_SOLVER", 0) + 1
                run = 0
    return {"pass_by_side": passes, "longest_pass_run_by_side": longest,
            "evaluated_by_side": evaluated, "hold_counts_by_side": holds,
            "arm_attempt_upper_bound": 2 * count}


def continuity_rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Balanced per-side feasible coverage, then total coverage and continuity."""
    passed = record["pass_by_side"]
    longest = record["longest_pass_run_by_side"]
    if len(passed) != 2 or len(longest) != 2:
        raise stage.FreshRobotError("CONTINUITY_BASE_SCORE_SHAPE")
    return (-min(passed), -sum(passed), -min(longest), -sum(longest),
            *record["original_sample_score"], record["original_candidate_index"])


def select_continuity_base(joints: np.ndarray, valid: np.ndarray, assets: Any,
                           mounts: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                           neutral: np.ndarray, original_search: Any
                           ) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Keep original 32 specs; rerank only original-score top four via full chronology."""
    original = original_search(joints, valid, assets, mounts, lower, upper, neutral)
    report = original[1]
    shortlisted = shortlist_indices(report)
    candidates = report["candidates"]
    audited = []
    calibrated_by_index = {}
    for index in shortlisted:
        calibrated = _calibrate_spec(joints, valid, assets, mounts, lower, upper,
                                     neutral, candidates[index]["spec"])
        measured = _chronological_score(joints, assets, mounts, lower, upper, neutral, calibrated)
        measured["original_candidate_index"] = index
        measured["spec"] = candidates[index]["spec"]
        measured["original_sample_score"] = candidates[index]["score"]
        measured["chronological_rank_key"] = list(continuity_rank_key(measured))
        audited.append(measured)
        calibrated_by_index[index] = calibrated
    chosen = min(audited, key=continuity_rank_key)
    selected = chosen["original_candidate_index"]
    base, offsets, q_anchor, neutral_roots, _ = calibrated_by_index[selected]
    new_report = dict(report)
    new_report.update({
        "rule": "same_frozen_grid_original_score_top4_full_chronology_v1",
        "original_selected_index": report["selected_index"],
        "original_selected_spec": report["selected_spec"],
        "selected_index": selected, "selected_spec": candidates[selected]["spec"],
        "T_camera_base": base.tolist(), "offsets_camera": offsets.tolist(),
        "shortlist_indices": shortlisted, "shortlist_size_limit": MAX_SHORTLIST,
        "scored_frame_count": int(joints.shape[1]),
        "chronological_attempt_bound": 2 * int(joints.shape[1]) * len(shortlisted),
        "chronological_candidates": audited,
        "approximation": "TOP4_ONLY_NO_GLOBAL_GRID_OPTIMALITY_CLAIM",
        "world_pose_verified": False, "visual_qualified": False,
    })
    return base, new_report, offsets, q_anchor, neutral_roots


@contextmanager
def _base_selection_override(audits: list[dict[str, Any]]) -> Iterator[None]:
    _check_pins()
    with _LOCK:
        original = stage._base_search

        def successor(joints: np.ndarray, valid: np.ndarray, assets: Any,
                      mounts: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      neutral: np.ndarray):
            result = select_continuity_base(joints, valid, assets, mounts, lower, upper,
                                            neutral, original)
            audits.append(result[1])
            return result

        stage._base_search = successor
        try:
            yield
        finally:
            stage._base_search = original


def run_continuity_base_stage(*, hawor_npz: Path, frame_provenance: Path,
                              input_audit: Path, raw_video: Path, source_kind: str,
                              session_id: str, output_root: Path) -> dict[str, Any]:
    """Future governed caller only; this module itself is intentionally unrouted."""
    _check_pins()
    flange.validate_sealed_development(
        hawor_npz=hawor_npz, frame_provenance=frame_provenance,
        input_audit=input_audit, raw_video=raw_video,
        source_kind=source_kind, session_id=session_id)
    output = output_root.resolve()
    if not output.is_relative_to(ROOT / "_run") or output.exists():
        raise stage.FreshRobotError("CONTINUITY_BASE_FRESH_PROJECT_OUTPUT_REQUIRED")
    audits: list[dict[str, Any]] = []
    with _base_selection_override(audits):
        child = first_anchor.run_first_anchor_stage(
            hawor_npz=hawor_npz, frame_provenance=frame_provenance,
            input_audit=input_audit, raw_video=raw_video,
            source_kind=source_kind, session_id=session_id,
            output_root=output / "first_anchor_stage")
    if len(audits) != 1 or child["status"] != "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS":
        raise stage.FreshRobotError("CONTINUITY_BASE_STAGE_AUDIT_DRIFT")
    audit_path = output / "CONTINUITY_BASE_SELECTION.json"
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audits[0], stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    result = {
        "schema_version": "exact78-continuity-base-result-v1",
        "status": "COMPLETE_CAMERA_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": session_id, "frame_count": child["frame_count"],
        "source_role": "DEVELOPMENT_ONLY",
        "sealed_hawor_result": child["sealed_hawor_result"],
        "child_result": file_ref(output / "first_anchor_stage" / "RESULT.json"),
        "continuity_base_selection": file_ref(audit_path),
        "verified_videos": child["verified_videos"],
        "single_change": "BASE_CANDIDATE_SELECTION_SCORE_USES_TOP4_CHRONOLOGICAL_ARM_FEASIBILITY",
        "candidate_grid_changed": False, "numeric_gates_changed": False,
        "ik_objective_changed": False, "flange_chain_changed": False,
        "hand_rule_changed": False, "renderer_camera_base_rule_changed": False,
        "global_grid_optimality_claim": False,
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
    }
    with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result
