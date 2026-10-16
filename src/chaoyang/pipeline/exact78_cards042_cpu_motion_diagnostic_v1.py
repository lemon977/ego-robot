"""Cards042 sealed-HaWoR CPU numeric diagnostic; no model or renderer calls.

Two separately governed modes are possible: FIRST_ANCHOR_ONLY (one admission
change from sealed flange run) and BASE_RERANK_ONLY (one selection change from
FIRST_ANCHOR_ONLY). This helper is intentionally not an operation by itself.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
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
from chaoyang.pipeline import exact78_fresh_robot_continuity_base_v1 as base_rerank


ROOT = Path(__file__).resolve().parents[3]
SESSION = "play_cards_0902_042"
N_FRAMES = 171
FPS = 30.0
MODES = {
    "FIRST_ANCHOR_ONLY": "FIRST_POST_F0_PASS_CHECKED_AGAINST_PREVIOUS_DISPLAY_Q",
    "BASE_RERANK_ONLY": "BASE_SELECTION_TOP4_CHRONOLOGICAL_SCORE_WITH_FIRST_ANCHOR_FROZEN",
}
_LOCK = threading.Lock()
_PINS = {
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v1.py":
        "9018d49be9030ec84c48e470f5b600ac828665bfe7e974b64b5f0736375b196c",
    "src/chaoyang/pipeline/exact78_fresh_robot_flange_successor_v1.py":
        "0df9c81e3ee2226212d673e3186cd6169fac2d315b1fc410ec3bfbf85556188c",
    "src/chaoyang/pipeline/exact78_fresh_robot_first_anchor_continuity_v1.py":
        "e026d0bfef05f44bf0d62e57340592b1032fd630db503438cdcaaafa841fd954",
    "src/chaoyang/pipeline/exact78_fresh_robot_continuity_base_v1.py":
        "c4da256cec689d5bbf9d745361b0f7b07fc82ec372860b90fcbfebc7d3c8ec7d",
}


def _check_pins() -> None:
    for relative, expected in _PINS.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise stage.FreshRobotError(f"CARDS042_CPU_DIAGNOSTIC_CODE_DRIFT:{relative}")


@contextmanager
def _flange_ik_only() -> Iterator[None]:
    """Use sealed flange FK for IK without invoking or patching the raster path."""
    with _LOCK:
        original = stage._tool_fk
        stage._tool_fk = flange.flange_fk
        try:
            yield
        finally:
            stage._tool_fk = original


def _per_frame(rows: list[dict[str, Any]], arm_detail: list[dict[str, Any]],
               q_arm: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 2 * N_FRAMES or len(arm_detail) != 2 * N_FRAMES:
        raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_ROW_COUNT")
    detail = {(row["frame"], row["side"]): row for row in arm_detail}
    output = []
    pass_arm = Counter()
    pass_hand = Counter()
    pass_both = Counter()
    hold_arm = {side: Counter() for side in stage.SIDES}
    hold_hand = {side: Counter() for side in stage.SIDES}
    first_anchor = {side: None for side in stage.SIDES}
    first_accepted = {side: None for side in stage.SIDES}
    max_display_step = {side: 0.0 for side in stage.SIDES}
    for row in rows:
        frame, side = row["frame"], row["side"]
        if not (isinstance(frame, int) and 0 <= frame < N_FRAMES and side in stage.SIDES):
            raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_ROW_IDENTITY")
        item = detail.get((frame, side))
        if item is None:
            raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_ARM_DETAIL_MISSING")
        arm = row["arm"]
        hand = row["hand"]
        arm_ok, hand_ok = bool(arm["pass_numeric"]), bool(hand["pass_numeric"])
        pass_arm[side] += arm_ok
        pass_hand[side] += hand_ok
        pass_both[side] += arm_ok and hand_ok
        if not arm_ok:
            hold_arm[side][arm["status"]] += 1
        if not hand_ok:
            hold_hand[side][hand["status"]] += 1
        if item["first_anchor_gate_applied"] and first_anchor[side] is None:
            first_anchor[side] = {"frame": frame, "candidate_max_abs_dq_rad":
                                  item["first_anchor_checked_max_abs_dq_rad"],
                                  "status": arm["status"]}
        if arm_ok and first_accepted[side] is None:
            first_accepted[side] = {"frame": frame,
                                    "candidate_max_abs_dq_rad": item["candidate_max_abs_dq_rad"],
                                    "display_max_abs_dq_rad":
                                        max((abs(value) for value in
                                             (item["display_minus_previous_display_q_rad"] or [])),
                                            default=None)}
        step = item["display_minus_previous_display_q_rad"]
        if step is not None:
            max_display_step[side] = max(max_display_step[side], max(abs(v) for v in step))
        output.append({
            "frame": frame, "side": side,
            "motion_source": row["motion_source"],
            "arm_status": arm["status"], "arm_pass_numeric": arm_ok,
            "hand_status": hand["status"], "hand_pass_numeric": hand_ok,
            "arm_and_hand_pass_numeric": arm_ok and hand_ok,
            "branch_margin_m_min": item["branch_margin_m_min"],
            "arm_position_mm": item["position_mm"],
            "arm_rotation_deg": item["rotation_deg"],
            "hand_bone_error_deg_max": hand.get("bone_error_deg_max"),
            "hand_tip_direction_error_deg_max": hand.get("tip_direction_error_deg_max"),
            "candidate_q_arm_rad": item["candidate_q_rad"],
            "previous_display_q_arm_rad": item["previous_display_q_rad"],
            "display_q_arm_rad": item["display_q_rad"],
            "candidate_minus_previous_display_q_rad":
                item["candidate_minus_previous_display_q_rad"],
            "candidate_max_abs_dq_rad": item["candidate_max_abs_dq_rad"],
            "display_minus_previous_display_q_rad":
                item["display_minus_previous_display_q_rad"],
            "first_anchor_gate_applied": item["first_anchor_gate_applied"],
        })
    identities = {(row["frame"], row["side"]) for row in output}
    if len(identities) != 2 * N_FRAMES or q_arm.shape != (N_FRAMES, 2, 7):
        raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_INCOMPLETE_GRID")
    return output, {
        "frame_count": N_FRAMES, "frame_side_count": 2 * N_FRAMES,
        "arm_pass_by_side": {side: pass_arm[side] for side in stage.SIDES},
        "hand_pass_by_side": {side: pass_hand[side] for side in stage.SIDES},
        "arm_and_hand_pass_by_side": {side: pass_both[side] for side in stage.SIDES},
        "worst_side_arm_pass": min(pass_arm.values()),
        "worst_side_arm_coverage": min(pass_arm.values()) / N_FRAMES,
        "worst_side_arm_and_hand_pass": min(pass_both.values()),
        "first_anchor_gate_attempt_by_side": first_anchor,
        "first_accepted_arm_anchor_by_side": first_accepted,
        "max_display_arm_step_rad_by_side": max_display_step,
        "arm_hold_counts_by_side": {side: dict(hold_arm[side]) for side in stage.SIDES},
        "hand_hold_counts_by_side": {side: dict(hold_hand[side]) for side in stage.SIDES},
    }


def run_cpu_diagnostic(*, hawor_npz: Path, frame_provenance: Path,
                       input_audit: Path, raw_video: Path, session_id: str,
                       source_kind: str, output_root: Path, mode: str) -> dict[str, Any]:
    """Only future gated operations may invoke this on a new project-local attempt."""
    _check_pins()
    if mode not in MODES or session_id != SESSION or source_kind != "tracker":
        raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_MODE_OR_SESSION")
    sealed = flange.validate_sealed_development(
        hawor_npz=hawor_npz, frame_provenance=frame_provenance,
        input_audit=input_audit, raw_video=raw_video,
        source_kind=source_kind, session_id=session_id)
    output = output_root.resolve()
    if not output.is_relative_to(ROOT / "_run") or output.exists():
        raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_FRESH_OUTPUT_REQUIRED")
    values, provenance, audit = stage.load_fresh_inputs(
        hawor_npz, frame_provenance, input_audit, raw_video, session_id)
    fps = float(values["fps_nominal"])
    if values["joints_3d_camera"].shape[1] != N_FRAMES or fps != FPS or sealed["frames"] != N_FRAMES:
        raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_TIMELINE_DRIFT")
    if provenance.get("frame_rows", [{}])[0].get("output_index") != 0 or audit.get("infiller_invoked") is not False:
        raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_SOURCE_DRIFT")
    arm_detail: list[dict[str, Any]] = []
    base_audits: list[dict[str, Any]] = []
    base_context = (base_rerank._base_selection_override(base_audits)
                    if mode == "BASE_RERANK_ONLY" else nullcontext())
    with _flange_ik_only(), base_context:
        states, base_report, rows = first_anchor._solve_with_first_anchor(
            values["joints_3d_camera"], values["prediction_valid"], fps, arm_detail)
    if (len(base_audits) != (1 if mode == "BASE_RERANK_ONLY" else 0)
            or (base_audits and base_report != base_audits[0])):
        raise stage.FreshRobotError("CARDS042_CPU_DIAGNOSTIC_BASE_MODE_DRIFT")
    frames, summary = _per_frame(rows, arm_detail, states["q_arm"])
    output.mkdir(parents=True)
    with (output / "FRAME_NUMERIC_DIAGNOSTIC.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "exact78-cards042-cpu-motion-frames-v1",
                   "session_id": session_id, "mode": mode,
                   "rows": frames}, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / "BASE_SELECTION.json").open("x", encoding="utf-8") as stream:
        json.dump(base_report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    np.savez_compressed(output / "ROBOT_NUMERIC_STATES.npz",
                        q_arm=states["q_arm"], q_hand=states["q_hand"],
                        T_camera_base=states["T_camera_base"],
                        T_tool_hand_root=states["T_tool_hand_root"])
    result = {
        "schema_version": "exact78-cards042-cpu-motion-result-v1",
        "status": "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS",
        "session_id": session_id, "mode": mode,
        "single_change": MODES[mode], "frame_count": N_FRAMES,
        "source_role": "DEVELOPMENT_ONLY",
        "sealed_hawor_result": file_ref(sealed["sealed_root"] / "RESULT.json"),
        "hawor_stage_seal": sealed["stage_seal"],
        "hawor_native_open_audit": sealed["native_open_audit"],
        "frame_numeric_diagnostic": file_ref(output / "FRAME_NUMERIC_DIAGNOSTIC.json"),
        "base_selection": file_ref(output / "BASE_SELECTION.json"),
        "numeric_states": file_ref(output / "ROBOT_NUMERIC_STATES.npz"),
        "summary": summary, "model_calls": 0, "render_calls": 0,
        "numeric_gates_changed": False, "world_pose_verified": False,
        "visual_qualified": False, "product_qualified": False,
        "training_eligible": False, "control_ground_truth": False,
    }
    with (output / "RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result
