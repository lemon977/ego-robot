"""Unrouted CPU-only display policy for a future 0901 Chips001 experiment.

Consumes a fresh solver's numeric q and frame rows. It never changes IK,
numeric PASS/HOLD decisions, source observations, or the sealed v2 result.
The returned q is diagnostic display state only, never an action label.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage


_SOURCES = {
    "RGB_HAWOR_PREDICTED", "DETERMINISTIC_INTERPOLATED", "EDGE_HELD",
    "DIAGNOSTIC_LONG_GAP", "DIAGNOSTIC_NEUTRAL",
}


def _expected_sources(valid: np.ndarray, count: int, limit: int
                      ) -> tuple[list[str], list[str]]:
    """Reconstruct the frozen display_tracks source/edge labels from anchors."""
    anchors = np.flatnonzero(valid).tolist()
    expected = ["DIAGNOSTIC_NEUTRAL"] * count
    edges = ["NONE"] * count
    if anchors:
        for anchor in anchors:
            expected[anchor] = "RGB_HAWOR_PREDICTED"
        for first, second in zip(anchors[:-1], anchors[1:]):
            label = ("DETERMINISTIC_INTERPOLATED" if second - first - 1 <= limit
                     else "DIAGNOSTIC_LONG_GAP")
            expected[first + 1:second] = [label] * (second - first - 1)
        for edge, indices in (("PREFIX", range(anchors[0])),
                              ("SUFFIX", range(anchors[-1] + 1, count))):
            label = "EDGE_HELD" if len(indices) <= limit else "DIAGNOSTIC_LONG_GAP"
            for frame in indices:
                expected[frame], edges[frame] = label, edge
    return expected, edges


def apply_display_policy(
    solver_q: np.ndarray, frame_rows: list[dict[str, Any]],
    neutral_q: np.ndarray, prediction_valid: np.ndarray, fps: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return safe display q and a complete per-frame, per-side diagnostic grid.

    Input rows must come from the same fresh solve as ``solver_q``. The caller
    must pin that provenance and must render the returned q, not solver q.
    """
    q = np.asarray(solver_q, dtype=np.float64)
    neutral = np.asarray(neutral_q, dtype=np.float64)
    valid = np.asarray(prediction_valid)
    if (q.ndim != 3 or q.shape[1:] != (2, 7) or q.shape[0] == 0
            or neutral.shape != (2, 7) or not np.isfinite(q).all()
            or not np.isfinite(neutral).all() or valid.shape != (2, q.shape[0])
            or valid.dtype != np.bool_ or not np.isfinite(fps) or fps <= 0):
        raise ValueError("DISPLAY_POLICY_INVALID_SHAPE_OR_VALUE")
    if len(frame_rows) != 2 * len(q):
        raise ValueError("DISPLAY_POLICY_INCOMPLETE_FRAME_GRID")
    limit = stage.short_gap_limit_frames(fps)  # Existing 10/30-second rule.
    gate = stage.ARM_VELOCITY_GATE_RAD  # Audit only; never alters numeric IK.
    expected_by_side = [_expected_sources(valid[index], len(q), limit)
                        for index in range(2)]
    for frame in range(len(q)):
        for side_index, side in enumerate(stage.SIDES):
            row = frame_rows[2 * frame + side_index]
            arm = row.get("arm")
            source = row.get("motion_source")
            edge = row.get("edge_gap_kind")
            if (row.get("frame") != frame or row.get("side") != side
                    or not isinstance(arm, dict)
                    or arm.get("frame") != frame or arm.get("side") != side
                    or arm.get("motion_source") != source
                    or not isinstance(arm.get("pass_numeric"), bool)
                    or not isinstance(arm.get("status"), str)
                    or arm["pass_numeric"] != arm["status"].startswith("PASS")
                    or source not in _SOURCES or edge not in {"NONE", "PREFIX", "SUFFIX"}
                    or source != expected_by_side[side_index][0][frame]
                    or edge != expected_by_side[side_index][1][frame]):
                raise ValueError("DISPLAY_POLICY_FRAME_ROW_DRIFT")
    displayed = np.empty_like(q)
    by_frame: list[list[dict[str, Any] | None]] = [[None, None] for _ in range(len(q))]
    for side_index, side in enumerate(stage.SIDES):
        # Only a direct predicted source with a numeric PASS supplies an arm-q
        # endpoint. A synthetic fill or solver HOLD cannot create an anchor.
        anchors = [frame for frame in range(len(q))
                   if valid[side_index, frame]
                   and frame_rows[2 * frame + side_index]["arm"]["pass_numeric"]]
        anchor_set = set(anchors)
        for frame in range(len(q)):
            row = frame_rows[2 * frame + side_index]
            if not anchors:
                chosen = neutral[side_index]
                status, display_source, display_edge = (
                    "NEUTRAL_NO_LEGAL_ARM_ANCHOR_DIAGNOSTIC", "NEUTRAL_DIAGNOSTIC", "NONE")
                left = right = gap = None
                future_donor = False
            elif frame < anchors[0]:
                chosen = q[anchors[0], side_index]
                gap = anchors[0]
                status = ("EDGE_HOLD_SHORT_DIAGNOSTIC" if gap <= limit
                          else "EDGE_HOLD_OVERLONG_DIAGNOSTIC")
                display_source, display_edge = "FIRST_LEGAL_ARM_ANCHOR_DIAGNOSTIC", "PREFIX"
                left, right, future_donor = None, anchors[0], True
            elif frame > anchors[-1]:
                chosen = q[anchors[-1], side_index]
                gap = len(q) - 1 - anchors[-1]
                status = ("EDGE_HOLD_SHORT_DIAGNOSTIC" if gap <= limit
                          else "EDGE_HOLD_OVERLONG_DIAGNOSTIC")
                display_source, display_edge = "LAST_LEGAL_ARM_ANCHOR_DIAGNOSTIC", "SUFFIX"
                left, right, future_donor = anchors[-1], None, False
            elif frame in anchor_set:
                chosen = q[frame, side_index]
                status, display_source, display_edge = (
                    "LEGAL_NUMERIC_ANCHOR_DIAGNOSTIC", "DIRECT_HAWOR_NUMERIC_PASS", "NONE")
                left = right = frame
                gap, future_donor = 0, False
            else:
                right_index = int(np.searchsorted(anchors, frame))
                left, right = anchors[right_index - 1], anchors[right_index]
                gap = right - left - 1
                ratio = (frame - left) / (right - left)
                chosen = (1 - ratio) * q[left, side_index] + ratio * q[right, side_index]
                status = ("TWO_END_SHORT_INTERPOLATION_DIAGNOSTIC" if gap <= limit
                          else "TWO_END_LONG_INTERPOLATION_DIAGNOSTIC")
                display_source, display_edge = "TWO_LEGAL_ARM_ANCHORS_DIAGNOSTIC", "NONE"
                future_donor = True
            displayed[frame, side_index] = chosen
            step = (None if frame == 0 else float(np.max(np.abs(
                chosen - displayed[frame - 1, side_index]))))
            candidate_step = (None if frame == 0 or not row["arm"]["pass_numeric"]
                              else float(np.max(np.abs(
                                  q[frame, side_index] - displayed[frame - 1, side_index]))))
            by_frame[frame][side_index] = {
                "frame": frame, "side": side,
                "motion_source": row["motion_source"],
                "edge_gap_kind": row["edge_gap_kind"],
                "display_edge_kind": display_edge,
                "short_gap_limit_frames": limit,
                "display_anchor_gap_frames": gap,
                "display_anchor_gap_within_short_limit": None if gap is None else gap <= limit,
                "display_left_anchor_frame": left,
                "display_right_anchor_frame": right,
                "diagnostic_future_donor": future_donor,
                "arm_numeric_status": row["arm"]["status"],
                "arm_numeric_pass": row["arm"]["pass_numeric"],
                "candidate_minus_previous_display_max_abs_rad": candidate_step,
                "display_minus_previous_display_max_abs_rad": step,
                "display_step_existing_velocity_gate": (
                    "NO_PREVIOUS_FRAME" if step is None else
                    "PASS" if step <= gate + 1e-9 else "FAIL"),
                "display_status": status,
                "display_source": display_source,
                "display_arm_visible": True,
                "robot_drawable": True,
                "direct_hawor_prediction": row["motion_source"] == "RGB_HAWOR_PREDICTED",
                "motion_observation_qualified": False,
                "motion_evidence_eligible": False,
                "visual_qualified": False, "training_eligible": False,
                "control_ground_truth": False,
            }
    if any(row is None for pair in by_frame for row in pair):
        raise ValueError("DISPLAY_POLICY_INCOMPLETE_OUTPUT_GRID")
    output = [row for pair in by_frame for row in pair if row is not None]
    return displayed, output
