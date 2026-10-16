"""CPU checks for the isolated displayed-trajectory acceleration candidate."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.ops import run_exact78_newraw_display_accel_cpu_candidate_v1 as op
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
from chaoyang.pipeline import exact78_newraw_display_accel_cpu_candidate_v1 as candidate
from chaoyang.pipeline import exact78_newraw_velocity_tube_cpu_candidate_v1 as tube


class DisplayAccelCpuTests(unittest.TestCase):
    def test_frozen_f3_difference_uses_actual_display_history(self) -> None:
        self.assertEqual(stage.ARM_ACCELERATION_GATE_RAD, .06)
        q0 = np.zeros(7)
        q1 = np.full(7, .055)
        q2_display = q1.copy()  # F2 was held.
        f3 = np.full(7, .03)
        self.assertAlmostEqual(candidate.displayed_acceleration(f3, q2_display, q1), .025)
        self.assertGreater(float(np.max(np.abs(f3 - 2 * q1 + q0))), .06)

    def test_full_grid_f2_hold_then_f3_accepts_without_changing_bounds(self) -> None:
        joints = np.zeros((2, 201, 21, 3))
        valid = np.ones((2, 201), dtype=bool)
        shadow_rows = [
            {"frame": frame, "side": side,
             "arm": {"pass_numeric": frame < (44 if side == "left" else 2)}}
            for frame in range(201) for side in stage.SIDES]
        shadow = {
            "display_joints_3d_camera": joints.copy(),
            "display_drawable": valid.copy(),
            "motion_source": np.full((2, 201), "HAWOR_PREDICTION"),
            "q_arm": np.zeros((201, 2, 7)), "q_hand": np.zeros((201, 2, 1)),
        }
        report = {"anchor_frames": [0, 0]}
        captured = (np.eye(4), report, np.zeros((2, 3)), np.zeros((2, 7)),
                    np.repeat(np.eye(4)[None], 2, axis=0))
        bounds: list[tuple[np.ndarray, np.ndarray]] = []
        calls = {0: 0, 1: 0}

        def attempt(_assets, side, _target, low, high, _seeds, previous=None):
            frame = calls[side]
            calls[side] += 1
            bounds.append((low.copy(), high.copy()))
            q = np.full(7, {0: 0., 1: .055, 2: 0., 3: .03}.get(frame, .03))
            return q, 0., 0., .01

        with (patch.object(tube, "_shadow_and_base", return_value=(
                shadow, report, shadow_rows, captured)),
              patch.object(stage, "load_pinned_robot_assets", return_value=object()),
              patch.object(stage, "_mounts", return_value=(
                  np.repeat(np.eye(4)[None], 2, axis=0), {})),
              patch.object(stage, "_arm_limits", return_value=(
                  np.full((2, 7), -1.), np.full((2, 7), 1.))),
              patch.object(stage, "_palm_basis", return_value=np.eye(3)),
              patch.object(stage, "_arm_attempt", side_effect=attempt)):
            states, _, rows, _ = candidate.solve_display_accel(joints, valid, 30.)
        self.assertEqual(len(rows), 402)
        self.assertEqual(calls, {0: 201, 1: 201})
        for side in stage.SIDES:
            side_rows = {row["frame"]: row for row in rows if row["side"] == side}
            self.assertEqual(side_rows[2]["status"], "HOLD_ARM_ACCELERATION")
            self.assertEqual(side_rows[3]["status"], "PASS_NUMERIC_CAMERA_DIAGNOSTIC")
            self.assertAlmostEqual(side_rows[2]["candidate_ddq_max_rad"], .11)
            self.assertAlmostEqual(side_rows[3]["candidate_ddq_max_rad"], .025)
            np.testing.assert_allclose(side_rows[2]["display_q_rad"], np.full(7, .055))
            np.testing.assert_allclose(side_rows[3]["display_q_rad"], np.full(7, .03))
        np.testing.assert_array_equal(bounds[0][0], np.full(7, -1.))
        np.testing.assert_allclose(bounds[3][0], np.full(7, -.065))
        np.testing.assert_allclose(bounds[3][1], np.full(7, .175))
        np.testing.assert_array_equal(states["q_hand"], shadow["q_hand"])

    def test_unregistered_operation_is_fenced_before_input_or_output(self) -> None:
        with patch.object(op, "authorize_exact78_operation",
                          side_effect=RuntimeError("NO_PUBLISHED_ROUTE")):
            with self.assertRaisesRegex(RuntimeError, "NO_PUBLISHED_ROUTE"):
                op.run_candidate(Path("/missing/hawor.json"), Path("/missing/output"))

    def test_check_only_rejects_output_path(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CHECK_ONLY_MUST_NOT_SET_OUTPUT"):
            op.main(["--hawor-result", "unused", "--output-root", "forbidden",
                     "--check-only"])

    def test_input_shape_rejected_before_shadow(self) -> None:
        with self.assertRaisesRegex(stage.FreshRobotError, "INPUT_SHAPE_OR_FPS_DRIFT"):
            candidate.solve_display_accel(np.zeros((2, 1, 21, 3)),
                                          np.ones((2, 1)), 30.)


if __name__ == "__main__":
    unittest.main()
