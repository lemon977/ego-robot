"""CPU-only admission and isolation checks; never run the Robot/model stage."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from chaoyang.pipeline import exact78_fresh_robot_first_anchor_continuity_v1 as candidate
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage


class FirstAnchorContinuityTests(unittest.TestCase):
    def test_parent_code_and_unchanged_velocity_gate_are_pinned(self) -> None:
        candidate._check_pins()
        self.assertEqual(stage.ARM_VELOCITY_GATE_RAD, 0.12)

    def test_first_post_zero_anchor_rejected_against_displayed_neutral(self) -> None:
        neutral = np.zeros(7)
        solved = np.asarray([1.5628, 0, 0, 0, 0, 0, 0])
        status, applied, dq = candidate.first_anchor_admission(
            "PASS", frame=5, previous_accepted=None,
            candidate_q=solved, previous_display_q=neutral)
        self.assertEqual(status, "HOLD_ARM_VELOCITY")
        self.assertTrue(applied)
        self.assertAlmostEqual(dq, 1.5628)

    def test_frame_zero_and_nonpass_do_not_acquire_a_new_gate(self) -> None:
        for status, frame in (("PASS", 0), ("HOLD_REACH", 5), ("HOLD_BRANCH", 5)):
            outcome = candidate.first_anchor_admission(
                status, frame=frame, previous_accepted=None,
                candidate_q=np.ones(7), previous_display_q=np.zeros(7))
            self.assertEqual(outcome, (status, False, None))

    def test_existing_accepted_branch_is_unmodified(self) -> None:
        previous = np.zeros(7)
        outcome = candidate.first_anchor_admission(
            "PASS", frame=5, previous_accepted=previous,
            candidate_q=np.ones(7), previous_display_q=previous)
        self.assertEqual(outcome, ("PASS", False, None))

    def test_boundary_is_same_existing_velocity_gate(self) -> None:
        for delta, expected in ((0.12, "PASS"), (0.120000002, "HOLD_ARM_VELOCITY")):
            outcome = candidate.first_anchor_admission(
                "PASS", frame=5, previous_accepted=None,
                candidate_q=np.full(7, delta), previous_display_q=np.zeros(7))
            self.assertEqual(outcome[0], expected)
            self.assertTrue(outcome[1])

    def test_invalid_first_anchor_q_fails_closed(self) -> None:
        for bad in (np.full(7, np.nan), np.zeros(6)):
            with self.assertRaisesRegex(stage.FreshRobotError, "FIRST_ANCHOR_Q_INVALID"):
                candidate.first_anchor_admission(
                    "PASS", frame=1, previous_accepted=None,
                    candidate_q=bad, previous_display_q=np.zeros(7))
        with self.assertRaisesRegex(stage.FreshRobotError, "PREVIOUS_DISPLAY_REQUIRED"):
            candidate.first_anchor_admission(
                "PASS", frame=1, previous_accepted=None,
                candidate_q=np.zeros(7), previous_display_q=None)

    def test_scoped_solver_override_restores_on_failure(self) -> None:
        old = stage.solve_camera_diagnostic
        rows = []
        with self.assertRaisesRegex(RuntimeError, "restore"):
            with candidate._first_anchor_override(rows):
                self.assertIsNot(stage.solve_camera_diagnostic, old)
                raise RuntimeError("restore")
        self.assertIs(stage.solve_camera_diagnostic, old)

    def test_holdout_or_historical_input_rejected_before_output_or_model(self) -> None:
        forbidden = candidate.ROOT / "_run" / "first_anchor_uncreated_test"
        with mock.patch.object(candidate.flange, "run_flange_robot_stage") as model:
            with self.assertRaisesRegex(stage.FreshRobotError, "DEVELOPMENT_ONLY"):
                candidate.run_first_anchor_stage(
                    hawor_npz=Path("old_motion.npz"),
                    frame_provenance=Path("old_provenance.json"),
                    input_audit=Path("old_audit.json"),
                    raw_video=Path("old_video.mp4"), source_kind="tracker",
                    session_id="play_cards_0901_001", output_root=forbidden)
            model.assert_not_called()
        self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
