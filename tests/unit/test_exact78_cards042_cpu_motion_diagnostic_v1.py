"""Model-free synthetic tests for the two separately packeted CPU modes."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from chaoyang.pipeline import exact78_cards042_cpu_motion_diagnostic_v1 as candidate
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage


def _synthetic_rows():
    rows, details = [], []
    for frame in range(candidate.N_FRAMES):
        for side in stage.SIDES:
            passed = frame >= 5
            status = "PASS_NUMERIC_CAMERA_DIAGNOSTIC" if passed else "HOLD_REACH"
            rows.append({"frame": frame, "side": side, "motion_source": "RGB_HAWOR_PREDICTED",
                         "arm": {"status": status, "pass_numeric": passed},
                         "hand": {"status": "PASS_NUMERIC_CAMERA_DIAGNOSTIC",
                                  "pass_numeric": True, "bone_error_deg_max": 0.0,
                                  "tip_direction_error_deg_max": 0.0}})
            details.append({"frame": frame, "side": side, "status": status,
                            "candidate_q_rad": [0.1] * 7,
                            "previous_display_q_rad": [0.0] * 7 if frame else None,
                            "display_q_rad": [0.1] * 7 if passed else [0.0] * 7,
                            "candidate_minus_previous_display_q_rad": [0.1] * 7 if frame else None,
                            "candidate_max_abs_dq_rad": 0.1 if frame else None,
                            "display_minus_previous_display_q_rad": [0.1] * 7 if passed else ([0.0] * 7 if frame else None),
                            "first_anchor_gate_applied": frame == 5,
                            "first_anchor_checked_max_abs_dq_rad": 0.1 if frame == 5 else None,
                            "branch_margin_m_min": 0.2, "position_mm": 0.0,
                            "rotation_deg": 0.0})
    return rows, details


class Cards042CpuMotionDiagnosticTests(unittest.TestCase):
    def test_pinned_inputs_and_distinct_one_change_modes(self) -> None:
        candidate._check_pins()
        self.assertEqual(set(candidate.MODES), {"FIRST_ANCHOR_ONLY", "BASE_RERANK_ONLY"})
        self.assertNotEqual(*candidate.MODES.values())

    def test_342_rows_arm_hand_branch_dq_and_worst_side_summary(self) -> None:
        rows, details = _synthetic_rows()
        frames, summary = candidate._per_frame(rows, details, np.zeros((171, 2, 7)))
        self.assertEqual(len(frames), 342)
        self.assertEqual(summary["worst_side_arm_pass"], 166)
        self.assertEqual(summary["worst_side_arm_coverage"], 166 / 171)
        self.assertEqual(summary["worst_side_arm_and_hand_pass"], 166)
        self.assertEqual(summary["first_anchor_gate_attempt_by_side"]["right"]["frame"], 5)
        self.assertEqual(summary["arm_hold_counts_by_side"]["left"], {"HOLD_REACH": 5})
        self.assertEqual(frames[11]["branch_margin_m_min"], 0.2)
        self.assertEqual(frames[11]["candidate_max_abs_dq_rad"], 0.1)

    def test_missing_frame_side_or_state_fails_closed(self) -> None:
        rows, details = _synthetic_rows()
        with self.assertRaisesRegex(stage.FreshRobotError, "ROW_COUNT"):
            candidate._per_frame(rows[:-1], details, np.zeros((171, 2, 7)))
        duplicate = [*rows[:-1], rows[0]]
        with self.assertRaisesRegex(stage.FreshRobotError, "INCOMPLETE_GRID"):
            candidate._per_frame(duplicate, details, np.zeros((171, 2, 7)))
        with self.assertRaisesRegex(stage.FreshRobotError, "INCOMPLETE_GRID"):
            candidate._per_frame(rows, details, np.zeros((170, 2, 7)))

    def test_flange_ik_override_restores_without_raster_or_render(self) -> None:
        original = stage._tool_fk
        with self.assertRaisesRegex(RuntimeError, "restore"):
            with candidate._flange_ik_only():
                self.assertIs(stage._tool_fk, candidate.flange.flange_fk)
                raise RuntimeError("restore")
        self.assertIs(stage._tool_fk, original)

    def test_bad_mode_and_holdout_rejected_before_solver(self) -> None:
        kwargs = dict(hawor_npz=Path("old.npz"), frame_provenance=Path("old.json"),
                      input_audit=Path("old_audit.json"), raw_video=Path("old.mp4"),
                      session_id="play_cards_0902_042", source_kind="tracker",
                      output_root=candidate.ROOT / "_run" / "never_created_cards042_cpu")
        with mock.patch.object(candidate.first_anchor, "_solve_with_first_anchor") as solve:
            with self.assertRaisesRegex(stage.FreshRobotError, "MODE_OR_SESSION"):
                candidate.run_cpu_diagnostic(**kwargs, mode="BOTH_CHANGES")
            with self.assertRaisesRegex(stage.FreshRobotError, "MODE_OR_SESSION"):
                candidate.run_cpu_diagnostic(**{**kwargs, "session_id": "play_cards_0901_001"},
                                             mode="FIRST_ANCHOR_ONLY")
            solve.assert_not_called()
        self.assertFalse(kwargs["output_root"].exists())

    def test_synthetic_run_writes_complete_cpu_only_result_without_render(self) -> None:
        rows, details = _synthetic_rows()
        states = {"q_arm": np.zeros((171, 2, 7)), "q_hand": np.zeros((171, 2, 22)),
                  "T_camera_base": np.eye(4),
                  "T_tool_hand_root": np.stack((np.eye(4), np.eye(4)))}
        values = {"joints_3d_camera": np.zeros((2, 171, 21, 3)),
                  "prediction_valid": np.ones((2, 171), bool),
                  "fps_nominal": np.asarray(30.0)}

        def solve(_joints, _valid, _fps, audit):
            audit.extend(details)
            return states, {"rule": "same_frozen_grid_for_all_sources_v1"}, rows

        @contextmanager
        def fail_if_base_override(_audits):
            raise AssertionError("base rerank must not run in FIRST_ANCHOR_ONLY")
            yield

        with tempfile.TemporaryDirectory(prefix="cards042_cpu_test_", dir=candidate.ROOT / "_run") as temp:
            output = Path(temp) / "fresh"
            with mock.patch.object(candidate.flange, "validate_sealed_development",
                                   return_value={"sealed_root": Path(temp), "frames": 171,
                                                 "stage_seal": {}, "native_open_audit": {}}), \
                 mock.patch.object(stage, "load_fresh_inputs",
                                   return_value=(values, {"frame_rows": [{"output_index": 0}]},
                                                 {"infiller_invoked": False})), \
                 mock.patch.object(candidate.first_anchor, "_solve_with_first_anchor",
                                   side_effect=solve), \
                 mock.patch.object(candidate.base_rerank, "_base_selection_override",
                                   side_effect=fail_if_base_override), \
                 mock.patch.object(candidate, "file_ref", side_effect=lambda p: {"path": str(p)}), \
                 mock.patch.object(stage, "render_camera_diagnostic") as render:
                result = candidate.run_cpu_diagnostic(
                    hawor_npz=Path("sealed.npz"), frame_provenance=Path("sealed.json"),
                    input_audit=Path("audit.json"), raw_video=Path("raw.mp4"),
                    session_id="play_cards_0902_042", source_kind="tracker",
                    output_root=output, mode="FIRST_ANCHOR_ONLY")
                render.assert_not_called()
            self.assertEqual(result["summary"]["frame_side_count"], 342)
            self.assertEqual(result["model_calls"], 0)
            self.assertEqual(result["render_calls"], 0)
            self.assertTrue((output / "FRAME_NUMERIC_DIAGNOSTIC.json").is_file())
            self.assertTrue((output / "BASE_SELECTION.json").is_file())
            self.assertTrue((output / "ROBOT_NUMERIC_STATES.npz").is_file())
            self.assertTrue((output / "RESULT.json").is_file())


if __name__ == "__main__":
    unittest.main()
