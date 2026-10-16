"""Synthetic CPU-only checks for frozen-grid chronological base reranking."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from chaoyang.pipeline import exact78_fresh_robot_continuity_base_v1 as candidate
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage


def _report() -> dict:
    rows = [{"spec": [float(i), 0.0, 0.0, 0.0],
             "score": [-12 + i, float(i)], "sample_rows": 16,
             "sample_passed": max(0, 12 - i)} for i in range(32)]
    return {"candidates": rows, "selected_index": 0, "selected_spec": rows[0]["spec"]}


class ContinuityBaseTests(unittest.TestCase):
    def test_pins_and_existing_numeric_gates(self) -> None:
        candidate._check_pins()
        self.assertEqual((stage.ARM_VELOCITY_GATE_RAD, stage.ARM_ACCELERATION_GATE_RAD), (.12, .06))

    def test_shortlist_is_stable_original_top_four_no_new_specs(self) -> None:
        report = _report()
        self.assertEqual(candidate.shortlist_indices(report), [0, 1, 2, 3])
        report["candidates"][2]["score"] = report["candidates"][1]["score"]
        self.assertEqual(candidate.shortlist_indices(report), [0, 1, 2, 3])
        report["candidates"].append(report["candidates"][0])
        with self.assertRaisesRegex(stage.FreshRobotError, "GRID_DRIFT"):
            candidate.shortlist_indices(report)

    def test_balanced_chronology_precedes_total_and_sample_score(self) -> None:
        a = {"pass_by_side": [100, 1], "longest_pass_run_by_side": [60, 1],
             "original_sample_score": [-16, 0.0], "original_candidate_index": 0}
        b = {"pass_by_side": [50, 50], "longest_pass_run_by_side": [20, 20],
             "original_sample_score": [-8, 100.0], "original_candidate_index": 1}
        self.assertLess(candidate.continuity_rank_key(b), candidate.continuity_rank_key(a))

    def test_chronological_score_enforces_first_anchor_then_accepts_contiguous_q(self) -> None:
        joints = np.zeros((2, 4, 21, 3))
        for frame in range(4):
            joints[:, frame, 0, 0] = frame
        identity = np.eye(4)
        calibrated = (identity, np.zeros((2, 3)), np.zeros((2, 7)),
                      np.stack((identity, identity)), [np.eye(3), np.eye(3)])

        def arm_attempt(_assets, _side, target, *_args, **_kwargs):
            frame = int(round(target[0, 3]))
            values = (0.0, 0.2, 0.08, 0.10)
            position = 20.0 if frame == 0 else 0.0
            return np.full(7, values[frame]), position, 0.0, 0.2

        with mock.patch.object(stage, "_palm_basis", return_value=np.eye(3)), \
             mock.patch.object(stage, "_arm_attempt", side_effect=arm_attempt):
            score = candidate._chronological_score(
                joints, object(), np.stack((identity, identity)),
                np.full((2, 7), -3.0), np.full((2, 7), 3.0),
                np.zeros((2, 7)), calibrated)
        self.assertEqual(score["pass_by_side"], [2, 2])
        self.assertEqual(score["longest_pass_run_by_side"], [2, 2])
        self.assertEqual(score["hold_counts_by_side"][0],
                         {"HOLD_REACH": 1, "HOLD_ARM_VELOCITY": 1})
        self.assertEqual(score["arm_attempt_upper_bound"], 8)

    def test_selector_changes_only_score_within_original_grid(self) -> None:
        report = _report()
        original = mock.Mock(return_value=(np.eye(4), report, np.zeros((2, 3)),
                                           np.zeros((2, 7)), np.zeros((2, 4, 4))))
        def calibrate(_j, _v, _a, _m, _lo, _hi, _q, spec):
            base = np.eye(4)
            base[0, 3] = spec[0]
            return base, np.zeros((2, 3)), np.zeros((2, 7)), np.zeros((2, 4, 4)), [None, None]

        def score(_j, _a, _m, _lo, _hi, _q, calibrated):
            index = int(calibrated[0][0, 3])
            by_side = ([1, 1], [2, 2], [4, 4], [3, 3])[index]
            return {"pass_by_side": by_side, "longest_pass_run_by_side": by_side,
                    "evaluated_by_side": [4, 4], "hold_counts_by_side": [{}, {}],
                    "arm_attempt_upper_bound": 8}

        with mock.patch.object(candidate, "_calibrate_spec", side_effect=calibrate) as cal, \
             mock.patch.object(candidate, "_chronological_score", side_effect=score):
            result = candidate.select_continuity_base(
                np.zeros((2, 4, 21, 3)), np.ones((2, 4), bool), object(),
                np.zeros((2, 4, 4)), np.zeros((2, 7)), np.ones((2, 7)),
                np.zeros((2, 7)), original)
        self.assertEqual(cal.call_count, 4)
        original.assert_called_once()
        self.assertEqual(result[1]["selected_index"], 2)
        self.assertEqual(result[1]["original_selected_index"], 0)
        self.assertEqual(result[1]["shortlist_indices"], [0, 1, 2, 3])
        self.assertEqual(result[1]["chronological_attempt_bound"], 32)
        self.assertEqual(result[1]["candidates"], report["candidates"])
        self.assertEqual(result[1]["approximation"], "TOP4_ONLY_NO_GLOBAL_GRID_OPTIMALITY_CLAIM")

    def test_override_restores_and_holdout_fails_before_stage(self) -> None:
        old = stage._base_search
        with self.assertRaisesRegex(RuntimeError, "restore"):
            with candidate._base_selection_override([]):
                self.assertIsNot(stage._base_search, old)
                raise RuntimeError("restore")
        self.assertIs(stage._base_search, old)
        forbidden = candidate.ROOT / "_run" / "continuity_base_uncreated_test"
        with mock.patch.object(candidate.first_anchor, "run_first_anchor_stage") as model:
            with self.assertRaisesRegex(stage.FreshRobotError, "DEVELOPMENT_ONLY"):
                candidate.run_continuity_base_stage(
                    hawor_npz=Path("old_hawor.npz"), frame_provenance=Path("old_frames.json"),
                    input_audit=Path("old_audit.json"), raw_video=Path("old_video.mp4"),
                    source_kind="tracker", session_id="play_cards_0901_001", output_root=forbidden)
            model.assert_not_called()
        self.assertFalse(forbidden.exists())


if __name__ == "__main__":
    unittest.main()
