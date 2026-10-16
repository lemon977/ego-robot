"""Read-only and fail-closed checks for the six-session T3 scene join candidate."""

from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest import mock

from chaoyang.governance import build_exact78_t3_six_scene_matrix_v1 as candidate


class T3SixSceneMatrixTests(unittest.TestCase):
    def test_sealed_six_join_keeps_capability_quality_review_separate(self) -> None:
        matrix = candidate.build_matrix()
        self.assertEqual(matrix["status"], "READ_ONLY_CANDIDATE_NOT_T3_SEAL")
        self.assertEqual(len(matrix["selection"]["development"]), 4)
        self.assertEqual(len(matrix["selection"]["holdout"]), 2)
        self.assertEqual(matrix["selection"]["source_holdout_gap"]["status"],
                         "SPLIT_NOT_ESTABLISHED")
        self.assertIsNone(matrix["selection"]["source_holdout_gap"]["session_id"])
        self.assertEqual(set(matrix["sessions"]), set(candidate.FRONTENDS))
        for session_id, row in matrix["sessions"].items():
            self.assertEqual(row["execution"]["action_status"], "PASSED")
            self.assertEqual(row["execution"]["scene_frontend_status"], "PASSED_EXECUTION_ONLY")
            self.assertEqual(row["execution"]["scene_mask_sealed_frames"], row["raw"]["frames"])
            self.assertEqual(row["execution"]["scene_clean_sealed_frames"], row["raw"]["frames"])
            self.assertEqual(row["mask"]["quality"], "NOT_YET_ADOPTED")
            self.assertEqual(row["clean"]["quality"], "NOT_YET_ADOPTED")
            self.assertEqual(row["mask"]["visible_object_identity"], "UNVERIFIED_TEXT_UNION")
            self.assertEqual(row["visual_qualification"], "NOT_QUALIFIED")
            self.assertEqual(row["full_product_qualification"], "NOT_QUALIFIED")
            self.assertEqual(row["training_label_qualification"], "NOT_QUALIFIED")
            self.assertEqual(row["human_review"], "PENDING")
            self.assertEqual(row["ai_review"], "NOT_EVALUATED")
            self.assertEqual(row["evidence"]["scene_terminal_result"]["sha256"],
                             candidate.FRONTENDS[session_id][1])
            for name in ("depth", "object6d"):
                self.assertEqual(row["geometry_prequalification"][name]["fresh_result"],
                                 "NOT_EVALUATED")
                self.assertEqual(row["geometry_prequalification"][name]["field_qualification"],
                                 "NOT_EVALUATED")
            self.assertEqual(row["geometry_prequalification"]["trusted_occlusion"]
                             ["geometry_decision_coverage"], "NOT_EVALUATED")
        self.assertEqual(matrix["sessions"]["get_potato_chips_0901_013"]["raw"]["nominal_fps"], 25)
        self.assertEqual(matrix["sessions"]["play_cards_0915_002"]["raw"]["frames"], 131)
        self.assertEqual(matrix["sessions"]["play_cards_0915_002"]["raw"]["t0_time_mode"],
                         "TIMEBASE_UNVERIFIED")

    def test_no_active_first_anchor_or_base_partial_is_opened(self) -> None:
        original_open = Path.open
        forbidden = ("exact78_cards042_first_anchor_cpu_v1_20261001",
                     "exact78_cards042_continuity_base_cpu_v1_20261001")

        def guarded_open(path: Path, *args, **kwargs):
            self.assertFalse(any(token in str(path) for token in forbidden), str(path))
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            candidate.build_matrix()

    def test_missing_frontend_is_not_evaluated_without_following_latest(self) -> None:
        row = candidate._terminal_frontend(candidate.ROOT, "play_cards_0902_042", None, 171)
        self.assertEqual(row["status"], "NOT_EVALUATED")
        self.assertIsNone(row["terminal_result"])
        self.assertEqual(row["mask"]["status"], "NOT_EVALUATED")
        self.assertEqual(row["clean"]["status"], "NOT_EVALUATED")

    def test_wrong_terminal_sha_fails_closed(self) -> None:
        state = candidate.action._json(candidate.ROOT / candidate.STATE_REL)
        task_id = candidate.FRONTENDS["play_cards_0902_042"][0]
        task = next(item for item in state["tasks"] if item.get("task_id") == task_id)
        wrong = copy.deepcopy(task)
        wrong["result"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "PINNED_SHA_DRIFT"):
            candidate._terminal_frontend(candidate.ROOT, "play_cards_0902_042", wrong, 171)

    def test_mask_row_loss_fails_closed_not_visual_pass(self) -> None:
        state = candidate.action._json(candidate.ROOT / candidate.STATE_REL)
        task_id = candidate.FRONTENDS["play_cards_0902_042"][0]
        task = next(item for item in state["tasks"] if item.get("task_id") == task_id)
        original = candidate.action._verified

        def modified(root, ref, **kwargs):
            path, value = original(root, ref, **kwargs)
            if path.name == "MASK.json":
                value = dict(value)
                value["rows"] = value["rows"][:-1]
            return path, value

        with mock.patch.object(candidate.action, "_verified", side_effect=modified):
            with self.assertRaisesRegex(ValueError, "MASK_ROW_MAP_DRIFT"):
                candidate._terminal_frontend(candidate.ROOT, "play_cards_0902_042", task, 171)

    def test_prequal_source_sha_drift_fails_before_matrix_join(self) -> None:
        original = candidate.action._ref

        def altered(path):
            value = original(path)
            if path.name == candidate.PREQUAL_REL.name:
                value["sha256"] = "0" * 64
            return value

        with mock.patch.object(candidate.action, "_ref", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "PREQUAL_EVIDENCE_SHA_DRIFT"):
                candidate.build_matrix()

    def test_none_is_not_evaluated_not_pass(self) -> None:
        self.assertEqual(candidate._qualification(None), "NOT_EVALUATED")
        self.assertEqual(candidate._qualification(False), "NOT_QUALIFIED")


if __name__ == "__main__":
    unittest.main()
