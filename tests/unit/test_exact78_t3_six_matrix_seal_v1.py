"""CPU-only dry-run and fail-closed tests for the unregistered T3 sealer."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

from chaoyang.governance import seal_exact78_t3_six_matrix_v1 as seal


class T3SealTest(unittest.TestCase):
    def _ready_report(self) -> dict:
        report = seal.candidate_report()
        # Synthetic fixture only: turn any pending row into a fully referenced
        # terminal row. This is never passed to register or written to disk.
        exemplar = next(row for row in report["sessions"].values()
                        if row["execution"]["status"] == "PASSED")
        for row in report["sessions"].values():
            if row["execution"]["status"] not in seal.TERMINAL:
                row["execution"] = {"status": "PASSED"}
                row["evidence"] = copy.deepcopy(exemplar["evidence"])
        return report

    def test_cohort_is_four_plus_two_not_three_holdouts(self) -> None:
        report = seal.candidate_report()
        self.assertEqual(report["cohort"]["development_sessions"], 4)
        self.assertEqual(report["cohort"]["pre_frozen_cross_group_holdout_sessions"], 2)
        self.assertEqual(report["cohort"]["0902_independent_holdout"], "SPLIT_NOT_ESTABLISHED")
        self.assertFalse(report["cohort"]["blind_test_claim"])
        for row in report["sessions"].values():
            self.assertEqual(row["review"]["ai_visual_review"], "NOT_EVALUATED")
            self.assertIn("user_visual_review", row["review"])
        self.assertEqual(report["external_evidence"]["c0_cleanup"], "NOT_BOUND_TO_THIS_TASK")

    def test_pending_session_blocks_packet(self) -> None:
        report = seal.candidate_report()
        session = next(iter(report["sessions"]))
        report["sessions"][session]["execution"] = {"status": "IN_PROGRESS"}
        ready, blockers = seal.readiness(report)
        self.assertFalse(ready)
        self.assertTrue(any(session in reason for reason in blockers))
        with self.assertRaisesRegex(seal.T3NotReady, "T3_SIX_TERMINAL_PROOF_NOT_READY"):
            seal.packet_value(report)

    def test_0902_split_must_remain_unestablished(self) -> None:
        from chaoyang.governance.common import load_json
        freeze = copy.deepcopy(load_json(seal.FREEZE))
        freeze["source_group_limits"]["0902"] = "FROZEN_INDEPENDENT_GROUP"
        with self.assertRaisesRegex(seal.T3NotReady, "T0_COHORT_OR_SPLIT_DRIFT"):
            seal._frozen_cohort(freeze)

    def test_passed_missing_seal_blocks(self) -> None:
        report = self._ready_report()
        session = next(iter(report["sessions"]))
        report["sessions"][session]["evidence"].pop("robot_seal", None)
        ready, blockers = seal.readiness(report)
        self.assertFalse(ready)
        self.assertTrue(any("PASSED_STAGE_SEALS_ABSENT" in reason for reason in blockers))

    def test_report_integrity_is_not_quality(self) -> None:
        report = self._ready_report()
        with patch.object(seal, "_validate_packet", return_value=[]):
            packet = seal.packet_value(report)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["workers_max"], 1)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(packet["commands"], [seal.COMMAND])
        self.assertTrue(all("GPU_LEASE" not in path for path in packet["write_set"]))
        self.assertIn("_run/current/" + seal.TASK_ID + "/attempts/attempt_0001",
                      packet["write_set"])
        self.assertEqual(packet["config"]["review_policy"],
                         "AI_AND_USER_SEPARATE_NOT_PREFILLED")

    def test_active_publisher_is_separate_from_evidence_readiness(self) -> None:
        blockers = seal.publisher_blockers(
            {"next_task": {"task_id": "other"},
             "tasks": [{"status": "RUNNING"}]},
            {"task_packets": [{"task_id": "other", "execution_allowed": True}]})
        self.assertEqual(len(blockers), 3)
        self.assertIn("NEXT_TASK_ACTIVE:other", blockers)

    def test_check_only_has_no_registration_side_effect(self) -> None:
        with patch.object(seal, "candidate_report", return_value={
                "execution_counts": {"IN_PROGRESS": 1},
                "cohort": {"development_sessions": 4},
                "inputs": {"task_state_for_terminal_eligibility_only":
                           {"path": "state", "bytes": 1, "sha256": "test"}},
                "sessions": {"s": {"execution": {"status": "IN_PROGRESS"},
                                   "evidence": {}}}}), \
                patch.object(seal, "register", side_effect=AssertionError("registered")):
            output = seal.dry_run()
        self.assertEqual(output["status"], "DRY_RUN_NOT_READY")
        self.assertFalse(output["registered"])


if __name__ == "__main__":
    unittest.main()
