"""No-write guard and stage-sequence tests for the new Cards004 submit candidate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.cli_v6 import exact78_cards004_submit_candidate_v1 as v6
from chaoyang.cli_v6 import exact78_cards004_cpu_child_v1 as child
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


class Cards004SubmitTests(unittest.TestCase):
    def test_exact_source_and_diagnostic_claims(self) -> None:
        row, ref = v6.v5._entry(v6.SESSION, v6.SOURCE_SHA)
        with patch.object(v6.v5, "plan_one", return_value={
                "status": "READY", "node": "CPU_ADMISSION", "task_id": "exact78_newraw_test"}):
            decision = v6.plan_one(v6.SESSION, v6.SOURCE_SHA)
        self.assertEqual(decision["node"], "CPU_ADMISSION")
        self.assertEqual(decision["manifest"], ref)
        self.assertEqual(decision["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(decision["capture_to_decode_alignment"], "INDEX_ALIGNMENT_UNVERIFIED")
        self.assertEqual(decision["physical_eye"], "left_sourceIndex_1")
        self.assertFalse(decision["control_ground_truth"])
        self.assertFalse(decision["training_eligible"])
        self.assertEqual(row["source_files"]["raw/vst.h264"]["sha256"],
                         "efe3c5b9a9656331c37c05015d39df4df842a41d2a91506f5c851af8380a8181")

    def test_no_v6_receipt_stops_driver_before_planning_or_child(self) -> None:
        with patch.object(Path, "is_file", return_value=False), \
             patch.object(v6, "plan_one") as plan, \
             patch.object(v6, "_launch_cpu") as launch:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V6_CODE_ACCEPTANCE_NOT_SEALED"):
                v6.submit_one(v6.SESSION, v6.SOURCE_SHA)
        plan.assert_not_called()
        launch.assert_not_called()

    def test_direct_cpu_child_cannot_bypass_v6_receipt(self) -> None:
        decision = {"status": "READY", "node": "CPU_ADMISSION", "task_id": "exact78_newraw_test"}
        with patch.object(child.driver, "plan_one", return_value=decision), \
             patch.object(child.driver, "accepted_code_for_execution",
                          side_effect=ColdStartInputError("NO_V6_RECEIPT")), \
             patch.object(child.phase_exact.producer, "register") as register:
            with self.assertRaisesRegex(ColdStartInputError, "NO_V6_RECEIPT"):
                child.submit_one(v6.SESSION, v6.SOURCE_SHA)
        register.assert_not_called()

    def test_budget_rejects_stage_without_full_wall_allowance(self) -> None:
        cutoff = datetime.now(timezone.utc) + timedelta(seconds=10)
        for node in ("CPU_ADMISSION", "HAWOR_V4", "ROBOT_V1"):
            with self.subTest(node=node):
                self.assertEqual(v6._budget(node, cutoff)["status"],
                                 "NOT_EVALUATED_BUDGET")

    def test_driver_does_not_start_child_after_budget_rejection(self) -> None:
        ready = {"status": "READY", "node": "CPU_ADMISSION", "task_id": "exact78_newraw_test"}
        with patch.object(v6, "accepted_code_for_execution", return_value=(
                {"sha256": "a" * 64}, datetime.now(timezone.utc) + timedelta(seconds=10))), \
             patch.object(v6, "plan_one", return_value=ready), \
             patch.object(v6, "_launch_cpu") as launch:
            result = v6.submit_one(v6.SESSION, v6.SOURCE_SHA)
        self.assertEqual(result["status"], "NOT_EVALUATED_BUDGET")
        launch.assert_not_called()

    def test_hawor_is_planned_only_from_sealed_cpu(self) -> None:
        cpu = {"status": "ATTACHED_TERMINAL", "result": {"path": "/fake/cpu.json"}}
        hawor_packet = {"inputs": {}}
        with patch.object(v6.v5, "plan_one", return_value=cpu), \
             patch.object(v6.prior.hawor, "packet_value", return_value=hawor_packet), \
             patch.object(v6.prior, "_sealed_cpu", return_value={}) as seal, \
             patch.object(v6.prior, "_node", return_value={
                 "status": "READY", "task_id": "exact78_newraw_hawor_test"}) as node:
            decision = v6.plan_one(v6.SESSION, v6.SOURCE_SHA)
        seal.assert_called_once_with(hawor_packet, cpu)
        self.assertEqual(node.call_args.args[1], v6.prior.PHASES[1])
        self.assertEqual(decision["node"], "HAWOR_V4")

    def test_robot_is_planned_only_from_sealed_hawor(self) -> None:
        cpu = {"status": "ATTACHED_TERMINAL", "result": {"path": "/fake/cpu.json"}}
        hawor = {"status": "ATTACHED_TERMINAL", "result": {"path": "/fake/hawor.json"}}
        robot = {"status": "READY", "task_id": "exact78_newraw_robot_test"}
        with patch.object(v6.v5, "plan_one", return_value=cpu), \
             patch.object(v6.prior.hawor, "packet_value", return_value={"inputs": {}}), \
             patch.object(v6.prior, "_sealed_cpu", return_value={}), \
             patch.object(v6.prior.robot, "packet_value", return_value={"inputs": {}}) as build_robot, \
             patch.object(v6.prior, "_node", side_effect=[hawor, robot]) as node:
            decision = v6.plan_one(v6.SESSION, v6.SOURCE_SHA)
        build_robot.assert_called_once_with(Path("/fake/hawor.json"))
        self.assertEqual(node.call_args.args[1], v6.prior.PHASES[2])
        self.assertEqual(decision["node"], "ROBOT_V1")

    def test_acceptance_never_uses_v4_code_receipt(self) -> None:
        with patch.object(Path, "is_file", return_value=False), \
             patch.object(v6.v5.v4, "accepted_code_for_execution") as old_gate:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V6_CODE_ACCEPTANCE_NOT_SEALED"):
                v6.accepted_code_for_execution(v6.SESSION, v6.SOURCE_SHA)
        old_gate.assert_not_called()

    def test_other_session_rejected_before_any_stage(self) -> None:
        other = Path(f"{v6.v5.VST_ROOT}/003")
        with patch.object(v6.v5, "plan_one") as old_plan:
            with self.assertRaisesRegex(ColdStartInputError, "CLI_V6_CARDS004_ONLY"):
                v6.plan_one(other, v6.v5.EXPECTED["play_cards_0915_003"][2])
        old_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
