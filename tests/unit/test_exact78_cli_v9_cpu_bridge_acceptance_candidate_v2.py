"""v2 successor tests: isolate idle READY while preserving live-writer WAIT."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_cli_v9_cpu_admission_candidate_v1 as admission
from chaoyang.governance import execute_exact78_cli_v9_cpu_bridge_acceptance_candidate_v1 as v1
from chaoyang.governance import execute_exact78_cli_v9_cpu_bridge_acceptance_candidate_v2 as v2
from tests.unit.test_exact78_cli_v9_cpu_admission_candidate_v1 import V9BridgeTests, TRACKER
from tests.unit.test_exact78_cli_v9_cpu_bridge_acceptance_candidate_v1 import V9AcceptanceTests


class V2AcceptanceTests(unittest.TestCase):
    def test_idle_ready_is_isolated_from_global_governance(self) -> None:
        state = {"tasks": [], "next_task": None}
        index = {"task_packets": []}
        with patch.object(admission, "load_json", side_effect=lambda path:
                          state if path == admission.cpu.TASK_STATE_PATH else index), \
             patch.object(admission.cpu, "register") as register, \
             patch.object(admission.cpu, "run_registered_child") as run:
            result = admission.check_only(TRACKER)
        self.assertEqual(result["status"], "READY_NOT_REGISTERED")
        self.assertFalse(result["registration_started"])
        register.assert_not_called()
        run.assert_not_called()

    def test_active_writer_waits_without_registering(self) -> None:
        state = {"tasks": [{"task_id": "acceptance_writer", "status": "RUNNING"}],
                 "next_task": {"task_id": "acceptance_writer"}}
        index = {"task_packets": [{"task_id": "acceptance_writer", "execution_allowed": True}]}
        with patch.object(admission, "load_json", side_effect=lambda path:
                          state if path == admission.cpu.TASK_STATE_PATH else index), \
             patch.object(admission.cpu, "register") as register:
            result = admission.check_only(TRACKER)
        self.assertEqual(result["status"], "WAIT_ACTIVE_WRITER")
        register.assert_not_called()

    def test_successor_packet_is_finite_and_distinct(self) -> None:
        packet = v2.packet_value({"code": {}, "examples": {}, "negative_tests": {}})
        self.assertNotEqual(packet["task_id"], v1.TASK_ID)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["model_calls"], 0)
        self.assertEqual(packet["budgets"]["render_calls"], 0)
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 600)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertIn("candidate_v2 --register-and-run", packet["commands"][0])

    def test_v1_failure_is_sealed_and_not_reused(self) -> None:
        self.assertEqual(v2.V1_FAILURE_RECEIPT_SHA,
                         "0e4ad1f575b500f0534e1efbfc26008391d415303bf0b0a4732b68d2ca74bffa")
        self.assertEqual(v2.V1_FAILURE_RESULT_SHA,
                         "d6985959f382886c3b86bd653d8aba3a752c58ffe657923f084cd5667fcb4a01")
        self.assertEqual(Path(v2.RECEIPT).name,
                         "EXACT78_CLI_V9_CPU_BRIDGE_CODE_ACCEPTANCE_V2.json")

    def test_v2_active_check_only_never_builds_packet(self) -> None:
        inputs = {"examples": {}, "negative_tests": {"status": "PASSED"}}
        with patch.object(v2, "inspect", return_value=(9, {}, {}, inputs, True)), \
             patch.object(v2, "packet_value") as packet, \
             patch.object(v2, "register") as register, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(v2.main(["--check-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "WAIT_ACTIVE_WRITER")
        packet.assert_not_called()
        register.assert_not_called()

    def test_v2_active_registration_rejects_before_write(self) -> None:
        with patch.object(v2, "_snapshot", return_value=(9, {}, {}, True)), \
             patch.object(v2, "atomic_json") as write:
            with self.assertRaisesRegex(RuntimeError, "ACTIVE_WRITER_OR_REVISION_CONFLICT"):
                v2.register({}, 9)
        write.assert_not_called()


def load_tests(loader: unittest.TestLoader, _tests: unittest.TestSuite,
               _pattern: str | None) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for test in loader.loadTestsFromTestCase(V9BridgeTests):
        if test._testMethodName != "test_check_only_does_not_register_or_run":
            suite.addTest(test)
    suite.addTests(loader.loadTestsFromTestCase(V9AcceptanceTests))
    suite.addTests(loader.loadTestsFromTestCase(V2AcceptanceTests))
    return suite


if __name__ == "__main__":
    unittest.main()
