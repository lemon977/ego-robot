"""No-registration tests for the independent Cards004 code acceptance executor."""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from chaoyang.cli_v6 import exact78_cards004_submit_candidate_v1 as driver
from chaoyang.governance import execute_exact78_cli_v6_cards004_acceptance_candidate_v1 as accept


class Cards004AcceptanceTests(unittest.TestCase):
    def test_independent_acceptance_identity_matches_driver_only(self) -> None:
        self.assertEqual(accept.TASK_ID, driver.ACCEPTANCE_TASK_ID)
        self.assertEqual(accept.PHASE, driver.ACCEPTANCE_PHASE)
        self.assertEqual(accept.PACKET, driver.ACCEPTANCE_PACKET)
        self.assertEqual(accept.RECEIPT, driver.ACCEPTANCE_RECEIPT)
        self.assertNotEqual(accept.TASK_ID, driver.v5.v4.TASK_ID)

    def test_exact_raw_roles_and_no_hardware_action_source(self) -> None:
        source = accept._source_evidence()
        self.assertEqual(set(source["source_files"]), driver.v5.RAW_ROLES)
        self.assertEqual(source["decoded_frames"], 152)
        self.assertEqual(source["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(source["capture_to_decode_alignment"], "INDEX_ALIGNMENT_UNVERIFIED")
        self.assertFalse(source["control_ground_truth"])
        for role in source["source_files"]:
            self.assertNotIn("pico", role)
            self.assertNotIn("manus", role)
            self.assertNotIn("tactile", role)
            self.assertNotIn("processed", role)

    def test_forbidden_role_injection_rejects(self) -> None:
        from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as producer
        actual = producer.preflight(driver.SESSION)
        changed = copy.deepcopy(actual)
        changed["descriptor"]["source_files"]["raw/pico.jsonl"] = {
            "path": str(driver.SESSION / "raw/pico.jsonl"), "bytes": 1,
            "sha256": "0" * 64}
        with patch.object(producer, "preflight", return_value=changed):
            with self.assertRaisesRegex(RuntimeError,
                                        "V6_ACCEPTANCE_SOURCE_OR_FORBIDDEN_ROLE_DRIFT"):
                accept._source_evidence()

    def test_operation_set_is_exact_three_published_routes(self) -> None:
        self.assertEqual(set(accept._published_operations()), set(driver.prior.ACCEPTANCE_OPERATIONS))

    def test_probe_cli_rejects_cross_source_even_if_ready(self) -> None:
        bad = {"status": "READY", "node": "CPU_ADMISSION",
               "source_session": "play_cards_0915_003",
               "submission_input_sha256": driver.SOURCE_SHA,
               "time_mode": "TIMEBASE_UNVERIFIED",
               "capture_to_decode_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
               "registration_started": False, "model_calls": 0}
        completed = subprocess.CompletedProcess([], 0, json.dumps(bad).encode(), b"")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "V6_ACCEPTANCE_CLI_CHECK_NOT_READY"):
                accept._probe_cli()

    def test_probe_tests_requires_all_eighteen(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"", b"Ran 17 tests in 1.0s\n\nOK\n")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "V6_ACCEPTANCE_NEGATIVE_TESTS_NOT_PASS"):
                accept._probe_tests()

    def test_packet_cpu_only_finite_and_result_matches_driver_gate(self) -> None:
        inputs = {"source_session": "play_cards_0915_004", "source_files": {},
                  "manifest": {"path": str(driver.v5.MANIFEST), "sha256": "a" * 64}}
        packet = accept.packet_value(inputs)
        self.assertEqual(packet["execution_mode"], "EXACT78_GOVERNANCE_ONLY")
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["model_calls"], 0)
        self.assertEqual(packet["budgets"]["render_calls"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 600)
        self.assertEqual(set(packet["write_set"]), {
            str(accept.ATTEMPT.relative_to(accept.REPO_ROOT)),
            str(accept.PACKET.relative_to(accept.REPO_ROOT)),
            str(accept.RECEIPT.relative_to(accept.REPO_ROOT)),
            str(accept.INDEX.relative_to(accept.REPO_ROOT)), "docs/governance"})
        self.assertIn("hard_cutoff_at", packet["inputs"])

    def test_duplicate_task_and_active_writer_block_registration_before_write(self) -> None:
        with patch.object(accept, "_governance_snapshot",
                          side_effect=RuntimeError("V6_ACCEPTANCE_TASK_ALREADY_USED")), \
             patch.object(accept, "atomic_json") as write, \
             patch.object(accept, "publish_bundle") as publish:
            with self.assertRaisesRegex(RuntimeError, "TASK_ALREADY_USED"):
                accept.register({}, 1)
        write.assert_not_called()
        publish.assert_not_called()
        with patch.object(accept, "_governance_snapshot", return_value=(1, {}, {}, True)), \
             patch.object(accept, "atomic_json") as write:
            with self.assertRaisesRegex(RuntimeError, "ACTIVE_TASK_OR_WRITER_CONFLICT"):
                accept.register({}, 1)
        write.assert_not_called()

    def test_fence_failure_prevents_terminal_write(self) -> None:
        with patch.object(accept, "_fence", side_effect=RuntimeError("FENCE_DRIFT")), \
             patch.object(accept, "atomic_json") as write:
            with self.assertRaisesRegex(RuntimeError, "FENCE_DRIFT"):
                accept.finish({})
        write.assert_not_called()

    def test_check_only_during_active_writer_cannot_register(self) -> None:
        inputs = {"source_session": "play_cards_0915_004", "source_files": {},
                  "negative_tests": {"status": "PASSED", "test_count": 18}}
        with patch.object(accept, "inspect", return_value=(12, {}, {}, inputs, True)), \
             patch.object(accept, "register") as register, \
             patch.object(accept, "finish") as finish, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(accept.main(["--check-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "WAIT_ACTIVE_WRITER")
        register.assert_not_called()
        finish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
