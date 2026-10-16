"""Negative boundaries for the v9 CPU-bridge code acceptance candidate."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_cli_v9_cpu_bridge_acceptance_candidate_v1 as accept


class V9AcceptanceTests(unittest.TestCase):
    def test_pins_and_two_raw_domains_are_exact(self) -> None:
        self.assertEqual(len(accept.CODE_SHA), 5)
        self.assertEqual(len(accept.DEPENDENCY_SHA), 3)
        self.assertEqual(len(accept.EXAMPLES["tracker"]["roles"]), 5)
        self.assertEqual(len(accept.EXAMPLES["vst"]["roles"]), 5)
        self.assertEqual(accept.EXAMPLES["vst"]["time_mode"], "TIMEBASE_UNVERIFIED")
        for row in accept.EXAMPLES.values():
            self.assertFalse(any("pico" in role or "manus" in role or "tactile" in role
                                 for role in row["roles"]))

    def test_test_probe_rejects_missing_negative_tests(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"", b"Ran 8 tests in 1.0s\n\nOK\n")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "TESTS_NOT_PASS"):
                accept._probe_tests()

    def test_example_probe_rejects_false_route_and_role(self) -> None:
        example = accept.EXAMPLES["tracker"]
        value = {key: example[key] for key in
                 ("source_kind", "decoded_frames", "physical_eye", "time_mode",
                  "v7_source_only_sha256", "descriptor_sha256", "task_id_preview",
                  "run_signature_preview")}
        value.update(status="READY_NOT_REGISTERED", route="UNPUBLISHED",
                     source_files={"pico_hand_motion": {"sha256": "0" * 64}},
                     registration_started=False, model_calls=0, gpu_count=0,
                     training_eligible=False, visual_qualified=False)
        completed = subprocess.CompletedProcess([], 0, json.dumps(value).encode(), b"")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_OR_CHECK_ONLY_DRIFT"):
                accept._probe_example("tracker")

    def test_packet_finite_governance_only(self) -> None:
        packet = accept.packet_value({"code": {}, "examples": {}, "negative_tests": {}})
        self.assertEqual(packet["execution_mode"], "EXACT78_GOVERNANCE_ONLY")
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 600)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["model_calls"], 0)
        self.assertEqual(packet["budgets"]["render_calls"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["required_outputs"], ["RESULT.json", "terminal_receipt"])

    def test_active_check_only_never_registers(self) -> None:
        inputs = {"examples": {}, "negative_tests": {"status": "PASSED"}}
        with patch.object(accept, "inspect", return_value=(1, {}, {}, inputs, True)), \
             patch.object(accept, "packet_value") as packet, \
             patch.object(accept, "register") as register, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(accept.main(["--check-only"]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "WAIT_ACTIVE_WRITER")
        packet.assert_not_called()
        register.assert_not_called()

    def test_active_registration_rejects_before_write(self) -> None:
        with patch.object(accept, "_snapshot", return_value=(4, {}, {}, True)), \
             patch.object(accept, "atomic_json") as write, \
             patch.object(accept, "publish_bundle") as publish:
            with self.assertRaisesRegex(RuntimeError, "ACTIVE_WRITER_OR_REVISION_CONFLICT"):
                accept.register({}, 4)
        write.assert_not_called()
        publish.assert_not_called()

    def test_fence_failure_prevents_terminal_write(self) -> None:
        with patch.object(accept, "_fence", side_effect=RuntimeError("FENCE_DRIFT")), \
             patch.object(accept, "atomic_json") as write:
            with self.assertRaisesRegex(RuntimeError, "FENCE_DRIFT"):
                accept.finish({})
        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
