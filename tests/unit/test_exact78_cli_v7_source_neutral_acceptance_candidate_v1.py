"""Negative boundaries for the v7 code-acceptance candidate."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_cli_v7_source_neutral_acceptance_candidate_v1 as accept


class V7AcceptanceTests(unittest.TestCase):
    def test_frozen_v7_and_cold_start_code_sha_table_is_complete(self) -> None:
        self.assertEqual(set(accept.CODE), set(accept.CODE_SHA))
        self.assertEqual(accept.CODE_SHA[accept.CODE[2]],
                         "28027c07b20593130ff63446f4ed688571d7642b7e7b1e3ea650403ae741c548")

    def test_exact_example_roles_and_provenance_limits(self) -> None:
        self.assertEqual(set(accept.EXAMPLES["tracker"]["roles"]),
                         {"video", "source_stereo_video", "camera_params", "slam", "clip_manifest"})
        self.assertEqual(len(accept.EXAMPLES["vst"]["roles"]), 5)
        self.assertEqual(accept.EXAMPLES["vst"]["time_mode"], "TIMEBASE_UNVERIFIED")
        for row in accept.EXAMPLES.values():
            self.assertFalse(any("pico" in role or "tactile" in role or "manus" in role
                                 for role in row["roles"]))

    def test_probe_rejects_false_ready_or_role_injection(self) -> None:
        value = {"status": "READY_FOR_FUTURE_CODE_ACCEPTANCE_ONLY",
                 "source": {"source_kind": "TRACKER_CLIP_RGB",
                            "session_root": accept.EXAMPLES["tracker"]["root"],
                            "physical_eye": "tracker_session_left_RGB",
                            "time_mode": "TRACKER_CLIP_METADATA_ONLY",
                            "source_files": {"hand": {"sha256": "0" * 64}}},
                 "submission_input_sha256": accept.EXAMPLES["tracker"]["signature"],
                 "registration_started": False, "model_calls": 0,
                 "decode_verified": False, "control_ground_truth": False}
        completed = subprocess.CompletedProcess([], 0, json.dumps(value).encode(), b"")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_OR_CLI_DRIFT"):
                accept._probe_example("tracker")

    def test_probe_tests_rejects_missing_negative_test(self) -> None:
        completed = subprocess.CompletedProcess([], 0, b"", b"Ran 9 tests in 1.0s\n\nOK\n")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "NEGATIVE_TESTS_NOT_PASS"):
                accept._probe_tests()

    def test_packet_is_finite_governance_only_and_not_a_driver(self) -> None:
        inputs = {"code": {}, "tests": {}, "examples": {}, "negative_tests": {}}
        packet = accept.packet_value(inputs)
        self.assertEqual(packet["execution_mode"], "EXACT78_GOVERNANCE_ONLY")
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["model_calls"], 0)
        self.assertEqual(packet["budgets"]["render_calls"], 0)
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 600)
        self.assertEqual(packet["required_outputs"], ["RESULT.json", "terminal_receipt"])

    def test_active_check_only_never_constructs_packet_or_registers(self) -> None:
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
