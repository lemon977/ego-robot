"""CPU-only negative tests for the unregistered CLI-v2 acceptance packet."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_cli_v2_acceptance_candidate_v1 as acceptance


class CliV2AcceptanceTests(unittest.TestCase):
    def test_cli_probe_requires_fresh_cpu_ready(self) -> None:
        value = {"status": "READY", "node": "CPU_ADMISSION",
                 "source_session": acceptance.SESSION_ID,
                 "submission_input_sha256": acceptance.EXPECTED_INPUT_SHA256,
                 "code_acceptance": "NOT_CHECKED_CHECK_ONLY"}
        output = (json.dumps(value) + "\n").encode()
        process = SimpleNamespace(returncode=0, stdout=output, stderr=b"")
        with patch.object(acceptance.subprocess, "run", return_value=process) as run:
            evidence = acceptance._probe_cli()
        self.assertEqual(evidence["stdout_sha256"], hashlib.sha256(output).hexdigest())
        self.assertIn("--check-only", run.call_args.args[0])
        self.assertIn("--expected-input-sha256", run.call_args.args[0])

    def test_cli_probe_rejects_wrong_node_or_source(self) -> None:
        for change in ({"node": "HAWOR_V4"}, {"source_session": "other"},
                       {"submission_input_sha256": "bad"}, {"status": "STOP_TERMINAL"}):
            value = {"status": "READY", "node": "CPU_ADMISSION",
                     "source_session": acceptance.SESSION_ID,
                     "submission_input_sha256": acceptance.EXPECTED_INPUT_SHA256}
            value.update(change)
            process = SimpleNamespace(returncode=0,
                stdout=json.dumps(value).encode(), stderr=b"")
            with self.subTest(change=change), patch.object(
                    acceptance.subprocess, "run", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "NOT_FRESH_CPU_READY"):
                    acceptance._probe_cli()

    def test_cli_probe_rejects_child_failure(self) -> None:
        process = SimpleNamespace(returncode=2, stdout=b"", stderr=b"blocked")
        with patch.object(acceptance.subprocess, "run", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "CLI_V2_CHECK_ONLY_FAILED"):
                acceptance._probe_cli()

    def test_negative_tests_require_nonzero_count_and_ok(self) -> None:
        process = SimpleNamespace(returncode=0, stdout=b"", stderr=b"Ran 11 tests\n\nOK\n")
        with patch.object(acceptance.subprocess, "run", return_value=process):
            self.assertEqual(acceptance._probe_tests()["test_count"], 11)
        process.stderr = b"Ran 10 tests\n\nOK\n"
        with patch.object(acceptance.subprocess, "run", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "NEGATIVE_TESTS_NOT_PASS"):
                acceptance._probe_tests()

    def test_published_operation_must_match_code_closure_sha(self) -> None:
        refs = {key: {"path": str(acceptance.REPO_ROOT / "src/chaoyang/ops" /
                            f"{operation}.py"), "bytes": 1, "sha256": key}
                for key, operation in acceptance.OPERATIONS.items()}
        routes = {key: {"candidate_operation": operation,
                        "task_packet_required": True,
                        "current_runnable_operation": None,
                        "code_closure": [refs[key]]}
                  for key, operation in acceptance.OPERATIONS.items()}
        with patch.object(acceptance, "load_json", return_value={
                "special_status_contracts": routes}), patch.object(
                acceptance, "artifact_ref", side_effect=lambda path: next(
                    ref for ref in refs.values() if ref["path"] == str(path))):
            accepted = acceptance._published_operations()
            self.assertEqual(set(accepted), set(acceptance.OPERATIONS))
            routes[next(iter(routes))]["code_closure"] = []
            with self.assertRaisesRegex(RuntimeError, "OPERATION_SHA_DRIFT"):
                acceptance._published_operations()

    def test_active_writer_blocks_check_only_registration_window(self) -> None:
        state = {"next_task": {"task_id": "other"},
                 "tasks": [{"task_id": "other", "status": "RUNNING"}]}
        index = {"task_packets": [{"task_id": "other", "execution_allowed": True}]}
        receipt = {"governance_revision": 7, "files": {
            "task_state": {"path": str(acceptance.TASK_STATE_PATH)},
            "task_packet_index_current": {"path": str(acceptance.INDEX)}}}
        def read(path: Path) -> dict:
            if path == acceptance.RECEIPT_PATH:
                return receipt
            return state if path == acceptance.TASK_STATE_PATH else index
        with patch.object(acceptance, "load_json", side_effect=read), patch.object(
                acceptance, "validate_artifact_ref", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "ACTIVE_TASK_OR_WRITER_CONFLICT"):
                acceptance._idle()

    def test_result_and_receipt_have_immutable_one_way_refs(self) -> None:
        packet = {"inputs": {"source_submission_input_sha256": acceptance.EXPECTED_INPUT_SHA256,
                  "cli_code": {name: {"path": name} for name in acceptance.CLI_PATHS},
                  "test_code": {"path": acceptance.TEST_PATH},
                  "published_operations": {},
                  "cli_check_only": {"status": "READY"},
                  "negative_tests": {"status": "PASSED", "test_count": 11}},
                  "run_signature": "signature"}
        packet_ref, result_ref = {"path": "packet", "sha256": "a"}, {
            "path": "result", "sha256": "b"}
        result = acceptance.result_value(packet, packet_ref)
        receipt = acceptance.receipt_value(packet, packet_ref, result_ref)
        self.assertEqual(result["schema_version"], acceptance.RESULT_SCHEMA)
        self.assertEqual(result["task_packet"], packet_ref)
        self.assertEqual(receipt["result"], result_ref)
        self.assertNotIn("terminal_receipt", result)
        self.assertEqual(result["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
