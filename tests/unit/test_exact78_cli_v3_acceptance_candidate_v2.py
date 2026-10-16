"""No-registration checks for the separate CLI-v3 code acceptance successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from chaoyang.cli import exact78_newraw_cpu_submit_candidate as old_adapter
from chaoyang.cli_v2 import exact78_raw_to_robot_candidate as phase_dag
from chaoyang.governance import execute_exact78_cli_v3_acceptance_candidate_v2 as acceptance


class CliV3AcceptanceTests(unittest.TestCase):
    def test_real_sealed_ledger_fixture_is_the_old_prefix_trap(self) -> None:
        fixture = json.loads((acceptance.REPO_ROOT / acceptance.FIXTURE_PATH).read_text())
        row = fixture["row"]
        self.assertEqual(fixture["captured_governance_revision"], 16420)
        self.assertEqual(row["result"]["sha256"],
                         "1125e50902cb28c8243cfab3647988fd16f52f11fbdf75e3953ea7624a826a28")
        self.assertTrue(old_adapter._is_cpu_admission_task(row))  # the real bug
        self.assertNotEqual(row["phase"], phase_dag.PHASES[0])
        self.assertEqual(acceptance._ledger_fixture()["path"],
                         str(acceptance.REPO_ROOT / acceptance.FIXTURE_PATH))

    def test_both_code_acceptances_are_not_cpu_phase(self) -> None:
        fixture = json.loads((acceptance.REPO_ROOT / acceptance.FIXTURE_PATH).read_text())
        successor = {"task_id": acceptance.TASK_ID, "phase": acceptance.PHASE,
                     "session": acceptance.previous.SESSION_ID, "status": "PASSED"}
        for row in (fixture["row"], successor):
            with self.subTest(task_id=row["task_id"]):
                self.assertTrue(row["task_id"].startswith("exact78_newraw_"))
                self.assertNotEqual(row["phase"], phase_dag.PHASES[0])

    def test_check_only_requires_new_entry_and_fresh_cpu(self) -> None:
        value = {"status": "READY", "node": "CPU_ADMISSION",
                 "source_session": acceptance.previous.SESSION_ID,
                 "submission_input_sha256": acceptance.previous.EXPECTED_INPUT_SHA256}
        output = (json.dumps(value) + "\n").encode()
        process = SimpleNamespace(returncode=0, stdout=output, stderr=b"")
        with patch.object(acceptance.subprocess, "run", return_value=process) as run:
            evidence = acceptance._probe_cli()
        self.assertEqual(evidence["stdout_sha256"], hashlib.sha256(output).hexdigest())
        self.assertEqual(run.call_args.args[0][3], acceptance.ENTRY)
        for drift in ({"node": "HAWOR_V4"}, {"source_session": "other"},
                      {"submission_input_sha256": "0" * 64}):
            wrong = {**value, **drift}
            with self.subTest(drift=drift), patch.object(acceptance.subprocess, "run",
                return_value=SimpleNamespace(returncode=0,
                    stdout=json.dumps(wrong).encode(), stderr=b"")):
                with self.assertRaisesRegex(RuntimeError, "NOT_FRESH_CPU_READY"):
                    acceptance._probe_cli()

    def test_result_and_receipt_bind_new_code_without_replacing_v1(self) -> None:
        inputs = {"source_submission_input_sha256": acceptance.previous.EXPECTED_INPUT_SHA256,
                  "cli_code": {name: {"path": name} for name in acceptance.CLI_PATHS},
                  "inherited_cli_v2_code": {name: {"path": name}
                                            for name in acceptance.INHERITED_PATHS},
                  "test_code": {"path": acceptance.TEST_PATH},
                  "ledger_fixture": {"path": acceptance.FIXTURE_PATH},
                  "published_operations": {},
                  "cli_check_only": {"status": "READY"},
                  "negative_tests": {"status": "PASSED", "test_count": 4}}
        packet = {"inputs": inputs, "run_signature": "v2-signature"}
        packet_ref, result_ref = {"path": "v2-packet"}, {"path": "v2-result"}
        result = acceptance.result_value(packet, packet_ref)
        receipt = acceptance.receipt_value(packet, packet_ref, result_ref)
        self.assertEqual(result["schema_version"], acceptance.RESULT_SCHEMA)
        self.assertEqual(result["accepted_entry"], "chaoyang.cli_v3")
        self.assertEqual(set(result["cli_code"]), set(acceptance.CLI_PATHS))
        self.assertEqual(receipt["task_id"], acceptance.TASK_ID)
        self.assertEqual(receipt["result"], result_ref)
        self.assertNotEqual(acceptance.RECEIPT, acceptance.previous.RECEIPT)

    def test_active_writer_rejects_successor_registration_window(self) -> None:
        state = {"next_task": {"task_id": "other"},
                 "tasks": [{"task_id": "other", "status": "RUNNING"}]}
        index = {"task_packets": [{"task_id": "other", "execution_allowed": True}]}
        receipt = {"governance_revision": 16420, "files": {
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

    def test_unsealed_task_root_blocks_reuse_before_first_write(self) -> None:
        state = {"next_task": None, "tasks": []}
        index = {"task_packets": []}
        receipt = {"governance_revision": 16420, "files": {
            "task_state": {"path": str(acceptance.TASK_STATE_PATH)},
            "task_packet_index_current": {"path": str(acceptance.INDEX)}}}
        def read(path: Path) -> dict:
            if path == acceptance.RECEIPT_PATH:
                return receipt
            return state if path == acceptance.TASK_STATE_PATH else index
        def exists(path: Path) -> bool:
            return path == acceptance.ATTEMPT.parents[1]
        with patch.object(acceptance, "load_json", side_effect=read), patch.object(
                acceptance, "validate_artifact_ref", return_value=[]), patch.object(
                Path, "exists", autospec=True, side_effect=exists):
            with self.assertRaisesRegex(RuntimeError, "TASK_ALREADY_USED"):
                acceptance._idle()


if __name__ == "__main__":
    unittest.main()
