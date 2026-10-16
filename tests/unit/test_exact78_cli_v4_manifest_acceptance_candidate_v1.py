"""CPU-only governance tests; never register the v4 acceptance task."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from chaoyang.cli_v4 import exact78_raw_to_robot_manifest_candidate_v1 as cli
from chaoyang.governance import execute_exact78_cli_v4_manifest_acceptance_candidate_v1 as accept


class ManifestAcceptanceTests(unittest.TestCase):
    def test_identity_and_receipt_contract_match_cli_gate(self) -> None:
        self.assertEqual(accept.TASK_ID, cli.TASK_ID)
        self.assertEqual(accept.PHASE, cli.PHASE)
        self.assertEqual(accept.PACKET, cli.PACKET)
        self.assertEqual(accept.RECEIPT, cli.RECEIPT)
        self.assertEqual(accept.RESULT_SCHEMA,
                         "exact78-newraw-cli-v4-manifest-acceptance-result-v1")
        self.assertEqual(accept.RECEIPT_SCHEMA,
                         "exact78-newraw-cli-v4-manifest-acceptance-receipt-v1")

    def test_manifest_source_set_is_exactly_two_and_vst_is_not_admitted(self) -> None:
        manifest, _ = cli.frozen_manifest()
        self.assertEqual({item["session_id"] for item in manifest["sessions"]},
                         {"get_potato_chips_0901_001", "play_cards_0902_002"})
        self.assertEqual(manifest["source_domains"]["0915_raw_vst"]["execution_status"],
                         "SCHEMA_RESERVED_NO_SESSION_ADMITTED")

    def test_existing_three_operation_routes_are_published(self) -> None:
        routes = accept._published_operations()
        self.assertEqual(set(routes), set(accept.OPERATIONS))
        self.assertEqual(set(routes), set(cli.OPERATIONS))

    def test_source_evidence_rejects_frame_rate_drift(self) -> None:
        manifest, _ = cli.frozen_manifest()
        def wrong_preflight(path: Path) -> dict:
            row = next(item for item in manifest["sessions"]
                       if item["source_root"] == str(path))
            return {"descriptor": {"session_id": row["session_id"],
                                   "submission_input_sha256": row["submission_input_sha256"],
                                   "source_kind": row["source_kind"],
                                   "physical_eye": row["physical_eye"],
                                   "time_mode": row["time_mode"],
                                   "decoded_frames": row["frame_count_decoded"],
                                   "fps_nominal": 30,
                                   "source_files": {}}}
        from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as producer
        with patch.object(producer, "preflight", side_effect=wrong_preflight):
            with self.assertRaisesRegex(RuntimeError, "V4_PREFROZEN_SOURCE_DRIFT:.*fps_nominal"):
                accept._source_evidence()

    def test_probe_cli_rejects_batch_execution_flag(self) -> None:
        manifest, ref = cli.frozen_manifest()
        rows = [{"source_session": item["session_id"],
                 "submission_input_sha256": item["submission_input_sha256"],
                 "node": "CPU_ADMISSION" if item["session_id"].startswith("get_") else "ROBOT_V1",
                 "status": "READY" if item["session_id"].startswith("get_") else
                           "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS"}
                for item in manifest["sessions"]]
        bad = {"status": "READ_ONLY_MATRIX", "manifest": ref, "sessions": rows,
               "registration_started": False, "model_calls": 0,
               "batch_execution_authorized": True}
        completed = subprocess.CompletedProcess([], 0, json.dumps(bad).encode(), b"")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "V4_CLI_CHECK_ONLY_MATRIX_DRIFT"):
                accept._probe_cli(ref)

    def test_probe_tests_rejects_failure(self) -> None:
        completed = subprocess.CompletedProcess([], 1, b"", b"FAILED")
        with patch.object(accept.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "V4_CLI_NEGATIVE_TESTS_NOT_PASS"):
                accept._probe_tests()

    def test_packet_is_cpu_only_one_attempt_ten_minutes(self) -> None:
        packet = accept.packet_value({"manifest": {"path": str(cli.MANIFEST),
                                                    "bytes": 1, "sha256": "a" * 64}})
        self.assertEqual(packet["task_id"], cli.TASK_ID)
        self.assertEqual(packet["execution_mode"], "EXACT78_GOVERNANCE_ONLY")
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["model_calls"], 0)
        self.assertEqual(packet["budgets"]["render_calls"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 600)

    def test_active_writer_rejects_before_register_writes(self) -> None:
        with patch.object(accept, "_idle", side_effect=RuntimeError("ACTIVE_TASK")), \
             patch.object(accept, "atomic_json") as write, \
             patch.object(accept, "publish_bundle") as publish:
            with self.assertRaisesRegex(RuntimeError, "ACTIVE_TASK"):
                accept.register({"inputs": {}, "runtime_code": []}, 1)
        write.assert_not_called()
        publish.assert_not_called()

    def test_idle_checks_actual_active_ledger_and_no_preflight_artifacts(self) -> None:
        receipt = {"governance_revision": 1, "files": {
            "task_state": {"path": str(accept.TASK_STATE_PATH)},
            "task_packet_index_current": {"path": str(accept.INDEX)}}}
        state = {"next_task": {"task_id": "other"},
                 "tasks": [{"task_id": "other", "status": "RUNNING"}]}
        index = {"task_packets": [{"task_id": "other", "execution_allowed": True}]}
        values = {accept.RECEIPT_PATH: receipt, accept.TASK_STATE_PATH: state,
                  accept.INDEX: index}
        with patch.object(accept, "load_json", side_effect=lambda path: values[path]), \
             patch.object(accept, "validate_artifact_ref", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "V4_ACTIVE_TASK_OR_WRITER_CONFLICT"):
                accept._idle()

    def test_fence_failure_rejects_before_result_write(self) -> None:
        with patch.object(accept, "_fence", side_effect=RuntimeError("FENCE_DRIFT")), \
             patch.object(accept, "atomic_json") as write:
            with self.assertRaisesRegex(RuntimeError, "FENCE_DRIFT"):
                accept.finish({})
        write.assert_not_called()

    def test_check_only_main_never_registers(self) -> None:
        inputs = {"manifest": {"path": str(cli.MANIFEST), "bytes": 1,
                               "sha256": "a" * 64}, "source_evidence": {},
                  "cli_code": {}, "inherited_cli_v3_code": {},
                  "inherited_cli_v2_code": {}, "test_code": {},
                  "acceptance_test_code": {}, "published_operations": {},
                  "cli_check_only": {}, "negative_tests": {}}
        with patch.object(accept, "inspect", return_value=(10, {}, {}, inputs)), \
             patch.object(accept, "packet_value", return_value={"run_signature": "s"}), \
             patch.object(accept, "register") as register, \
             patch.object(accept, "finish") as finish, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(accept.main(["--check-only"]), 0)
        register.assert_not_called()
        finish.assert_not_called()

    def test_result_and_receipt_keep_model_and_visual_false(self) -> None:
        inputs = {"manifest": {}, "source_evidence": {}, "cli_code": {},
                  "inherited_cli_v3_code": {}, "inherited_cli_v2_code": {},
                  "test_code": {}, "acceptance_test_code": {},
                  "published_operations": {}, "cli_check_only": {},
                  "negative_tests": {}}
        packet = {"inputs": inputs, "run_signature": "sig"}
        result = accept.result_value(packet, {}, status="PASSED")
        receipt = accept.receipt_value(packet, {}, {}, status="PASSED")
        self.assertFalse(result["visual_qualified"])
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(receipt["status"], "PASSED")

    def test_packet_write_set_is_finite_and_has_no_wrapper_log_root(self) -> None:
        packet = accept.packet_value({"manifest": {"path": str(cli.MANIFEST),
                                                    "bytes": 1, "sha256": "a" * 64}})
        self.assertEqual(set(packet["write_set"]), {
            str(accept.ATTEMPT.relative_to(accept.REPO_ROOT)),
            str(accept.PACKET.relative_to(accept.REPO_ROOT)),
            str(accept.RECEIPT.relative_to(accept.REPO_ROOT)),
            str(accept.INDEX.relative_to(accept.REPO_ROOT)), "docs/governance"})
        self.assertNotIn("diagnostics", " ".join(packet["write_set"]))


if __name__ == "__main__":
    unittest.main()
