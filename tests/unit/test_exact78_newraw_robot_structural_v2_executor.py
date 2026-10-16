"""Read-only and model-free tests for the unrouted 0901 Chips001 v2 executor."""

from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from chaoyang.governance import execute_exact78_newraw_robot_structural_v2_candidate as executor
from chaoyang.pipeline import exact78_newraw_robot_structural_v2_candidate as structural


class StructuralV2ExecutorTests(unittest.TestCase):
    def test_real_packet_is_new_signature_schema_and_finite_799_cpu(self) -> None:
        packet = executor.packet_value()
        self.assertEqual(packet["task_id"], structural.TASK_ID)
        self.assertEqual(packet["phase"], "EXACT78_NEWRAW_ROBOT_STRUCTURAL_V2_CANDIDATE")
        self.assertEqual(packet["config"]["frame_count"], 799)
        self.assertEqual(packet["config"]["fps_nominal"], 25.0)
        self.assertEqual(packet["budgets"]["workers_max"], 1)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 5400)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["config"]["model_calls"], 0)
        self.assertLessEqual(datetime.fromisoformat(packet["deadline_at"]),
                             executor.H20_DEADLINE)
        self.assertIn(executor.OPERATION, packet["commands"][0])
        self.assertNotIn("--hawor-result", packet["commands"][0])
        self.assertIn("candidate/robot_v2", packet["commands"][0])
        self.assertNotEqual(packet["run_signature"],
                            json.loads(structural._V1_PACKET.read_text())["run_signature"])
        self.assertEqual(packet["inputs"]["hawor_npz"]["sha256"],
                         structural._SOURCE_SHA["hawor_npz"])
        self.assertNotIn("old_robot", packet["inputs"])
        self.assertFalse(packet["config"]["numeric_gates_changed"])
        self.assertFalse(packet["config"]["base_search_rule_changed"])
        self.assertFalse(packet["config"]["display_acceleration_changed"])
        self.assertEqual(packet["config"]["arm_velocity_gate_rad"], .12)
        self.assertEqual(packet["config"]["arm_acceleration_gate_rad"], .06)
        self.assertEqual(packet["config"]["hand_velocity_gate_rad"], .08)
        self.assertEqual(packet["config"]["hand_acceleration_gate_rad"], .06)

    def test_check_only_is_ready_without_registering_or_rendering(self) -> None:
        attempt, packet_path, receipt = executor._paths()
        with mock.patch.object(executor, "publish_bundle") as publish, \
             mock.patch.object(structural, "run_candidate") as render:
            result = executor.check_only()
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["scope"], "CODE_INPUT_PACKET_PREP_ONLY")
        self.assertEqual(result["route_status"], "ROUTE_TEMPLATE_READY_BUILDER_NOT_STAGED")
        self.assertEqual((result["render_calls"], result["model_calls"]), (0, 0))
        self.assertFalse(result["registration_started"])
        self.assertFalse(any(path.exists() for path in (attempt, packet_path, receipt)))
        publish.assert_not_called()
        render.assert_not_called()

    def test_route_template_is_exact_and_staged_route_is_admissible(self) -> None:
        packet = executor.packet_value()
        contract = {"special_status_contracts": {
            executor.ROUTE_KEY: executor.route_contract(packet["runtime_code"])}}
        self.assertTrue(executor._route_valid(contract, packet))
        for key, bad in (("candidate_operation", "old_v1"),
                         ("output_schema", "exact78-newraw-robot-cpu-execution-result-v1"),
                         ("model_calls_authorized", 1),
                         ("current_runnable_operation", executor.OPERATION)):
            altered = copy.deepcopy(contract)
            altered["special_status_contracts"][executor.ROUTE_KEY][key] = bad
            self.assertFalse(executor._route_valid(altered, packet), key)
        altered = copy.deepcopy(contract)
        altered["special_status_contracts"][executor.ROUTE_KEY]["code_closure"][0]["sha256"] = "0" * 64
        self.assertFalse(executor._route_valid(altered, packet))

    def test_register_requires_staged_route_before_any_write(self) -> None:
        packet = executor.packet_value()
        with mock.patch.object(executor, "_route_readiness",
                               side_effect=RuntimeError("STRUCTURAL_V2_ROUTE_NOT_STAGED")), \
             mock.patch.object(executor, "publish_bundle") as publish:
            with self.assertRaisesRegex(RuntimeError, "ROUTE_NOT_STAGED"):
                executor.register(packet)
        attempt, packet_path, receipt = executor._paths()
        self.assertFalse(any(path.exists() for path in (attempt, packet_path, receipt)))
        publish.assert_not_called()

    def test_staged_registration_uses_one_packet_index_cas(self) -> None:
        packet = executor.packet_value()
        with tempfile.TemporaryDirectory(prefix="structural_v2_register_",
                                         dir=executor.REPO_ROOT / "_run/maintenance") as temp:
            attempt = Path(temp) / "attempt_0001"
            packet_path = Path(temp) / "task" / "TASK_PACKET.json"
            receipt = Path(temp) / "RECEIPT.json"
            state = {"tasks": [], "next_task": None}
            index = {"task_packets": [], "schema_version": "chaoyang-v71-task-packet-index-v3"}
            def read(path: Path) -> dict:
                return state if path == executor.TASK_STATE_PATH else {}
            with mock.patch.object(executor, "_route_readiness",
                                   return_value="READY_TO_PUBLISH_ROUTE_WITH_PACKET"), \
                 mock.patch.object(executor, "_paths",
                                   return_value=(attempt, packet_path, receipt)), \
                 mock.patch.object(executor, "packet_value", return_value=packet), \
                 mock.patch.object(executor, "_index_copy", return_value=index), \
                 mock.patch.object(executor, "_receipt_revision", return_value=12), \
                 mock.patch.object(executor, "load_json", side_effect=read), \
                 mock.patch.object(executor, "validate_task_state", return_value=[]), \
                 mock.patch.object(executor, "publish_bundle") as publish:
                executor.register(packet)
            self.assertTrue(packet_path.is_file())
            self.assertTrue((attempt / "WRITER.json").is_file())
            self.assertFalse(receipt.exists())
            self.assertEqual(index["task_packets"][0]["task_id"], structural.TASK_ID)
            self.assertTrue(index["task_packets"][0]["execution_allowed"])
            publish.assert_called_once()
            self.assertEqual(publish.call_args.kwargs["expected_revision"], 12)
            self.assertIs(publish.call_args.kwargs["task_packet_index_value"], index)

    def test_stage_verifier_rejects_old_v1_schema_without_consuming_output(self) -> None:
        packet = executor.packet_value()
        with mock.patch.object(executor, "load_json", return_value={
            "schema_version": "exact78-fresh-robot-ffmpeg44-successor-result-v1"}):
            with self.assertRaisesRegex(RuntimeError, "RESULT_CLAIM_DRIFT"):
                executor._verify_stage(packet)

    def test_failed_worker_closes_without_adopting_partial(self) -> None:
        packet = executor.packet_value()
        with tempfile.TemporaryDirectory(prefix="structural_v2_executor_",
                                         dir=executor.REPO_ROOT / "_run/maintenance") as temp:
            attempt = Path(temp) / "attempt_0001"
            attempt.mkdir()
            log = attempt / "worker.log"
            log.write_text("fixture failure\n", encoding="utf-8")
            packet_path, receipt = Path(temp) / "TASK_PACKET.json", Path(temp) / "RECEIPT.json"
            state = {"tasks": [{"task_id": structural.TASK_ID, "status": "RUNNING"}],
                     "next_task": {"task_id": structural.TASK_ID}}
            index = {"task_packets": [{"task_id": structural.TASK_ID,
                                        "execution_allowed": True}]}
            with mock.patch.object(executor, "_paths", return_value=(attempt, packet_path, receipt)), \
                 mock.patch.object(executor, "_fence"), \
                 mock.patch.object(executor, "_index_copy", return_value=index), \
                 mock.patch.object(executor, "_receipt_revision", return_value=6), \
                 mock.patch.object(executor, "validate_task_state", return_value=[]), \
                 mock.patch.object(executor, "publish_bundle",
                                   return_value={"governance_revision": 7}) as publish, \
                 mock.patch.object(executor, "load_json", return_value=state):
                terminal = executor.finish(packet, worker_returncode=9,
                                           worker_log=log, stage=None, error="MODEL_ABSENT")
            result = json.loads((attempt / "RESULT.json").read_text())
            self.assertEqual(terminal["status"], "FAILED_RUNTIME_FINAL")
            self.assertEqual(result["schema_version"], executor.OUTPUT_SCHEMA)
            self.assertFalse(result["render_complete"])
            self.assertFalse(result["visual_qualified"])
            self.assertFalse(result["training_eligible"])
            self.assertEqual(result["stage"], {})
            self.assertFalse(index["task_packets"][0]["execution_allowed"])
            self.assertTrue(receipt.is_file())
            publish.assert_called_once()


if __name__ == "__main__":
    unittest.main()
