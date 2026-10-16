"""Read-only packet/operation guards for separate Cards042 CPU diagnostics."""

from __future__ import annotations

import copy
import unittest
from unittest import mock

from chaoyang.governance import execute_exact78_cards042_cpu_motion_diagnostic_v1 as executor
from chaoyang.ops import run_exact78_cards042_cpu_motion_diagnostic_v1 as operation


class Cards042CpuPacketTests(unittest.TestCase):
    def test_first_anchor_packet_is_single_change_and_pins_exact_source(self) -> None:
        spec = executor.frozen_spec("FIRST_ANCHOR_ONLY")
        packet = executor.packet_value("FIRST_ANCHOR_ONLY", spec)
        self.assertEqual(spec["single_change"], executor.cpu.MODES["FIRST_ANCHOR_ONLY"])
        self.assertFalse(spec["base_selection_changed"])
        self.assertTrue(spec["first_anchor_rule_changed"])
        self.assertFalse(spec["old_robot_result_input"])
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(packet["execution_mode"], "EXACT78_CANDIDATE")
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["model_calls"], 0)
        self.assertEqual(packet["budgets"]["render_calls"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["commands"], [
            f"PYTHONPATH=src {executor.PYTHON} -B -m chaoyang.cli run "
            f"{executor.OPERATION} --spec {executor.paths('FIRST_ANCHOR_ONLY')['spec']}"])
        self.assertEqual(packet["inputs"]["source_refs"], spec["inputs"])
        self.assertNotIn("comparison_parent_evidence_only", packet["inputs"])
        self.assertTrue(all("0916" not in entry for entry in packet["read_set"] + packet["write_set"]))
        self.assertTrue(all("/candidate/robot/" not in ref["path"] for ref in spec["inputs"].values()))

    def test_base_packet_waits_for_sealed_first_anchor_result(self) -> None:
        spec = executor.frozen_spec("BASE_RERANK_ONLY")
        self.assertTrue(spec["base_selection_changed"])
        self.assertFalse(spec["first_anchor_rule_changed"])
        with mock.patch.object(executor, "_first_anchor_parent_ref", return_value=None):
            self.assertEqual(executor.readiness("BASE_RERANK_ONLY"),
                             (False, "WAIT_FIRST_ANCHOR_SEALED_RESULT"))
            with self.assertRaisesRegex(RuntimeError, "WAIT_FIRST_ANCHOR_SEALED_RESULT"):
                executor.packet_value("BASE_RERANK_ONLY", spec)

    def test_packet_preflight_and_writer_are_one_fresh_attempt(self) -> None:
        packet = executor.packet_value("FIRST_ANCHOR_ONLY", executor.frozen_spec("FIRST_ANCHOR_ONLY"))
        state, index = executor._registration_values("FIRST_ANCHOR_ONLY", packet)
        task = executor.MODES["FIRST_ANCHOR_ONLY"]
        self.assertEqual(state["next_task"]["task_id"], task)
        self.assertEqual(index["task_packets"][-1]["task_id"], task)
        self.assertEqual(index["task_packets"][-1]["packet_sha256"],
                         executor._virtual_ref(executor.paths("FIRST_ANCHOR_ONLY")["packet"], packet)["sha256"])

    def test_route_requires_exact_operation_and_cpu_code_closure(self) -> None:
        refs = executor.runtime_refs()
        closure = [ref for ref in refs if "/src/chaoyang/ops/" in ref["path"]
                   or "/src/chaoyang/pipeline/" in ref["path"]]
        route = {"special_status_contracts": {executor.ROUTE_KEY: {
            "execution_status": "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION",
            "candidate_operation": executor.OPERATION,
            "task_packet_required": True, "current_runnable_operation": None,
            "output_schema": "exact78-cards042-cpu-motion-result-v1",
            "model_calls_authorized": 0, "render_calls_authorized": 0,
            "code_closure": closure}}}
        self.assertTrue(executor._route_valid(route))
        route["special_status_contracts"][executor.ROUTE_KEY]["code_closure"] = closure[:-1]
        self.assertFalse(executor._route_valid(route))

    def test_direct_operation_rejects_before_spec_or_algorithm_import(self) -> None:
        with mock.patch.object(operation, "authorize_exact78_operation",
                               side_effect=RuntimeError("NO_ACTIVE_PACKET")) as gate, \
             mock.patch.object(operation, "_packet_and_spec") as packet:
            with self.assertRaisesRegex(RuntimeError, "NO_ACTIVE_PACKET"):
                operation.main()
            gate.assert_called_once()
            packet.assert_not_called()

    def test_historical_motion_path_is_not_a_valid_frozen_input(self) -> None:
        spec = executor.frozen_spec("FIRST_ANCHOR_ONLY")
        operation._allowed_input_paths(executor.REPO_ROOT, spec["inputs"])
        wrong = copy.deepcopy(spec)
        wrong["inputs"]["hawor_npz"]["path"] = str(executor.REPO_ROOT / "old_hawor.npz")
        with self.assertRaisesRegex(RuntimeError, "HISTORICAL_OR_WRONG_SOURCE_PATH"):
            operation._allowed_input_paths(executor.REPO_ROOT, wrong["inputs"])
        wrong = copy.deepcopy(spec)
        wrong["inputs"]["raw_video"]["path"] = "/path/to/private_data/egodata/processed/old_remap.mp4"
        with self.assertRaisesRegex(RuntimeError, "RAW_SOURCE_DRIFT"):
            operation._allowed_input_paths(executor.REPO_ROOT, wrong["inputs"])


if __name__ == "__main__":
    unittest.main()
