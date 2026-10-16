"""Model-free guards for the bounded Cards042 flange-link packet/operation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from chaoyang.governance import execute_exact78_robot_flange_link_cards042_v1 as executor
from chaoyang.ops import run_exact78_robot_flange_link_cards042_v1 as operation


class FlangePacketTests(unittest.TestCase):
    def test_frozen_spec_uses_only_sealed_development_hawor_and_raw(self) -> None:
        spec = executor.frozen_spec()
        self.assertEqual(spec["frame_count"], 171)
        self.assertEqual(spec["source_role"], "DEVELOPMENT_ONLY")
        self.assertFalse(spec["old_robot_result_input"])
        self.assertEqual(spec["single_change"],
                         "IK_AND_RASTER_HAND_ROOT_PARENT_FLANGE_INSTEAD_OF_SIDE_TOOL")
        operation._allowed_input_paths(executor.REPO_ROOT, spec["inputs"])
        paths = [ref["path"] for ref in spec["inputs"].values()]
        self.assertIn("exact78_raw_to_robot_t2_0902_cards042_v1_20260930",
                      spec["inputs"]["hawor_npz"]["path"])
        self.assertFalse(any("exact78_raw_to_robot_t1b_cards042_v1_20260930" in path
                             for path in paths))
        self.assertFalse(any("holdout" in path or "0916" in path for path in paths))
        self.assertFalse(any("/robot/" in path and "assets/robot/" not in path
                             for path in paths))

    def test_packet_pins_code_assets_signature_and_one_fresh_writer(self) -> None:
        spec = executor.frozen_spec()
        packet = executor.packet_value(spec)
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(packet["execution_mode"], "EXACT78_CANDIDATE")
        self.assertEqual(packet["commands"], [executor.COMMAND])
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["config"]["model_calls"], 0)
        self.assertIn("src/chaoyang/governance/current_r3_contracts.py",
                      executor.CODE_PATHS)
        self.assertIn("src/chaoyang/pipeline/exact78_robot_raster_v1.py",
                      executor.CODE_PATHS)
        self.assertEqual(packet["inputs"]["source_refs"], spec["inputs"])
        self.assertEqual(packet["inputs"]["flange_spec"],
                         executor._virtual_ref(executor.SPEC, spec))
        self.assertEqual(packet["write_set"][0],
                         str(executor.OUT.relative_to(executor.REPO_ROOT)))
        self.assertFalse(any("0916" in item for item in packet["read_set"]
                             + packet["write_set"]))

    def test_old_motion_and_wrong_raw_fail_before_algorithm_import(self) -> None:
        spec = executor.frozen_spec()
        wrong = copy.deepcopy(spec["inputs"])
        wrong["hawor_npz"]["path"] = str(executor.REPO_ROOT / "old_hawor.npz")
        with self.assertRaisesRegex(RuntimeError, "SEALED_PATH_DRIFT"):
            operation._allowed_input_paths(executor.REPO_ROOT, wrong)
        wrong = copy.deepcopy(spec["inputs"])
        wrong["raw_video"]["path"] = "/path/to/private_data/egodata/processed/old_remap.mp4"
        with self.assertRaisesRegex(RuntimeError, "RAW_SOURCE_DRIFT"):
            operation._allowed_input_paths(executor.REPO_ROOT, wrong)

    def test_direct_operation_gate_rejects_before_spec_or_algorithm(self) -> None:
        with (mock.patch.object(operation, "authorize_exact78_operation",
                                side_effect=RuntimeError("NO_PACKET")) as gate,
              mock.patch.object(operation, "_packet_and_spec") as packet):
            with self.assertRaisesRegex(RuntimeError, "NO_PACKET"):
                operation.main()
            gate.assert_called_once()
            packet.assert_not_called()

    def test_staged_route_requires_exact_candidate_code_closure(self) -> None:
        refs = executor.runtime_refs()
        expected = [ref for ref in refs if "/src/chaoyang/ops/" in ref["path"]
                    or "/src/chaoyang/pipeline/" in ref["path"]]
        route = {"special_status_contracts": {executor.CONTRACT_KEY: {
            "execution_status": "CANDIDATE_FENCED_ROBOT_ONLY_SINGLE_SESSION",
            "candidate_operation": executor.OPERATION,
            "task_packet_required": True,
            "current_runnable_operation": None,
            "output_schema": "exact78-flange-link-robot-successor-result-v1",
            "model_calls_authorized": 0,
            "training_steps_authorized": 0,
            "code_closure": expected,
        }}}
        self.assertTrue(executor._route_valid(route))
        route["special_status_contracts"][executor.CONTRACT_KEY]["code_closure"] = expected[:-1]
        self.assertFalse(executor._route_valid(route))

    def test_registration_state_preflight_has_exactly_one_new_writer(self) -> None:
        packet = executor.packet_value(executor.frozen_spec())
        state, index = executor._registration_values(packet)
        self.assertEqual(state["next_task"]["task_id"], executor.TASK)
        new = [row for row in state["tasks"] if row["task_id"] == executor.TASK]
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]["run_signature"], packet["run_signature"])
        self.assertEqual(index["task_packets"][-1]["packet_sha256"],
                         executor._virtual_ref(executor.PACKET, packet)["sha256"])


if __name__ == "__main__":
    unittest.main()
