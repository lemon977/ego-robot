"""Model-free negative checks for the isolated two-seed CPU candidate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
from chaoyang.pipeline import exact78_newraw_two_seed_cpu_candidate_v1 as two
from chaoyang.ops import run_exact78_newraw_two_seed_cpu_candidate_v1 as op
from chaoyang.governance import execute_exact78_newraw_two_seed_cpu_candidate_v1 as executor
from chaoyang.governance import compare_exact78_newraw_two_seed_cpu_v1 as evaluator


def candidate(value: float, score: float) -> dict:
    return {"q": np.full(7, value), "position_mm": value,
            "rotation_deg": value, "branch_margin_m_min": .01,
            "score": (False, 0.0, 0.0, score)}


class TwoSeedCpuTests(unittest.TestCase):
    @staticmethod
    def _candidate_stub(packet: dict) -> dict:
        source = packet["inputs"]
        root = executor.OUTPUT / "candidate/two_seed_cpu"
        return {
            "schema_version": "exact78-newraw-two-seed-cpu-result-v1",
            "status": "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS",
            "session_id": op.EXPECTED_SESSION, "frame_count": 201,
            "old_robot_result_input": False, "numeric_gates_changed": False,
            "inputs": {"hawor_npz": source["hawor_npz"],
                       "frame_provenance": source["hawor_frame_provenance"],
                       "input_audit": source["hawor_input_audit"],
                       "raw_video": source["raw_video"]},
            "outputs": {name: {"path": str(root / name), "bytes": 1,
                               "sha256": "0" * 64} for name in (
                "BASE_SEARCH.json", "TWO_SEED_FRAMES.json",
                "SHADOW_BASELINE.json", "NUMERIC_STATES.npz")},
        }

    def test_only_temporal_rejection_can_choose_other_seed(self) -> None:
        pass_ = ("PASS_NUMERIC_CAMERA_DIAGNOSTIC", .02, .01)
        velocity = ("HOLD_ARM_VELOCITY", .13, .01)
        acceleration = ("HOLD_ARM_ACCELERATION", .04, .07)
        reach = ("HOLD_REACH", .02, .01)
        self.assertEqual(two.choose_seed_on_temporal_rejection(0, [velocity, pass_]), 1)
        self.assertEqual(two.choose_seed_on_temporal_rejection(1, [pass_, acceleration]), 0)
        self.assertEqual(two.choose_seed_on_temporal_rejection(0, [reach, pass_]), 0)
        self.assertEqual(two.choose_seed_on_temporal_rejection(0, [velocity, reach]), 0)
        self.assertEqual(two.choose_seed_on_temporal_rejection(0, [pass_, pass_]), 0)

    def test_original_temporal_gate_is_unchanged(self) -> None:
        previous = np.zeros(7)
        previous_previous = np.zeros(7)
        value = candidate(.12, .12)
        self.assertEqual(two._temporal_status(value, previous, previous_previous,
                                              3, 4)[0], "HOLD_ARM_ACCELERATION")
        self.assertEqual(two._temporal_status(value, previous, previous_previous,
                                              2, 4)[0], "PASS_NUMERIC_CAMERA_DIAGNOSTIC")
        value = candidate(.120000002, .12)
        self.assertEqual(two._temporal_status(value, previous, None, 3, 4)[0],
                         "HOLD_ARM_VELOCITY")

    def test_default_seed_score_reproduced_from_frozen_attempt(self) -> None:
        calls = iter((.1, .2))

        def fake_least_squares(*args, **kwargs):
            return SimpleNamespace(x=np.full(7, next(calls)))

        def frozen_attempt(assets, side, target, lower, upper, seeds, previous=None):
            stage.least_squares(None, None)
            stage.least_squares(None, None)
            return np.full(7, .1), .1, .1, .01

        with (patch.object(stage, "least_squares", fake_least_squares),
              patch.object(stage, "_arm_attempt", frozen_attempt),
              patch.object(two, "_candidate_metrics",
                           side_effect=[candidate(.1, .1), candidate(.2, .2)])):
            index, rows = two._capture_attempt(None, 0, np.eye(4), np.full(7, -1.),
                                               np.full(7, 1.), [np.zeros(7), np.ones(7)], None)
        self.assertEqual(index, 0)
        self.assertEqual(len(rows), 2)
        self.assertIs(stage.least_squares, two.stage.least_squares)

    def test_default_drift_fails_closed_and_restores_solver(self) -> None:
        original = stage.least_squares

        def frozen_attempt(assets, side, target, lower, upper, seeds, previous=None):
            stage.least_squares(None, None)
            stage.least_squares(None, None)
            return np.full(7, .9), .1, .1, .01

        with (patch.object(stage, "least_squares",
                           side_effect=[SimpleNamespace(x=np.full(7, .1)),
                                        SimpleNamespace(x=np.full(7, .2))]),
              patch.object(stage, "_arm_attempt", frozen_attempt),
              patch.object(two, "_candidate_metrics",
                           side_effect=[candidate(.1, .1), candidate(.2, .2)])):
            with self.assertRaisesRegex(stage.FreshRobotError,
                                        "TWO_SEED_DEFAULT_SCORE_REPRODUCTION_FAILED"):
                two._capture_attempt(None, 0, np.eye(4), np.full(7, -1.),
                                     np.full(7, 1.), [np.zeros(7), np.ones(7)], None)
        self.assertIs(stage.least_squares, original)

    def test_unregistered_operation_cannot_run(self) -> None:
        with patch.object(op, "authorize_exact78_operation",
                          side_effect=RuntimeError("NO_ACTIVE_PACKET")):
            with self.assertRaisesRegex(RuntimeError, "NO_ACTIVE_PACKET"):
                op.run_candidate(Path("/project/fake_hawor.json"),
                                 Path("/project/fake_output"))

    def test_wrong_hawor_sha_rejected_before_handoff(self) -> None:
        with patch("chaoyang.governance.exact78_cold_start_guard_v1.file_ref",
                   return_value={"sha256": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "TWO_SEED_HAWOR_V4_SHA_DRIFT"):
                op._handoff(Path(__file__))

    def test_packet_is_finite_and_has_no_old_robot_input(self) -> None:
        packet = executor.packet_value()
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 1200)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertLessEqual(len(packet["read_set"]), 8)
        self.assertFalse(packet["config"]["old_robot_result_input"])
        self.assertFalse(any("exact78_newraw_robot_play_cards_0902_001" in
                             str(item) for item in packet["inputs"].values()))
        self.assertIn("--hawor-result", packet["commands"][0])

    def test_candidate_route_requires_exact_code_closure(self) -> None:
        packet = executor.packet_value()
        route = {"candidate_operation": executor.OPERATION,
                 "execution_status": "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION",
                 "task_packet_required": True, "current_runnable_operation": None,
                 "output_schema": "exact78-newraw-two-seed-cpu-execution-result-v1",
                 "model_calls_authorized": 0, "render_calls_authorized": 0,
                 "code_closure": packet["runtime_code"]}
        contract = {"special_status_contracts": {executor.ROUTE_KEY: route}}
        self.assertTrue(executor._route_valid(contract, packet))
        route["code_closure"] = route["code_closure"][:-1]
        self.assertFalse(executor._route_valid(contract, packet))
        route["code_closure"] = [ref for ref in packet["runtime_code"] if not
                                 ref["path"].endswith("exact78_fresh_robot_stage_v1.py")]
        self.assertFalse(executor._route_valid(contract, packet))

    def test_terminal_rejects_missing_output_ref_before_reading_children(self) -> None:
        packet = executor.packet_value()
        value = self._candidate_stub(packet)
        del value["outputs"]["NUMERIC_STATES.npz"]
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(executor, "load_json", return_value=value):
            candidate_result, errors = executor.verify_candidate(packet)
        self.assertIsNone(candidate_result)
        self.assertIn("TWO_SEED_OUTPUT_REF_SET_DRIFT", errors[0])

    def test_terminal_rejects_replaced_output_ref(self) -> None:
        packet = executor.packet_value()
        value = self._candidate_stub(packet)
        value["outputs"]["NUMERIC_STATES.npz"]["sha256"] = "1" * 64
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(executor, "load_json", return_value=value), \
             patch.object(executor, "validate_artifact_ref",
                          return_value=["sha256 mismatch"]):
            candidate_result, errors = executor.verify_candidate(packet)
        self.assertIsNone(candidate_result)
        self.assertIn("TWO_SEED_OUTPUT_SHA_DRIFT", errors[0])

    def test_terminal_rejects_replaced_producer_input_ref(self) -> None:
        packet = executor.packet_value()
        value = self._candidate_stub(packet)
        value["inputs"]["hawor_npz"] = {**value["inputs"]["hawor_npz"],
                                          "sha256": "1" * 64}
        with patch.object(Path, "is_file", return_value=True), \
             patch.object(executor, "load_json", return_value=value):
            candidate_result, errors = executor.verify_candidate(packet)
        self.assertIsNone(candidate_result)
        self.assertIn("TWO_SEED_CANDIDATE_INPUT_REF_DRIFT", errors[0])

    def test_evaluator_must_see_terminal_before_old_robot(self) -> None:
        with patch.object(evaluator, "_new_terminal",
                          side_effect=RuntimeError("NOT_TERMINAL")):
            with patch.object(evaluator, "file_ref") as old_ref:
                with self.assertRaisesRegex(RuntimeError, "NOT_TERMINAL"):
                    evaluator.compare_sealed(Path("/missing/new.json"))
                old_ref.assert_not_called()

    def test_numeric_equality_contract(self) -> None:
        self.assertTrue(evaluator._same_number(.1, .1000005))
        self.assertFalse(evaluator._same_number(.1, .100002))
        self.assertFalse(evaluator._same_number(None, 0.0))


if __name__ == "__main__":
    unittest.main()
