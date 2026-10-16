"""Read-only and fencing checks for the finite displayed-acceleration executor."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_newraw_display_accel_cpu_candidate_v1 as executor
from chaoyang.ops import run_exact78_newraw_display_accel_cpu_candidate_v1 as operation
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage


class DisplayAccelerationExecutorTests(unittest.TestCase):
    def test_original_numeric_thresholds_are_unchanged(self) -> None:
        self.assertEqual((stage.POSITION_GATE_MM, stage.ROTATION_GATE_DEG,
                          stage.ARM_VELOCITY_GATE_RAD, stage.ARM_ACCELERATION_GATE_RAD),
                         (10.0, 5.0, 0.12, 0.06))

    def test_packet_pins_one_cpu_attempt_and_only_sealed_hawor(self) -> None:
        packet = executor.packet_value()
        self.assertEqual(packet["task_id"], executor.TASK_ID)
        self.assertEqual(packet["phase"], operation.PHASE)
        self.assertEqual(packet["execution_mode"], "EXACT78_CANDIDATE")
        self.assertEqual(packet["budgets"], {
            "workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
            "model_calls": 0, "render_calls": 0, "training_steps": 0,
            "wall_seconds_max": 1800, "session_attempts_max": 1})
        self.assertEqual(packet["inputs"]["hawor_outer_result"]["sha256"],
                         operation.EXPECTED_HAWOR_SHA)
        self.assertEqual(packet["config"]["single_change"], operation.SINGLE_CHANGE)
        for key in ("numeric_gates_changed", "base_selection_changed", "hand_mount_changed",
                    "old_robot_result_input"):
            self.assertIs(packet["config"][key], False)
        self.assertFalse(any("exact78_newraw_robot_play_cards_0902_001" in str(value)
                             for value in packet["inputs"].values()))
        self.assertEqual(packet["required_outputs"][-2:], ["RESULT.json", "terminal_receipt"])

    def test_route_requires_full_exact_code_closure_and_zero_calls(self) -> None:
        packet = executor.packet_value()
        route = {"candidate_operation": operation.OPERATION,
                 "execution_status": "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION",
                 "task_packet_required": True, "current_runnable_operation": None,
                 "output_schema": "exact78-newraw-display-accel-cpu-execution-result-v1",
                 "model_calls_authorized": 0, "render_calls_authorized": 0,
                 "training_steps_authorized": 0, "code_closure": packet["runtime_code"]}
        contract = {"special_status_contracts": {executor.ROUTE_KEY: route}}
        self.assertTrue(executor._route_valid(contract, packet))
        route["code_closure"] = route["code_closure"][:-1]
        self.assertFalse(executor._route_valid(contract, packet))
        route["code_closure"] = packet["runtime_code"]
        route["model_calls_authorized"] = 1
        self.assertFalse(executor._route_valid(contract, packet))

    def test_active_task_prevents_registration(self) -> None:
        packet = executor.packet_value()
        with patch.object(executor, "load_json", return_value={
                "next_task": {"task_id": "another"}, "tasks": []}):
            with self.assertRaisesRegex(RuntimeError, "ACTIVE_TASK_CONFLICT"):
                executor.readiness(packet)

    def test_missing_stage_output_ref_fails_before_array_read(self) -> None:
        packet = executor.packet_value()
        value = {"schema_version": "exact78-newraw-display-accel-cpu-result-v1",
                 "status": "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS",
                 "session_id": operation.EXPECTED_SESSION, "frame_count": 201,
                 "fps_nominal": 30.0, "single_change": operation.SINGLE_CHANGE,
                 "model_calls": 0, "render_calls": 0, "gpu_count": 0,
                 "visual_qualified": False, "training_eligible": False,
                 "inputs": {"hawor_npz": packet["inputs"]["hawor_npz"],
                            "frame_provenance": packet["inputs"]["hawor_frame_provenance"],
                            "input_audit": packet["inputs"]["hawor_input_audit"],
                            "raw_video": packet["inputs"]["raw_video"]},
                 "outputs": {}}
        with (patch.object(Path, "is_file", return_value=True),
              patch.object(executor, "load_json", return_value=value)):
            result, errors = executor.verify_candidate(packet)
        self.assertIsNone(result)
        self.assertIn("DISPLAY_ACCEL_OUTPUT_REF_SET_DRIFT", errors[0])

    def test_writer_fence_rejects_other_process_before_ledger_read(self) -> None:
        packet = executor.packet_value()
        packet["writer"]["pid"] = -1
        with self.assertRaisesRegex(RuntimeError, "WRITER_FENCE_DRIFT"):
            executor.verify_fence(packet)

    def test_check_only_does_not_publish_or_start_worker(self) -> None:
        with (patch.object(executor, "packet_value", return_value={"run_signature": "abc"}),
              patch.object(executor, "readiness", side_effect=RuntimeError("ACTIVE_TASK_CONFLICT")),
              patch.object(executor, "route_readiness", side_effect=RuntimeError("ROUTE_NOT_STAGED")),
              patch.object(executor, "register") as register,
              patch.object(executor, "run_worker") as worker,
              patch("sys.argv", ["executor", "--check-only"])):
            self.assertEqual(executor.main(), 0)
        register.assert_not_called()
        worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
