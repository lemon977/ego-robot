"""Model-free checks for the isolated 201-frame velocity-tube IK shadow."""

from __future__ import annotations

from pathlib import Path
import signal
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.ops import run_exact78_newraw_velocity_tube_cpu_candidate_v1 as op
from chaoyang.governance import execute_exact78_newraw_velocity_tube_cpu_candidate_v1 as executor
from chaoyang.pipeline import exact78_fresh_robot_stage_v1 as stage
from chaoyang.pipeline import exact78_newraw_velocity_tube_cpu_candidate_v1 as tube


class VelocityTubeCpuTests(unittest.TestCase):
    def test_first_frame_uses_original_joint_limits(self) -> None:
        low, high = np.full(7, -1.), np.full(7, 1.)
        narrowed = tube.velocity_bounds(low, high, None)
        np.testing.assert_array_equal(narrowed[0], low)
        np.testing.assert_array_equal(narrowed[1], high)
        narrowed[0][0] = 99.
        self.assertEqual(low[0], -1.)

    def test_later_bounds_intersect_original_limits_at_exact_existing_gate(self) -> None:
        self.assertEqual(stage.ARM_VELOCITY_GATE_RAD, .12)
        low, high = np.full(7, -.5), np.full(7, .5)
        display = np.array((-.49, .49, 0., 0., 0., 0., 0.))
        narrowed_low, narrowed_high = tube.velocity_bounds(low, high, display)
        np.testing.assert_allclose(narrowed_low[:3], (-.5, .37, -.12))
        np.testing.assert_allclose(narrowed_high[:3], (-.37, .5, .12))
        np.testing.assert_array_equal(low, np.full(7, -.5))
        np.testing.assert_array_equal(high, np.full(7, .5))

    def test_invalid_tube_fails_closed(self) -> None:
        with self.assertRaisesRegex(stage.FreshRobotError, "VELOCITY_TUBE_BOUNDS_INVALID"):
            tube.velocity_bounds(np.zeros(7), np.ones(7), np.full(7, 2.))

    def test_full_grid_changes_only_later_arm_bounds(self) -> None:
        joints = np.zeros((2, 201, 21, 3), dtype=float)
        valid = np.ones((2, 201), dtype=bool)
        shadow_rows = [
            {"frame": frame, "side": side,
             "arm": {"pass_numeric": frame < (44 if side == "left" else 2)},
             "hand": {"pass_numeric": True}}
            for frame in range(201) for side in stage.SIDES]
        shadow = {
            "display_joints_3d_camera": joints.copy(),
            "display_drawable": valid.copy(),
            "motion_source": np.full((2, 201), "HAWOR_PREDICTION"),
            "q_arm": np.zeros((201, 2, 7)), "q_hand": np.zeros((201, 2, 1)),
        }
        report = {"anchor_frames": [0, 0]}
        captured = (np.eye(4), report, np.zeros((2, 3)), np.zeros((2, 7)),
                    np.repeat(np.eye(4)[None], 2, axis=0))
        seen: list[tuple[int, np.ndarray, np.ndarray]] = []

        def attempt(_assets, side, _target, low, high, _seeds, previous=None):
            seen.append((side, low.copy(), high.copy()))
            return np.zeros(7), 0., 0., .01

        with (patch.object(tube, "_shadow_and_base", return_value=(
                shadow, report, shadow_rows, captured)),
              patch.object(stage, "load_pinned_robot_assets", return_value=object()),
              patch.object(stage, "_mounts", return_value=(
                  np.repeat(np.eye(4)[None], 2, axis=0), {})),
              patch.object(stage, "_arm_limits", return_value=(
                  np.full((2, 7), -1.), np.full((2, 7), 1.))),
              patch.object(stage, "_palm_basis", return_value=np.eye(3)),
              patch.object(stage, "_arm_attempt", side_effect=attempt)):
            states, base, rows, baseline = tube.solve_velocity_tube(joints, valid, 30.)
        self.assertEqual(len(seen), 402)
        self.assertEqual(len(rows), 402)
        self.assertEqual(len(baseline), 402)
        self.assertEqual(base, report)
        for index in (0, 201):
            np.testing.assert_array_equal(seen[index][1], np.full(7, -1.))
            np.testing.assert_array_equal(seen[index][2], np.full(7, 1.))
        for index in (1, 202):
            np.testing.assert_array_equal(seen[index][1], np.full(7, -.12))
            np.testing.assert_array_equal(seen[index][2], np.full(7, .12))
        np.testing.assert_array_equal(states["q_hand"], shadow["q_hand"])
        np.testing.assert_array_equal(states["q_arm_shadow"], shadow["q_arm"])
        self.assertIsNone(rows[0]["previous_display_q_rad"])
        self.assertEqual(rows[2]["previous_display_q_rad"], [0.] * 7)

    def test_shape_and_shadow_drift_rejected(self) -> None:
        with self.assertRaisesRegex(stage.FreshRobotError, "INPUT_SHAPE_OR_FPS_DRIFT"):
            tube.solve_velocity_tube(np.zeros((2, 1, 21, 3)), np.ones((2, 1)), 30.)
        wrong = [{"side": side, "arm": {"pass_numeric": False}}
                 for side in stage.SIDES]
        with patch.object(tube, "_shadow_and_base", return_value=({}, {}, wrong, ())):
            with self.assertRaisesRegex(stage.FreshRobotError, "SHADOW_BASELINE_ARM_COUNT_DRIFT"):
                tube.solve_velocity_tube(np.zeros((2, 201, 21, 3)),
                                         np.ones((2, 201)), 30.)

    def test_automatic_base_capture_restores_frozen_solver_on_failure(self) -> None:
        original = stage._base_search
        with patch.object(stage, "solve_camera_diagnostic",
                          side_effect=stage.FreshRobotError("BASE_FAIL")):
            with self.assertRaisesRegex(stage.FreshRobotError, "BASE_FAIL"):
                tube._shadow_and_base(np.zeros((2, 201, 21, 3)),
                                      np.ones((2, 201)), 30.)
        self.assertIs(stage._base_search, original)

    def test_unregistered_execution_rejected_before_algorithm_import(self) -> None:
        with patch.object(op, "authorize_exact78_operation",
                          side_effect=RuntimeError("NO_PUBLISHED_ROUTE")):
            with self.assertRaisesRegex(RuntimeError, "NO_PUBLISHED_ROUTE"):
                op.run_candidate(Path("/missing/hawor.json"), Path("/missing/output"))

    def test_wrong_outer_hawor_sha_rejected_before_handoff(self) -> None:
        with patch("chaoyang.governance.exact78_cold_start_guard_v1.file_ref",
                   return_value={"sha256": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "HAWOR_V4_SHA_DRIFT"):
                op._handoff(Path(__file__))

    def test_check_only_cannot_choose_output_path(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CHECK_ONLY_MUST_NOT_SET_OUTPUT"):
            op.main(["--hawor-result", "unused", "--output-root", "forbidden",
                     "--check-only"])

    def test_finite_packet_pins_v4_and_old_robot_is_not_input(self) -> None:
        packet = executor.packet_value()
        self.assertEqual(packet["budgets"]["wall_seconds_max"], 1800)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertFalse(packet["config"]["old_robot_result_input"])
        self.assertFalse(packet["config"]["numeric_gates_changed"])
        self.assertFalse(packet["config"]["base_selection_changed"])
        self.assertFalse(packet["config"]["hand_mount_changed"])
        self.assertEqual(packet["inputs"]["hawor_outer_result"]["sha256"],
                         op.EXPECTED_HAWOR_SHA)
        self.assertFalse(any("exact78_newraw_robot_play_cards_0902_001" in str(value)
                             for value in packet["inputs"].values()))
        self.assertEqual(packet["write_set"][0],
                         str(executor.OUTPUT.relative_to(executor.REPO_ROOT)))

    def test_route_requires_exact_code_closure_and_no_model(self) -> None:
        packet = executor.packet_value()
        route = {"candidate_operation": op.OPERATION,
                 "execution_status": "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION",
                 "task_packet_required": True, "current_runnable_operation": None,
                 "output_schema": "exact78-newraw-velocity-tube-cpu-execution-result-v1",
                 "model_calls_authorized": 0, "render_calls_authorized": 0,
                 "code_closure": packet["runtime_code"]}
        contract = {"special_status_contracts": {executor.ROUTE_KEY: route}}
        self.assertTrue(executor._route_valid(contract, packet))
        route["code_closure"] = route["code_closure"][:-1]
        self.assertFalse(executor._route_valid(contract, packet))

    def test_terminal_missing_output_ref_fails_before_stage_consumption(self) -> None:
        packet = executor.packet_value()
        value = {"schema_version": "exact78-newraw-velocity-tube-cpu-result-v1",
                 "status": "COMPLETE_CPU_NUMERIC_DIAGNOSTIC_NOT_VISUAL_PASS",
                 "session_id": op.EXPECTED_SESSION, "frame_count": 201,
                 "fps_nominal": 30.0, "model_calls": 0, "render_calls": 0,
                 "gpu_count": 0, "visual_qualified": False,
                 "training_eligible": False,
                 "inputs": {"hawor_npz": packet["inputs"]["hawor_npz"],
                            "frame_provenance": packet["inputs"]["hawor_frame_provenance"],
                            "input_audit": packet["inputs"]["hawor_input_audit"],
                            "raw_video": packet["inputs"]["raw_video"]},
                 "outputs": {}}
        with (patch.object(Path, "is_file", return_value=True),
              patch.object(executor, "load_json", return_value=value)):
            result, errors = executor.verify_candidate(packet)
        self.assertIsNone(result)
        self.assertIn("VELOCITY_TUBE_OUTPUT_REF_SET_DRIFT", errors[0])

    def test_timeout_terms_only_fenced_child_group_and_proves_exit(self) -> None:
        class Worker:
            pid = 39171
            returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.returncode = -signal.SIGTERM
                return self.returncode

        worker = Worker()

        def kill_group(pgid, sent):
            self.assertEqual(pgid, worker.pid)
            if sent == 0:
                raise ProcessLookupError()
            self.assertEqual(sent, signal.SIGTERM)

        with (patch.object(executor, "process_identity",
                           return_value={"start_ticks": 456}),
              patch.object(executor.os, "getpgid", return_value=worker.pid),
              patch.object(executor.os, "killpg", side_effect=kill_group) as sent):
            executor._stop_fenced_group(worker, 456)
        self.assertEqual([call.args[1] for call in sent.call_args_list],
                         [signal.SIGTERM, 0])

    def test_timeout_kills_same_group_after_bounded_term(self) -> None:
        class Worker:
            pid = 39172
            returncode = None
            waits = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                self.waits += 1
                if self.waits == 1:
                    raise subprocess.TimeoutExpired("worker", timeout)
                self.returncode = -signal.SIGKILL
                return self.returncode

        def kill_group(_pgid, sent):
            if sent == 0:
                raise ProcessLookupError()

        with (patch.object(executor, "process_identity",
                           return_value={"start_ticks": 456}),
              patch.object(executor.os, "getpgid", return_value=Worker.pid),
              patch.object(executor.os, "killpg", side_effect=kill_group) as sent):
            executor._stop_fenced_group(Worker(), 456)
        self.assertEqual([call.args[1] for call in sent.call_args_list],
                         [signal.SIGTERM, signal.SIGKILL, 0])

    def test_unknown_pid_or_surviving_group_cannot_publish_terminal(self) -> None:
        class Worker:
            pid = 39173

            def poll(self):
                return None

        with (patch.object(executor, "process_identity",
                           return_value={"start_ticks": 999}),
              patch.object(executor.os, "getpgid", return_value=Worker.pid),
              patch.object(executor.os, "killpg") as sent):
            with self.assertRaisesRegex(executor.WorkerNotStoppedError, "FENCE_DRIFT"):
                executor._stop_fenced_group(Worker(), 456)
            sent.assert_not_called()

        class Exited:
            pid = 39174

            def poll(self):
                return 0

        with (patch.object(executor.os, "killpg", return_value=None),
              patch.object(executor.time, "sleep", return_value=None)):
            with self.assertRaisesRegex(executor.WorkerNotStoppedError,
                                        "GROUP_STILL_LIVE"):
                executor._stop_fenced_group(Exited(), 456)


if __name__ == "__main__":
    unittest.main()
