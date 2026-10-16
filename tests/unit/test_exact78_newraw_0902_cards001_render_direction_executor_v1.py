"""CPU-only admission and fail-closed checks; never register or render."""

from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest import mock

from chaoyang.governance import execute_exact78_newraw_0902_cards001_render_direction_v1 as runner
from chaoyang.governance import prepare_exact78_newraw_0902_cards001_render_direction_v1 as prep
from chaoyang.ops import run_exact78_newraw_0902_cards001_render_direction_v1 as operation


class Cards001DirectionExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec, _, _ = prep.frozen_spec()

    def test_packet_exact_single_change_and_30_minute_budget(self) -> None:
        packet = runner.packet_value(self.spec)
        self.assertEqual(packet["task_id"], prep.TASK_ID)
        self.assertEqual(packet["commands"], [runner.COMMAND])
        self.assertEqual(packet["config"]["change_only"],
                         "EGO_CAMERA_INV_T_CAMERA_BASE_TO_T_CAMERA_BASE")
        for field in ("flange_parent_changed", "q_base_mount_recomputed",
                      "numeric_gates_changed"):
            self.assertIs(packet["config"][field], False)
        self.assertEqual(packet["budgets"], {
            "workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0,
            "training_steps": 0, "wall_seconds_max": 1800,
            "session_attempts_max": 1})
        self.assertEqual(packet["write_set"][0], str(prep.OUT.relative_to(prep.REPO_ROOT)))
        self.assertEqual(packet["required_outputs"][1:3],
                         ["WORKER_PROCESS.json", "WORKER_GROUP_CLEANUP.json"])
        self.assertEqual(set(packet["required_outputs"][4:7]),
                         {f"candidate/render/{name}" for name in prep.VIDEO_FILES.values()})
        self.assertEqual(packet["inputs"]["sealed_stage"]["sha256"], prep.STAGE_SHA)
        self.assertEqual(packet["inputs"]["fixed_inputs"]["states"]["sha256"], prep.STATES_SHA)
        self.assertEqual(packet["inputs"]["fixed_inputs"]["hawor_npz"]["sha256"], prep.HAWOR_SHA)
        self.assertEqual(packet["inputs"]["fixed_inputs"]["raw_video"]["sha256"], prep.RAW_SHA)
        self.assertEqual(packet["inputs"]["robot_asset_pin"]["sha256"], prep.ASSET_PIN_SHA)

    def test_runtime_closure_pins_dedicated_op_executor_and_renderer(self) -> None:
        refs = runner.runtime_refs()
        paths = {Path(ref["path"]) for ref in refs}
        self.assertIn(runner.SCRIPT, paths)
        self.assertIn(prep.REPO_ROOT / "src/chaoyang/ops" / f"{runner.OPERATION}.py", paths)
        self.assertIn(Path(prep.__file__), paths)
        for relative in runner.direction.PINNED_CODE_SHA:
            self.assertIn(prep.REPO_ROOT / relative, paths)
        self.assertNotIn(prep.REPO_ROOT / "src/chaoyang/governance/execute_exact78_robot_render_direction_cards042_v1.py", paths)

    def test_synthetic_active_packet_is_busy_and_check_only_writes_nothing(self) -> None:
        before = tuple(path.exists() for path in (prep.OUT, prep.PACKET, prep.RECEIPT))
        real_load = runner.load_json

        def busy_state(path: Path):
            if path == runner.TASK_STATE_PATH:
                return {"next_task": {"task_id": "another_task"}, "tasks": []}
            if path == runner.INDEX:
                return {"task_packets": []}
            return real_load(path)

        with mock.patch.object(runner, "load_json", side_effect=busy_state), \
             mock.patch.object(runner, "route_readiness", return_value=(True, "STAGED")):
            ready, reason = runner.readiness()
        self.assertFalse(ready)
        self.assertEqual(reason, "ACTIVE_TASK_CONFLICT")
        with mock.patch.object(sys, "argv", ["executor", "--check-only"]), \
             mock.patch.object(runner, "readiness", return_value=(False, "ACTIVE_TASK_CONFLICT")), \
             mock.patch.object(runner, "register", side_effect=AssertionError("REGISTER_CALLED")), \
             mock.patch.object(runner, "run_worker", side_effect=AssertionError("WORKER_CALLED")):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(tuple(path.exists() for path in (prep.OUT, prep.PACKET,
                                                          prep.RECEIPT)), before)

    def test_route_contract_requires_exact_new_operation_and_closure(self) -> None:
        refs = [runner.artifact_ref(prep.REPO_ROOT / relative) for relative in (
            f"src/chaoyang/ops/{runner.OPERATION}.py",
            "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
            *runner.direction.PINNED_CODE_SHA)]
        route = {"candidate_operation": runner.OPERATION,
                 "task_packet_required": True, "current_runnable_operation": None,
                 "output_schema": "exact78-robot-render-direction-only-v1",
                 "model_calls_authorized": 0, "training_steps_authorized": 0,
                 "code_closure": refs}
        contract = {"special_status_contracts": {runner.CONTRACT_KEY: route}}
        self.assertTrue(runner._route_valid(contract))
        route["candidate_operation"] = "run_exact78_robot_render_direction_v1"
        self.assertFalse(runner._route_valid(contract))

    def test_not_staged_and_published_route_drift_fail_closed(self) -> None:
        with mock.patch("chaoyang.cli._maintained_operations", return_value=set()), \
             mock.patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                        return_value={"special_status_contracts": {}}):
            self.assertEqual(runner.route_readiness(), (False, "CANDIDATE_ROUTE_NOT_STAGED"))
        with mock.patch("chaoyang.cli._maintained_operations", return_value={runner.OPERATION}), \
             mock.patch.object(runner, "load_json", return_value={"special_status_contracts": {}}):
            self.assertEqual(runner.route_readiness(), (False, "PUBLISHED_ROUTE_DRIFT"))

    def test_hard_cutoff_caps_deadline_and_rejects_late_packet(self) -> None:
        stamp = prep.CAMPAIGN_END - timedelta(seconds=100)
        packet = runner.packet_value(self.spec, stamp=stamp)
        self.assertEqual(packet["deadline_at"], prep.CAMPAIGN_END.isoformat())
        with self.assertRaisesRegex(RuntimeError, "CAMPAIGN_HARD_CUTOFF_PASSED"):
            runner.packet_value(self.spec, stamp=prep.CAMPAIGN_END)

    def test_missing_candidate_is_not_success(self) -> None:
        candidate, errors = runner.verify_candidate(self.spec)
        self.assertIsNone(candidate)
        self.assertEqual(errors, ["CANDIDATE_RESULT_MISSING"])

    def test_external_zero_pixels_fails_execution_before_video_acceptance(self) -> None:
        result = {"status": "COMPLETE_RENDER_DIAGNOSTIC_NOT_VISUAL_PASS",
                  "session_id": prep.SESSION_ID, "frame_count": prep.FRAMES,
                  "inputs": self.spec["inputs"],
                  "q_base_mount_recomputed": False, "numeric_gates_changed": False,
                  "ego_camera_rule": "T_CAMERA_BASE_BASE_TO_CAMERA_NO_INVERSION",
                  "external_camera_rule": "PINNED_FIXED_EXTERNAL_FROM_BASE_TO_CAMERA",
                  "clean_or_depth_consumed": False, "visual_qualified": False,
                  "product_qualified": False, "training_eligible": False,
                  "control_ground_truth": False,
                  "frame_rows": [{"frame": i, "raw_decoded_display_index": i,
                                  "geometry_decision": "UNKNOWN",
                                  "display_policy": "DIAGNOSTIC_ROBOT_OVER_RAW_NO_OCCLUSION_PROOF",
                                  "ego_robot_pixels": 1, "external_robot_pixels": 0,
                                  "final_visible": True} for i in range(prep.FRAMES)]}
        real_is_file = Path.is_file
        result_path = prep.OUT / "candidate/RESULT.json"
        with mock.patch.object(Path, "is_file", lambda path: path == result_path or real_is_file(path)), \
             mock.patch.object(runner, "load_json", return_value=result):
            candidate, errors = runner.verify_candidate(self.spec)
        self.assertIsNone(candidate)
        self.assertEqual(errors, ["CANDIDATE_VALIDATION:EXTERNAL_ROBOT_ZERO_PIXELS:0"])

    def test_deliberate_worker_timeout_clears_child_encoder_group(self) -> None:
        # Simulates a Python renderer with an inherited-process-group encoder.
        script = ("import subprocess,time; "
                  "subprocess.Popen(['/usr/bin/sleep','20']); time.sleep(20)")
        worker = subprocess.Popen([sys.executable, "-B", "-c", script],
                                  start_new_session=True, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        ticks = runner.process_identity(worker.pid)["start_ticks"]
        try:
            deadline = time.monotonic() + 3
            while len(runner._group_members(worker.pid)) < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertGreaterEqual(len(runner._group_members(worker.pid)), 2)
            receipt = runner._settle_group(worker, ticks, term_seconds=0.5,
                                           kill_seconds=0.5)
            self.assertEqual(receipt["live_members_after_cleanup"], 0)
            self.assertIn("SIGTERM", receipt["signals"])
            self.assertEqual(runner._group_members(worker.pid), [])
        finally:
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait(timeout=2)

    def test_unproven_group_cannot_terminalize_or_release_writer(self) -> None:
        with mock.patch.object(sys, "argv", ["executor", "--execute"]), \
             mock.patch.object(runner, "readiness", return_value=(True, "STAGED")), \
             mock.patch.object(runner, "register"), \
             mock.patch.object(runner, "run_worker",
                               side_effect=runner.WorkerGroupUncleared("ENCODER_STILL_LIVE")), \
             mock.patch.object(runner, "atomic_json") as record, \
             mock.patch.object(runner, "publish_bundle") as publish, \
             mock.patch.object(runner, "finish",
                               side_effect=AssertionError("WRITER_RELEASED")):
            with self.assertRaisesRegex(runner.WorkerGroupUncleared,
                                        "ENCODER_STILL_LIVE"):
                runner.main()
        self.assertEqual(record.call_args.args[0], prep.OUT / "WORKER_GROUP_HOLD.json")
        self.assertTrue(record.call_args.args[1]["writer_remains_active"])
        publish.assert_not_called()

    def test_finish_requires_actual_group_cleanup_proof(self) -> None:
        with mock.patch.object(runner, "verify_fence"):
            with self.assertRaisesRegex(runner.WorkerGroupUncleared,
                                        "WORKER_GROUP_CLEANUP_PROOF_ABSENT"):
                runner.finish({}, self.spec, 0, prep.OUT / "worker.log")

    def test_execute_rejects_busy_without_registering(self) -> None:
        with mock.patch.object(sys, "argv", ["executor", "--execute"]), \
             mock.patch.object(runner, "readiness", return_value=(False, "ACTIVE_TASK_CONFLICT")), \
             mock.patch.object(runner, "register", side_effect=AssertionError("REGISTER_CALLED")):
            with self.assertRaisesRegex(RuntimeError, "ACTIVE_TASK_CONFLICT"):
                runner.main()

    def test_direct_operation_rejects_wrong_or_absent_spec(self) -> None:
        with self.assertRaises(SystemExit):
            operation.run_candidate(Path("/dev/null"))

    def test_direct_import_call_is_not_a_writer_descendant(self) -> None:
        with self.assertRaisesRegex(SystemExit, "EXACT78_GATE_DENIED"):
            operation.run_candidate(prep.SPEC)

    def test_main_gate_blocks_before_algorithm_import_or_render(self) -> None:
        with mock.patch.object(operation, "authorize_exact78_operation",
                               side_effect=RuntimeError("DENIED")), \
             mock.patch.object(operation, "run_candidate",
                               side_effect=AssertionError("RENDER_REACHED")), \
             mock.patch.object(sys, "argv", ["operation", "--spec", str(prep.SPEC)]):
            with self.assertRaisesRegex(RuntimeError, "DENIED"):
                operation.main()


if __name__ == "__main__":
    unittest.main()
