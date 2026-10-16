"""CPU-only packet/operation negatives; never register or render."""

from __future__ import annotations

import io
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_robot_render_direction_cards042_v1 as task
from chaoyang.governance.common import artifact_ref
from chaoyang.governance.exact78_operation_gate import _command_tail
from chaoyang.ops import run_exact78_robot_render_direction_v1 as operation


class RenderDirectionPacketTests(unittest.TestCase):
    @staticmethod
    def route_contract() -> dict:
        closure = [artifact_ref(task.REPO_ROOT / relative) for relative in (
            "src/chaoyang/ops/run_exact78_robot_render_direction_v1.py",
            "src/chaoyang/pipeline/exact78_robot_render_direction_v1.py",
            *task.render_direction.PINNED_CODE_SHA)]
        return {"special_status_contracts": {task.CONTRACT_KEY: {
            "candidate_operation": task.OPERATION, "task_packet_required": True,
            "current_runnable_operation": None,
            "output_schema": "exact78-robot-render-direction-only-v1",
            "model_calls_authorized": 0, "training_steps_authorized": 0,
            "code_closure": closure}}}

    def test_full_cards042_spec_and_unique_new_write_set(self):
        spec = task.frozen_spec()
        self.assertEqual((spec["task_id"], spec["session_id"]),
                         (task.TASK, "play_cards_0902_042"))
        self.assertEqual(spec["output_root"], str(task.OUT / "candidate"))
        self.assertNotIn("render", spec["inputs"])
        self.assertNotIn("clean", spec["inputs"])
        self.assertEqual(_command_tail(task.COMMAND),
                         [task.OPERATION, "--spec", str(task.SPEC)])
        packet = task.packet_value(task.virtual_json_ref(task.SPEC, spec), spec)
        self.assertEqual(packet["commands"], [task.COMMAND])
        self.assertEqual(packet["write_set"][0], str(task.OUT.relative_to(task.REPO_ROOT)))
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["config"]["ik_calls"], 0)
        self.assertEqual(packet["config"]["model_calls"], 0)
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertNotIn("GPU_LEASE", str(packet["write_set"]))
        self.assertNotIn("0916", str(packet["write_set"]))
        self.assertTrue(all("old" not in name.lower() for name in packet["required_outputs"]))
        self.assertIn("candidate/render/ego_robot_diagnostic.mp4", packet["required_outputs"])
        self.assertIn(str(task.REPO_ROOT / "src/chaoyang/governance/current_r3_contracts.py"),
                      [ref["path"] for ref in packet["runtime_code"]])

    def test_active_task_prevents_registration(self):
        with patch.object(task, "load_json", return_value={"tasks": [{"status": "RUNNING"}]}):
            self.assertEqual(task.readiness(), (False, "ACTIVE_TASK_CONFLICT"))

    def test_exact_staged_builder_route_can_publish_with_first_packet(self):
        contract = self.route_contract()
        with patch("chaoyang.cli._maintained_operations", return_value=set()), \
             patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                   return_value=contract) as build, \
             patch.object(task, "load_json", return_value={}):
            self.assertEqual(task._candidate_route_readiness(),
                (True, "READY_TO_PUBLISH_CANDIDATE_ROUTE_WITH_PACKET"))
        build.assert_called_once()

    def test_published_route_must_have_exact_closure(self):
        contract = self.route_contract()
        with patch("chaoyang.cli._maintained_operations", return_value={task.OPERATION}), \
             patch.object(task, "load_json", return_value=contract):
            self.assertEqual(task._candidate_route_readiness(),
                             (True, "PUBLISHED_CANDIDATE_ROUTE"))
        contract["special_status_contracts"][task.CONTRACT_KEY]["code_closure"].pop()
        with patch("chaoyang.cli._maintained_operations", return_value={task.OPERATION}), \
             patch.object(task, "load_json", return_value=contract):
            self.assertEqual(task._candidate_route_readiness(),
                             (False, "RENDER_DIRECTION_PUBLISHED_ROUTE_DRIFT"))

    def test_unstaged_route_is_not_silently_published(self):
        contract = self.route_contract()
        contract["special_status_contracts"][task.CONTRACT_KEY]["task_packet_required"] = False
        with patch("chaoyang.cli._maintained_operations", return_value=set()), \
             patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                   return_value=contract), \
             patch.object(task, "load_json", return_value={}):
            self.assertEqual(task._candidate_route_readiness(),
                             (False, "RENDER_DIRECTION_CANDIDATE_ROUTE_NOT_STAGED"))

    def test_check_only_never_registers_or_renders(self):
        with patch.object(task, "frozen_spec", return_value={}), \
             patch.object(task, "virtual_json_ref", return_value={"sha256": "x"}), \
             patch.object(task, "packet_value", return_value={"run_signature": "y", "write_set": ["new"]}), \
             patch.object(task, "readiness", return_value=(False, "ACTIVE_TASK_CONFLICT")), \
             patch.object(task, "register") as register, \
             patch.object(task, "run_worker") as run_worker, \
             patch("sys.argv", ["task", "--check-only"]), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            self.assertEqual(task.main(), 0)
        register.assert_not_called()
        run_worker.assert_not_called()
        self.assertIn("DRY_RUN_NOT_ROUTABLE", output.getvalue())

    def test_execute_rejected_without_new_route(self):
        with patch.object(task, "frozen_spec", return_value={}), \
             patch.object(task, "virtual_json_ref", return_value={}), \
             patch.object(task, "packet_value", return_value={}), \
             patch.object(task, "readiness", return_value=(False, "RENDER_DIRECTION_CANDIDATE_ROUTE_NOT_STAGED")), \
             patch.object(task, "register") as register, \
             patch("sys.argv", ["task", "--execute"]):
            with self.assertRaisesRegex(RuntimeError, "RENDER_DIRECTION_CANDIDATE_ROUTE_NOT_STAGED"):
                task.main()
        register.assert_not_called()

    def test_operation_without_active_packet_denied_before_render(self):
        with patch.object(operation, "load_json", return_value={"task_packets": []}), \
             patch.object(operation, "run_render_only") as render:
            with self.assertRaisesRegex(RuntimeError, "RENDER_DIRECTION_NOT_ROUTABLE"):
                operation.authorize(Path("absent-spec.json"))
        render.assert_not_called()

    def test_operation_packet_sha_drift_denied_before_render(self):
        index = {"task_packets": [{"task_id": task.TASK, "execution_allowed": True,
                                   "packet_path": str(task.PACKET.relative_to(task.REPO_ROOT)),
                                   "packet_sha256": "a" * 64}]}
        with patch.object(operation, "load_json", return_value=index), \
             patch.object(operation, "artifact_ref", return_value={"sha256": "b" * 64}), \
             patch.object(operation, "run_render_only") as render:
            with self.assertRaisesRegex(RuntimeError, "RENDER_DIRECTION_PACKET_SHA_DRIFT"):
                operation.authorize(task.SPEC)
        render.assert_not_called()

    def test_runtime_pin_drift_rejected(self):
        wrong = dict(task.render_direction.PINNED_CODE_SHA)
        wrong["src/chaoyang/pipeline/robot_scene_state_cpu.py"] = "0" * 64
        with patch.object(task.render_direction, "PINNED_CODE_SHA", wrong):
            with self.assertRaisesRegex(RuntimeError, "RENDER_DIRECTION_RUNTIME_CODE_DRIFT"):
                task.runtime_refs()

    def test_missing_candidate_does_not_fake_success(self):
        with patch.object(task, "OUT", task.REPO_ROOT / "_run/current/nonexistent_render_direction_test"):
            result, errors = task.verify_candidate({"inputs": {}, "session_id": "play_cards_0902_042"})
        self.assertIsNone(result)
        self.assertEqual(errors, ["CANDIDATE_RESULT_MISSING"])


if __name__ == "__main__":
    unittest.main()
