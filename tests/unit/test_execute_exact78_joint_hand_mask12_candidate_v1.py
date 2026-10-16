"""Negative CPU tests for the unregistered finite Mask12 GPU executor."""

from __future__ import annotations

from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from chaoyang.governance import exact78_joint_hand_mask12_candidate_v1 as design
from chaoyang.governance import execute_exact78_joint_hand_mask12_candidate_v1 as executor
from chaoyang.governance.common import (
    REPO_ROOT, TASK_STATE_PATH, load_json, process_identity, validate_task_state,
)


class Mask12ExecutorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = design.packet_value()

    def test_executor_is_in_signed_code_closure(self) -> None:
        paths = {row["path"] for row in self.packet["runtime_code"]}
        self.assertIn(str(executor.__file__), paths)
        self.assertEqual(self.packet["budgets"]["wall_seconds_max"], 2700)
        self.assertEqual(self.packet["budgets"]["model_calls_max"], 42)

    def test_check_only_is_read_only_even_with_active_task(self) -> None:
        output = io.StringIO()
        with patch.object(design, "packet_value", return_value=self.packet), \
             patch.object(executor, "readiness", side_effect=RuntimeError("MASK12_ACTIVE_TASK_CONFLICT")), \
             patch.object(executor, "register") as register, redirect_stdout(output):
            self.assertEqual(executor.main(["--check-only"]), 0)
            register.assert_not_called()
        value = json.loads(output.getvalue())
        self.assertFalse(value["registered"])
        self.assertFalse(value["gpu_called"])
        self.assertFalse(value["old_mask_clean_read"])
        self.assertEqual(value["anchor_count"], 12)
        self.assertEqual(value["model_calls_max"], 42)
        self.assertEqual(value["status"], "MASK12_ACTIVE_TASK_CONFLICT")

    def test_registration_cannot_bypass_readiness(self) -> None:
        with patch.object(executor, "readiness", side_effect=RuntimeError("MASK12_ACTIVE_TASK_CONFLICT")), \
             patch.object(executor, "publish_bundle") as publish:
            with self.assertRaisesRegex(RuntimeError, "MASK12_ACTIVE_TASK_CONFLICT"):
                executor.register(self.packet)
            publish.assert_not_called()

    def test_route_rejects_unstaged_cli_before_registration(self) -> None:
        from chaoyang import cli

        with patch.object(cli, "_maintained_operations", return_value=set()):
            with self.assertRaisesRegex(RuntimeError, "MASK12_CLI_OPERATION_NOT_STAGED"):
                executor.route_readiness(self.packet)

    def test_route_rejects_wrong_staged_code_closure(self) -> None:
        from chaoyang import cli
        from chaoyang.governance import current_r3_contracts

        staged = {"special_status_contracts": {design.ROUTE_KEY: {"code_closure": []}}}
        with patch.object(cli, "_maintained_operations", return_value={design.OPERATION}), \
             patch.object(current_r3_contracts, "build_algorithm_contract", return_value=staged):
            with self.assertRaisesRegex(RuntimeError, "MASK12_ROUTE_NOT_STAGED_OR_CODE_SHA_DRIFT"):
                executor.route_readiness(self.packet)

    def test_fence_rejects_wrong_writer_before_touching_current(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["writer"]["pid"] = -1
        with self.assertRaisesRegex(RuntimeError, "MASK12_WRITER_OR_PACKET_FENCE_DRIFT"):
            executor.verify_fence(packet)

    def test_lease_command_is_one_exact_gpu_candidate(self) -> None:
        command = executor.worker_command(self.packet, 300, 2300)
        self.assertEqual(command.count("run_gpu_command_with_v71_lease"), 1)
        self.assertEqual(command.count(design.OPERATION), 1)
        self.assertEqual(command[command.index("--wall-seconds") + 1], "2300")
        self.assertEqual(command[-2:], ["--output-root", str(executor.ATTEMPT / "candidate")])
        self.assertNotIn("ProPainter", " ".join(command))

    def test_one_prompt_delta_for_tracker_and_vst(self) -> None:
        for scene in self.packet["inputs"]["sessions"]:
            prompts = executor.expected_prompts(scene, self.packet["config"])
            self.assertEqual(prompts["human"], design.NEW_PROMPT)
            self.assertEqual(prompts["object"], self.packet["config"]["other_role_prompts"][
                "poker_object" if scene["task"] == "poker" else "chips_object"])
            self.assertEqual(set(prompts), ({"human", "object", "equipment", "finger_sleeve", "yellow_cable"}
                                            if scene["source_family"] == "VST_0915"
                                            else {"human", "object", "equipment"}))

    def test_object_protection_accepts_only_exact_mask_algebra(self) -> None:
        base = np.zeros((2, 2), dtype=bool)
        human, object_mask = base.copy(), base.copy()
        human[1, 1], object_mask[0, 0] = True, True
        images = {name: base.copy() for name in executor.MASK_ROLES}
        images.update(human=human, object=object_mask, human_equipment=human.copy(),
                      removal=human.copy(), protected=object_mask.copy())
        self.assertEqual(executor.check_mask_algebra(images, "TRACKER_0902", "session", 0), 1)
        images["removal"][0, 0] = True
        with self.assertRaisesRegex(RuntimeError, "MASK12_PROTECTION_ALGEBRA_FAIL"):
            executor.check_mask_algebra(images, "TRACKER_0902", "session", 0)

    def test_vst_sleeve_joins_human_equipment_without_changing_object(self) -> None:
        base = np.zeros((2, 2), dtype=bool)
        sleeve, object_mask = base.copy(), base.copy()
        sleeve[1, 0], object_mask[0, 1] = True, True
        images = {name: base.copy() for name in executor.MASK_ROLES |
                  {"finger_sleeve", "yellow_cable"}}
        images.update(finger_sleeve=sleeve, object=object_mask,
                      human_equipment=sleeve.copy(), removal=sleeve.copy(),
                      protected=object_mask.copy())
        self.assertEqual(executor.check_mask_algebra(images, "VST_0915", "session", 75), 1)

    def test_proc_startticks_and_process_group_are_read_from_kernel(self) -> None:
        observed = executor._proc_evidence(os.getpid())
        self.assertIsNotNone(observed)
        self.assertEqual(observed.start_ticks, process_identity(os.getpid())["start_ticks"])
        self.assertEqual(observed.process_group, os.getpgrp())

    def test_signal_refuses_reused_group_leader(self) -> None:
        old = executor.ProcessGroupIdentity(12345, 100, 1, 12345, 12345, "S")
        reused = executor.ProcessGroupIdentity(12345, 200, 1, 12345, 12345, "S")
        with patch.object(executor, "_group_members", return_value=[reused]), \
             patch.object(executor.os, "killpg") as kill:
            with self.assertRaisesRegex(executor.CleanupUnproven, "PROCESS_GROUP_IDENTITY_DRIFT"):
                executor._signal_exact_group(old, signal.SIGTERM)
            kill.assert_not_called()

    def test_gpu_child_state_change_does_not_look_like_pid_reuse(self) -> None:
        wrapper = executor.ProcessGroupIdentity(12345, 100, 1, 12345, 12345, "S")
        known = executor.ProcessGroupIdentity(12346, 101, 12345, 12346, 12346, "R")
        same_child_now_sleeping = executor.ProcessGroupIdentity(12346, 101, 12345, 12346, 12346, "S")
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "_run/maintenance") as name:
            lease_path = Path(name) / "GPU_LEASE.json"
            lease_path.write_text(json.dumps({
                "schema_version": "chaoyang-gpu-lease-v71", "status": "ACQUIRED",
                "task_id": design.TASK_ID, "attempt_id": executor.GPU_ATTEMPT_ID,
                "pid": wrapper.pid, "process_startticks": wrapper.start_ticks,
                "gpu_process_pid": known.pid, "fencing_token": "synthetic",
            }), encoding="utf-8")
            with patch.object(executor, "GPU_LEASE", lease_path), \
                 patch.object(executor, "_proc_evidence", return_value=same_child_now_sleeping):
                observed, lease = executor._capture_owned_gpu_child(wrapper, known)
        self.assertEqual(observed.start_ticks, known.start_ticks)
        self.assertEqual(observed.state, "S")
        self.assertEqual(lease["fencing_token"], "synthetic")

    def test_real_cpu_child_gets_its_own_group_and_term_is_bounded(self) -> None:
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                       "TMPDIR": str(REPO_ROOT / "_run/maintenance")}
        process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                                   start_new_session=True, env=environment,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            identity = executor._proc_evidence(process.pid)
            self.assertIsNotNone(identity)
            self.assertEqual(identity.parent_pid, os.getpid())
            self.assertEqual(identity.process_group, process.pid)
            self.assertEqual(identity.session_id, process.pid)
            self.assertTrue(executor._signal_exact_group(identity, signal.SIGTERM))
            self.assertTrue(executor._wait_group_empty(identity, 2))
            process.wait(timeout=2)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2)

    def test_deadline_cleanup_terms_then_kills_both_exact_groups(self) -> None:
        wrapper = executor.ProcessGroupIdentity(12345, 100, 1, 12345, 12345, "S")
        child = executor.ProcessGroupIdentity(12346, 101, 12345, 12346, 12346, "S")
        process = MagicMock()
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "_run/maintenance") as name, \
             patch.object(executor, "ATTEMPT", Path(name)), \
             patch.object(executor, "GPU_LEASE", Path(name) / "absent-lease.json"), \
             patch.object(executor, "_capture_owned_gpu_child", return_value=(child, None)), \
             patch.object(executor, "_signal_exact_group", return_value=True) as send, \
             patch.object(executor, "_wait_group_empty", side_effect=[False, True, False, True]), \
             patch.object(executor, "_group_members", return_value=[]), \
             patch.object(executor, "_lease_released_or_safely_release", return_value=True):
            executor._cleanup_deadline(process, wrapper, child, None)
            audit = json.loads((Path(name) / "DEADLINE_CLEANUP_AUDIT.json").read_text())
        self.assertEqual(audit["status"], "PROVEN")
        self.assertEqual([(item["pid"], item["signal"]) for item in audit["signals"]],
                         [(child.pid, "TERM"), (child.pid, "KILL"),
                          (wrapper.pid, "TERM"), (wrapper.pid, "KILL")])
        self.assertEqual(send.call_count, 4)
        process.wait.assert_called_once_with(timeout=2)

    def test_unproven_lease_writes_failed_audit_and_holds_next_packet(self) -> None:
        wrapper = executor.ProcessGroupIdentity(12345, 100, 1, 12345, 12345, "S")
        process = MagicMock()
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "_run/maintenance") as name, \
             patch.object(executor, "ATTEMPT", Path(name)), \
             patch.object(executor, "GPU_LEASE", Path(name) / "absent-lease.json"), \
             patch.object(executor, "_capture_owned_gpu_child", return_value=(None, None)), \
             patch.object(executor, "_signal_exact_group", return_value=True), \
             patch.object(executor, "_wait_group_empty", return_value=True), \
             patch.object(executor, "_group_members", return_value=[]), \
             patch.object(executor, "_lease_released_or_safely_release", return_value=False):
            with self.assertRaisesRegex(executor.CleanupUnproven, "GPU_OR_LEASE_RELEASE_UNPROVEN"):
                executor._cleanup_deadline(process, wrapper, None, None)
            audit = json.loads((Path(name) / "DEADLINE_CLEANUP_AUDIT.json").read_text())
        self.assertEqual(audit["status"], "UNPROVEN")
        self.assertFalse(audit["new_task_packet_allowed"])
        state = {"blockers": [], "next_task": None}
        executor.apply_unproven_cleanup_hold(state)
        self.assertEqual(state["next_task"]["task_id"], design.TASK_ID)
        self.assertEqual(state["blockers"][0]["scope"], "ALL_NEW_TASK_PACKETS")
        governed = copy.deepcopy(load_json(TASK_STATE_PATH))
        governed["tasks"].append({
            "task_id": design.TASK_ID, "phase": self.packet["phase"],
            "attempt": 1, "status": "FAILED_RUNTIME_RETRYABLE",
            "updated_at": governed["generated_at"],
        })
        executor.apply_unproven_cleanup_hold(governed)
        self.assertEqual(validate_task_state(governed), [])

    def test_no_owned_lease_needs_no_gpu_probe(self) -> None:
        from chaoyang.governance import gpu_lease_v71

        wrapper = executor.ProcessGroupIdentity(12345, 100, 1, 12345, 12345, "S")
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "_run/maintenance") as name, \
             patch.object(executor, "ATTEMPT", Path(name)), \
             patch.object(executor, "GPU_LEASE", Path(name) / "absent-lease.json"), \
             patch.object(executor, "_group_members", return_value=[]), \
             patch.object(gpu_lease_v71, "query_gpu_pids") as query:
            self.assertFalse(executor._lease_released_or_safely_release(wrapper, None, None))
            (Path(name) / "GPU_RECEIPT.json").write_text(json.dumps({
                "schema_version": "v71-gpu-command-receipt-v1",
                "status": "BLOCKED_RESOURCE", "task_id": design.TASK_ID,
                "attempt_id": executor.GPU_ATTEMPT_ID,
            }), encoding="utf-8")
            self.assertTrue(executor._lease_released_or_safely_release(wrapper, None, None))
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
