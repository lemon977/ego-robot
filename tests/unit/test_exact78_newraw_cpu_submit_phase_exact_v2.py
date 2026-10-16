"""Model-free phase-exact CPU admission and v3 successor contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import io

from chaoyang.cli_v2 import exact78_raw_to_robot_candidate as prior
from chaoyang.cli_v3 import exact78_newraw_cpu_submit_phase_exact_v2 as adapter
from chaoyang.cli_v3 import exact78_raw_to_robot_candidate_v2 as successor
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


ROOT = prior.REPO_ROOT
FIXTURE = ROOT / "tests/fixtures/exact78_cards002_cli_v2_ledger_prefix_trap_v1.json"
SESSION = prior.ACCEPTANCE_SOURCE_ROOT
INPUT_SHA = prior.ACCEPTANCE_INPUT_SHA


def cpu_packet() -> dict:
    return {"task_id": "exact78_newraw_play_cards_0902_002_50b73360d82cd4a7_v1",
            "phase": prior.PHASES[0], "execution_mode": "EXACT78_CANDIDATE",
            "inputs": {"source_session": SESSION.name, "source_root": str(SESSION),
                       "submission_input_sha256": INPUT_SHA, "descriptor_sha256": "x"},
            "runtime_code": [], "config": {}, "run_signature": "cpu-signature",
            "commands": ["one exact command"]}


class PhaseExactCpuSubmissionTests(unittest.TestCase):
    def _adapter(self, state: dict, index: dict, *, root: Path | None = None):
        target = root or Path("/no-such-exact78-v3-test")
        return (patch.object(prior.legacy_submit, "_raw_session_path", return_value=SESSION),
                patch.object(prior.legacy_submit, "_frozen_t0_selected_paths", return_value=set()),
                patch.object(adapter.producer, "preflight", return_value={}),
                patch.object(adapter.producer, "packet_value", return_value=cpu_packet()),
                patch.object(adapter.producer, "_assert_cli_admission_ready", return_value="PUBLISHED"),
                patch.object(prior, "REPO_ROOT", target),
                patch.object(prior, "load_json", side_effect=lambda path: (
                    state if path == prior.TASK_STATE_PATH else index)))

    def test_real_ledger_acceptance_row_and_downstream_rows_are_not_cpu(self) -> None:
        frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(frozen["captured_governance_revision"], 16420)
        self.assertEqual(frozen["row"]["session"], SESSION.name)
        rows = [frozen["row"],
                {"task_id": successor.TASK_ID, "phase": successor.PHASE,
                 "session": SESSION.name, "status": "PASSED"},
                {"task_id": "exact78_newraw_hawor_play_cards_0902_002_v4",
                 "phase": prior.PHASES[1], "session": SESSION.name, "status": "PASSED"},
                {"task_id": "exact78_newraw_robot_play_cards_0902_002_v1",
                 "phase": prior.PHASES[2], "session": SESSION.name, "status": "PASSED"}]
        state, index = {"next_task": None, "tasks": rows}, {"task_packets": []}
        with ExitStack() as stack:
            for item in self._adapter(state, index):
                stack.enter_context(item)
            decision, packet = adapter.plan_one(SESSION, INPUT_SHA)
        self.assertEqual(decision["status"], "READY")
        self.assertEqual(packet["phase"], prior.PHASES[0])

    def test_same_cpu_phase_different_signature_is_rejected(self) -> None:
        state = {"next_task": None, "tasks": [{"task_id": "different-cpu-id",
                 "phase": prior.PHASES[0], "session": SESSION.name, "status": "PASSED"}]}
        index = {"task_packets": []}
        with ExitStack() as stack:
            for item in self._adapter(state, index):
                stack.enter_context(item)
            with self.assertRaisesRegex(ColdStartInputError, "DIFFERENT_SIGNATURE"):
                adapter.plan_one(SESSION, INPUT_SHA)

    def test_different_session_active_writer_is_rejected(self) -> None:
        state = {"next_task": {"task_id": "other"}, "tasks": [{"task_id": "other",
                 "phase": prior.PHASES[1], "session": "elsewhere", "status": "RUNNING"}]}
        index = {"task_packets": [{"task_id": "other", "execution_allowed": True}]}
        with ExitStack() as stack:
            for item in self._adapter(state, index):
                stack.enter_context(item)
            with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_DIFFERENT_WRITER"):
                adapter.plan_one(SESSION, INPUT_SHA)

    def test_unsealed_partial_cpu_row_is_rejected(self) -> None:
        packet = cpu_packet()
        state = {"next_task": None, "tasks": [{"task_id": packet["task_id"],
                 "phase": prior.PHASES[0], "session": SESSION.name, "status": "RUNNING",
                 "run_signature": packet["run_signature"], "task_packet": {"path": "p"}}]}
        index = {"task_packets": []}
        with ExitStack() as stack:
            for item in self._adapter(state, index):
                stack.enter_context(item)
            stack.enter_context(patch.object(prior, "_ref", return_value={"path": "p"}))
            stack.enter_context(patch.object(prior, "load_json", side_effect=lambda path: (
                state if path == prior.TASK_STATE_PATH else
                packet if path.name == "TASK_PACKET.json" else index)))
            with self.assertRaisesRegex(ColdStartInputError, "NONTERMINAL_WITHOUT_LIVE_WRITER"):
                adapter.plan_one(SESSION, INPUT_SHA)

    def test_same_signature_terminal_attaches_only_with_sealed_refs(self) -> None:
        packet = cpu_packet()
        synthetic_root = Path("/no-such-exact78-v3-test")
        task = packet["task_id"]
        def ref(path: Path) -> dict:
            return {"path": str(path), "bytes": 1, "sha256": "0" * 64}
        packet_ref = ref(synthetic_root / "tasks/current" / task / "TASK_PACKET.json")
        result_ref = ref(synthetic_root / "_run/current" / task /
                         "attempts/attempt_0001/RESULT.json")
        receipt_ref = ref(synthetic_root / "tasks/receipts" / f"{task.upper()}.json")
        row = {"task_id": packet["task_id"], "phase": prior.PHASES[0],
               "session": SESSION.name, "status": "PASSED",
               "run_signature": packet["run_signature"], "task_packet": packet_ref,
               "result": result_ref, "terminal_receipt": receipt_ref}
        state, index = {"next_task": None, "tasks": [row]}, {"task_packets": []}
        result = {"schema_version": prior.SCHEMAS[0], "task_id": packet["task_id"],
                  "status": "PASSED", "source_session": SESSION.name,
                  "run_signature": packet["run_signature"]}
        receipt = {"schema_version": prior.RECEIPT_SCHEMAS[0],
                   "task_id": packet["task_id"], "status": "PASSED",
                   "result": row["result"]}
        def read(path: Path) -> dict:
            if path == prior.TASK_STATE_PATH:
                return state
            if path == prior.INDEX:
                return index
            if path.name == "TASK_PACKET.json":
                return packet
            if path.name == "RESULT.json":
                return result
            return receipt
        with ExitStack() as stack:
            for item in self._adapter(state, index):
                stack.enter_context(item)
            stack.enter_context(patch.object(prior, "load_json", side_effect=read))
            stack.enter_context(patch.object(prior, "validate_artifact_ref", return_value=[]))
            decision, _ = adapter.plan_one(SESSION, INPUT_SHA)
        self.assertEqual(decision["status"], "ATTACHED_TERMINAL")

    def test_direct_cpu_execution_requires_v3_receipt_before_register(self) -> None:
        packet = cpu_packet()
        with patch.object(adapter, "plan_one", return_value=({"status": "READY"}, packet)), \
             patch.object(successor, "accepted_code_for_execution",
                          side_effect=ColdStartInputError("NOT_SEALED")), \
             patch.object(adapter.producer, "register") as register:
            with self.assertRaisesRegex(ColdStartInputError, "NOT_SEALED"):
                adapter.submit_one(SESSION, INPUT_SHA)
        register.assert_not_called()

    def test_v3_dag_without_v2_receipt_stops_before_planning_or_dispatch(self) -> None:
        with patch.object(prior.legacy_submit, "_raw_session_path", return_value=SESSION), \
             patch.object(successor, "load_json", side_effect=lambda path: (
                 {"tasks": []} if path == successor.TASK_STATE_PATH else
                 {"task_packets": []})), \
             patch.object(successor, "plan_one") as plan, \
             patch.object(successor, "_launch_cpu") as launch:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V3_CODE_ACCEPTANCE_NOT_PASSED"):
                successor.submit_one(SESSION, INPUT_SHA)
        plan.assert_not_called()
        launch.assert_not_called()

    def test_v3_cpu_launcher_inherits_output_and_writes_no_wrapper_files(self) -> None:
        class FinishedChild:
            pid = 4321
            def wait(self, timeout=None):
                return 0
            def poll(self):
                return 0
        stream = io.StringIO()
        with patch.object(successor.subprocess, "Popen", return_value=FinishedChild()) as popen, \
             patch.object(successor, "process_identity", return_value={"start_ticks": 321}), \
             patch.object(successor.sys, "stderr", stream):
            result = successor._launch_cpu(SESSION, INPUT_SHA, cpu_packet()["task_id"])
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("stdout", popen.call_args.kwargs)
        self.assertNotIn("stderr", popen.call_args.kwargs)
        self.assertIn("CPU_CHILD_STARTED", stream.getvalue())

    def test_cpu_writes_only_through_original_fenced_producer(self) -> None:
        packet = cpu_packet()
        with patch.object(adapter, "plan_one", return_value=({"status": "READY"}, packet)), \
             patch.object(successor, "accepted_code_for_execution", return_value={}), \
             patch.object(prior, "_budget", return_value=None), \
             patch.object(adapter.producer, "register") as register, \
             patch.object(adapter.producer, "run_registered_child",
                          return_value={"status": "PASSED"}) as run:
            self.assertEqual(adapter.submit_one(SESSION, INPUT_SHA)["status"], "PASSED")
        register.assert_called_once_with(packet)
        run.assert_called_once_with(packet)


if __name__ == "__main__":
    unittest.main()
