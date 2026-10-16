"""Synthetic, no-ledger tests for one-session submit/CLI CPU admission wiring."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.cli import exact78_newraw_cpu_submit_candidate as submit
from chaoyang.governance import common
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as producer
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError, file_ref
from chaoyang.governance.exact78_operation_gate import Exact78GateDenied
from chaoyang.ops import run_exact78_newraw_cpu_admission_candidate_v1 as operation
from tests.unit import test_exact78_first_submit_cpu_producer_candidate as _producer_tests


class NewrawCpuSubmitCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _producer_tests.FirstSubmitCpuProducerCandidateTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.prepared = self.fixture._prepared()
        self.repo = self.fixture.root / "repo"
        self.index = self.repo / "tasks/current/INDEX.json"
        self.state = self.repo / "docs/governance/LONG_HORIZON_TASK_STATE.json"
        self._write(self.index, {"task_packets": []})
        self._write(self.state, {"tasks": [], "next_task": None})

    def _write(self, path: Path, value: str | dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) if isinstance(value, dict) else value,
                        encoding="utf-8")

    def _paths(self, task_id: str) -> tuple[Path, Path, Path]:
        return (self.repo / "_run/current" / task_id / "attempts/attempt_0001",
                self.repo / "tasks/current" / task_id / "TASK_PACKET.json",
                self.repo / "tasks/receipts" / f"{task_id.upper()}.json")

    def _plan(self) -> tuple[dict, dict]:
        with patch.object(submit, "INDEX", self.index), patch.object(submit, "STATE", self.state), \
                patch.object(submit, "REPO_ROOT", self.repo), \
                patch.object(producer, "_paths", side_effect=self._paths), \
                patch.object(producer, "preflight", return_value=self.prepared):
            return submit.plan_one(self.fixture.session, allowed_root=self.fixture.data)

    def _active(self, packet: dict) -> None:
        attempt, packet_path, _ = self._paths(packet["task_id"])
        self._write(packet_path, packet)
        self._write(attempt / "WRITER.json", packet["writer"])
        packet_ref = file_ref(packet_path)
        self._write(self.index, {"task_packets": [{"task_id": packet["task_id"],
            "packet_path": str(packet_path.relative_to(self.repo)),
            "packet_sha256": packet_ref["sha256"], "execution_allowed": True}]})
        self._write(self.state, {"tasks": [{"task_id": packet["task_id"],
            "session": packet["inputs"]["source_session"], "status": "RUNNING",
            "pid": packet["writer"]["pid"],
            "proc_start_ticks": packet["writer"]["proc_start_ticks"],
            "run_signature": packet["run_signature"], "task_packet": packet_ref}],
            "next_task": {"task_id": packet["task_id"]}})

    def test_first_ready_is_read_only(self) -> None:
        decision, packet = self._plan()
        self.assertEqual(decision["status"], "FIRST_RUN_READY_NOT_REGISTERED")
        self.assertEqual(packet["inputs"]["descriptor_sha256"],
                         self.prepared["descriptor_sha256"])
        self.assertFalse(self._paths(packet["task_id"])[0].exists())

    def test_multiple_old_same_session_phases_do_not_block_first_cpu_admission(self) -> None:
        session = self.fixture.session.name
        self._write(self.state, {"tasks": [
            {"task_id": "exact78_raw_to_robot_t2_0915_cards011_v1_20260930",
             "phase": "EXACT78_RAW_TO_ROBOT_T2_TWO_SOURCE", "session": session,
             "status": "PASSED"},
            {"task_id": "exact78_cards011_fresh_robot_v1_20260930",
             "phase": "EXACT78_FRESH_ROBOT", "session": session,
             "status": "FAILED_QUALITY_FINAL"},
            {"task_id": "exact78_cards011_scene_review_v1_20260930",
             "phase": "EXACT78_VISUAL_SCENE", "session": session,
             "status": "PASSED"},
        ], "next_task": None})
        decision, _ = self._plan()
        self.assertEqual(decision["status"], "FIRST_RUN_READY_NOT_REGISTERED")

    def test_old_phases_do_not_hide_same_domain_different_signature(self) -> None:
        session = self.fixture.session.name
        self._write(self.state, {"tasks": [
            {"task_id": "exact78_raw_to_robot_t2_0915_cards011_v1_20260930",
             "phase": "EXACT78_RAW_TO_ROBOT_T2_TWO_SOURCE", "session": session,
             "status": "PASSED"},
            {"task_id": "exact78_newraw_play_cards_0915_011_otherdigest_v1",
             "phase": submit.CPU_ADMISSION_PHASE, "session": session,
             "status": "PASSED"},
        ], "next_task": None})
        with self.assertRaisesRegex(ColdStartInputError,
                                    "SESSION_PRIOR_DIFFERENT_SIGNATURE_REQUIRES_SUCCESSOR"):
            self._plan()

    def test_same_signature_attaches_live_writer(self) -> None:
        _, packet = self._plan()
        self._active(packet)
        decision, _ = self._plan()
        self.assertEqual(decision["status"], "ATTACHED_TO_ACTIVE_CPU_ADMISSION")
        self.assertFalse(decision["execution_started_here"])

    def test_active_different_signature_or_writer_rejects(self) -> None:
        _, packet = self._plan()
        different = copy.deepcopy(packet)
        different["run_signature"] = "different-signature"
        self._active(different)
        with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_DIFFERENT_SIGNATURE"):
            self._plan()
        different["inputs"]["source_root"] = "/different/raw/session"
        self._active(different)
        with patch.object(producer, "preflight") as preflight:
            with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_DIFFERENT_WRITER"):
                self._plan()
        preflight.assert_not_called()

    def test_stale_writer_rejects_attachment(self) -> None:
        _, packet = self._plan()
        self._active(packet)
        attempt, _, _ = self._paths(packet["task_id"])
        writer = json.loads((attempt / "WRITER.json").read_text())
        writer["fencing_token"] = "wrong"
        self._write(attempt / "WRITER.json", writer)
        with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_WRITER_FENCE_DRIFT"):
            self._plan()

    def test_terminal_same_signature_attaches_without_rerun(self) -> None:
        _, packet = self._plan()
        attempt, packet_path, receipt_path = self._paths(packet["task_id"])
        self._write(packet_path, packet)
        self._write(attempt / "RESULT.json", {"task_id": packet["task_id"],
            "status": "PASSED", "run_signature": packet["run_signature"]})
        result_ref = file_ref(attempt / "RESULT.json")
        self._write(receipt_path, {"task_id": packet["task_id"], "status": "PASSED",
                                   "result": result_ref})
        self._write(self.state, {"tasks": [{"task_id": packet["task_id"],
            "session": packet["inputs"]["source_session"], "status": "PASSED",
            "task_packet": file_ref(packet_path), "result": result_ref,
            "terminal_receipt": file_ref(receipt_path)}], "next_task": None})
        decision, _ = self._plan()
        self.assertEqual(decision["status"], "ATTACHED_TO_TERMINAL_CPU_ADMISSION")
        self.assertFalse(decision["execution_started_here"])

    def test_submit_preview_does_not_register_or_run(self) -> None:
        with patch.object(submit, "plan_one", return_value=self._plan()), \
                patch.object(producer, "register") as register, \
                patch.object(producer, "run_registered_child") as worker:
            decision = submit.submit_one(self.fixture.session, preview=True)
        self.assertEqual(decision["status"], "FIRST_RUN_READY_NOT_REGISTERED")
        register.assert_not_called()
        worker.assert_not_called()

    def test_future_submit_registers_once_then_uses_gated_child(self) -> None:
        ready = self._plan()
        terminal = {"status": "PASSED", "task_id": ready[1]["task_id"]}
        with patch.object(submit, "plan_one", return_value=ready), \
                patch.object(producer, "register") as register, \
                patch.object(producer, "run_registered_child", return_value=terminal) as worker:
            result = submit.submit_one(self.fixture.session)
        self.assertEqual(result, terminal)
        register.assert_called_once_with(ready[1])
        worker.assert_called_once_with(ready[1])

    def test_operation_gate_precedes_producer_import_or_raw_read(self) -> None:
        with patch("chaoyang.governance.exact78_operation_gate.authorize_exact78_operation",
                   side_effect=Exact78GateDenied("blocked")), \
                patch.object(producer, "preflight") as preflight:
            with self.assertRaises(Exact78GateDenied):
                operation.main(["--session", str(self.fixture.session),
                                "--output-root", str(self.repo / "attempt")])
        preflight.assert_not_called()

    def test_operation_consumes_only_bound_source_after_gate(self) -> None:
        _, packet = self._plan()
        attempt, packet_path, _ = self._paths(packet["task_id"])
        attempt.mkdir(parents=True)
        packet["writer"]["pid"] = os.getpid() + 100000
        self._write(packet_path, packet)
        self._write(self.index, {"task_packets": [{"task_id": packet["task_id"],
            "execution_allowed": True}]})
        refs = {name: {"path": str(attempt / name), "bytes": 1, "sha256": "0" * 64}
                for name in ("descriptor", "frame_time_map", "seal")}
        with patch("chaoyang.governance.exact78_operation_gate.authorize_exact78_operation"), \
                patch.object(common, "REPO_ROOT", self.repo), \
                patch.object(producer, "_paths", side_effect=self._paths), \
                patch.object(producer, "preflight", return_value=self.prepared), \
                patch.object(producer, "produce_admission", return_value=refs) as produce:
            self.assertEqual(operation.main(["--session", str(self.fixture.session),
                "--output-root", str(attempt)]), 0)
        produce.assert_called_once()
        self.assertTrue(produce.call_args.kwargs["cli_worker"])
        self.assertEqual(json.loads((attempt / "ADMISSION_STAGE.json").read_text())["admission"], refs)

    def test_operation_rejects_source_drift_before_sealing(self) -> None:
        _, packet = self._plan()
        attempt, packet_path, _ = self._paths(packet["task_id"])
        attempt.mkdir(parents=True)
        packet["writer"]["pid"] = os.getpid() + 100000
        self._write(packet_path, packet)
        self._write(self.index, {"task_packets": [{"task_id": packet["task_id"],
            "execution_allowed": True}]})
        changed = copy.deepcopy(self.prepared)
        changed["descriptor_sha256"] = "0" * 64
        with patch("chaoyang.governance.exact78_operation_gate.authorize_exact78_operation"), \
                patch.object(common, "REPO_ROOT", self.repo), \
                patch.object(producer, "_paths", side_effect=self._paths), \
                patch.object(producer, "preflight", return_value=changed), \
                patch.object(producer, "produce_admission") as produce:
            with self.assertRaisesRegex(RuntimeError, "CPU_ADMISSION_WORKER_SOURCE_DRIFT"):
                operation.main(["--session", str(self.fixture.session),
                                "--output-root", str(attempt)])
        produce.assert_not_called()
        self.assertFalse((attempt / "ADMISSION_STAGE.json").exists())

    def test_registered_writer_launches_only_cli_worker_with_project_local_cache(self) -> None:
        _, packet = self._plan()
        attempt, _, _ = self._paths(packet["task_id"])
        attempt.mkdir(parents=True)
        refs = {name: {"path": str(attempt / name), "bytes": 1, "sha256": "0" * 64}
                for name in ("descriptor", "frame_time_map", "seal")}
        self._write(attempt / "ADMISSION_STAGE.json", {
            "schema_version": "exact78-first-submit-cpu-admission-stage-v1",
            "task_id": packet["task_id"],
            "source_session": packet["inputs"]["source_session"],
            "descriptor_sha256": packet["inputs"]["descriptor_sha256"],
            "status": "SEALED_CPU_ADMISSION_STAGE", "admission": refs})

        class DoneChild:
            returncode = 0

            def poll(self) -> int:
                return 0

        with patch.object(producer, "REPO_ROOT", self.repo), \
                patch.object(producer, "_paths", side_effect=self._paths), \
                patch.object(producer, "_fence"), \
                patch.object(producer, "validate_artifact_ref", return_value=[]), \
                patch.object(producer, "finish", return_value={"status": "PASSED"}) as finish, \
                patch.object(producer.subprocess, "Popen", return_value=DoneChild()) as popen:
            result = producer.run_registered_child(packet)
        self.assertEqual(result["status"], "PASSED")
        command = popen.call_args.args[0]
        self.assertEqual(command[2:6], ["-m", "chaoyang.cli", "run", producer.OPERATION])
        environment = popen.call_args.kwargs["env"]
        self.assertEqual(environment["TMPDIR"], str(attempt / "cache"))
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        finish.assert_called_once_with(packet, refs, error=None)


if __name__ == "__main__":
    unittest.main()
