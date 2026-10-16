"""No-model negative and idempotence tests for the T0-only submit v2 planner."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chaoyang.cli import exact78_submit_v2_candidate as submit
from chaoyang.governance import exact78_raw_two_source_executor_v1 as executor
from chaoyang.governance.common import artifact_ref
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


class SubmitV2CandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix="exact78-submit-v2-test-", dir=submit.REPO_ROOT / "_run/current")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module in (submit, executor):
            item = patch.object(module, "REPO_ROOT", self.root)
            item.start()
            self.addCleanup(item.stop)
        self.target = executor.Target(
            task_id="exact78_test_six_one", session=self.root / "raw/session_0",
            session_id="session_0", frame_count=2, fps_nominal=25,
            time_mode="TIMEBASE_UNVERIFIED", physical_eye="left_sourceIndex_1",
            receipt_name="TEST_SIX_ONE.json", execution_revision="TEST_SIX_ONE",
            objective="test", claim_limit="candidate", wall_seconds_max=10)
        self.script = self._write("src/fenced_wrapper.py", "# pinned wrapper\n")
        self.request = {"submission_input_sha256": "input-a"}
        self.expected = {"run_signature": "run-a"}
        self._write_json("tasks/current/INDEX.json", {"task_packets": []})
        self._write_json("docs/governance/LONG_HORIZON_TASK_STATE.json",
                         {"tasks": [], "next_task": None})
        item = patch.object(submit, "_expected_packet", return_value=(
            self.target, self.request, self.expected, self.script))
        item.start()
        self.addCleanup(item.stop)
        item = patch.object(submit, "_candidate_route_readiness", return_value=(True, "READY"))
        item.start()
        self.addCleanup(item.stop)

    def _write(self, relative: str, text: str) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _write_json(self, relative: str, value: dict) -> Path:
        return self._write(relative, json.dumps(value, sort_keys=True))

    def _writer(self) -> dict:
        pid = os.getpid()
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        ticks = int(stat[stat.rfind(")") + 2:].split()[19])
        return {"pid": pid, "proc_start_ticks": ticks, "executor_epoch": 1,
                "fencing_token": "test-token", "scope": self.target.task_id}

    def _packet(self, *, run_signature: str = "run-a") -> dict:
        return {"task_id": self.target.task_id, "execution_mode": "EXACT78_CANDIDATE",
                "run_signature": run_signature, "inputs": {"submission_input_sha256": "input-a"},
                "commands": [self.target.command],
                "write_set": [str(self.target.output.relative_to(self.root))],
                "writer": self._writer(), "runtime_code": [artifact_ref(self.script)]}

    def _set_active(self, *, status: str = "WAIT_GPU_RESOURCE", run_signature: str = "run-a",
                    stale_writer: bool = False) -> None:
        packet = self._packet(run_signature=run_signature)
        self.target.output.mkdir(parents=True)
        writer = dict(packet["writer"])
        if stale_writer:
            writer["proc_start_ticks"] += 1
            packet["writer"] = writer
        self._write_json(str((self.target.output / "WRITER.json").relative_to(self.root)), writer)
        self._write_json(str(self.target.packet.relative_to(self.root)), packet)
        packet_ref = artifact_ref(self.target.packet)
        self._write_json("tasks/current/INDEX.json", {"task_packets": [{
            "task_id": self.target.task_id,
            "packet_path": str(self.target.packet.relative_to(self.root)),
            "packet_sha256": packet_ref["sha256"], "execution_allowed": True}]})
        self._write_json("docs/governance/LONG_HORIZON_TASK_STATE.json", {
            "next_task": {"task_id": self.target.task_id},
            "tasks": [{"task_id": self.target.task_id, "session": self.target.session_id,
                       "status": status, "run_signature": run_signature,
                       "task_packet": packet_ref, "pid": writer["pid"],
                       "proc_start_ticks": writer["proc_start_ticks"]}]})

    def _set_terminal(self) -> None:
        packet = self._packet()
        self._write_json(str(self.target.packet.relative_to(self.root)), packet)
        packet_ref = artifact_ref(self.target.packet)
        self._write_json(str((self.target.output / "RESULT.json").relative_to(self.root)), {
            "task_id": self.target.task_id, "source_session": self.target.session_id,
            "status": "PASSED", "visual_qualified": False,
            "product_qualified": False, "training_eligible": False})
        result_ref = artifact_ref(self.target.output / "RESULT.json")
        self._write_json(str(self.target.receipt.relative_to(self.root)), {
            "task_id": self.target.task_id, "status": "PASSED", "result": result_ref})
        self._write_json("docs/governance/LONG_HORIZON_TASK_STATE.json", {
            "next_task": None,
            "tasks": [{"task_id": self.target.task_id, "session": self.target.session_id,
                       "status": "PASSED", "task_packet": packet_ref,
                       "result": result_ref, "terminal_receipt": artifact_ref(self.target.receipt)}]})

    def test_first_run_is_ready_but_never_dispatched(self) -> None:
        result = submit.plan_single(self.target.session)
        self.assertEqual(result["status"], "FIRST_RUN_READY_NOT_REGISTERED")
        self.assertTrue(result["requires_fenced_successor_packet"])
        self.assertFalse(self.target.output.exists())
        self.assertFalse(self.target.packet.exists())

    def test_waiting_gpu_same_full_signature_attaches(self) -> None:
        self._set_active()
        result = submit.plan_single(self.target.session)
        self.assertEqual(result["status"], "ATTACHED_TO_ACTIVE_CANDIDATE")
        self.assertEqual(result["task_state"], "WAIT_GPU_RESOURCE")
        self.assertFalse(result["execution_started_here"])

    def test_active_different_full_signature_rejects(self) -> None:
        self._set_active(run_signature="run-b")
        with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_PACKET_SIGNATURE_CONFLICT"):
            submit.plan_single(self.target.session)

    def test_stale_writer_rejects(self) -> None:
        self._set_active(stale_writer=True)
        with self.assertRaisesRegex(ColdStartInputError, "WRITER_NOT_LIVE"):
            submit.plan_single(self.target.session)

    def test_terminal_same_signature_returns_receipt(self) -> None:
        self._set_terminal()
        result = submit.plan_single(self.target.session)
        self.assertEqual(result["status"], "ATTACHED_TO_TERMINAL_CANDIDATE")
        self.assertFalse(result["visual_qualified"])
        self.assertEqual(result["receipt"]["sha256"], artifact_ref(self.target.receipt)["sha256"])

    def test_terminal_changed_full_signature_rejects(self) -> None:
        self._set_terminal()
        self.expected["run_signature"] = "run-b"
        with self.assertRaisesRegex(ColdStartInputError, "TERMINAL_PACKET_SIGNATURE_CONFLICT"):
            submit.plan_single(self.target.session)

    def test_terminal_lookup_does_not_spawn_when_another_task_is_active(self) -> None:
        self._set_terminal()
        state_path = self.root / "docs/governance/LONG_HORIZON_TASK_STATE.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["next_task"] = {"task_id": "exact78_other"}
        state["tasks"].append({"task_id": "exact78_other", "status": "RUNNING"})
        self._write_json("docs/governance/LONG_HORIZON_TASK_STATE.json", state)
        self._write_json("tasks/current/INDEX.json", {"task_packets": [{
            "task_id": "exact78_other", "execution_allowed": True}]})
        result = submit.plan_single(self.target.session)
        self.assertEqual(result["status"], "ATTACHED_TO_TERMINAL_CANDIDATE")
        self.assertFalse(result["execution_started_here"])

    def test_unsealed_attempt_is_not_reused(self) -> None:
        self.target.output.mkdir(parents=True)
        with self.assertRaisesRegex(ColdStartInputError, "UNSEALED_ATTEMPT_CONFLICT"):
            submit.plan_single(self.target.session)

    def test_multi_manifest_only_returns_batch_design(self) -> None:
        with patch.object(submit, "build_request", return_value={
                "manifest": {"sha256": "manifest-a"},
                "submission_input_sha256": "batch-a",
                "sources": [{"session_id": "play_cards_0902_042"},
                            {"session_id": "play_cards_0915_001"}]}), \
                patch.object(submit, "_validate_t0_target", return_value={}):
            result = submit.plan_manifest(self.root / "manifest.json")
        self.assertEqual(result["status"], "BATCH_REQUIRES_PER_SESSION_TASK_PACKETS")
        self.assertFalse(result["execution_started_here"])

    def test_manifest_non_t0_id_rejects(self) -> None:
        with patch.object(submit, "build_request", return_value={
                "sources": [{"session_id": "unfrozen_new_session"}]}):
            with self.assertRaisesRegex(ColdStartInputError, "T0_FROZEN_TARGET_NOT_AUTHORIZED"):
                submit.plan_manifest(self.root / "manifest.json")

    def test_non_t0_id_rejects_before_wrapper_import(self) -> None:
        with self.assertRaisesRegex(ColdStartInputError, "T0_FROZEN_TARGET_NOT_AUTHORIZED"):
            submit._wrapper_name("unknown_new_session")


if __name__ == "__main__":
    unittest.main()
