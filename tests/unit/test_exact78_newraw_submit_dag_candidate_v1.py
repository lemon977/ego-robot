"""No-ledger, no-model checks for the unwired v2 three-node entry."""

from __future__ import annotations

from pathlib import Path
from datetime import timedelta
import io
import json
import unittest
from unittest.mock import patch

from chaoyang.cli_v2 import main
from chaoyang.cli_v2 import exact78_raw_to_robot_candidate as dag
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


SESSION = Path("/raw/play_cards_0902_002")
EXPECTED_SHA = "9e9de7fe6cc21209af6d9c96c1f6df356855d26c86b1b9227930a5a854223f63"


def packet(phase: str = dag.PHASES[0]) -> dict:
    return {"task_id": "exact78_newraw_play_cards_0902_002_digest_v1",
            "phase": phase, "execution_mode": "EXACT78_CANDIDATE",
            "inputs": {"source_session": "play_cards_0902_002"},
            "runtime_code": [], "config": {}, "run_signature": "same-signature",
            "commands": ["one exact command"]}


def decision(status: str, node: str, **extra) -> dict:
    return {"status": status, "node": node, "task_id": node,
            "scene_clean": "NOT_WIRED", "general_operation": None, **extra}


class VersionedDagCandidateTests(unittest.TestCase):
    def test_missing_code_acceptance_receipt_blocks_execution_before_planning(self) -> None:
        with patch.object(dag.legacy_submit, "_raw_session_path",
                          return_value=dag.ACCEPTANCE_SOURCE_ROOT), \
             patch.object(dag, "load_json", side_effect=lambda path: (
                 {"tasks": []} if path == dag.TASK_STATE_PATH else {"task_packets": []})), \
             patch.object(dag, "plan_one") as plan, \
             patch.object(dag, "_launch") as launch:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V2_CODE_ACCEPTANCE_NOT_PASSED"):
                dag.submit_one(dag.ACCEPTANCE_SOURCE_ROOT, EXPECTED_SHA)
        plan.assert_not_called()
        launch.assert_not_called()

    def test_changed_accepted_cli_code_sha_blocks_execution(self) -> None:
        task = dag.ACCEPTANCE_TASK_ID
        def ref(path: Path) -> dict:
            return {"path": str(path), "bytes": 1, "sha256": "0" * 64}
        packet_ref, result_ref, receipt_ref = (
            ref(dag.ACCEPTANCE_PACKET), ref(dag.ACCEPTANCE_RESULT),
            ref(dag.ACCEPTANCE_RECEIPT))
        row = {"task_id": task, "phase": dag.ACCEPTANCE_PHASE,
               "session": dag.ACCEPTANCE_SESSION, "status": "PASSED",
               "run_signature": "sig", "task_packet": packet_ref,
               "result": result_ref, "terminal_receipt": receipt_ref}
        packet_value = {"task_id": task, "phase": dag.ACCEPTANCE_PHASE,
                        "inputs": {"source_session": dag.ACCEPTANCE_SESSION,
                                   "source_root": str(dag.ACCEPTANCE_SOURCE_ROOT),
                                   "source_submission_input_sha256": EXPECTED_SHA},
                        "run_signature": "sig"}
        result_value = {"schema_version": "exact78-newraw-cli-v2-acceptance-result-v1",
                        "task_id": task, "status": "PASSED",
                        "source_session": dag.ACCEPTANCE_SESSION,
                        "source_submission_input_sha256": EXPECTED_SHA,
                        "run_signature": "sig", "task_packet": packet_ref,
                        "cli_code": {name: ref(dag.REPO_ROOT / name)
                                     for name in dag.CLI_CODE}}
        result_value["test_code"] = ref(dag.REPO_ROOT / dag.TEST_CODE)
        result_value["published_operations"] = {}
        result_value["cli_check_only"] = {}
        packet_value["inputs"].update(
            cli_code=result_value["cli_code"], test_code=result_value["test_code"],
            published_operations=result_value["published_operations"],
            cli_check_only=result_value["cli_check_only"])
        receipt_value = {"schema_version": "exact78-newraw-cli-v2-acceptance-receipt-v1",
                         "task_id": task, "status": "PASSED",
                         "source_session": dag.ACCEPTANCE_SESSION,
                         "run_signature": "sig", "task_packet": packet_ref,
                         "result": result_ref}
        values = {dag.TASK_STATE_PATH: {"tasks": [row]},
                  dag.INDEX: {"task_packets": [{"task_id": task,
                     "execution_allowed": False,
                     "packet_path": str(dag.ACCEPTANCE_PACKET.relative_to(dag.REPO_ROOT)),
                     "packet_sha256": packet_ref["sha256"]}]},
                  dag.ACCEPTANCE_PACKET: packet_value,
                  dag.ACCEPTANCE_RESULT: result_value,
                  dag.ACCEPTANCE_RECEIPT: receipt_value}
        def ref_errors(expected: dict) -> list[str]:
            return ["STALE_SHA"] if expected.get("path") == str(
                dag.REPO_ROOT / dag.CLI_CODE[0]) else []
        with patch.object(dag, "load_json", side_effect=lambda path: values[path]), \
             patch.object(dag, "validate_artifact_ref", side_effect=ref_errors):
            with self.assertRaisesRegex(ColdStartInputError, "CLI_V2_ACCEPTED_CODE_REF"):
                dag._accepted_code_for_execution(dag.ACCEPTANCE_SOURCE_ROOT, EXPECTED_SHA)
        # The same synthetic seal passes once the accepted SHA and published
        # operation closure are self-consistent; this guards against a gate
        # that would reject every legitimate receipt.
        candidates = {
            "exact78_newraw_cpu_admission_candidate_v1":
                "run_exact78_newraw_cpu_admission_candidate_v1",
            "exact78_newraw_hawor_gpu_candidate_v4":
                "run_exact78_newraw_hawor_gpu_candidate_v4",
            "exact78_newraw_robot_cpu_candidate_v1":
                "run_exact78_newraw_robot_cpu_candidate_v1",
        }
        published = {key: {"candidate_operation": name,
                    "operation_ref": ref(dag.REPO_ROOT / "src/chaoyang/ops" / f"{name}.py")}
                     for key, name in candidates.items()}
        contracts = {key: {"candidate_operation": name,
                     "task_packet_required": True,
                     "current_runnable_operation": None,
                     "code_closure": [published[key]["operation_ref"]]}
                     for key, name in candidates.items()}
        result_value["published_operations"] = published
        result_value["cli_check_only"] = {"status": "READY", "node": "CPU_ADMISSION",
                                          "source_session": dag.ACCEPTANCE_SESSION,
                                          "stdout_sha256": "0" * 64}
        packet_value["inputs"]["published_operations"] = published
        packet_value["inputs"]["cli_check_only"] = result_value["cli_check_only"]
        values[dag.ALGORITHM_CONTRACT] = {"special_status_contracts": contracts}
        with patch.object(dag, "load_json", side_effect=lambda path: values[path]), \
             patch.object(dag, "validate_artifact_ref", return_value=[]):
            self.assertEqual(dag._accepted_code_for_execution(
                dag.ACCEPTANCE_SOURCE_ROOT, EXPECTED_SHA), result_ref)

    def test_prefrozen_input_sha_is_required_before_node_admission(self) -> None:
        with patch.object(dag.legacy_submit, "_raw_session_path", return_value=SESSION), \
             patch.object(dag.legacy_submit, "_frozen_t0_selected_paths", return_value=set()), \
             patch.object(dag, "load_json", return_value={"general_raw_to_robot": {
                 "current_runnable_operation": None}}), \
             patch.object(dag.cpu, "preflight", return_value={}), \
             patch.object(dag.cpu, "packet_value", return_value={"inputs": {
                 "submission_input_sha256": EXPECTED_SHA}}), \
             patch.object(dag, "_node") as node:
            with self.assertRaisesRegex(ColdStartInputError, "PREFROZEN_SUBMISSION"):
                dag.plan_one(SESSION, "0" * 64)
        node.assert_not_called()

    def test_h20_h24_and_full_stage_budget_are_fail_closed(self) -> None:
        self.assertIsNone(dag._budget("HAWOR_V4", dag.H20_STOP_HEAVY - timedelta(seconds=1)))
        self.assertEqual(dag._budget("HAWOR_V4", dag.H20_STOP_HEAVY)["reason"],
                         "H20_STOP_NEW_HEAVY_STAGES")
        self.assertEqual(dag._budget("CPU_ADMISSION", dag.H24_HARD_CUTOFF)["reason"],
                         "H24_HARD_CUTOFF")
        nearly_late = dag.H24_HARD_CUTOFF - timedelta(
            seconds=dag.STAGE_WALL_SECONDS["CPU_ADMISSION"] + 119)
        self.assertEqual(dag._budget("CPU_ADMISSION", nearly_late)["reason"],
                         "INSUFFICIENT_FULL_STAGE_WALL_BUDGET")

    def test_interrupt_waits_for_one_child_and_stops_followup_dispatch(self) -> None:
        plans = [decision("READY", "CPU_ADMISSION"),
                 decision("READY", "HAWOR_V4")]
        with patch.object(dag.legacy_submit, "_raw_session_path", return_value=SESSION), \
             patch.object(dag, "_accepted_code_for_execution", return_value={"sha256": "accepted"}), \
             patch.object(dag, "plan_one", side_effect=plans), \
             patch.object(dag, "_budget", return_value=None), \
             patch.object(dag, "_launch", return_value={"exit_code": 0,
                 "interrupted": True, "child_pid": 4321,
                 "child_start_ticks": 321}) as launch:
            result = dag.submit_one(SESSION, EXPECTED_SHA)
        self.assertEqual(result["status"], "INTERRUPTED_AFTER_FENCED_CHILD_TERMINAL")
        self.assertEqual(result["nodes_launched_here"], ["CPU_ADMISSION"])
        launch.assert_called_once()

    def test_launch_supervises_separate_session_without_wrapper_files(self) -> None:
        class FinishedChild:
            pid = 4321
            def wait(self, timeout=None):
                return 0
            def poll(self):
                return 0
        stream = io.StringIO()
        with patch.object(dag.subprocess, "Popen", return_value=FinishedChild()) as popen, \
             patch.object(dag, "process_identity", return_value={"start_ticks": 321}), \
             patch.object(dag.sys, "stderr", stream):
            result = dag._launch("CPU_ADMISSION", SESSION, None,
                                 "exact78_newraw_play_cards_0902_002_digest_v1")
            events = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("stdout", popen.call_args.kwargs)
        self.assertNotIn("stderr", popen.call_args.kwargs)
        self.assertNotIn("diagnostic_dir", result)
        self.assertEqual([event["event"] for event in events],
                         ["PREPARED", "CHILD_STARTED", "CHILD_TERMINAL"])
        self.assertEqual(events[1]["pid"], 4321)
        self.assertEqual(events[1]["proc_start_ticks"], 321)

    def test_manifest_is_explicitly_rejected_without_planning(self) -> None:
        with patch("chaoyang.cli_v2.plan_one") as plan:
            with self.assertRaises(SystemExit) as error:
                main(["submit", "exact78-raw-to-robot", "--manifest", "/manifest.json",
                      "--expected-input-sha256", EXPECTED_SHA])
        self.assertEqual(error.exception.code, 2)
        plan.assert_not_called()

    def test_check_only_never_launches_a_node(self) -> None:
        with patch("chaoyang.cli_v2.plan_one", return_value=decision("READY", "CPU_ADMISSION")), \
             patch("chaoyang.cli_v2.submit_one") as submit:
            self.assertEqual(main(["submit", "exact78-raw-to-robot", "--session",
                                   str(SESSION), "--check-only",
                                   "--expected-input-sha256", EXPECTED_SHA]), 0)
        submit.assert_not_called()

    def test_conflict_returns_nonzero_without_launching(self) -> None:
        with patch("chaoyang.cli_v2.plan_one", side_effect=ColdStartInputError(
                "SAME_SESSION_DIFFERENT_SIGNATURE_REQUIRES_SUCCESSOR")), \
             patch("chaoyang.cli_v2.submit_one") as submit:
            self.assertEqual(main(["submit", "exact78-raw-to-robot", "--session",
                                   str(SESSION), "--check-only",
                                   "--expected-input-sha256", EXPECTED_SHA]), 2)
        submit.assert_not_called()

    def test_only_exact_cpu_phase_matches_same_session(self) -> None:
        state = {"next_task": None, "tasks": [
            {"task_id": "exact78_newraw_hawor_play_cards_0902_002_digest_v4",
             "phase": dag.PHASES[1], "session": "play_cards_0902_002", "status": "PASSED"},
            {"task_id": "exact78_newraw_robot_play_cards_0902_002_digest_v1",
             "phase": dag.PHASES[2], "session": "play_cards_0902_002", "status": "PASSED"}]}
        with patch.object(dag, "load_json", side_effect=lambda path: (
                state if path == dag.TASK_STATE_PATH else {"task_packets": []})), \
             patch.object(dag, "REPO_ROOT", Path("/no-such-exact78-dag-test")):
            self.assertEqual(dag._node(packet(), dag.PHASES[0], dag.SCHEMAS[0])["status"],
                             "READY")

    def test_same_session_different_cpu_signature_is_rejected(self) -> None:
        state = {"next_task": None, "tasks": [{"task_id": "other_cpu_digest",
            "phase": dag.PHASES[0], "session": "play_cards_0902_002", "status": "PASSED"}]}
        with patch.object(dag, "load_json", side_effect=lambda path: (
                state if path == dag.TASK_STATE_PATH else {"task_packets": []})):
            with self.assertRaisesRegex(ColdStartInputError, "DIFFERENT_SIGNATURE"):
                dag._node(packet(), dag.PHASES[0], dag.SCHEMAS[0])

    def test_unrelated_active_writer_is_rejected(self) -> None:
        state = {"next_task": {"task_id": "other"}, "tasks": [{
            "task_id": "other", "phase": dag.PHASES[1],
            "session": "different_session", "status": "RUNNING"}]}
        index = {"task_packets": [{"task_id": "other", "execution_allowed": True}]}
        with patch.object(dag, "load_json", side_effect=lambda path: (
                state if path == dag.TASK_STATE_PATH else index)):
            with self.assertRaisesRegex(ColdStartInputError, "ACTIVE_DIFFERENT_WRITER"):
                dag._node(packet(), dag.PHASES[0], dag.SCHEMAS[0])

    def test_unsealed_partial_is_not_admitted(self) -> None:
        state = {"next_task": None, "tasks": [{"task_id": packet()["task_id"],
            "phase": dag.PHASES[0], "session": "play_cards_0902_002",
            "status": "RUNNING", "run_signature": "same-signature",
            "task_packet": {"path": "p"}}]}
        with patch.object(dag, "load_json", side_effect=lambda path: (
                state if path == dag.TASK_STATE_PATH else
                packet() if path.name == "TASK_PACKET.json" else {"task_packets": []})), \
             patch.object(dag, "_ref", return_value={"path": "p"}):
            with self.assertRaisesRegex(ColdStartInputError, "NONTERMINAL_WITHOUT_LIVE_WRITER"):
                dag._node(packet(), dag.PHASES[0], dag.SCHEMAS[0])

    def test_terminal_same_signature_requires_result_receipt_and_phase(self) -> None:
        expected = packet()
        result = {"schema_version": dag.SCHEMAS[0], "task_id": expected["task_id"],
                  "status": "PASSED", "source_session": "play_cards_0902_002",
                  "run_signature": expected["run_signature"]}
        row = {"task_id": expected["task_id"], "phase": dag.PHASES[0],
               "session": "play_cards_0902_002", "status": "PASSED",
               "run_signature": expected["run_signature"],
               "task_packet": {"path": "packet"},
               "result": {"path": "result"}, "terminal_receipt": {"path": "receipt"}}
        state = {"next_task": None, "tasks": [row]}
        def read(path: Path) -> dict:
            if path == dag.TASK_STATE_PATH:
                return state
            if path == dag.INDEX:
                return {"task_packets": []}
            if path.name == "TASK_PACKET.json":
                return expected
            if path.name == "RESULT.json":
                return result
            return {"schema_version": dag.RECEIPT_SCHEMAS[0],
                    "task_id": expected["task_id"], "status": "PASSED",
                    "result": row["result"]}
        with patch.object(dag, "load_json", side_effect=read), \
             patch.object(dag, "_ref", side_effect=lambda path, ref, label: ref):
            attached = dag._node(expected, dag.PHASES[0], dag.SCHEMAS[0])
            self.assertEqual(attached["status"], "ATTACHED_TERMINAL")
            result["run_signature"] = "different"
            with self.assertRaisesRegex(ColdStartInputError, "CPU_TERMINAL_SIGNATURE_DRIFT"):
                dag._node(expected, dag.PHASES[0], dag.SCHEMAS[0])

    def test_gpu_wait_zero_attempts_stops_without_robot(self) -> None:
        with patch.object(dag.legacy_submit, "_raw_session_path", return_value=SESSION), \
             patch.object(dag, "_accepted_code_for_execution", return_value={"sha256": "accepted"}), \
             patch.object(dag, "plan_one", return_value=decision(
                 "STOP_TERMINAL", "HAWOR_V4", terminal_status="BLOCKED_RESOURCE")), \
             patch.object(dag, "_launch") as launch:
            result = dag.submit_one(SESSION, EXPECTED_SHA)
        self.assertEqual(result["nodes_launched_here"], [])
        launch.assert_not_called()

    def test_one_submit_launches_only_one_writer_at_a_time(self) -> None:
        plans = iter((decision("READY", "CPU_ADMISSION"),
                      decision("READY", "HAWOR_V4"),
                      decision("READY", "HAWOR_V4"),
                      decision("READY", "ROBOT_V1"),
                      decision("READY", "ROBOT_V1"),
                      decision("CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS",
                               "ROBOT_V1"),
                      decision("CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS",
                               "ROBOT_V1")))
        events: list[str] = []
        def launch(node: str, session: Path, upstream: Path | None, task_id: str) -> dict:
            events.append(node)
            return {"exit_code": 0, "interrupted": False,
                    "child_pid": 123, "child_start_ticks": 321}
        with patch.object(dag.legacy_submit, "_raw_session_path", return_value=SESSION), \
             patch.object(dag, "_accepted_code_for_execution", return_value={"sha256": "accepted"}), \
             patch.object(dag, "plan_one", side_effect=lambda session, sha: next(plans)), \
             patch.object(dag, "_launch", side_effect=launch), \
             patch.object(dag, "_budget", return_value=None), \
             patch.object(dag.cpu, "preflight", return_value={}), \
             patch.object(dag.cpu, "packet_value", return_value={"task_id": "cpu"}), \
             patch.object(dag.cpu, "_paths", return_value=(Path("/sealed/cpu"), None, None)), \
             patch.object(dag.hawor, "packet_value", return_value={"task_id": "hawor"}), \
             patch.object(dag.hawor, "_paths", return_value=(Path("/sealed/hawor"), None, None)):
            result = dag.submit_one(SESSION, EXPECTED_SHA)
        self.assertEqual(events, ["CPU_ADMISSION", "HAWOR_V4", "ROBOT_V1"])
        self.assertEqual(result["nodes_launched_here"], events)

    def test_failed_child_without_terminal_seal_is_not_success(self) -> None:
        with patch.object(dag.legacy_submit, "_raw_session_path", return_value=SESSION), \
             patch.object(dag, "_accepted_code_for_execution", return_value={"sha256": "accepted"}), \
             patch.object(dag, "plan_one", side_effect=[decision("READY", "CPU_ADMISSION"),
                                                        decision("READY", "CPU_ADMISSION")]), \
             patch.object(dag, "_launch", return_value={"exit_code": 2,
                "interrupted": False, "child_pid": 123, "child_start_ticks": 321}), \
             patch.object(dag, "_budget", return_value=None):
            with self.assertRaisesRegex(ColdStartInputError, "WITHOUT_TERMINAL"):
                dag.submit_one(SESSION, EXPECTED_SHA)


if __name__ == "__main__":
    unittest.main()
