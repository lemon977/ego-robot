"""No-registration tests for the two-session manifest-gated CLI-v4 candidate."""

from __future__ import annotations

import copy
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang import cli_v4
from chaoyang.cli_v4 import exact78_newraw_cpu_submit_manifest_v1 as adapter
from chaoyang.cli_v4 import exact78_raw_to_robot_manifest_candidate_v1 as candidate
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError


class ManifestAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest, self.ref = candidate.frozen_manifest()
        self.chips = next(row for row in self.manifest["sessions"]
                          if row["session_id"] == "get_potato_chips_0901_001")
        self.cards = next(row for row in self.manifest["sessions"]
                          if row["session_id"] == "play_cards_0902_002")

    def test_exact_two_source_whitelist_and_reserved_vst_are_separate(self) -> None:
        self.assertEqual(len(self.manifest["sessions"]), 2)
        self.assertEqual((self.chips["frame_count_decoded"], self.chips["fps_nominal"]),
                         (799, 25))
        self.assertEqual((self.cards["frame_count_decoded"], self.cards["fps_nominal"]),
                         (189, 30))
        vst = self.manifest["source_domains"]["0915_raw_vst"]
        self.assertEqual(vst["physical_eye"], "left_sourceIndex_1")
        self.assertEqual(vst["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(vst["execution_status"], "SCHEMA_RESERVED_NO_SESSION_ADMITTED")
        self.assertEqual(vst["glove_motion_input"], "FORBIDDEN")

    def test_manifest_sha_drift_rejects_before_planning(self) -> None:
        with patch.object(candidate, "MANIFEST_SHA", "0" * 64), \
             patch.object(candidate.prior, "plan_one") as plan:
            with self.assertRaisesRegex(ColdStartInputError, "CLI_V4_MANIFEST_SHA_DRIFT"):
                candidate.plan_one(Path(self.chips["source_root"]),
                                   self.chips["submission_input_sha256"])
        plan.assert_not_called()

    def test_nonmember_or_wrong_sha_rejects_before_old_planner(self) -> None:
        with patch.object(candidate.prior.legacy_submit, "_raw_session_path",
                          side_effect=lambda value: Path(value)), \
             patch.object(candidate.prior, "plan_one") as plan:
            for source, digest in ((Path(self.chips["source_root"]), "0" * 64),
                                   (Path("/path/to/private_data/egodata/other/unknown"), "0" * 64)):
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ColdStartInputError,
                                                "CLI_V4_SESSION_NOT_IN_FROZEN_MANIFEST"):
                        candidate.plan_one(source, digest)
        plan.assert_not_called()

    def test_member_delegates_to_phase_dag_with_manifest_ref(self) -> None:
        source = Path(self.chips["source_root"])
        with patch.object(candidate.prior.legacy_submit, "_raw_session_path", return_value=source), \
             patch.object(candidate.prior, "plan_one", return_value={"status": "READY",
                                                                 "node": "CPU_ADMISSION"}) as plan:
            value = candidate.plan_one(source, self.chips["submission_input_sha256"])
        plan.assert_called_once_with(source, self.chips["submission_input_sha256"])
        self.assertEqual(value["candidate_cli"], "chaoyang.cli_v4")
        self.assertEqual(value["manifest"], self.ref)

    def test_cards002_attaches_sealed_terminal_without_old_packet_rebuild(self) -> None:
        source = Path(self.cards["source_root"])
        with patch.object(candidate.prior, "plan_one") as plan, \
             patch.object(candidate.prior.cpu, "packet_value") as packet_value:
            value = candidate.plan_one(source, self.cards["submission_input_sha256"])
        plan.assert_not_called()
        packet_value.assert_not_called()
        self.assertEqual(value["node"], "ROBOT_V1")
        self.assertEqual(value["status"],
                         "CANDIDATE_ROBOT_EXECUTION_COMPLETE_NOT_VISUAL_PASS")
        self.assertTrue(value["sealed_attachment"])
        self.assertFalse(value["visual_qualified"])

    def test_cards002_manifest_terminal_sha_drift_rejects(self) -> None:
        altered = copy.deepcopy(self.cards)
        altered["sealed_robot_terminal"]["result"]["sha256"] = "0" * 64
        with patch.object(candidate, "_entry", return_value=(altered, self.ref)), \
             patch.object(candidate.prior, "plan_one") as plan:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V4_FROZEN_ROBOT_TRIPLE_DRIFT"):
                candidate.plan_one(Path(altered["source_root"]),
                                   altered["submission_input_sha256"])
        plan.assert_not_called()

    def test_cards002_hawor_chain_drift_rejects(self) -> None:
        original = candidate._sealed_stage

        def changed_stage(task_id, *args, **kwargs):
            packet, result, refs = original(task_id, *args, **kwargs)
            if "_hawor_" in task_id:
                result = copy.deepcopy(result)
                result["producer_result"]["sha256"] = "0" * 64
            return packet, result, refs

        with patch.object(candidate, "_sealed_stage", side_effect=changed_stage), \
             patch.object(candidate.prior, "plan_one") as plan:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V4_CARDS002_SEALED_CHAIN_DRIFT"):
                candidate.plan_one(Path(self.cards["source_root"]),
                                   self.cards["submission_input_sha256"])
        plan.assert_not_called()

    def test_cards002_receipt_ledger_ref_drift_rejects(self) -> None:
        original = candidate.load_json

        def changed_json(path):
            value = original(path)
            if path == candidate.TASK_STATE_PATH:
                value = copy.deepcopy(value)
                row = next(item for item in value["tasks"] if item.get("task_id") ==
                           "exact78_newraw_robot_play_cards_0902_002_50b73360d82cd4a7_v1")
                row["terminal_receipt"]["sha256"] = "0" * 64
            return value

        with patch.object(candidate, "load_json", side_effect=changed_json), \
             patch.object(candidate.prior, "plan_one") as plan:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V4_STAGE_RECEIPT_SHA_DRIFT"):
                candidate.plan_one(Path(self.cards["source_root"]),
                                   self.cards["submission_input_sha256"])
        plan.assert_not_called()

    def test_execution_has_no_receipt_and_stops_before_planning_or_spawning(self) -> None:
        source = Path(self.chips["source_root"])
        with patch.object(candidate.prior.legacy_submit, "_raw_session_path", return_value=source), \
             patch.object(candidate.prior, "plan_one") as plan, \
             patch.object(candidate, "_launch_cpu") as launch:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V4_CODE_ACCEPTANCE_NOT_SEALED"):
                candidate.submit_one(source, self.chips["submission_input_sha256"])
        plan.assert_not_called()
        launch.assert_not_called()

    def test_direct_cpu_adapter_requires_v4_receipt_before_register(self) -> None:
        source = Path(self.chips["source_root"])
        packet = {"inputs": {"source_root": str(source)}}
        with patch.object(adapter, "plan_one", return_value=({"status": "READY"}, packet)), \
             patch.object(candidate, "accepted_code_for_execution",
                          side_effect=ColdStartInputError("NO_V4_RECEIPT")), \
             patch.object(adapter.phase_exact.producer, "register") as register:
            with self.assertRaisesRegex(ColdStartInputError, "NO_V4_RECEIPT"):
                adapter.submit_one(source, self.chips["submission_input_sha256"])
        register.assert_not_called()

    def test_cpu_adapter_check_only_never_checks_receipt_or_registers(self) -> None:
        source = Path(self.chips["source_root"])
        with patch.object(adapter, "plan_one", return_value=({"status": "READY"}, {})), \
             patch.object(candidate, "accepted_code_for_execution") as accept, \
             patch.object(adapter.phase_exact.producer, "register") as register:
            self.assertEqual(adapter.submit_one(source, self.chips["submission_input_sha256"],
                                                check_only=True)["status"], "READY")
        accept.assert_not_called()
        register.assert_not_called()

    def test_manifest_cli_is_check_only_never_batch_dispatches(self) -> None:
        with patch.object(cli_v4, "plan_manifest", return_value={"status": "READ_ONLY_MATRIX"}), \
             patch.object(cli_v4, "submit_one") as submit, \
             contextlib.redirect_stdout(io.StringIO()):
            code = cli_v4.main(["submit", "exact78-raw-to-robot", "--manifest",
                                str(candidate.MANIFEST), "--check-only"])
        self.assertEqual(code, 0)
        submit.assert_not_called()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli_v4.main(["submit", "exact78-raw-to-robot", "--manifest",
                             str(candidate.MANIFEST)])
        self.assertEqual(caught.exception.code, 2)

    def test_unsealed_acceptance_does_not_consume_old_v3_receipt(self) -> None:
        source = Path(self.cards["source_root"])
        with patch.object(candidate.v3, "accepted_code_for_execution") as old_gate:
            with self.assertRaisesRegex(ColdStartInputError,
                                        "CLI_V4_CODE_ACCEPTANCE_NOT_SEALED"):
                candidate.accepted_code_for_execution(source,
                                                      self.cards["submission_input_sha256"])
        old_gate.assert_not_called()

    def test_synthetic_sealed_v4_receipt_can_authorize_both_manifest_members(self) -> None:
        def ref(path: Path) -> dict:
            return {"path": str(path), "bytes": 1, "sha256": "a" * 64}
        code_sets = {
            "cli_code": {name: ref(candidate.REPO_ROOT / name) for name in candidate.CLI_CODE},
            "inherited_cli_v3_code": {name: ref(candidate.REPO_ROOT / name)
                                      for name in candidate.v3.CLI_CODE},
            "inherited_cli_v2_code": {name: ref(candidate.REPO_ROOT / name)
                                      for name in candidate.prior.CLI_CODE},
        }
        operation_refs = {
            key: {"candidate_operation": "run_exact78_newraw_" + key,
                  "operation_ref": ref(candidate.REPO_ROOT / "src/chaoyang/ops" /
                                       ("run_exact78_newraw_" + key + ".py"))}
            for key in candidate.OPERATIONS
        }
        source_evidence = {
            item["session_id"]: {"source_root": item["source_root"],
                                 "submission_input_sha256": item["submission_input_sha256"],
                                 "decoded_frames": item["frame_count_decoded"],
                                 "fps_nominal": item["fps_nominal"],
                                 "time_mode": item["time_mode"],
                                 "physical_eye": item["physical_eye"],
                                 "source_files": {role: ref(Path("/fake") / role) for role in (
                                     "video", "source_stereo_video", "camera_params",
                                     "slam", "clip_manifest")}}
            for item in self.manifest["sessions"]
        }
        check = {"status": "READ_ONLY_MATRIX", "registration_started": False,
                 "model_calls": 0, "stdout_sha256": "b" * 64}
        negative = {"status": "PASSED", "test_count": 10}
        contracts = {"special_status_contracts": {
            key: {"candidate_operation": value["candidate_operation"],
                  "task_packet_required": True, "current_runnable_operation": None,
                  "code_closure": [value["operation_ref"]]}
            for key, value in operation_refs.items()}}
        packet_ref, result_ref, receipt_ref = (ref(path) for path in (
            candidate.PACKET, candidate.RESULT, candidate.RECEIPT))
        inputs = {"accepted_entry": "chaoyang.cli_v4", "manifest": self.ref,
                  **code_sets, "test_code": ref(candidate.REPO_ROOT / candidate.TEST_CODE),
                  "acceptance_test_code": ref(candidate.REPO_ROOT /
                                              "tests/unit/test_exact78_cli_v4_manifest_acceptance_candidate_v1.py"),
                  "source_evidence": source_evidence, "cli_check_only": check,
                  "negative_tests": negative, "published_operations": operation_refs}
        packet = {"task_id": candidate.TASK_ID, "phase": candidate.PHASE,
                  "run_signature": "signature", "execution_mode": "EXACT78_GOVERNANCE_ONLY",
                  "inputs": inputs}
        result = {"schema_version": "exact78-newraw-cli-v4-manifest-acceptance-result-v1",
                  "task_id": candidate.TASK_ID, "status": "PASSED",
                  "run_signature": "signature", "accepted_entry": "chaoyang.cli_v4",
                  "manifest": self.ref, "task_packet": packet_ref,
                  **code_sets, "test_code": inputs["test_code"],
                  "acceptance_test_code": inputs["acceptance_test_code"],
                  "source_evidence": source_evidence, "cli_check_only": check,
                  "negative_tests": negative,
                  "published_operations": operation_refs}
        receipt = {"schema_version": "exact78-newraw-cli-v4-manifest-acceptance-receipt-v1",
                   "task_id": candidate.TASK_ID, "status": "PASSED",
                   "run_signature": "signature", "task_packet": packet_ref,
                   "result": result_ref}
        state = {"tasks": [{"task_id": candidate.TASK_ID, "phase": candidate.PHASE,
                            "status": "PASSED", "run_signature": "signature",
                            "task_packet": packet_ref, "result": result_ref,
                            "terminal_receipt": receipt_ref}]}
        index = {"task_packets": [{"task_id": candidate.TASK_ID,
                                   "execution_allowed": False,
                                   "packet_path": str(candidate.PACKET.relative_to(candidate.REPO_ROOT)),
                                   "packet_sha256": packet_ref["sha256"]}]}
        sources = {candidate.TASK_STATE_PATH: state, candidate.prior.INDEX: index,
                   candidate.PACKET: packet, candidate.RESULT: result,
                   candidate.RECEIPT: receipt,
                   candidate.MANIFEST: self.manifest,
                   candidate.prior.ALGORITHM_CONTRACT: contracts}
        for member in (self.chips, self.cards):
            with patch.object(candidate, "_entry", return_value=(member, self.ref)), \
                 patch.object(Path, "is_file", return_value=True), \
                 patch.object(candidate, "_ref", side_effect=lambda path, value, reason: value), \
                 patch.object(candidate, "validate_artifact_ref", return_value=[]), \
                 patch.object(candidate, "load_json", side_effect=lambda path: sources[path]):
                self.assertEqual(candidate.accepted_code_for_execution(
                    Path(member["source_root"]), member["submission_input_sha256"]), result_ref)


if __name__ == "__main__":
    unittest.main()
