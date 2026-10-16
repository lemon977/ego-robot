"""CPU-only tests for the unregistered new-raw Robot packet/operation seam."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from copy import deepcopy

from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT, file_ref
from chaoyang.governance import execute_exact78_newraw_robot_cpu_candidate_v1 as builder
from chaoyang.ops import run_exact78_newraw_robot_cpu_candidate_v1 as operation


class NewrawRobotCpuCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory(prefix="newraw-robot-packet-",
                                           dir=PROJECT_ROOT / "_run/maintenance")
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "HAWOR_OUTER_RESULT.json"
        self.path.write_text("{}", encoding="utf-8")
        self.ref = file_ref(self.path)
        self.session = "play_cards_0902_001"
        self.cpu = {"source_session": self.session, "source_root": str(self.path.parent),
                    "descriptor_sha256": "a" * 64, "submission_input_sha256": "b" * 64}
        self.cpu_refs = {name: self.ref for name in (
            "upstream_admission_descriptor", "upstream_admission_frame_time_map",
            "upstream_admission_seal", "upstream_admission_result",
            "upstream_admission_receipt")}
        self.handoff = {
            "session_id": self.session, "frame_count": 201, "source_kind": "tracker",
            "time_mode": "CLIP_FRAME_INDEX_ONLY", "physical_eye": "left",
            "source_files": {"video": self.ref},
            **{name: self.ref for name in (
                "hawor_outer_result", "hawor_receipt", "hawor_packet",
                "hawor_stage_seal", "hawor_native_open_audit", "hawor_inner_result",
                "hawor_npz", "hawor_frame_provenance", "hawor_input_audit",
                "raw_video")},
        }

    def packet(self) -> dict:
        with patch.object(builder, "_cpu_probe", return_value=(self.cpu, self.cpu_refs, {})), \
                patch("chaoyang.governance.exact78_newraw_robot_handoff_candidate_v1.verify_handoff",
                      return_value=self.handoff), \
                patch.object(builder, "_asset_refs", return_value={"asset_pin": self.ref}):
            return builder.packet_value(self.path)

    def test_201_frame_tracker_packet_has_one_cpu_writer_and_no_model(self) -> None:
        packet = self.packet()
        self.assertEqual(packet["task_id"],
                         "exact78_newraw_robot_play_cards_0902_001_aaaaaaaaaaaaaaaa_v1")
        self.assertEqual(packet["phase"], "EXACT78_NEWRAW_ROBOT_CPU_CANDIDATE")
        self.assertEqual(packet["config"]["frame_count"], 201)
        self.assertEqual(packet["config"]["source_kind"], "tracker")
        self.assertEqual(packet["budgets"]["workers_max"], 1)
        self.assertEqual(packet["budgets"]["cpu_threads_total"], 2)
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["config"]["model_calls"], 0)
        self.assertIn("--hawor-result", packet["commands"][0])
        self.assertNotIn("register-and-run", packet["commands"][0])

    def test_vst0915_source_kind_and_unverified_timebase_are_not_upgraded(self) -> None:
        self.handoff["source_kind"] = "vst0915"
        self.handoff["time_mode"] = "TIMEBASE_UNVERIFIED"
        self.handoff["frame_count"] = 131
        packet = self.packet()
        self.assertEqual(packet["config"]["source_kind"], "vst0915")
        self.assertEqual(packet["config"]["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertFalse(packet["config"]["timebase_authority"])

    def test_operation_denial_precedes_robot_stage(self) -> None:
        with patch.object(operation, "authorize_exact78_operation",
                          side_effect=RuntimeError("GATE_DENIED")), \
                patch.object(operation, "run_candidate") as stage:
            with self.assertRaisesRegex(RuntimeError, "GATE_DENIED"):
                operation.main(["--hawor-result", str(self.path),
                                "--output-root", str(self.path.parent / "robot")])
        stage.assert_not_called()

    def test_unregistered_operation_cannot_find_exact_active_packet(self) -> None:
        # The candidate is intentionally not published in CLI or the current
        # Task Packet index. Isolate that state because another task may run.
        with patch.object(Path, "read_text",
                          return_value=json.dumps({"task_packets": []})):
            with self.assertRaisesRegex(RuntimeError, "EXACT_ROBOT_CANDIDATE_PACKET_REQUIRED"):
                operation._active_packet()

    def test_route_requires_exact_cpu_contract_and_code_closure(self) -> None:
        packet = self.packet()
        route = {
            "execution_status": "CANDIDATE_FENCED_CPU_ONE_SESSION",
            "candidate_operation": builder.OPERATION,
            "task_packet_required": True,
            "current_runnable_operation": None,
            "output_schema": "exact78-newraw-robot-cpu-execution-result-v1",
            "model_calls_authorized": 0,
            "render_calls_authorized": "ONE_FULL_SESSION_ONLY_WHEN_EXACT_PACKET_ACTIVE",
            "training_steps_authorized": 0,
            "code_closure": packet["runtime_code"],
        }
        self.assertTrue(builder._route_valid(
            {"special_status_contracts": {builder.ROUTE_KEY: route}}, packet))
        for key, bad in (("model_calls_authorized", 1),
                         ("training_steps_authorized", 1),
                         ("current_runnable_operation", builder.OPERATION),
                         ("candidate_operation", "old_robot_operation")):
            altered = deepcopy(route)
            altered[key] = bad
            self.assertFalse(builder._route_valid(
                {"special_status_contracts": {builder.ROUTE_KEY: altered}}, packet), key)
        altered = deepcopy(route)
        altered["code_closure"][0]["sha256"] = "0" * 64
        self.assertFalse(builder._route_valid(
            {"special_status_contracts": {builder.ROUTE_KEY: altered}}, packet))

    def test_route_absent_cannot_register_even_with_synthetic_packet(self) -> None:
        packet = self.packet()
        with patch.object(builder, "_assert_route_ready",
                          side_effect=RuntimeError("ROBOT_ROUTE_NOT_STAGED")), \
                patch.object(builder, "publish_bundle") as publish:
            with self.assertRaisesRegex(RuntimeError, "ROBOT_ROUTE_NOT_STAGED"):
                builder.register(packet)
        publish.assert_not_called()

    def test_missing_stage_cannot_pass_terminal_verifier(self) -> None:
        packet = self.packet()
        with patch.object(builder, "_paths", return_value=(
                self.path.parent, self.path.parent / "PACKET.json",
                self.path.parent / "RECEIPT.json")):
            with self.assertRaises((FileNotFoundError, ValueError, KeyError)):
                builder._verify_stage(packet)

    def test_failed_worker_seals_failed_result_without_promoting_partial(self) -> None:
        packet = self.packet()
        attempt = self.path.parent / "attempt"
        attempt.mkdir()
        log = attempt / "worker.log"
        log.write_text("fixture failure\n", encoding="utf-8")
        packet_path = attempt / "TASK_PACKET.json"
        receipt = attempt / "TERMINAL_RECEIPT.json"
        state = {"tasks": [{"task_id": packet["task_id"], "status": "RUNNING"}],
                 "next_task": {"task_id": packet["task_id"]}}
        index = {"task_packets": [{"task_id": packet["task_id"],
                                    "execution_allowed": True}]}
        published = {"governance_revision": 7}
        with patch.object(builder, "_paths", return_value=(attempt, packet_path, receipt)), \
                patch.object(builder, "_fence"), \
                patch.object(builder, "_index_copy", return_value=index), \
                patch.object(builder, "_receipt_revision", return_value=6), \
                patch.object(builder, "validate_task_state", return_value=[]), \
                patch.object(builder, "publish_bundle", return_value=published) as publish, \
                patch.object(builder, "load_json", return_value=state):
            terminal = builder.finish(packet, worker_returncode=9,
                                      worker_log=log, stage=None,
                                      error="UPSTREAM_NOT_SEALED")
        self.assertEqual(terminal["status"], "FAILED_RUNTIME_FINAL")
        result = json.loads((attempt / "RESULT.json").read_text(encoding="utf-8"))
        self.assertFalse(result["render_complete"])
        self.assertFalse(result["visual_qualified"])
        self.assertFalse(result["training_eligible"])
        self.assertEqual(result["stage"], {})
        self.assertFalse(index["task_packets"][0]["execution_allowed"])
        publish.assert_called_once()

    def test_no_success_without_independent_stage_verification(self) -> None:
        packet = self.packet()
        attempt = self.path.parent / "attempt"
        attempt.mkdir()
        log = attempt / "worker.log"
        log.write_text("fixture\n", encoding="utf-8")
        with patch.object(builder, "_paths", return_value=(
                attempt, attempt / "TASK_PACKET.json", attempt / "RECEIPT.json")), \
                patch.object(builder, "_fence"), \
                patch.object(builder, "_verify_stage",
                             side_effect=RuntimeError("ROBOT_VIDEO_SHA_OR_FRAME_DRIFT")):
            with self.assertRaisesRegex(RuntimeError, "ROBOT_VIDEO_SHA_OR_FRAME_DRIFT"):
                builder.finish(packet, worker_returncode=0, worker_log=log,
                               stage={"result": self.ref})
        self.assertFalse((attempt / "RESULT.json").exists())


if __name__ == "__main__":
    unittest.main()
