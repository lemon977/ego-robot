"""No-ledger tests for the future single-session CPU admission producer."""

from __future__ import annotations

import json
from io import StringIO
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from chaoyang.cli import exact78_first_submit_bridge_candidate as bridge
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as producer
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError, PROJECT_ROOT, file_ref


class FirstSubmitCpuProducerCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix="exact78-cpu-producer-test-", dir=PROJECT_ROOT / "_run/maintenance")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "source_data"
        self.data.mkdir()
        self.session = self.data / "tracker" / "play_cards_0915_011"
        self._write(self.session / "rgb.mp4", "synthetic RGB")
        self._write(self.session / "source_stereo/stereo.mp4", "synthetic stereo")
        self._write(self.session / "camera_params.json", "{}")
        self._write(self.session / "slam_trajectory_play_cards_0915_011.jsonl", "{}\n")
        self._write(self.session / "clip_manifest.json", {
            "clip_name": "play_cards_0915_011",
            "files": {"video": "rgb.mp4", "source_stereo_video": "source_stereo/stereo.mp4",
                      "camera_params": "camera_params.json",
                      "slam": "slam_trajectory_play_cards_0915_011.jsonl"},
            "video": {"frame_count": 2, "fps": 25, "width": 1280, "height": 960},
            "video_rectification": {"eye": "left", "rectified": True,
                                    "resolution": [1280, 960],
                                    "intrinsics": [[800, 0, 640], [0, 800, 480], [0, 0, 1]]},
        })

    def _write(self, path: Path, value: str | dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) if isinstance(value, dict) else value,
                        encoding="utf-8")

    def _prepared(self) -> dict:
        frames = [{"width": 1280, "height": 960, "pts": i,
                   "best_effort_timestamp": i, "pict_type": "P"} for i in range(2)]
        with patch.object(bridge, "_decode_frame_rows", return_value=frames):
            return producer.preflight(self.session, allowed_root=self.data)

    def test_preflight_one_explicit_session_and_no_registration(self) -> None:
        prepared = self._prepared()
        self.assertEqual(prepared["descriptor"]["session_id"], self.session.name)
        self.assertEqual(prepared["descriptor"]["decoded_frames"], 2)
        self.assertFalse((PROJECT_ROOT / prepared["target_draft"]["packet_path"]).exists())

    def test_derived_source_is_rejected_before_registration(self) -> None:
        prepared = self._prepared()
        descriptor = prepared["descriptor"]
        descriptor["source_files"]["video"]["path"] = str(self.session / "preprocess/all_data.mp4")
        with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_SOURCE_FORBIDDEN"):
            producer._reject_derived(descriptor, allowed_root=self.data)

    def test_derived_manifest_video_rejects_before_bridge_reads_it(self) -> None:
        manifest_path = self.session / "clip_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["video"] = "preprocess/all_data.mp4"
        self._write(manifest_path, manifest)
        with patch.object(producer, "plan_single") as bridge_plan:
            with self.assertRaisesRegex(ColdStartInputError, "OLD_DERIVED_SOURCE_FORBIDDEN"):
                producer.preflight(self.session, allowed_root=self.data)
        bridge_plan.assert_not_called()

    def test_real_tracker_manifest_derived_sidecar_is_not_an_input(self) -> None:
        manifest_path = self.session / "clip_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The actual tracker format can retain this legacy field. Its target
        # deliberately does not exist: success proves it was not opened/hash'd.
        manifest["files"]["humanego_all_data"] = "preprocess/all_data.npz"
        self._write(manifest_path, manifest)
        prepared = self._prepared()
        source_files = prepared["descriptor"]["source_files"]
        self.assertEqual(set(source_files), {
            "video", "source_stereo_video", "camera_params", "slam", "clip_manifest"})
        self.assertNotIn("humanego_all_data", producer.packet_value(prepared)["inputs"]["source_files"])

    def test_consumed_manifest_role_cannot_point_to_derived_data(self) -> None:
        manifest_path = self.session / "clip_manifest.json"
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        for role in ("video", "source_stereo_video", "camera_params", "slam"):
            with self.subTest(role=role):
                manifest = json.loads(json.dumps(original))
                manifest["files"][role] = "preprocess/derived_source.mp4"
                self._write(manifest_path, manifest)
                with patch.object(producer, "plan_single") as bridge_plan:
                    with self.assertRaisesRegex(ColdStartInputError, f"OLD_DERIVED_SOURCE_FORBIDDEN:{role}"):
                        producer.preflight(self.session, allowed_root=self.data)
                bridge_plan.assert_not_called()

    def test_packet_is_cpu_only_and_source_bound(self) -> None:
        prepared = self._prepared()
        packet = producer.packet_value(prepared)
        self.assertEqual(packet["execution_mode"], "EXACT78_CANDIDATE")
        self.assertEqual(packet["budgets"]["gpu_count"], 0)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["inputs"]["descriptor_sha256"], prepared["descriptor_sha256"])
        self.assertEqual(packet["inputs"]["source_files"], prepared["descriptor"]["source_files"])
        self.assertIn("--session", packet["commands"][0])
        self.assertIn("-m chaoyang.cli run " + producer.OPERATION, packet["commands"][0])
        self.assertNotIn("--register-and-run", packet["commands"][0])
        self.assertNotIn("--manifest", packet["commands"][0])

    def test_registration_builds_one_fenced_cas_packet_in_local_fixture(self) -> None:
        prepared = self._prepared()
        packet = producer.packet_value(prepared)
        repo = self.root / "repo"
        index = repo / "tasks/current/INDEX.json"
        state = repo / "docs/governance/LONG_HORIZON_TASK_STATE.json"
        receipt = repo / "docs/governance/CURRENT_STATUS_RECEIPT.json"
        authority = repo / "docs/governance/CURRENT_AUTHORITY_INDEX.json"
        route = repo / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
        self._write(index, {"task_packets": []})
        self._write(state, {"tasks": [], "next_task": None})
        self._write(receipt, {"governance_revision": 7, "files": {
            "task_state": file_ref(state), "task_packet_index_current": file_ref(index)}})
        self._write(authority, {})
        self._write(route, {"general_raw_to_robot": {"current_runnable_operation": None}})
        with patch.object(producer, "REPO_ROOT", repo), patch.object(producer, "INDEX", index), \
                patch.object(producer, "TASK_STATE_PATH", state), \
                patch.object(producer, "RECEIPT_PATH", receipt), \
                patch.object(producer, "AUTHORITY_PATH", authority), \
                patch.object(producer, "ROUTE", route), patch.object(producer, "DATA_ROOT", self.data), \
                patch.object(producer, "_assert_cli_admission_ready"), \
                patch.object(producer, "validate_task_state", return_value=[]), \
                patch.object(producer, "publish_bundle") as publish:
            producer.register(packet)
        attempt = repo / "_run/current" / packet["task_id"] / "attempts/attempt_0001"
        self.assertEqual(json.loads((attempt / "WRITER.json").read_text()), packet["writer"])
        self.assertTrue((repo / "tasks/current" / packet["task_id"] / "TASK_PACKET.json").is_file())
        self.assertEqual(publish.call_args.kwargs["expected_revision"], 7)
        self.assertEqual(publish.call_args.args[1]["tasks"][0]["status"], "RUNNING")
        allowed = [row for row in publish.call_args.kwargs["task_packet_index_value"]["task_packets"]
                   if row["execution_allowed"]]
        self.assertEqual(len(allowed), 1)
        self.assertEqual(allowed[0]["task_id"], packet["task_id"])

    def test_produce_admission_writes_three_same_attempt_refs(self) -> None:
        prepared = self._prepared()
        task_id = prepared["target_draft"]["task_id"]
        attempt = self.root / "repo/_run/current" / task_id / "attempts/attempt_0001"
        attempt.mkdir(parents=True)
        packet = {"task_id": task_id, "deadline_at": "2099-01-01T00:00:00+00:00", "inputs": {
            "descriptor_sha256": prepared["descriptor_sha256"],
            "source_files": prepared["descriptor"]["source_files"],
            "submission_input_sha256": prepared["descriptor"]["submission_input_sha256"]}}
        paths = (attempt, self.root / "repo/TASK_PACKET.json", self.root / "repo/RECEIPT.json")
        with patch.object(producer, "_fence"), patch.object(producer, "_paths", return_value=paths), \
                patch.object(producer, "DATA_ROOT", self.data):
            refs = producer.produce_admission(prepared, packet)
        self.assertEqual(set(refs), {"descriptor", "frame_time_map", "seal"})
        self.assertEqual({Path(ref["path"]).parent for ref in refs.values()}, {attempt / "admission"})
        seal = json.loads(Path(refs["seal"]["path"]).read_text(encoding="utf-8"))
        self.assertEqual(seal["status"], "SEALED_CPU_ADMISSION")
        self.assertEqual(seal["descriptor"], refs["descriptor"])
        self.assertEqual(seal["frame_time_map"], refs["frame_time_map"])

    def test_source_mutation_prevents_seal(self) -> None:
        prepared = self._prepared()
        task_id = prepared["target_draft"]["task_id"]
        attempt = self.root / "repo/_run/current" / task_id / "attempts/attempt_0001"
        attempt.mkdir(parents=True)
        packet = {"task_id": task_id, "deadline_at": "2099-01-01T00:00:00+00:00", "inputs": {
            "descriptor_sha256": prepared["descriptor_sha256"],
            "source_files": prepared["descriptor"]["source_files"],
            "submission_input_sha256": prepared["descriptor"]["submission_input_sha256"]}}
        (self.session / "rgb.mp4").write_text("source changed", encoding="utf-8")
        paths = (attempt, self.root / "repo/TASK_PACKET.json", self.root / "repo/RECEIPT.json")
        with patch.object(producer, "_fence"), patch.object(producer, "_paths", return_value=paths), \
                patch.object(producer, "DATA_ROOT", self.data):
            with self.assertRaisesRegex(ColdStartInputError, "SOURCE_SHA_DRIFT"):
                producer.produce_admission(prepared, packet)
        self.assertFalse((attempt / "admission").exists())

    def test_direct_registration_entry_rejects_before_raw_preflight(self) -> None:
        with patch.object(producer, "preflight") as preflight, \
                patch.object(producer, "register") as register:
            with self.assertRaisesRegex(RuntimeError, "DIRECT_PRODUCER_EXECUTION_FORBIDDEN"):
                producer.main(["--session", str(self.session), "--register-and-run"])
        preflight.assert_not_called()
        register.assert_not_called()

    def test_uncontracted_candidate_cannot_register(self) -> None:
        packet = producer.packet_value(self._prepared())
        contract = self.root / "ALGORITHM_CONTRACT.json"
        self._write(contract, {"special_status_contracts": {}})
        with patch.object(producer, "_registration_window") as window, \
                patch.object(producer, "ALGORITHM_CONTRACT", contract), \
                patch("chaoyang.cli._maintained_operations", return_value=set()), \
                patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                      return_value={"special_status_contracts": {}}):
            with self.assertRaisesRegex(RuntimeError, "ALGORITHM_CONTRACT_CLI_ROUTE_NOT_STAGED"):
                producer.register(packet)
        window.assert_not_called()

    def test_operation_file_alone_does_not_satisfy_algorithm_contract(self) -> None:
        packet = producer.packet_value(self._prepared())
        repo = self.root / "repo"
        operation = repo / "src/chaoyang/ops" / f"{producer.OPERATION}.py"
        contract = repo / "docs/governance/ALGORITHM_CONTRACT.json"
        self._write(operation, "def main():\n    return 0\n")
        self._write(contract, {"special_status_contracts": {}})
        packet["runtime_code"].append(file_ref(operation))
        with patch.object(producer, "REPO_ROOT", repo), \
                patch.object(producer, "ALGORITHM_CONTRACT", contract), \
                patch("chaoyang.cli._maintained_operations", return_value=set()), \
                patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                      return_value={"special_status_contracts": {}}):
            with self.assertRaisesRegex(RuntimeError, "ALGORITHM_CONTRACT_CLI_ROUTE_NOT_STAGED"):
                producer._assert_cli_admission_ready(packet)

    def test_staged_route_can_publish_with_packet_but_wrong_sha_rejects(self) -> None:
        packet = producer.packet_value(self._prepared())
        contract = self.root / "ALGORITHM_CONTRACT.json"
        self._write(contract, {"special_status_contracts": {}})
        route = {
            "candidate_operation": producer.OPERATION,
            "execution_status": "CANDIDATE_FENCED_CPU_ONLY_SINGLE_SESSION",
            "task_packet_required": True, "current_runnable_operation": None,
            "output_schema": "exact78-first-submit-cpu-admission-result-v1",
            "model_calls_authorized": 0, "render_calls_authorized": 0,
            "training_steps_authorized": 0,
            "code_closure": packet["runtime_code"],
        }
        staged = {"special_status_contracts": {producer.ROUTE_KEY: route}}
        with patch.object(producer, "ALGORITHM_CONTRACT", contract), \
                patch("chaoyang.cli._maintained_operations", return_value=set()), \
                patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                      return_value=staged):
            self.assertEqual(producer._assert_cli_admission_ready(packet),
                             "READY_TO_PUBLISH_CPU_ROUTE_WITH_PACKET")
            wrong = json.loads(json.dumps(staged))
            wrong_refs = wrong["special_status_contracts"][producer.ROUTE_KEY]["code_closure"]
            next(ref for ref in wrong_refs if ref["path"].endswith(
                f"/{producer.OPERATION}.py"))["sha256"] = "0" * 64
            with patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                       return_value=wrong):
                with self.assertRaisesRegex(RuntimeError, "ALGORITHM_CONTRACT_CLI_ROUTE_NOT_STAGED"):
                    producer._assert_cli_admission_ready(packet)
        published_bad = {"special_status_contracts": {producer.ROUTE_KEY: {"candidate_operation": "wrong"}}}
        self._write(contract, published_bad)
        with patch.object(producer, "ALGORITHM_CONTRACT", contract):
            with self.assertRaisesRegex(RuntimeError, "ALGORITHM_CONTRACT_PUBLISHED_ROUTE_DRIFT"):
                producer._assert_cli_admission_ready(packet)

    def test_check_only_reports_summary_without_registration(self) -> None:
        prepared = self._prepared()
        printed = StringIO()
        with patch.object(producer, "preflight", return_value=prepared), \
                patch.object(producer, "register") as register, redirect_stdout(printed):
            code = producer.main(["--session", str(self.session), "--check-only"])
        self.assertEqual(code, 0)
        result = json.loads(printed.getvalue())
        self.assertEqual(result["status"], "DRY_RUN_CPU_ADMISSION_READY_NOT_REGISTERED")
        self.assertFalse(result["registration_started"])
        register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
