"""CPU-only tests for the unregistered sealed-admission HaWoR consumer."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH, AUTHORITY_PATH, BASELINE_REGISTRY_PATH, REPO_ROOT,
    load_json,
)
# Import the module itself so tests can patch its read-only boundaries without
# invoking its register/run path.
from chaoyang.governance import execute_exact78_newraw_hawor_gpu_candidate_v1 as consumer
from chaoyang.governance.exact78_cold_start_guard_v1 import ColdStartInputError
from chaoyang.ops import run_exact78_newraw_hawor_gpu_candidate_v1 as operation
from chaoyang.ops import run_exact78_hawor_raw_v3 as frozen_v3
from chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1 import (
    checked_spec, run_stage,
)


SEALED_RESULT = (REPO_ROOT / "_run/current/"
    "exact78_newraw_play_cards_0902_001_9541579d8de29395_v1/"
    "attempts/attempt_0001/RESULT.json")


class NewRawHaWoRGpuCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = consumer.packet_value(SEALED_RESULT)

    def test_real_terminal_cpu_admission_produces_finite_gpu_packet(self) -> None:
        packet = self.packet
        self.assertEqual(packet["inputs"]["source_session"], "play_cards_0902_001")
        self.assertEqual(packet["config"]["frame_count"], 201)
        self.assertEqual(packet["config"]["model_input_size"], [1280, 960])
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(len(packet["inputs"]["model_assets"]), 6)
        self.assertLessEqual(len(packet["read_set"]), 8)
        self.assertIn("SAME_SIGNATURE_SEALED_STAGE_ONLY", packet["config"]["resume_policy"])
        self.assertEqual(checked_spec(packet)["frame_count"], 201)

    def test_frozen_v3_frame_status_preserves_sealed_order_and_time(self) -> None:
        spec = checked_spec(self.packet)
        for index in (0, 100, 200):
            row = frozen_v3.frame_status(spec["frame_rows_sealed"][index], index,
                                         spec, True, False)
            self.assertEqual(row["decoded_display_index"], index)
            self.assertEqual(row["model_input_index"], index)
            self.assertEqual(row["output_index"], index)
            self.assertEqual(row["best_effort_timestamp"],
                             spec["frame_rows_sealed"][index]["best_effort_timestamp"])
            self.assertEqual(row["time_mode"], "CLIP_FRAME_INDEX_ONLY")
            self.assertEqual(row["hands"]["left"]["motion_source"], "HAWOR_MODEL_PREDICTION")
            self.assertEqual(row["hands"]["right"]["motion_source"], "MISSING")

    def test_sealed_result_sha_drift_rejected_before_model(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["inputs"]["upstream_admission_result"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ColdStartInputError, "SEALED_REF_SHA_OR_SCOPE_DRIFT"):
            checked_spec(packet)

    def test_unsealed_partial_cannot_resume(self) -> None:
        packet = copy.deepcopy(self.packet)
        with tempfile.TemporaryDirectory(prefix="newraw-hawor-partial-",
                                         dir=REPO_ROOT / "_run/maintenance") as temp:
            path = Path(temp)
            with patch.object(consumer, "_paths", return_value=(path, path / "p", path / "r")):
                (path / "candidate").mkdir()
                with self.assertRaisesRegex(RuntimeError, "UNSEALED_PARTIAL_CANNOT_RESUME"):
                    consumer._sealed_stage(packet)

    def test_stage_rejects_output_outside_exact_attempt(self) -> None:
        with self.assertRaisesRegex(ColdStartInputError, "OUTPUT_OUTSIDE_PROJECT_ATTEMPT"):
            run_stage(self.packet, REPO_ROOT / "_run/maintenance/forbidden-hawor-output")

    def _staged_contract(self) -> dict:
        from chaoyang.governance.current_r3_contracts import build_algorithm_contract
        return build_algorithm_contract(load_json(BASELINE_REGISTRY_PATH),
                                        load_json(AUTHORITY_PATH))

    def test_exact_staged_route_is_ready_for_same_cas_packet(self) -> None:
        staged = self._staged_contract()
        self.assertTrue(consumer._route_valid(staged, self.packet))
        self.assertEqual(consumer._assert_route_ready(self.packet),
                         "READY_TO_PUBLISH_ROUTE_WITH_PACKET")

    def test_staged_route_rejects_model_authorization_or_closure_drift(self) -> None:
        staged = self._staged_contract()
        route = staged["special_status_contracts"][consumer.ROUTE_KEY]
        route["model_calls_authorized"] = "UNBOUNDED"
        self.assertFalse(consumer._route_valid(staged, self.packet))
        route["model_calls_authorized"] = "ONE_FULL_HAWOR_SESSION_ONLY_WHEN_EXACT_PACKET_ACTIVE"
        route["code_closure"][0]["sha256"] = "0" * 64
        self.assertFalse(consumer._route_valid(staged, self.packet))

    def test_published_route_drift_never_falls_back_to_staged(self) -> None:
        published = self._staged_contract()
        published["special_status_contracts"][consumer.ROUTE_KEY]["output_schema"] = "WRONG"
        original_load = consumer.load_json

        def read_with_drift(path: Path) -> dict:
            if Path(path) == ALGORITHM_CONTRACT_PATH:
                return published
            return original_load(path)

        with patch.object(consumer, "load_json", side_effect=read_with_drift):
            with self.assertRaisesRegex(RuntimeError, "PUBLISHED_CANDIDATE_ROUTE_DRIFT"):
                consumer._assert_route_ready(self.packet)

    def test_published_route_rejects_builder_drift_before_cas(self) -> None:
        published = self._staged_contract()
        staged = copy.deepcopy(published)
        staged["special_status_contracts"][consumer.ROUTE_KEY]["claim_limit"] = "DRIFT"
        original_load = consumer.load_json

        def read_published(path: Path) -> dict:
            if Path(path) == ALGORITHM_CONTRACT_PATH:
                return published
            return original_load(path)

        with patch.object(consumer, "load_json", side_effect=read_published), \
             patch("chaoyang.cli._maintained_operations", return_value={consumer.OPERATION}), \
             patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                   return_value=staged):
            with self.assertRaisesRegex(RuntimeError, "PUBLISHED_CANDIDATE_BUILDER_DRIFT"):
                consumer._assert_route_ready(self.packet)

    def test_staged_route_does_not_allow_production_or_command_drift(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["commands"] = [packet["commands"][0] + " --legacy-mask old.npy"]
        with self.assertRaisesRegex(RuntimeError, "CANDIDATE_COMMAND_DRIFT"):
            consumer._assert_route_ready(packet)
        original_load = consumer.load_json

        def read_with_production(path: Path) -> dict:
            if Path(path) == consumer.ROUTE:
                value = copy.deepcopy(original_load(path))
                value["general_raw_to_robot"]["current_runnable_operation"] = "some_production_op"
                return value
            return original_load(path)

        with patch.object(consumer, "load_json", side_effect=read_with_production):
            with self.assertRaisesRegex(RuntimeError, "PRODUCTION_ROUTE_ALREADY_SELECTED"):
                consumer._assert_route_ready(self.packet)

    def _audit(self, lines: str) -> dict:
        packet = {
            "task_id": "exact78_newraw_hawor_synthetic_v1",
            "inputs": {
                "source_files": {"video": {"path": "/path/to/private_data/egodata/synthetic/raw.mp4"}},
                "raw_video": {"path": "/path/to/private_data/egodata/synthetic/raw.mp4"},
                "upstream_admission_result": {
                    "path": str(REPO_ROOT / "_run/current/exact78_newraw_synthetic_v1/attempts/attempt_0001/RESULT.json")},
            },
        }
        with tempfile.TemporaryDirectory(prefix="newraw-hawor-audit-",
                                         dir=REPO_ROOT / "_run/maintenance") as temp:
            trace = Path(temp) / "trace.log"
            trace.write_text(lines, encoding="utf-8")
            return consumer._native_open_audit(packet, trace, Path(temp) / "AUDIT.json")

    def test_native_open_allows_only_declared_raw(self) -> None:
        audit = self._audit('openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n')
        self.assertEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")

    def test_native_open_rejects_external_write_and_old_result(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_WRONLY|O_CREAT) = -1 EACCES\n'
            f'openat(AT_FDCWD, "{REPO_ROOT}/_run/current/exact78_old_hawor/RESULT.json", O_RDONLY) = 4\n')
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertTrue(any("EXTERNAL_WRITE" in item for item in audit["forbidden_paths"]))
        self.assertTrue(any("OTHER_EXACT78_ATTEMPT_READ" in item for item in audit["forbidden_paths"]))

    def test_native_open_rejects_undeclared_external_read(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/old_mask.npy", O_RDONLY) = 4\n')
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertTrue(any("UNDECLARED_EXTERNAL_READ" in item
                            for item in audit["forbidden_paths"]))

    def test_native_open_rejects_project_external_cache_write(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/tmp/torch-cache.bin", O_WRONLY|O_CREAT) = 4\n')
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertTrue(any("OUTSIDE_PROJECT_WRITE" in item
                            for item in audit["forbidden_paths"]))

    def test_operation_denial_precedes_model_stage(self) -> None:
        with patch.object(operation, "authorize_exact78_operation", side_effect=RuntimeError("GATE_DENIED")):
            with patch("chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1.run_stage") as model:
                with self.assertRaisesRegex(RuntimeError, "GATE_DENIED"):
                    operation.main(["--producer-result", str(SEALED_RESULT),
                                    "--output-root", str(REPO_ROOT / "_run/maintenance/not-used")])
                model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
