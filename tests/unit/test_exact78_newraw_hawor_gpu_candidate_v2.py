"""CPU-only successor tests; never register a packet or invoke HaWoR."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.governance import execute_exact78_newraw_hawor_gpu_candidate_v2 as executor
from chaoyang.ops import run_exact78_newraw_hawor_gpu_candidate_v2 as operation


SEALED_CPU_RESULT = (REPO_ROOT / "_run/current/"
    "exact78_newraw_play_cards_0902_001_9541579d8de29395_v1/"
    "attempts/attempt_0001/RESULT.json")
FAILED_V1_RESULT = (REPO_ROOT / "_run/current/"
    "exact78_newraw_hawor_play_cards_0902_001_9541579d8de29395_v1/"
    "attempts/attempt_0001/RESULT.json")


class NewRawHaWoRGpuV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = executor.packet_value(SEALED_CPU_RESULT)

    def _contract(self) -> dict:
        route = {
            "execution_status": "CANDIDATE_FENCED_ONE_GPU_ONE_SESSION",
            "candidate_operation": executor.OPERATION,
            "task_packet_required": True,
            "current_runnable_operation": None,
            "output_schema": "exact78-newraw-hawor-gpu-execution-result-v2",
            "model_calls_authorized": "ONE_FULL_HAWOR_SESSION_ONLY_WHEN_EXACT_PACKET_ACTIVE",
            "training_steps_authorized": 0,
            "code_closure": [artifact_ref(REPO_ROOT / path)
                             for path in executor.ROUTE_CLOSURE_RELATIVE],
        }
        return {"special_status_contracts": {executor.ROUTE_KEY: route}}

    def test_new_packet_has_independent_task_and_attempt(self) -> None:
        packet = self.packet
        self.assertEqual(packet["inputs"]["source_session"], "play_cards_0902_001")
        self.assertEqual(packet["config"]["frame_count"], 201)
        self.assertTrue(packet["task_id"].endswith("_v2"))
        self.assertNotEqual(packet["task_id"], FAILED_V1_RESULT.parents[2].name)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(len(packet["inputs"]["model_assets"]), 6)
        self.assertEqual(len(packet["commands"]), 1)
        self.assertIn(operation.OPERATION, packet["commands"][0])
        self.assertFalse(executor._paths(packet["task_id"])[0].exists())

    def test_worker_imports_in_frozen_py310_without_common(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TMPDIR"] = str(REPO_ROOT / "_run")
        env["XDG_CACHE_HOME"] = str(REPO_ROOT / "_run")
        python = executor.MODEL_PYTHON
        script = (
            "import sys; "
            "import chaoyang.ops.run_exact78_newraw_hawor_gpu_candidate_v2; "
            "import chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1; "
            "import chaoyang.ops.run_exact78_hawor_raw_v3; "
            "assert 'chaoyang.governance.common' not in sys.modules; "
            "print('PY310_IMPORT_SMOKE_PASS')"
        )
        completed = subprocess.run([str(python), "-B", "-c", script], cwd=REPO_ROOT,
                                   env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PY310_IMPORT_SMOKE_PASS", completed.stdout)

    def test_v2_route_requires_exact_new_operation_closure_and_authority(self) -> None:
        contract = self._contract()
        self.assertTrue(executor._route_valid(contract, self.packet))
        for field, drift in (("model_calls_authorized", "UNBOUNDED"),
                             ("output_schema", "WRONG"),
                             ("current_runnable_operation", "production")):
            altered = copy.deepcopy(contract)
            altered["special_status_contracts"][executor.ROUTE_KEY][field] = drift
            self.assertFalse(executor._route_valid(altered, self.packet), field)
        altered = copy.deepcopy(contract)
        altered["special_status_contracts"][executor.ROUTE_KEY]["code_closure"][0]["sha256"] = "0" * 64
        self.assertFalse(executor._route_valid(altered, self.packet))

    def test_no_model_import_before_gate_or_bad_binding(self) -> None:
        with patch.object(operation, "authorize_exact78_operation", side_effect=RuntimeError("GATE_DENIED")):
            with patch("chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1.run_stage") as model:
                with self.assertRaisesRegex(RuntimeError, "GATE_DENIED"):
                    operation.main(["--producer-result", str(SEALED_CPU_RESULT),
                                    "--output-root", str(REPO_ROOT / "_run/maintenance/unused")])
                model.assert_not_called()
        with patch.object(operation, "authorize_exact78_operation", return_value=None):
            with patch.object(operation, "_packet_for_worker", return_value=self.packet):
                with patch("chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1.run_stage") as model:
                    with self.assertRaisesRegex(RuntimeError, "V2_OUTPUT_ATTEMPT_DRIFT"):
                        operation.main(["--producer-result", str(SEALED_CPU_RESULT),
                                        "--output-root", str(REPO_ROOT / "_run/maintenance/unused")])
                    model.assert_not_called()

    def test_native_open_rejects_derived_read_and_external_write(self) -> None:
        packet = {
            "task_id": self.packet["task_id"],
            "inputs": {
                "source_files": {"video": {"path": "/path/to/private_data/egodata/synthetic/raw.mp4"}},
                "raw_video": {"path": "/path/to/private_data/egodata/synthetic/raw.mp4"},
                "upstream_admission_result": {"path": str(SEALED_CPU_RESULT)},
            },
        }
        with tempfile.TemporaryDirectory(prefix="newraw-hawor-v2-audit-",
                                         dir=REPO_ROOT / "_run/maintenance") as temporary:
            root = Path(temporary)
            trace = root / "trace.log"
            trace.write_text(
                'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
                'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/old_mask.npy", O_RDONLY) = 4\n'
                'openat(AT_FDCWD, "/tmp/rogue.bin", O_WRONLY|O_CREAT) = 5\n',
                encoding="utf-8")
            audit = executor._native_open_audit(packet, trace, root / "AUDIT.json")
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertTrue(any("UNDECLARED_EXTERNAL_READ" in path for path in audit["forbidden_paths"]))
        self.assertTrue(any("OUTSIDE_PROJECT_WRITE" in path for path in audit["forbidden_paths"]))

    def test_failure_terminal_never_invites_downstream_consumption(self) -> None:
        self.assertIn("Consume only sealed HaWoR", executor._next_action("PASSED"))
        for status in ("FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE"):
            self.assertNotIn("Consume", executor._next_action(status))
            self.assertIn("No sealed HaWoR", executor._next_action(status))


if __name__ == "__main__":
    unittest.main()
