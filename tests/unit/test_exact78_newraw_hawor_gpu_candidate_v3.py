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
from chaoyang.governance import execute_exact78_newraw_hawor_gpu_candidate_v3 as executor
from chaoyang.ops import run_exact78_newraw_hawor_gpu_candidate_v3 as operation


SEALED_CPU_RESULT = (REPO_ROOT / "_run/current/"
    "exact78_newraw_play_cards_0902_001_9541579d8de29395_v1/"
    "attempts/attempt_0001/RESULT.json")
FAILED_V1_RESULT = (REPO_ROOT / "_run/current/"
    "exact78_newraw_hawor_play_cards_0902_001_9541579d8de29395_v1/"
    "attempts/attempt_0001/RESULT.json")
FAILED_V2_RESULT = (REPO_ROOT / "_run/current/"
    "exact78_newraw_hawor_play_cards_0902_001_9541579d8de29395_v2/"
    "attempts/attempt_0001/RESULT.json")


class NewRawHaWoRGpuV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.packet = executor.packet_value(SEALED_CPU_RESULT)

    def _contract(self) -> dict:
        route = {
            "execution_status": "CANDIDATE_FENCED_ONE_GPU_ONE_SESSION",
            "candidate_operation": executor.OPERATION,
            "task_packet_required": True,
            "current_runnable_operation": None,
            "output_schema": "exact78-newraw-hawor-gpu-execution-result-v3",
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
        self.assertTrue(packet["task_id"].endswith("_v3"))
        self.assertNotEqual(packet["task_id"], FAILED_V1_RESULT.parents[2].name)
        self.assertNotEqual(packet["task_id"], FAILED_V2_RESULT.parents[2].name)
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
            "import chaoyang.ops.run_exact78_newraw_hawor_gpu_candidate_v3; "
            "import chaoyang.pipeline.exact78_newraw_hawor_stage_candidate_v1; "
            "import chaoyang.ops.run_exact78_hawor_raw_v3; "
            "assert 'chaoyang.governance.common' not in sys.modules; "
            "print('PY310_IMPORT_SMOKE_PASS')"
        )
        completed = subprocess.run([str(python), "-B", "-c", script], cwd=REPO_ROOT,
                                   env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PY310_IMPORT_SMOKE_PASS", completed.stdout)

    def test_v3_route_requires_exact_new_operation_closure_and_authority(self) -> None:
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
                    with self.assertRaisesRegex(RuntimeError, "V3_OUTPUT_ATTEMPT_DRIFT"):
                        operation.main(["--producer-result", str(SEALED_CPU_RESULT),
                                        "--output-root", str(REPO_ROOT / "_run/maintenance/unused")])
                    model.assert_not_called()

    def _audit(self, lines: str) -> dict:
        packet = {
            "task_id": self.packet["task_id"],
            "inputs": {
                "source_files": {"video": {"path": "/path/to/private_data/egodata/synthetic/raw.mp4"}},
                "raw_video": {"path": "/path/to/private_data/egodata/synthetic/raw.mp4"},
                "upstream_admission_result": {"path": str(SEALED_CPU_RESULT)},
            },
        }
        with tempfile.TemporaryDirectory(prefix="newraw-hawor-v3-audit-",
                                         dir=REPO_ROOT / "_run/maintenance") as temporary:
            root = Path(temporary)
            trace = root / "trace.log"
            trace.write_text(lines, encoding="utf-8")
            return executor._native_open_audit(packet, trace, root / "AUDIT.json")

    def test_native_open_rejects_derived_read_and_successful_persistent_writable_open(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/old_mask.npy", O_RDONLY) = 4\n'
            'openat(AT_FDCWD, "/root/.nv/ComputeCache/index", O_RDWR) = 5\n')
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertTrue(any("UNDECLARED_EXTERNAL_READ" in path for path in audit["forbidden_paths"]))
        self.assertTrue(any("OUTSIDE_PROJECT_PERSISTENT_WRITABLE_OPEN" in path
                            for path in audit["forbidden_paths"]))
        self.assertEqual(audit["write_open_counts"]["successful_persistent_external"], 1)

    def test_failed_opens_are_disclosed_and_exact_kernel_ipc_is_not_persistent(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/cpfs_infra/vendor/libuuid/clock.txt", O_RDWR|O_CREAT) = -1 ENOENT\n'
            'openat(AT_FDCWD, "/dev/shm/cuda_injection_path_shm", O_RDWR) = -1 ENOENT\n'
            'openat(AT_FDCWD, "/dev/shm/sem.8fLqBp", O_RDWR|O_CREAT|O_EXCL) = 4\n'
            'openat(AT_FDCWD, "/proc/self/task/12345/comm", O_WRONLY|O_TRUNC) = 5\n'
            'openat(AT_FDCWD, "/dev/nvidia0", O_RDWR) = 6\n')
        self.assertEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertEqual(audit["write_open_counts"]["failed_external_attempts"], 2)
        self.assertEqual(audit["write_open_counts"]["successful_transient_kernel_ipc"], 2)
        self.assertEqual(audit["write_open_counts"]["successful_persistent_external"], 0)
        self.assertEqual(audit["write_open_counts"]["successful_runtime_device"], 1)
        self.assertIn("actual write syscalls are not proven", audit["scope_limit"])

    def test_mnt_data_write_attempt_fails_even_when_denied(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_WRONLY) = -1 EACCES\n')
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertTrue(any("EXTERNAL_DATA_WRITE_ATTEMPT" in path
                            for path in audit["forbidden_paths"]))

    def test_other_dev_shm_or_relative_write_is_not_exempt(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/dev/shm/other-state", O_RDWR|O_CREAT) = 4\n'
            'openat(AT_FDCWD, "/dev/nvidiaevil", O_RDWR) = 6\n'
            'openat(AT_FDCWD, "../escaped.bin", O_WRONLY|O_CREAT) = 5\n')
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertEqual(audit["write_open_counts"]["successful_persistent_external"], 2)
        self.assertTrue(any("UNRESOLVED_RELATIVE_WRITABLE_OPEN" in path
                            for path in audit["forbidden_paths"]))

    def test_unfinished_external_writable_open_fails_closed(self) -> None:
        audit = self._audit(
            'openat(AT_FDCWD, "/path/to/private_data/egodata/synthetic/raw.mp4", O_RDONLY) = 3\n'
            'openat(AT_FDCWD, "/root/.nv/ComputeCache/index", O_RDWR <unfinished ...>\n')
        self.assertNotEqual(audit["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertEqual(audit["write_open_counts"]["unresolved_external_returns"], 1)
        self.assertTrue(any("UNRESOLVED_EXTERNAL_WRITABLE_OPEN_RETURN" in path
                            for path in audit["forbidden_paths"]))

    def test_worker_cuda_cache_path_is_project_attempt_local(self) -> None:
        with tempfile.TemporaryDirectory(prefix="newraw-hawor-v3-env-",
                                         dir=REPO_ROOT / "_run/maintenance") as temporary:
            env = executor._worker_environment(Path(temporary))
            cache = Path(env["CUDA_CACHE_PATH"])
            self.assertTrue(cache.is_dir())
            self.assertTrue(cache.is_relative_to(Path(temporary)))
            self.assertTrue(cache.is_relative_to(REPO_ROOT))

    def test_failure_terminal_never_invites_downstream_consumption(self) -> None:
        self.assertIn("Consume only sealed HaWoR", executor._next_action("PASSED"))
        for status in ("FAILED_RUNTIME_FINAL", "BLOCKED_RESOURCE"):
            self.assertNotIn("Consume", executor._next_action(status))
            self.assertIn("No sealed HaWoR", executor._next_action(status))


if __name__ == "__main__":
    unittest.main()
