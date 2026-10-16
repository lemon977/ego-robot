"""Read-only checks for the v5 acceptance preparation gate."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_cli_v5_four_session_acceptance_candidate_v1 as accept


class V5AcceptanceTests(unittest.TestCase):
    def test_exact_four_rows_and_source_sha(self) -> None:
        rows, ref = accept._source_rows()
        self.assertEqual(tuple(row["session_id"] for row in rows), accept.EXPECTED_IDS)
        self.assertEqual(ref["sha256"], accept.cli.MANIFEST_SHA)
        self.assertEqual(rows[3]["submission_input_sha256"],
                         "fbd99999a28849a3f47ccedf43912f955afa720c2ae94ccaa93c25c47a07ae41")

    def test_sealed_003_attaches_with_limited_claim(self) -> None:
        rows, _ = accept._source_rows()
        result = accept._sealed_003(rows[2])
        self.assertTrue(result["sealed_attachment"])
        self.assertFalse(result["visual_qualified"])
        self.assertFalse(result["training_eligible"])

    def test_fresh_004_rejects_nonready_decision(self) -> None:
        rows, _ = accept._source_rows()
        with patch.object(accept.cpu, "check_one", return_value={
                "status": "READY", "node": "ROBOT_V1",
                "registration_started": False, "model_calls": 0,
                "code_acceptance": "NOT_SEALED_CODE_PREP_ONLY"}):
            with self.assertRaisesRegex(RuntimeError, "V5_004_CPU_CHECK_ONLY_NOT_READY"):
                accept._fresh_004(rows[3])

    def test_active_writer_skips_fresh_cpu_probe(self) -> None:
        rows = [{"session_id": name, "source_root": "/unused",
                 "submission_input_sha256": "a" * 64, "source_files": {}}
                for name in accept.EXPECTED_IDS]
        rows[2]["sealed_robot_terminal"] = {}
        with patch.object(accept, "_source_rows", return_value=(rows, {})), \
             patch.object(accept, "_sealed_003", return_value={}), \
             patch.object(accept, "load_json", return_value={"next_task": {"task_id": "other"},
                                                           "tasks": [{"task_id": "other", "status": "RUNNING"}]}), \
             patch.object(accept, "_fresh_004") as fresh, \
             patch.object(accept, "artifact_ref", return_value={}), \
             patch.object(accept, "_published_operations", return_value={}):
            result = accept.inspect()
        fresh.assert_not_called()
        self.assertEqual(result["status"], "BLOCKED_ACTIVE_WRITER_AND_REAL_SUBMIT_PATH_ABSENT")
        self.assertFalse(result["registration_started"])

    def test_register_fails_before_any_inspection_or_write(self) -> None:
        with patch.object(accept, "inspect") as inspect:
            with self.assertRaisesRegex(RuntimeError, "V5_REAL_SUBMIT_PATH_ABSENT"):
                accept.main(["--register-and-run"])
        inspect.assert_not_called()

    def test_check_only_never_registers(self) -> None:
        with patch.object(accept, "inspect", return_value={"status": "BLOCKED_REAL_SUBMIT_PATH_ABSENT",
                                                             "registration_started": False}), \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(accept.main(["--check-only"]), 0)
        self.assertIn('"registration_started": false', stream.getvalue())


if __name__ == "__main__":
    unittest.main()
