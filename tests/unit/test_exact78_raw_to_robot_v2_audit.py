"""Model-free tests for the T1-B native source-open boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from chaoyang.ops.run_exact78_raw_to_robot_v2 import _audit_native_source_opens
from chaoyang.governance.exact78_cold_start_guard_v1 import PROJECT_ROOT


class NativeOpenAuditTests(unittest.TestCase):
    VIDEO = "/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0902/playing_cards/play_cards_0902_042/rgb.mp4"

    def _audit(self, extra: str) -> dict:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "_run/current") as raw:
            root = Path(raw)
            trace = root / "trace.log"
            trace.write_text(
                f'123 openat(AT_FDCWD, "{self.VIDEO}", O_RDONLY|O_CLOEXEC) = 5\n'
                + extra, encoding="utf-8")
            source = {"kind": "TRACKER_CLIP_RGB", "files": {
                "video": {"path": self.VIDEO, "bytes": 1, "sha256": "a" * 64}}}
            return _audit_native_source_opens(trace, source, root / "audit.json")

    def test_allowed_source_and_own_output_write(self) -> None:
        own = PROJECT_ROOT / "_run/current/exact78_raw_to_robot_t1b_cards042_v1_20260930/attempts/attempt_0001/candidate/hawor/result.json"
        result = self._audit(f'123 openat(AT_FDCWD, "{own}", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 6\n')
        self.assertEqual(result["status"], "PASS_NATIVE_OPEN_SCOPE")
        self.assertEqual(result["forbidden_paths"], [])

    def test_old_result_read_is_rejected(self) -> None:
        old = PROJECT_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
        result = self._audit(f'123 openat(AT_FDCWD, "{old}", O_RDONLY|O_CLOEXEC) = 6\n')
        self.assertEqual(result["status"], "FAILED_OR_INCOMPLETE_NATIVE_OPEN_SCOPE")
        self.assertIn(str(old), result["forbidden_paths"])

    def test_external_write_is_rejected(self) -> None:
        result = self._audit(f'123 openat(AT_FDCWD, "{self.VIDEO}", O_WRONLY|O_TRUNC) = 6\n')
        self.assertEqual(result["status"], "FAILED_OR_INCOMPLETE_NATIVE_OPEN_SCOPE")
        self.assertIn("EXTERNAL_WRITE_OPEN:" + self.VIDEO, result["forbidden_paths"])


if __name__ == "__main__":
    unittest.main()
