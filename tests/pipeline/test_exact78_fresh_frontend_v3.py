"""CPU-only closure, adapter, and process-local patch tests for candidate v3."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from chaoyang.ops import run_exact78_fresh_frontend_v2 as v2
from chaoyang.ops import run_exact78_fresh_frontend_v3 as v3


class FrontendV3Tests(unittest.TestCase):
    def test_pinned_v2_and_acceptance_code_are_exact(self):
        v3.verify_pinned_closure()

    def test_pin_drift_stops_before_v2_main(self):
        with patch.object(v3, "PINNED_V2_SHA256", "0" * 64), \
             patch.object(v2, "main") as v2_main:
            with self.assertRaisesRegex(RuntimeError, "V3_PINNED_CODE_SHA_DRIFT:FRONTEND_V2"):
                v3.main()
            v2_main.assert_not_called()

    def test_acceptance_pin_drift_stops_before_v2_main(self):
        with patch.object(v3, "PINNED_ACCEPTANCE_SHA256", "0" * 64), \
             patch.object(v2, "main") as v2_main:
            with self.assertRaisesRegex(RuntimeError, "V3_PINNED_CODE_SHA_DRIFT:MP4_ACCEPTANCE"):
                v3.main()
            v2_main.assert_not_called()

    def test_adapter_preserves_exact_v2_return_fields(self):
        expected = {"artifact": {"path": "video.mp4", "sha256": "a" * 64, "bytes": 1},
                    "decoded_frames": 131}
        with patch.object(v3.mp4_acceptance, "verify_propainter_mp4", return_value=expected) as verify:
            actual = v3.verify_internal_mp4_v3(Path("video.mp4"), 131, 30)
        verify.assert_called_once_with(Path("video.mp4"), 131, 30)
        self.assertEqual(actual, {
            "artifact": expected["artifact"], "decoded_frames": 131, "fps": 30,
            "duration_seconds": 131 / 30, "dimensions_wh": [960, 720],
            "pts_mode": "CFR_INDEX_ALIGNED",
        })

    def test_only_v2_verifier_is_replaced_and_then_restored(self):
        original = v2.verify_internal_mp4
        original_clean = v2.clean
        original_map = v2.map_stage
        observed = []

        def fake_main():
            observed.append(v2.verify_internal_mp4)
            self.assertIs(v2.clean, original_clean)
            self.assertIs(v2.map_stage, original_map)
            return 0

        with patch.object(v3, "verify_pinned_closure"), patch.object(v2, "main", side_effect=fake_main):
            self.assertEqual(v3.main(), 0)
        self.assertEqual(observed, [v3.verify_internal_mp4_v3])
        self.assertIs(v2.verify_internal_mp4, original)

    def test_patch_restored_if_v2_main_fails(self):
        original = v2.verify_internal_mp4
        with patch.object(v3, "verify_pinned_closure"), \
             patch.object(v2, "main", side_effect=RuntimeError("packet gate rejected")):
            with self.assertRaisesRegex(RuntimeError, "packet gate rejected"):
                v3.main()
        self.assertIs(v2.verify_internal_mp4, original)

    def test_preexisting_patch_is_rejected(self):
        with patch.object(v3, "verify_pinned_closure"), \
             patch.object(v2, "verify_internal_mp4", v3.verify_internal_mp4_v3), \
             patch.object(v2, "main") as v2_main:
            with self.assertRaisesRegex(RuntimeError, "V3_V2_VERIFIER_ALREADY_PATCHED"):
                v3.main()
            v2_main.assert_not_called()

    def test_module_entry_is_a_separate_process_and_argparse_rejects_missing_spec(self):
        original = v2.verify_internal_mp4
        result = subprocess.run(
            [sys.executable, "-B", "-m", "chaoyang.ops.run_exact78_fresh_frontend_v3",
             "--phase", "clean"], capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--spec", result.stderr)
        self.assertIs(v2.verify_internal_mp4, original)

    def test_mp4_rejection_prevents_adapter_result(self):
        with patch.object(v3.mp4_acceptance, "verify_propainter_mp4",
                          side_effect=v3.mp4_acceptance.Mp4AcceptanceError("FRAME_PTS:75")):
            with self.assertRaisesRegex(v3.mp4_acceptance.Mp4AcceptanceError, "FRAME_PTS:75"):
                v3.verify_internal_mp4_v3(Path("video.mp4"), 131, 30)


if __name__ == "__main__":
    unittest.main()
