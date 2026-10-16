"""CPU-only frozen-role/frame/fps and fail-closed wrapper tests."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.governance import execute_exact78_raw_t2_0902_cards042_v1 as cards042
from chaoyang.governance import execute_exact78_raw_t2_0902_chips103_v1 as chips103
from chaoyang.governance import execute_exact78_raw_t2_0901_cards001_holdout_v1 as cards001
from chaoyang.governance import execute_exact78_raw_t2_0915_cards002_holdout_v1 as cards002
from chaoyang.ops.run_exact78_raw_to_robot_v3 import TARGETS


MODULES = (cards042, chips103, cards001, cards002)
EXPECTED = {
    "play_cards_0902_042": ("development", 171, 30, "tracker_session_left_RGB",
                            "TRACKER_CLIP_METADATA_ONLY"),
    "get_potato_chips_0902_103": ("development", 284, 30, "tracker_session_left_RGB",
                                  "TRACKER_CLIP_METADATA_ONLY"),
    "play_cards_0901_001": ("holdout", 645, 30, "tracker_session_left_RGB",
                            "TRACKER_CLIP_METADATA_ONLY"),
    "play_cards_0915_002": ("holdout", 131, 30, "left_sourceIndex_1",
                            "TIMEBASE_UNVERIFIED"),
}


class RemainingWrapperTests(unittest.TestCase):
    def test_four_independent_task_and_receipt_identities(self) -> None:
        self.assertEqual(len({module.TARGET.task_id for module in MODULES}), 4)
        self.assertEqual(len({module.TARGET.receipt_name for module in MODULES}), 4)
        self.assertEqual(len({module.TARGET.output for module in MODULES}), 4)

    def test_roles_frames_fps_eye_and_time_match_t0(self) -> None:
        root = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
        frozen = json.loads((root / "INPUT_FREEZE.json").read_text(encoding="utf-8"))["selected"]
        mapped = json.loads((root / "FRAME_TIME_MAP.json").read_text(encoding="utf-8"))["sessions"]
        for module in MODULES:
            with self.subTest(session=module.TARGET.session_id):
                target = module.TARGET
                role, frames, fps, eye, time_mode = EXPECTED[target.session_id]
                self.assertEqual(module.ROLE, role)
                self.assertEqual(TARGETS[target.session_id]["role"], role)
                self.assertEqual(frozen[target.session_id]["role"], role)
                self.assertEqual(frozen[target.session_id]["source_path"], str(target.session))
                self.assertEqual(target.frame_count, mapped[target.session_id]["decoded_frames"])
                self.assertEqual(target.frame_count, frames)
                self.assertEqual(target.fps_nominal, fps)
                self.assertEqual(mapped[target.session_id]["nominal_fps_from_capture_metadata"], fps)
                self.assertEqual(target.physical_eye, eye)
                self.assertEqual(mapped[target.session_id]["frame_rows"][0]["sbs_physical_eye"], eye)
                self.assertEqual(target.time_mode, time_mode)
                self.assertEqual(mapped[target.session_id]["time_mode"], time_mode)

    def test_every_wrapper_pins_same_pre_holdout_recipe(self) -> None:
        expected = MODULES[0].RECIPE_SHA
        for module in MODULES:
            with self.subTest(session=module.TARGET.session_id):
                self.assertEqual(module.RECIPE_SHA, expected)
                for relative, sha in module.RECIPE_SHA.items():
                    self.assertEqual(artifact_ref(REPO_ROOT / relative)["sha256"], sha)

    def test_recipe_sha_drift_stops_before_packet_builder(self) -> None:
        for module in MODULES:
            with self.subTest(session=module.TARGET.session_id):
                wrong = dict(module.RECIPE_SHA)
                first = next(iter(wrong))
                wrong[first] = "0" * 64
                with mock.patch.object(module, "RECIPE_SHA", wrong):
                    with mock.patch.object(module, "main_for") as builder:
                        with self.assertRaisesRegex(RuntimeError, "FROZEN_RECIPE_SHA_DRIFT"):
                            module.main()
                        builder.assert_not_called()

    def test_holdout_role_spoof_stops_before_packet_builder(self) -> None:
        for module in (cards001, cards002):
            with self.subTest(session=module.TARGET.session_id):
                wrong = dict(TARGETS[module.TARGET.session_id], role="development")
                with mock.patch.dict(TARGETS, {module.TARGET.session_id: wrong}):
                    with mock.patch.object(module, "main_for") as builder:
                        with self.assertRaisesRegex(RuntimeError, "T0_SOURCE_ROLE_DRIFT"):
                            module.main()
                        builder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
