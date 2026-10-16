"""CPU-only pre-registration tests for the frozen 0901 Cards001 holdout scene."""

from __future__ import annotations

import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from chaoyang.governance import execute_exact78_raw_t2_0901_cards001_frontend_v1 as t2
from chaoyang.governance.exact78_operation_gate import _command_tail
from chaoyang.ops.run_exact78_fresh_frontend_v2 import validate_spec, verify_internal_mp4


class Cards001HoldoutFrontendTests(unittest.TestCase):
    def test_preselected_cross_group_identity_and_frozen_recipe(self):
        inputs = t2.frozen_inputs()
        frozen = t2.load_json(t2.T0 / "INPUT_FREEZE.json")
        holdout = frozen["selected"][t2.SESSION.name]
        development = frozen["selected"]["get_potato_chips_0901_013"]
        self.assertEqual(holdout["role"], "holdout")
        self.assertEqual(holdout["split_claim"], "PRE_FROZEN_CROSS_GROUP_NOT_BLIND")
        self.assertNotEqual(holdout["source_group"], development["source_group"])
        self.assertEqual(inputs["frozen_development_recipe_packet"]["sha256"], t2.DEV_PACKET_SHA)
        self.assertEqual(inputs["models"], t2.load_json(t2.DEV_PACKET)["inputs"]["models"])
        self.assertEqual(inputs["source_files"]["stereo"]["sha256"], t2.EXPECTED_SHA["stereo"])

    def test_original_sbs_not_identity_mono_or_processed_is_model_input(self):
        inputs = t2.frozen_inputs()
        spec = t2.spec_value(inputs)
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v2", "--spec", str(t2.SPEC), "--phase", "all"])
        self.assertEqual((spec["frame_count"], spec["source_fps"], spec["source_index"]),
                         (645, 30, 0))
        self.assertEqual(spec["task"], "poker")
        self.assertEqual(spec["source_video"], inputs["source_files"]["stereo"])
        self.assertNotEqual(spec["source_video"], inputs["source_files"]["video"])
        self.assertEqual(spec["camera_params"], inputs["source_files"]["camera"])
        self.assertNotIn("extrinsics", t2.load_json(t2.SESSION / "camera_params.json"))
        self.assertEqual(spec["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertNotIn("camera_maps", spec)
        wrong = copy.deepcopy(spec)
        wrong["source_video"] = inputs["source_files"]["video"]
        with self.assertRaisesRegex(ValueError, "TRACKER_SOURCE_MUST_BE_ORIGINAL_STEREO"):
            validate_spec(wrong)
        self.assertNotIn("processed", str(spec).lower())
        self.assertNotIn("clean", str(spec).lower())

    def test_packet_is_independent_one_lease_and_no_holdout_tuning(self):
        inputs = t2.frozen_inputs()
        spec = t2.spec_value(inputs)
        packet = t2.packet_value(inputs, t2.virtual_json_ref(t2.SPEC, spec))
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["config"]["split_claim"], "PRE_FROZEN_CROSS_GROUP_NOT_BLIND")
        self.assertIs(packet["config"]["tuning_from_holdout"], False)
        self.assertEqual(packet["config"]["recipe_sha256"], t2.DEV_PACKET_SHA)
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertIn("_run/current/GPU_LEASE.json", packet["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", packet["write_set"])
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertIn(str(t2.DEV_PACKET.relative_to(t2.REPO_ROOT)), packet["read_set"])
        development_result = (t2.REPO_ROOT / "_run/current" /
            "exact78_raw_to_robot_t2_chips013_frontend_v1_20260930/attempts/attempt_0001/RESULT.json")
        self.assertNotIn(str(development_result), str(packet["inputs"]))
        self.assertTrue(all("0916" not in path for path in packet["write_set"]))

    def test_identity_or_recipe_drift_rejected(self):
        with patch.object(t2, "DEV_PACKET_SHA", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "DEVELOPMENT_RECIPE_PACKET_SHA_DRIFT"):
                t2.frozen_inputs()
        real_load = t2.load_json
        frozen = copy.deepcopy(real_load(t2.T0 / "INPUT_FREEZE.json"))
        frozen["selected"][t2.SESSION.name]["role"] = "development"
        with patch.object(t2, "load_json", side_effect=lambda path: (
                frozen if path == t2.T0 / "INPUT_FREEZE.json" else real_load(path))):
            with self.assertRaisesRegex(RuntimeError, "CARDS001_T0_IDENTITY_DRIFT"):
                t2.frozen_inputs()
        inputs = t2.frozen_inputs()
        original_ref = t2.artifact_ref
        def altered_ref(path):
            ref = original_ref(path)
            if str(path).endswith("/run_exact78_fresh_frontend_v2.py"):
                return {**ref, "sha256": "0" * 64}
            return ref
        with patch.object(t2, "artifact_ref", side_effect=altered_ref):
            with self.assertRaisesRegex(RuntimeError, "HOLDOUT_RECIPE_CODE_DIFFER_FROM_DEVELOPMENT"):
                t2.packet_value(inputs, t2.virtual_json_ref(t2.SPEC, t2.spec_value(inputs)))

    def test_active_task_blocks_registration(self):
        with patch.object(t2, "load_json", return_value={"tasks": [{"status": "RUNNING"}]}):
            self.assertEqual(t2.readiness(), (False, "ACTIVE_TASK_CONFLICT"))

    def test_check_only_does_not_publish_or_start_gpu(self):
        with patch.object(t2, "frozen_inputs", return_value={}), \
             patch.object(t2, "spec_value", return_value={}), \
             patch.object(t2, "packet_value", return_value={"inputs": {"frontend_spec": {"sha256": "x"}},
                                                            "run_signature": "y"}), \
             patch.object(t2, "readiness", return_value=(False, "ACTIVE_TASK_CONFLICT")), \
             patch.object(t2, "register") as register, \
             patch.object(t2, "run_worker") as worker, \
             patch.object(t2, "virtual_json_ref", return_value={"sha256": "x"}), \
             patch("sys.argv", ["executor", "--check-only"]), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(t2.main(), 0)
        register.assert_not_called()
        worker.assert_not_called()
        self.assertIn("DRY_RUN_NOT_ROUTABLE", stdout.getvalue())

    def test_v2_mp4_accepts_exact_645_at_30_and_rejects_wrong_count(self):
        with tempfile.TemporaryDirectory(dir=t2.REPO_ROOT / "_run/current") as temporary:
            path = Path(temporary) / "internal.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     30, (960, 720))
            self.assertTrue(writer.isOpened())
            frame = np.zeros((720, 960, 3), dtype=np.uint8)
            try:
                for _ in range(645):
                    writer.write(frame)
            finally:
                writer.release()
            evidence = verify_internal_mp4(path, 645, 30)
            self.assertEqual((evidence["decoded_frames"], evidence["fps"],
                              evidence["duration_seconds"]), (645, 30, 21.5))
            with self.assertRaisesRegex(RuntimeError, "PROPAINTER_VIDEO_TIME_OR_DOMAIN"):
                verify_internal_mp4(path, 644, 30)


if __name__ == "__main__":
    unittest.main()
