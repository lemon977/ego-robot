"""CPU-only 0901 Chips013 packet/source checks; never register or invoke GPU."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_raw_t2_chips013_frontend_v1 as t2
from chaoyang.governance.exact78_operation_gate import _command_tail
from chaoyang.ops.run_exact78_fresh_frontend_v2 import _timing_rows, validate_spec


class Chips013T2ExecutorTests(unittest.TestCase):
    @staticmethod
    def sample_inputs():
        frozen = t2.load_json(t2.T0 / "INPUT_FREEZE.json")
        files = frozen["selected"][t2.SESSION.name]["files"]
        return {
            "source_files": {key: files[key] for key in ("stereo", "video", "camera", "clip_manifest")},
            "map_generator_code": t2.artifact_ref(t2.REPO_ROOT /
                "vendor/FoundationStereo/scripts/pico_stereo_depth.py"),
            "t0_frame_time_map": t2.artifact_ref(t2.T0 / "FRAME_TIME_MAP.json"),
            "models": {"sam_checkpoint": frozen["models"]["sam31_checkpoint"],
                       "propainter_weights": {
                           "raft-things.pth": frozen["models"]["raft_weights"],
                           "recurrent_flow_completion.pth": frozen["models"]["flow_weights"],
                           "ProPainter.pth": frozen["models"]["propainter_weights"]}},
            "propainter_entry": t2.artifact_ref(t2.REPO_ROOT /
                "vendor/ProPainter/inference_propainter.py"),
        }

    def test_command_is_one_full_v2_candidate_and_native_fps(self):
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v2", "--spec", str(t2.SPEC), "--phase", "all",
        ])
        self.assertEqual(t2.STAGES, ("MAP", "RAW", "MASK", "PREPARE", "CLEAN"))
        spec = t2.spec_value(self.sample_inputs())
        self.assertEqual((spec["frame_count"], spec["source_fps"]), (870, 25))
        self.assertEqual(spec["source_family"], "TRACKER_0901")
        self.assertEqual(spec["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertNotIn("camera_maps", spec)

    def test_original_stereo_is_producer_mono_only_identity(self):
        inputs = self.sample_inputs()
        spec = t2.spec_value(inputs)
        self.assertEqual(spec["source_video"], inputs["source_files"]["stereo"])
        self.assertNotEqual(spec["source_video"], inputs["source_files"]["video"])
        self.assertEqual(inputs["source_files"]["stereo"]["sha256"], t2.EXPECTED_SHA["stereo"])
        self.assertEqual(inputs["source_files"]["video"]["sha256"], t2.EXPECTED_SHA["video"])
        self.assertNotIn("preprocess", str(spec))
        spec["source_video"] = inputs["source_files"]["video"]
        with self.assertRaisesRegex(ValueError, "TRACKER_SOURCE_MUST_BE_ORIGINAL_STEREO"):
            validate_spec(spec)

    def test_absent_extrinsics_do_not_block_visual_map_spec(self):
        camera = t2.load_json(t2.SESSION / "camera_params.json")
        self.assertNotIn("extrinsics", camera)
        validate_spec(t2.spec_value(self.sample_inputs()))

    def test_t0_editor_index_cannot_be_promoted_to_sync(self):
        spec = t2.spec_value(self.sample_inputs())
        rows, mode = _timing_rows(spec)
        self.assertEqual((len(rows), mode), (870, "TIMEBASE_UNVERIFIED"))
        self.assertEqual([row["decode_display_index"] for row in rows], list(range(870)))
        self.assertTrue(all(row["capture_frame_id"] is None for row in rows))
        self.assertTrue(all(row["capture_to_decode_alignment"] == "CLIP_INDEX_ONLY" for row in rows))

    def test_packet_preview_has_separate_write_set_and_one_writer(self):
        fake_ref = lambda path: {"path": str(Path(path)), "bytes": 0, "sha256": "a" * 64}
        with patch.object(t2, "artifact_ref", side_effect=fake_ref), \
             patch.object(t2, "process_identity", return_value={"start_ticks": 77}):
            packet = t2.packet_value({"source_files": {}}, fake_ref(t2.SPEC))
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["config"]["gpu_lease_count"], 1)
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertEqual(packet["config"]["geometry"], "CALIBRATION_EXTRINSICS_ABSENT")
        self.assertEqual(packet["config"]["mono_clip_role"], "IDENTITY_ONLY_NOT_MODEL_INPUT")
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertNotIn("0916", str(packet["write_set"]))

    def test_t0_sha_drift_is_fail_closed_before_registration(self):
        with patch.object(t2, "T0_SHA", {t2.T0 / "INPUT_FREEZE.json": "0" * 64}):
            with self.assertRaisesRegex(RuntimeError, "T0_SHA_DRIFT"):
                t2.frozen_inputs()


if __name__ == "__main__":
    unittest.main()
