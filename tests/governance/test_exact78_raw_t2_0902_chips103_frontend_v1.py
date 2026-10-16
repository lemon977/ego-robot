"""CPU-only contract tests for one original-stereo Chips103 frontend candidate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_raw_t2_0902_chips103_frontend_v1 as t2
from chaoyang.governance.exact78_operation_gate import _command_tail
from chaoyang.ops.run_exact78_fresh_frontend_v2 import _timing_rows, validate_spec


class Chips103FrontendPacketTests(unittest.TestCase):
    def inputs(self):
        frozen = t2.load_json(t2.T0 / "INPUT_FREEZE.json")
        row = frozen["selected"][t2.SESSION.name]
        return {"source_files": {key: row["files"][key] for key in
                                 ("stereo", "camera", "clip_manifest")},
                "map_generator_code": t2.artifact_ref(t2.REPO_ROOT /
                    "vendor/FoundationStereo/scripts/pico_stereo_depth.py"),
                "t0_frame_time_map": t2.artifact_ref(t2.T0 / "FRAME_TIME_MAP.json"),
                "models": {"sam_checkpoint": frozen["models"]["sam31_checkpoint"],
                           "propainter_weights": {
                               "raft-things.pth": frozen["models"]["raft_weights"],
                               "recurrent_flow_completion.pth": frozen["models"]["flow_weights"],
                               "ProPainter.pth": frozen["models"]["propainter_weights"]}},
                "propainter_entry": t2.artifact_ref(t2.REPO_ROOT /
                    "vendor/ProPainter/inference_propainter.py")}

    def test_original_stereo_same_session_camera_and_284_frames(self):
        spec = t2.spec_value(self.inputs())
        self.assertEqual(spec["task"], "chips")
        self.assertEqual((spec["frame_count"], spec["source_fps"]), (284, 30))
        self.assertEqual(spec["source_index"], 0)
        self.assertEqual(Path(spec["source_video"]["path"]).parent,
                         t2.SESSION / "source_stereo")
        self.assertEqual(spec["camera_params"]["path"], str(t2.SESSION / "camera_params.json"))
        self.assertEqual(spec["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertNotIn("camera_maps", spec)
        rows, mode = _timing_rows(spec)
        self.assertEqual((len(rows), mode), (284, "TIMEBASE_UNVERIFIED"))
        self.assertEqual([row["decode_display_index"] for row in rows], list(range(284)))

    def test_old_mono_or_old_map_are_forbidden(self):
        spec = t2.spec_value(self.inputs())
        frozen = t2.load_json(t2.T0 / "INPUT_FREEZE.json")
        mono = frozen["selected"][t2.SESSION.name]["files"]["video"]
        wrong = copy.deepcopy(spec)
        wrong["source_video"] = mono
        with self.assertRaisesRegex(ValueError, "TRACKER_SOURCE_MUST_BE_ORIGINAL_STEREO"):
            validate_spec(wrong)
        wrong = copy.deepcopy(spec)
        wrong["camera_maps"] = {"path": "old_chips103_map.npz"}
        with self.assertRaisesRegex(ValueError, "TRACKER_MAP_MUST_BE_GENERATED_IN_STAGE"):
            validate_spec(wrong)

    def test_packet_has_one_command_gpu_lease_and_independent_write_set(self):
        fake_ref = lambda path: {"path": str(Path(path)), "bytes": 0, "sha256": "a" * 64}
        with patch.object(t2, "artifact_ref", side_effect=fake_ref), \
             patch.object(t2, "process_identity", return_value={"start_ticks": 77}):
            packet = t2.packet_value(self.inputs(), fake_ref(t2.SPEC))
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v2", "--spec", str(t2.SPEC), "--phase", "all"])
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertNotIn("cards042", str(packet["write_set"]).lower())
        self.assertNotIn("0916", str(packet["write_set"]))
        self.assertIn("_run/current/GPU_LEASE.json", packet["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", packet["write_set"])
        self.assertIn("candidate/CAMERA_MAPS.npz", packet["required_outputs"])
        self.assertIn("candidate/propainter_model/frames/inpaint_out.mp4", packet["required_outputs"])
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(set(packet["inputs"]["source_files"]),
                         {"stereo", "camera", "clip_manifest"})
        self.assertEqual(len(packet["read_set"]), 8)

    def test_finish_rechecks_both_internal_videos_at_native_fps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = {"artifact": {"sha256": "a"}}
            second = {"artifact": {"sha256": "b"}}
            (root / "CLEAN.json").write_text(json.dumps({"internal_videos": {
                "masked_in.mp4": first, "inpaint_out.mp4": second}}))
            def checked_video(path, frames, fps):
                self.assertEqual((frames, fps), (284, 30))
                return first if path.name == "masked_in.mp4" else second
            with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.verify_internal_mp4",
                       side_effect=checked_video) as verify:
                t2.verify_internal_video_records(root, 284, 30)
                self.assertEqual(verify.call_count, 2)
                verify.side_effect = lambda *_: {"artifact": {"sha256": "drift"}}
                with self.assertRaisesRegex(RuntimeError, "INTERNAL_VIDEO_REF_DRIFT"):
                    t2.verify_internal_video_records(root, 284, 30)


if __name__ == "__main__":
    unittest.main()
