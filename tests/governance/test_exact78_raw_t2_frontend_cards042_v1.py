"""CPU-only dry-run contract checks; never register or invoke model workers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_raw_t2_cards042_frontend_v1 as t2
from chaoyang.governance.exact78_operation_gate import _command_tail


class Cards042T2ExecutorTests(unittest.TestCase):
    def test_exact_one_cli_command_and_one_gpu_lease(self):
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v2", "--spec", str(t2.SPEC), "--phase", "all",
        ])
        self.assertEqual(t2.STAGES, ("MAP", "RAW", "MASK", "PREPARE", "CLEAN"))

    def test_spec_is_full_original_stereo_and_same_session_map_rule(self):
        frozen = t2.load_json(t2.T0 / "INPUT_FREEZE.json")
        row = frozen["selected"]["play_cards_0902_042"]
        inputs = {
            "source_files": {key: row["files"][key] for key in ("stereo", "camera", "clip_manifest")},
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
        spec = t2.spec_value(inputs)
        self.assertEqual(spec["source_video"], row["files"]["stereo"])
        self.assertEqual(spec["camera_params"], row["files"]["camera"])
        self.assertEqual(spec["source_index"], 0)
        self.assertEqual((spec["frame_count"], spec["source_fps"]), (171, 30))
        self.assertEqual(spec["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertNotIn("camera_maps", spec)

    def test_packet_preview_is_finite_and_keeps_one_writer(self):
        fake_ref = lambda path: {"path": str(Path(path)), "bytes": 0, "sha256": "a" * 64}
        with patch.object(t2, "artifact_ref", side_effect=fake_ref), \
             patch.object(t2, "process_identity", return_value={"start_ticks": 77}):
            packet = t2.packet_value({"source_files": {}}, fake_ref(t2.SPEC), "ABSENT")
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["config"]["gpu_lease_count"], 1)
        self.assertEqual(packet["config"]["stages"], list(t2.STAGES))
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertIn("_run/current/GPU_LEASE.json", packet["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", packet["write_set"])
        self.assertIn("candidate/propainter_model/frames/masked_in.mp4", packet["required_outputs"])
        self.assertIn("candidate/propainter_model/frames/inpaint_out.mp4", packet["required_outputs"])
        self.assertNotIn("0916", str(packet["write_set"]))
        self.assertEqual(len(packet["read_set"]), 8)

    def test_active_t1b_prevents_registration(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(t2, "T1B", Path(temporary) / "not_terminal.json"):
            ready, reason, terminal = t2.readiness()
        self.assertFalse(ready)
        self.assertEqual(reason, "WAIT_T1B_TERMINAL")
        self.assertEqual(terminal, "ABSENT")

    def test_finish_video_record_rechecks_both_internal_mp4s(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = {"artifact": {"path": str(root / "masked_in.mp4"), "sha256": "a"}}
            second = {"artifact": {"path": str(root / "inpaint_out.mp4"), "sha256": "b"}}
            (root / "CLEAN.json").write_text(json.dumps({"internal_videos": {
                "masked_in.mp4": first, "inpaint_out.mp4": second}}))
            def checked_video(path, frame_count, fps):
                self.assertEqual((frame_count, fps), (171, 30))
                return first if path.name == "masked_in.mp4" else second
            with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.verify_internal_mp4",
                       side_effect=checked_video) as verify:
                t2.verify_internal_video_records(root, 171, 30)
                self.assertEqual(verify.call_count, 2)
                verify.side_effect = lambda path, *_: {"artifact": {"sha256": "drift"}}
                with self.assertRaisesRegex(RuntimeError, "INTERNAL_VIDEO_REF_DRIFT"):
                    t2.verify_internal_video_records(root, 171, 30)


if __name__ == "__main__":
    unittest.main()
