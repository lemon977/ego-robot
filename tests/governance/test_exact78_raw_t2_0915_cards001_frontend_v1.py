"""CPU-only dry-run and negative-domain tests; no task registration or model run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from chaoyang.governance import execute_exact78_raw_t2_0915_cards001_frontend_v1 as t2
from chaoyang.governance.exact78_operation_gate import _command_tail
from chaoyang.ops.run_exact78_fresh_frontend_v2 import validate_spec, verify_internal_mp4


class Cards001VstFrontendPacketTests(unittest.TestCase):
    def inputs(self):
        frozen = t2.load_json(t2.T0 / "INPUT_FREEZE.json")
        row = frozen["selected"][t2.SESSION_ID]
        return {"source_files": {name: row["files"][name] for name in t2.SOURCE_NAMES},
                "t0_frame_time_map": t2.artifact_ref(t2.T0 / "FRAME_TIME_MAP.json"),
                "models": {"sam_checkpoint": frozen["models"]["sam31_checkpoint"],
                           "propainter_weights": {
                               "raft-things.pth": frozen["models"]["raft_weights"],
                               "recurrent_flow_completion.pth": frozen["models"]["flow_weights"],
                               "ProPainter.pth": frozen["models"]["propainter_weights"]}},
                "propainter_entry": t2.artifact_ref(t2.REPO_ROOT /
                    "vendor/ProPainter/inference_propainter.py")}

    def test_raw_vst_physical_left_and_unverified_timebase_spec(self):
        spec = t2.spec_value(self.inputs())
        self.assertEqual((spec["frame_count"], spec["source_fps"]), (150, 30))
        self.assertEqual(spec["source_index"], 1)
        self.assertEqual(spec["source_video"]["path"], str(t2.SESSION / "raw/vst.h264"))
        self.assertEqual(spec["camera_params"]["path"], str(t2.SESSION / "raw/camera_params.json"))
        self.assertEqual(spec["map_rule"], "NO_MAP_VST_ENCODED")
        self.assertEqual(spec["map_generator_code"], "ABSENT")
        self.assertEqual(spec["transform"], "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY")
        self.assertNotIn("camera_maps", spec)

    def test_processed_video_wrong_eye_or_old_remap_are_rejected(self):
        spec = t2.spec_value(self.inputs())
        wrong = copy.deepcopy(spec)
        wrong["source_index"] = 0
        with self.assertRaisesRegex(ValueError, "VST_PHYSICAL_LEFT"):
            validate_spec(wrong)
        wrong = copy.deepcopy(spec)
        wrong["transform"] = "SAME_SESSION_EQUIDIS62_TO_PINHOLE"
        with self.assertRaises(ValueError):
            validate_spec(wrong)
        wrong = copy.deepcopy(spec)
        wrong["source_video"] = {"path": str(t2.SESSION / "processed/vst.mp4")}
        with self.assertRaises((FileNotFoundError, ValueError)):
            validate_spec(wrong)

    def test_packet_is_independent_and_single_lease(self):
        fake_ref = lambda path: {"path": str(Path(path)), "bytes": 0, "sha256": "a" * 64}
        with patch.object(t2, "artifact_ref", side_effect=fake_ref), \
             patch.object(t2, "process_identity", return_value={"start_ticks": 77}):
            packet = t2.packet_value(self.inputs(), fake_ref(t2.SPEC))
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v2", "--spec", str(t2.SPEC), "--phase", "all"])
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["config"]["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(packet["config"]["qpc_head_sync"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(packet["config"]["physical_left_source_index"], 1)
        self.assertIn("_run/current/GPU_LEASE.json", packet["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", packet["write_set"])
        self.assertNotIn("candidate/CAMERA_MAPS.npz", packet["required_outputs"])
        self.assertEqual(set(packet["inputs"]["source_files"]), set(t2.SOURCE_NAMES))
        self.assertTrue(all(Path(ref["path"]).parent == t2.SESSION / "raw"
                            for ref in packet["inputs"]["source_files"].values()))
        self.assertNotIn("0916", str(packet["write_set"]))
        self.assertEqual(len(packet["read_set"]), 8)

    def test_finish_rechecks_two_mp4_refs(self):
        with tempfile.TemporaryDirectory(dir=t2.REPO_ROOT / "_run/current") as temporary:
            root = Path(temporary)
            first = {"artifact": {"sha256": "a"}}
            second = {"artifact": {"sha256": "b"}}
            (root / "CLEAN.json").write_text(json.dumps({"internal_videos": {
                "masked_in.mp4": first, "inpaint_out.mp4": second}}))
            def checked_video(path, frame_count, fps):
                self.assertEqual((frame_count, fps), (150, 30))
                return first if path.name == "masked_in.mp4" else second
            with patch("chaoyang.ops.run_exact78_fresh_frontend_v2.verify_internal_mp4",
                       side_effect=checked_video) as verify:
                t2.verify_internal_video_records(root)
                self.assertEqual(verify.call_count, 2)
                verify.side_effect = lambda *_: {"artifact": {"sha256": "drift"}}
                with self.assertRaisesRegex(RuntimeError, "INTERNAL_VIDEO_REF_DRIFT"):
                    t2.verify_internal_video_records(root)

    def test_v2_mp4_accepts_exact_150_at_30_and_rejects_frame_drift(self):
        with tempfile.TemporaryDirectory(dir=t2.REPO_ROOT / "_run/current") as temporary:
            path = Path(temporary) / "internal.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     30, (960, 720))
            self.assertTrue(writer.isOpened())
            frame = np.zeros((720, 960, 3), dtype=np.uint8)
            try:
                for _ in range(150):
                    writer.write(frame)
            finally:
                writer.release()
            evidence = verify_internal_mp4(path, 150, 30)
            self.assertEqual((evidence["decoded_frames"], evidence["fps"],
                              evidence["duration_seconds"]), (150, 30, 5.0))
            with self.assertRaisesRegex(RuntimeError, "PROPAINTER_VIDEO_TIME_OR_DOMAIN"):
                verify_internal_mp4(path, 149, 30)


if __name__ == "__main__":
    unittest.main()
