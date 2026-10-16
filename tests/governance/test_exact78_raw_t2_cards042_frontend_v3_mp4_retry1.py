"""CPU-only preview tests; never register or invoke a model worker."""

from __future__ import annotations

import io
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_raw_t2_cards042_frontend_v3_mp4_retry1 as t2
from chaoyang.governance.exact78_operation_gate import _command_tail


class Cards042FrontendV3Mp4Retry1Tests(unittest.TestCase):
    def test_exact_new_cli_command_and_full_171_frame_source(self):
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v3", "--spec", str(t2.SPEC), "--phase", "all",
        ])
        inputs = t2.frozen_inputs()
        spec = t2.spec_value(inputs)
        self.assertEqual((spec["frame_count"], spec["source_fps"], spec["source_index"]),
                         (171, 30, 0))
        self.assertEqual(spec["source_video"], inputs["source_files"]["stereo"])
        self.assertEqual(spec["camera_params"], inputs["source_files"]["camera"])
        self.assertEqual(spec["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertEqual(spec["output_root"], str(t2.OUT / "candidate"))
        self.assertNotIn("camera_maps", spec)
        self.assertNotIn(str(t2.OLD_RESULT), str(spec))

    def test_algorithm_delta_pins_and_new_packet_write_set(self):
        inputs = t2.frozen_inputs()
        spec = t2.spec_value(inputs)
        packet = t2.packet_value(inputs, t2.virtual_json_ref(t2.SPEC, spec))
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["config"]["stages"], list(t2.STAGES))
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertIn("_run/current/GPU_LEASE.json", packet["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", packet["write_set"])
        self.assertIn("candidate/propainter_model/frames/masked_in.mp4", packet["required_outputs"])
        self.assertIn("candidate/propainter_model/frames/inpaint_out.mp4", packet["required_outputs"])
        self.assertNotIn(str(t2.OLD_RESULT), str(packet["inputs"]))
        self.assertNotIn(str(t2.OLD_RESULT), str(packet["read_set"]))
        self.assertEqual(packet["post_result_ab_reference"], {
            "path": str(t2.OLD_RESULT), "use": "EVALUATOR_ONLY_AFTER_FRESH_RESULT"})
        self.assertTrue(all("0916" not in entry for entry in packet["write_set"]))
        refs = {Path(ref["path"]).relative_to(t2.REPO_ROOT).as_posix(): ref["sha256"]
                for ref in packet["runtime_code"]}
        for relative, expected in t2.ALGORITHM_SHA.items():
            self.assertEqual(refs[relative], expected)

    def test_algorithm_sha_drift_rejected_before_packet(self):
        wrong = dict(t2.ALGORITHM_SHA)
        wrong["src/chaoyang/ops/run_exact78_fresh_frontend_v3.py"] = "0" * 64
        with patch.object(t2, "ALGORITHM_SHA", wrong):
            with self.assertRaisesRegex(RuntimeError, "V3_ALGORITHM_SHA_DRIFT"):
                t2.runtime_refs()

    def test_current_active_task_cannot_register(self):
        with patch.object(t2, "load_json", return_value={"tasks": [{"status": "RUNNING"}]}):
            self.assertEqual(t2.readiness(), (False, "ACTIVE_TASK_CONFLICT"))

    def test_execute_rejected_without_route_and_without_writes(self):
        with patch.object(t2, "frozen_inputs", return_value={}), \
             patch.object(t2, "spec_value", return_value={}), \
             patch.object(t2, "packet_value", return_value={}), \
             patch.object(t2, "readiness", return_value=(False, "V3_CANDIDATE_ROUTE_NOT_PUBLISHED")), \
             patch.object(t2, "register") as register, \
             patch("sys.argv", ["executor", "--execute"]):
            with self.assertRaisesRegex(RuntimeError, "V3_CANDIDATE_ROUTE_NOT_PUBLISHED"):
                t2.main()
        register.assert_not_called()

    def test_check_only_does_not_register_or_run(self):
        with patch.object(t2, "frozen_inputs", return_value={}), \
             patch.object(t2, "spec_value", return_value={}), \
             patch.object(t2, "packet_value", return_value={"run_signature": "x"}), \
             patch.object(t2, "readiness", return_value=(False, "ACTIVE_TASK_CONFLICT")), \
             patch.object(t2, "register") as register, \
             patch.object(t2, "run_worker") as run_worker, \
             patch.object(t2, "virtual_json_ref", return_value={"sha256": "y"}), \
             patch("sys.argv", ["executor", "--check-only"]), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(t2.main(), 0)
        register.assert_not_called()
        run_worker.assert_not_called()
        self.assertIn("DRY_RUN_NOT_ROUTABLE", stdout.getvalue())

    def test_finish_validator_rechecks_both_mp4_records_with_v3(self):
        first = {"artifact": {"sha256": "a"}}
        second = {"artifact": {"sha256": "b"}}
        clean_record = {"internal_videos": {"masked_in.mp4": first, "inpaint_out.mp4": second}}

        def exact_video(path, frame_count, fps):
            self.assertEqual((frame_count, fps), (171, 30))
            return first if path.name == "masked_in.mp4" else second

        with patch.object(t2, "load_json", return_value=clean_record), \
             patch("chaoyang.ops.run_exact78_fresh_frontend_v3.verify_internal_mp4_v3",
                   side_effect=exact_video) as verify:
            t2.verify_internal_video_records(Path("candidate"))
            self.assertEqual(verify.call_count, 2)
            verify.side_effect = lambda *_: {"artifact": {"sha256": "drift"}}
            with self.assertRaisesRegex(RuntimeError, "INTERNAL_VIDEO_REF_DRIFT"):
                t2.verify_internal_video_records(Path("candidate"))


if __name__ == "__main__":
    unittest.main()
