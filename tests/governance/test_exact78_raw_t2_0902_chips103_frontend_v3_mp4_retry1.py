"""CPU-only tests for the unregistered Chips103 duration-quantization retry."""

from __future__ import annotations

import io
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_raw_t2_0902_chips103_frontend_v3_mp4_retry1 as t2
from chaoyang.governance.exact78_operation_gate import _command_tail


class Chips103FrontendV3Mp4Retry1Tests(unittest.TestCase):
    def test_exact_command_and_frozen_full_source(self):
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v3", "--spec", str(t2.SPEC), "--phase", "all",
        ])
        inputs = t2.frozen_inputs()
        spec = t2.spec_value(inputs)
        self.assertEqual((spec["frame_count"], spec["source_fps"], spec["source_index"]),
                         (284, 30, 0))
        self.assertEqual(spec["task"], "chips")
        self.assertEqual(spec["source_video"], inputs["source_files"]["stereo"])
        self.assertEqual(spec["camera_params"], inputs["source_files"]["camera"])
        self.assertEqual(spec["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertNotIn("camera_maps", spec)
        self.assertNotIn("old_result", str(spec).lower())
        self.assertEqual(inputs["source_files"]["stereo"]["sha256"], t2.SOURCE_SHA["stereo"])

    def test_packet_is_separate_and_exactly_one_model_lease(self):
        inputs = t2.frozen_inputs()
        packet = t2.packet_value(inputs, t2.virtual_json_ref(t2.SPEC, t2.spec_value(inputs)))
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["config"]["frame_count"], 284)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertIn("_run/current/GPU_LEASE.json", packet["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", packet["write_set"])
        self.assertNotIn("post_result_ab_reference", packet)
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

    def test_staged_route_requires_exact_code_closure(self):
        def state_or_index(path):
            if path == t2.TASK_STATE_PATH:
                return {"tasks": [], "next_task": None}
            if path == t2.INDEX:
                return {"task_packets": []}
            return {}

        route = {"candidate_operation": t2.OPERATION,
                 "execution_status": "CANDIDATE_FENCED_SINGLE_SESSION_MP4_ACCEPTANCE_REVALIDATION",
                 "current_runnable_operation": None,
                 "output_schema": "exact78-fresh-frontend-stage-seal-v2",
                 "model_calls_authorized": "ONE_FULL_CHIPS103_SAM31_AND_PROPAINTER_ONLY_WHEN_EXACT_PACKET_ACTIVE",
                 "training_steps_authorized": 0,
                 "task_packet_required": True,
                 "code_closure": [t2.artifact_ref(t2.REPO_ROOT / key)
                                  for key in t2.ALGORITHM_SHA]}
        with patch.object(t2, "load_json", side_effect=state_or_index), \
             patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                   return_value={"special_status_contracts": {t2.CONTRACT_KEY: route}}):
            self.assertEqual(t2.readiness(), (True, "READY_STAGED_ROUTE_AND_PACKET_SAME_CAS"))
            route["code_closure"] = route["code_closure"][:-1]
            self.assertEqual(t2.readiness(), (False, "V3_CANDIDATE_ROUTE_NOT_STAGED"))

    def test_active_task_and_unstaged_route_fail_closed(self):
        with patch.object(t2, "load_json", return_value={"tasks": [{"status": "RUNNING"}]}):
            self.assertEqual(t2.readiness(), (False, "ACTIVE_TASK_CONFLICT"))
        with patch.object(t2, "load_json", return_value={"tasks": [], "next_task": None}), \
             patch.object(Path, "exists", return_value=True):
            self.assertEqual(t2.readiness(), (False, "FRESH_T2_V3_TASK_REQUIRED"))
        with patch.object(t2, "load_json", side_effect=lambda path: (
                {"tasks": [], "next_task": None} if path == t2.TASK_STATE_PATH
                else {"task_packets": []} if path == t2.INDEX else {})), \
             patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                   return_value={"special_status_contracts": {}}):
            self.assertEqual(t2.readiness(), (False, "V3_CANDIDATE_ROUTE_NOT_STAGED"))

    def test_worker_rejects_missing_published_route_before_gpu(self):
        with patch.object(t2, "verify_fence"), \
             patch("chaoyang.cli._maintained_operations", return_value=set()), \
             patch.object(t2.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "V3_CANDIDATE_ROUTE_NOT_PUBLISHED_WITH_PACKET"):
                t2.run_worker({})
        popen.assert_not_called()

    def test_execute_without_route_does_not_register(self):
        with patch.object(t2, "frozen_inputs", return_value={}), \
             patch.object(t2, "spec_value", return_value={}), \
             patch.object(t2, "packet_value", return_value={}), \
             patch.object(t2, "readiness", return_value=(False, "V3_CANDIDATE_ROUTE_NOT_STAGED")), \
             patch.object(t2, "register") as register, \
             patch("sys.argv", ["executor", "--execute"]):
            with self.assertRaisesRegex(RuntimeError, "V3_CANDIDATE_ROUTE_NOT_STAGED"):
                t2.main()
        register.assert_not_called()

    def test_check_only_cannot_register_or_run(self):
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

    def test_terminal_validator_rechecks_both_284_frame_mp4s(self):
        first = {"artifact": {"sha256": "a"}}
        second = {"artifact": {"sha256": "b"}}
        clean = {"internal_videos": {"masked_in.mp4": first, "inpaint_out.mp4": second}}

        def exact_video(path, frame_count, fps):
            self.assertEqual((frame_count, fps), (284, 30))
            return first if path.name == "masked_in.mp4" else second

        with patch.object(t2, "load_json", return_value=clean), \
             patch("chaoyang.ops.run_exact78_fresh_frontend_v3.verify_internal_mp4_v3",
                   side_effect=exact_video) as verify:
            t2.verify_internal_video_records(Path("candidate"))
            self.assertEqual(verify.call_count, 2)
            verify.side_effect = lambda *_: {"artifact": {"sha256": "drift"}}
            with self.assertRaisesRegex(RuntimeError, "INTERNAL_VIDEO_REF_DRIFT"):
                t2.verify_internal_video_records(Path("candidate"))


if __name__ == "__main__":
    unittest.main()
