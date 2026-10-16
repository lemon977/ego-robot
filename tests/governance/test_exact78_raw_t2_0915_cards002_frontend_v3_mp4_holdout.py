"""CPU-only tests for the unregistered 0915 Cards002 MP4 holdout."""

from __future__ import annotations

import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from chaoyang.governance import execute_exact78_raw_t2_0915_cards002_frontend_v3_mp4_holdout as t2
from chaoyang.governance.exact78_operation_gate import _command_tail
from chaoyang.ops.run_exact78_fresh_frontend_v2 import validate_spec


class Cards002VstMp4HoldoutTests(unittest.TestCase):
    def test_frozen_raw_identity_and_distinct_group(self):
        inputs = t2.frozen_inputs()
        frozen = t2.load_json(t2.T0 / "INPUT_FREEZE.json")
        holdout = frozen["selected"][t2.SESSION_ID]
        dev = frozen["selected"]["play_cards_0915_001"]
        self.assertEqual(holdout["role"], "holdout")
        self.assertEqual(holdout["prior_exposure"], "UNKNOWN")
        self.assertNotEqual(holdout["source_group"], dev["source_group"])
        self.assertEqual(set(inputs["source_files"]), set(t2.SOURCE_NAMES))
        for name in t2.SOURCE_NAMES:
            self.assertEqual(inputs["source_files"][name]["sha256"], t2.SOURCE_SHA[name])
            self.assertEqual(Path(inputs["source_files"][name]["path"]).parent,
                             t2.SESSION / "raw")
        self.assertEqual(inputs["models"], t2.load_json(t2.DEV_PACKET)["inputs"]["models"])

    def test_spec_uses_physical_left_raw_h264_not_processed_or_hand_motion(self):
        inputs = t2.frozen_inputs()
        spec = t2.spec_value(inputs)
        self.assertEqual(_command_tail(t2.COMMAND), [
            "run_exact78_fresh_frontend_v3", "--spec", str(t2.SPEC), "--phase", "all"])
        self.assertEqual((spec["frame_count"], spec["source_fps"], spec["source_index"]),
                         (131, 30, 1))
        self.assertEqual(spec["source_video"]["path"], str(t2.SESSION / "raw/vst.h264"))
        self.assertEqual(spec["camera_params"]["path"], str(t2.SESSION / "raw/camera_params.json"))
        self.assertEqual(spec["transform"], "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY")
        self.assertEqual(spec["map_rule"], "NO_MAP_VST_ENCODED")
        self.assertNotIn("camera_maps", spec)
        self.assertNotIn("pico", str(spec).lower())
        self.assertNotIn("glove", str(spec).lower())
        self.assertNotIn("processed", str(spec).lower())
        wrong = copy.deepcopy(spec)
        wrong["source_index"] = 0
        with self.assertRaisesRegex(ValueError, "VST_PHYSICAL_LEFT"):
            validate_spec(wrong)
        wrong = copy.deepcopy(spec)
        wrong["source_video"] = {"path": str(t2.SESSION / "processed/vst.mp4")}
        with self.assertRaises((FileNotFoundError, ValueError)):
            validate_spec(wrong)

    def test_t0_131_raw_rows_are_not_truncated_to_legacy_130(self):
        timing = t2.load_json(t2.T0 / "FRAME_TIME_MAP.json")["sessions"][t2.SESSION_ID]
        self.assertEqual(timing["decoded_frames"], 131)
        self.assertEqual(timing["raw_qpc_count"], 131)
        self.assertEqual(timing["raw_capture_timestamp_count"], 131)
        self.assertEqual(timing["frame_count_provenance"]["legacy_processed_group_manifest"], 130)
        self.assertEqual(timing["time_mode"], "TIMEBASE_UNVERIFIED")
        self.assertEqual(timing["capture_to_decode_alignment"], "INDEX_ALIGNMENT_UNVERIFIED")
        self.assertEqual(timing["qpc_unit"], "UNVERIFIED")
        self.assertEqual(timing["head_pose_clock_domain"], "UNVERIFIED")

    def test_packet_has_new_write_set_pinned_acceptance_and_no_tuning(self):
        inputs = t2.frozen_inputs()
        packet = t2.packet_value(inputs, t2.virtual_json_ref(t2.SPEC, t2.spec_value(inputs)))
        self.assertEqual(packet["commands"], [t2.COMMAND])
        self.assertEqual(packet["config"]["legacy_processed_group_manifest_frames"], 130)
        self.assertEqual(packet["config"]["frame_count"], 131)
        self.assertIs(packet["config"]["tuning_from_holdout"], False)
        self.assertEqual(packet["config"]["mp4_acceptance"],
                         "EXACT_STREAM_TICKS_AND_MVHD_QUANTIZATION_V2")
        self.assertEqual(packet["budgets"]["gpu_count"], 1)
        self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
        self.assertEqual(packet["write_set"][0], str(t2.OUT.relative_to(t2.REPO_ROOT)))
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertNotIn("0916", str(packet["write_set"]))
        refs = {Path(ref["path"]).relative_to(t2.REPO_ROOT).as_posix(): ref["sha256"]
                for ref in packet["runtime_code"]}
        for relative, sha in t2.ALGORITHM_SHA.items():
            self.assertEqual(refs[relative], sha)
        dev_result = (t2.REPO_ROOT / "_run/current" /
            "exact78_raw_to_robot_t2_0915_cards001_frontend_v1_20260930/attempts/attempt_0001/RESULT.json")
        self.assertNotIn(str(dev_result), str(packet["inputs"]))

    def test_algorithm_or_timebase_drift_rejected(self):
        inputs = t2.frozen_inputs()
        wrong = dict(t2.ALGORITHM_SHA)
        wrong["src/chaoyang/pipeline/propainter_mp4_acceptance_v2.py"] = "0" * 64
        with patch.object(t2, "ALGORITHM_SHA", wrong):
            with self.assertRaisesRegex(RuntimeError, "V3_ALGORITHM_SHA_DRIFT"):
                t2.packet_value(inputs, t2.virtual_json_ref(t2.SPEC, t2.spec_value(inputs)))
        original = t2.load_json
        timing = copy.deepcopy(original(t2.T0 / "FRAME_TIME_MAP.json"))
        timing["sessions"][t2.SESSION_ID]["time_mode"] = "SOURCE_TIMEBASE_VERIFIED"
        with patch.object(t2, "load_json", side_effect=lambda path: (
                timing if path == t2.T0 / "FRAME_TIME_MAP.json" else original(path))):
            with self.assertRaisesRegex(RuntimeError, "CARDS002_TIMEBASE_FREEZE_DRIFT"):
                t2.frozen_inputs()

    def test_staged_specific_route_required_not_only_global_v3_operation(self):
        def state_or_index(path):
            if path == t2.TASK_STATE_PATH:
                return {"tasks": [], "next_task": None}
            if path == t2.INDEX:
                return {"task_packets": []}
            return {}
        closure = [t2.artifact_ref(t2.REPO_ROOT / name) for name in t2.ALGORITHM_SHA]
        route = {"candidate_operation": t2.OPERATION,
                 "execution_status": "CANDIDATE_FENCED_PRE_FROZEN_HOLDOUT_MP4_ACCEPTANCE",
                 "current_runnable_operation": None,
                 "output_schema": "exact78-fresh-frontend-stage-seal-v2",
                 "model_calls_authorized": "ONE_FULL_0915_CARDS002_SAM31_AND_PROPAINTER_ONLY_WHEN_EXACT_PACKET_ACTIVE",
                 "training_steps_authorized": 0, "task_packet_required": True,
                 "code_closure": closure}
        with patch.object(t2, "load_json", side_effect=state_or_index), \
             patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                   return_value={"special_status_contracts": {t2.CONTRACT_KEY: route}}):
            self.assertEqual(t2.readiness(), (True, "READY_STAGED_ROUTE_AND_PACKET_SAME_CAS"))
            route["code_closure"] = closure[:-1]
            self.assertEqual(t2.readiness(), (False, "V3_CARDS002_HOLDOUT_ROUTE_NOT_STAGED"))

    def test_active_task_and_no_route_fail_closed(self):
        with patch.object(t2, "load_json", return_value={"tasks": [{"status": "RUNNING"}]}):
            self.assertEqual(t2.readiness(), (False, "ACTIVE_TASK_CONFLICT"))
        with patch.object(t2, "load_json", side_effect=lambda path: (
                {"tasks": [], "next_task": None} if path == t2.TASK_STATE_PATH
                else {"task_packets": []} if path == t2.INDEX else {})), \
             patch("chaoyang.governance.current_r3_contracts.build_algorithm_contract",
                   return_value={"special_status_contracts": {}}):
            self.assertEqual(t2.readiness(), (False, "V3_CARDS002_HOLDOUT_ROUTE_NOT_STAGED"))

    def test_check_only_never_registers_or_starts_model(self):
        with patch.object(t2, "frozen_inputs", return_value={}), \
             patch.object(t2, "spec_value", return_value={}), \
             patch.object(t2, "packet_value", return_value={"run_signature": "x"}), \
             patch.object(t2, "readiness", return_value=(False, "ACTIVE_TASK_CONFLICT")), \
             patch.object(t2, "virtual_json_ref", return_value={"sha256": "y"}), \
             patch.object(t2, "register") as register, \
             patch.object(t2, "run_worker") as worker, \
             patch("sys.argv", ["executor", "--check-only"]), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(t2.main(), 0)
        register.assert_not_called()
        worker.assert_not_called()
        self.assertIn("DRY_RUN_NOT_ROUTABLE", stdout.getvalue())

    def test_finish_rechecks_both_131_frame_v3_mp4_records(self):
        first = {"artifact": {"sha256": "a"}}
        second = {"artifact": {"sha256": "b"}}
        clean = {"internal_videos": {"masked_in.mp4": first, "inpaint_out.mp4": second}}
        def exact_video(path, frames, fps):
            self.assertEqual((frames, fps), (131, 30))
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
