"""CPU-only contract tests; never import SAM weights or invoke a model."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chaoyang.governance import exact78_joint_hand_mask12_candidate_v1 as candidate
from chaoyang.governance import common
from chaoyang.ops import run_exact78_joint_hand_mask12_candidate_v1 as operation


class Mask12CandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Four SEALED RAW records and their twelve PNGs are read-only fixtures.
        cls.packet = candidate.packet_value()

    def test_exact_development_domain_and_finite_budget(self) -> None:
        packet = self.packet
        self.assertEqual(packet["execution_mode"], "EXACT78_CANDIDATE")
        self.assertEqual(packet["task_id"], candidate.TASK_ID)
        self.assertEqual(packet["budgets"], {
            "workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
            "model_calls_max": 42, "training_steps": 0,
            "wall_seconds_max": 2700, "session_attempts_max": 1,
        })
        self.assertEqual(len(packet["read_set"]), 8)
        self.assertEqual(len(packet["inputs"]["sessions"]), 4)
        anchors = {item["session_id"]: [row["frame"] for row in item["anchors"]]
                   for item in packet["inputs"]["sessions"]}
        self.assertEqual(anchors, {
            "play_cards_0902_042": [0, 85, 170],
            "get_potato_chips_0902_103": [0, 142, 283],
            "get_potato_chips_0901_013": [0, 435, 869],
            "play_cards_0915_001": [0, 75, 149],
        })
        self.assertEqual(sum(len(row["anchors"]) * (5 if row["source_family"] == "VST_0915" else 3)
                             for row in packet["inputs"]["sessions"]), 42)
        self.assertEqual(packet["config"]["old_human_prompt"], candidate.OLD_PROMPT)
        self.assertEqual(packet["config"]["new_human_prompt"], candidate.NEW_PROMPT)
        self.assertFalse(packet["config"]["old_mask_clean_algorithm_inputs"])
        self.assertEqual(packet["config"]["clean_calls_authorized"], 0)
        for item in packet["inputs"]["sessions"]:
            self.assertEqual(set(item), {
                "session_id", "source_family", "task", "frame_count", "source_fps",
                "anchors", "source_video", "frontend_spec", "sealed_result",
                "sealed_raw", "sam_checkpoint",
            })
            for frame in item["anchors"]:
                self.assertEqual(set(frame), {"frame", "raw_image", "time_mode", "physical_eye"})

    def test_check_only_does_not_register_or_run(self) -> None:
        output = io.StringIO()
        with patch.object(candidate, "packet_value", return_value=self.packet), redirect_stdout(output):
            self.assertEqual(candidate.main(["--check-only"]), 0)
        value = json.loads(output.getvalue())
        self.assertIn(value["status"], {
            "ROUTE_NOT_STAGED_NO_EXECUTION",
            "PUBLISHED_ROUTE_PACKET_STILL_REQUIRED",
        })
        self.assertEqual(value["model_calls"], 0)
        self.assertIs(value["registration_started"], False)
        self.assertIs(value["old_mask_clean_read"], False)

    def test_route_and_runtime_closure_not_current_production(self) -> None:
        route = candidate.route_contract(self.packet["runtime_code"])
        self.assertEqual(route["candidate_operation"], candidate.OPERATION)
        self.assertIsNone(route["current_runnable_operation"])
        self.assertTrue(route["task_packet_required"])
        code_paths = {Path(item["path"]).name for item in self.packet["runtime_code"]}
        self.assertIn("run_exact78_joint_hand_mask12_candidate_v1.py", code_paths)
        self.assertIn("vst_encoded_video_domain.py", code_paths)
        self.assertIn("exact78_operation_gate.py", code_paths)
        self.assertEqual(Path(self.packet["inputs"]["sam_vendor_manifest"]["path"]).name,
                         "vendor.json")

    def test_only_common_human_prompt_changes(self) -> None:
        from chaoyang.ops import run_exact78_fresh_frontend_v2 as frontend

        config = self.packet["config"]
        for scene in self.packet["inputs"]["sessions"]:
            original = frontend._prompts(scene)
            self.assertEqual(original["human"], config["old_human_prompt"])
            changed = dict(original)
            changed["human"] = config["new_human_prompt"]
            self.assertEqual(set(changed), set(original))
            for key in original.keys() - {"human"}:
                self.assertEqual(changed[key], original[key])
            self.assertEqual(changed["object"], config["other_role_prompts"][
                "poker_object" if scene["task"] == "poker" else "chips_object"])
            self.assertEqual(changed["equipment"], config["other_role_prompts"][
                "vst_equipment" if scene["source_family"] == "VST_0915"
                else "tracker_equipment"])

    def test_source_sha_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=common.REPO_ROOT / "_run/maintenance") as name:
            path = Path(name) / "input.raw"
            path.write_bytes(b"safe test fixture")
            with self.assertRaisesRegex(RuntimeError, "INPUT_SHA_DRIFT"):
                candidate._exact_ref(path, "0" * 64)

    def test_cross_session_and_frame_map_drift_fail_closed(self) -> None:
        row = candidate.FROZEN[0]
        original = candidate._object

        def cross_session(path: Path) -> dict:
            value = original(path)
            if path.name == "FRONTEND_SPEC.json":
                value["session_id"] = "wrong_session"
            return value

        with patch.object(candidate, "_object", side_effect=cross_session):
            with self.assertRaisesRegex(RuntimeError, "SEALED_SCENE_IDENTITY_DRIFT"):
                candidate._freeze_session(row)

        def wrong_frame(path: Path) -> dict:
            value = original(path)
            if path.name == "RAW.json":
                value["rows"][0]["model_input_index"] = 1
            return value

        with patch.object(candidate, "_object", side_effect=wrong_frame):
            with self.assertRaisesRegex(RuntimeError, "ANCHOR_FRAME_MAP_DRIFT"):
                candidate._freeze_session(row)

    def test_no_packet_denies_before_algorithm_import(self) -> None:
        def deny(*_args, **_kwargs):
            raise RuntimeError("GATE_DENIED_BEFORE_ALGORITHM")

        with patch.object(operation, "authorize_exact78_operation", side_effect=deny):
            with self.assertRaisesRegex(RuntimeError, "GATE_DENIED_BEFORE_ALGORITHM"):
                operation.run_candidate(output_root=common.REPO_ROOT / "_run/maintenance/never-created")
        self.assertNotIn("torch", operation.__dict__)

    def test_staged_route_required_even_with_matching_packet(self) -> None:
        attempt = common.REPO_ROOT / "_run/maintenance/synthetic-mask12-attempt"
        with patch.object(operation, "authorize_exact78_operation"), \
             patch.object(operation, "_active_packet", return_value=(self.packet, attempt)), \
             patch.object(candidate, "packet_value", return_value=self.packet), \
             patch.object(candidate, "route_status", return_value="ROUTE_NOT_STAGED_NO_EXECUTION"):
            with self.assertRaisesRegex(RuntimeError, "MASK12_ALGORITHM_ROUTE_NOT_PUBLISHED"):
                operation.run_candidate(output_root=attempt / "candidate")
        self.assertNotIn("torch", operation.__dict__)


if __name__ == "__main__":
    unittest.main()
