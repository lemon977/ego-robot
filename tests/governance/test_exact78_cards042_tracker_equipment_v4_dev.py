"""CPU-only packet/route and negative checks for an unregistered v4 dev run."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.governance import execute_exact78_raw_t2_cards042_tracker_equipment_v4_dev as runner


class Cards042TrackerEquipmentV4PrepTests(unittest.TestCase):
    def test_frozen_spec_is_exact_raw_cards042_and_not_old_clean(self):
        inputs = runner.frozen_inputs()
        spec = runner.spec_value(inputs)
        self.assertEqual(spec["session_id"], "play_cards_0902_042")
        self.assertEqual((spec["frame_count"], spec["source_fps"], spec["source_index"]), (171, 30, 0))
        self.assertEqual(spec["source_video"]["sha256"], runner.base.SOURCE_SHA["stereo"])
        self.assertEqual(spec["map_rule"], "GENERATE_FROM_CAPTURE_CAMERA_PARAMS")
        self.assertEqual(spec["output_root"], str(runner.OUT / "candidate"))
        self.assertNotIn("clean", json.dumps(spec).lower())

    def test_packet_has_independent_attempt_exact_lease_and_ab_after_seal(self):
        inputs = runner.frozen_inputs()
        spec = runner.spec_value(inputs)
        packet = runner.packet_value(inputs, runner.base.virtual_json_ref(runner.SPEC, spec))
        self.assertEqual(packet["write_set"][0], str(runner.OUT.relative_to(REPO_ROOT)))
        self.assertIn("_run/current/GPU_LEASE.json", packet["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", packet["write_set"])
        self.assertEqual(packet["config"]["algorithm_delta"],
                         "TRACKER_EQUIPMENT_PROMPT_UNION_ONLY")
        self.assertEqual(packet["config"]["probability_threshold"], 0.5)
        self.assertEqual(packet["commands"], [runner.COMMAND])
        self.assertEqual(packet["post_result_ab_reference"]["use"],
                         "EVALUATOR_ONLY_AFTER_FRESH_RESULT")
        self.assertNotIn(str(runner.OLD_RESULT), json.dumps(packet["inputs"]))
        refs = {ref["path"] for ref in packet["runtime_code"]}
        self.assertIn(str(runner.V4_PATH), refs)
        self.assertIn(str(REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v3.py"), refs)

    def test_staged_route_requires_new_operation_and_exact_closure(self):
        code = [artifact_ref(path) for path in (
            runner.V4_PATH,
            REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v3.py",
            REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
            REPO_ROOT / "src/chaoyang/pipeline/propainter_mp4_acceptance_v2.py",
        )]
        route = {"candidate_operation": runner.OPERATION, "task_packet_required": True,
                 "current_runnable_operation": None,
                 "output_schema": "exact78-fresh-frontend-stage-seal-v2",
                 "model_calls_authorized": "ONE_FULL_CARDS042_SAM31_AND_PROPAINTER_ONLY_WHEN_EXACT_PACKET_ACTIVE",
                 "training_steps_authorized": 0, "code_closure": code}
        contract = {"special_status_contracts": {runner.CONTRACT_KEY: route}}
        self.assertTrue(runner.route_valid(contract))
        self.assertFalse(runner.route_valid({"special_status_contracts": {
            runner.CONTRACT_KEY: {**route, "code_closure": code[1:]}}}))
        self.assertFalse(runner.route_valid({"special_status_contracts": {
            runner.CONTRACT_KEY: {**route, "current_runnable_operation": runner.OPERATION}}}))
        self.assertFalse(runner.route_valid({"special_status_contracts": {
            runner.CONTRACT_KEY: {**route, "model_calls_authorized": "UNBOUNDED"}}}))

    def test_mask_schema_failure_is_closed_before_success(self):
        with patch.object(runner, "load_json", side_effect=[{"schema_version": "v2", "rows": []},
                                                         {"rows": []}]):
            with self.assertRaisesRegex(RuntimeError, "V4_MASK_SCHEMA_OR_FRAME_COUNT"):
                runner.verify_mask_and_protection(runner.OUT / "candidate")

    def test_check_only_has_no_attempt_or_packet_write(self):
        before = tuple(path.exists() for path in (runner.OUT, runner.PACKET, runner.RECEIPT))
        result = subprocess.run(
            [sys.executable, "-B", "-m",
             "chaoyang.governance.execute_exact78_raw_t2_cards042_tracker_equipment_v4_dev",
             "--check-only"], capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-1000:])
        data = json.loads(result.stdout.strip())
        self.assertEqual(data["task_id"], runner.TASK)
        self.assertEqual(data["source_frames"], 171)
        self.assertEqual(data["gpu_lease_count"], 1)
        self.assertEqual(before, tuple(path.exists() for path in
                                       (runner.OUT, runner.PACKET, runner.RECEIPT)))


if __name__ == "__main__":
    unittest.main()
