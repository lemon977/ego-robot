"""CPU-only, no-registration checks for four finite white-device templates."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

from chaoyang.governance.common import RECEIPT_PATH, artifact_ref
from chaoyang.governance import execute_exact78_white_device_v5_dev as dev


class WhiteDeviceFinitePacketTests(unittest.TestCase):
    def test_cards042_real_dry_run_is_read_only_across_route_staging(self):
        ctx = dev.context("0902_cards042")
        before = artifact_ref(RECEIPT_PATH)
        self.assertFalse(ctx["out"].exists())
        with patch.object(sys, "argv", ["executor", "--session", "0902_cards042", "--check-only"]), \
             patch.object(dev, "register") as register, \
             patch.object(dev, "run_worker") as worker, \
             contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(dev.main(), 0)
        result = json.loads(captured.getvalue())
        self.assertIn(result["status"], {"DRY_RUN_NOT_ROUTABLE", "DRY_RUN_READY"})
        self.assertIn(result["reason"], {"ACTIVE_TASK_CONFLICT", "V5_CANDIDATE_ROUTE_NOT_STAGED",
                                        "READY_TO_PUBLISH_CANDIDATE_ROUTE_WITH_PACKET"})
        self.assertEqual(result["source_frames"], 171)
        self.assertEqual(result["fps"], 30)
        self.assertEqual(result["source_video"]["sha256"],
                         "f1baaa4c0ea6201b9ad00fc2947abd4b0816a6e4c9d38d4fae3df4f35de4f53d")
        self.assertEqual(result["write_set"][0], str(ctx["out"].relative_to(dev.REPO_ROOT)))
        self.assertIn("_run/current/GPU_LEASE.json", result["write_set"])
        self.assertIn("_run/current/GPU_LEASE.lock", result["write_set"])
        self.assertEqual(artifact_ref(RECEIPT_PATH), before)
        self.assertFalse(ctx["out"].exists())
        register.assert_not_called()
        worker.assert_not_called()

    def test_four_specs_have_distinct_attempts_and_same_one_prompt(self):
        attempts = set()
        specs = []
        for key, cfg in dev.SESSION_CONFIGS.items():
            ctx = dev.context(key)
            inputs = dev.frozen_inputs(ctx)
            spec = dev.spec_value(ctx, inputs)
            spec_ref = dev.virtual_json_ref(ctx["spec"], spec)
            packet = dev.packet_value(ctx, inputs, spec_ref)
            self.assertEqual(packet["write_set"][0], str(ctx["out"].relative_to(dev.REPO_ROOT)))
            self.assertEqual(packet["config"]["equipment_prompt"], dev.PROMPT)
            self.assertEqual(packet["config"]["probability_threshold"], 0.5)
            self.assertEqual(packet["config"]["gpu_lease_count"], 1)
            self.assertEqual(packet["budgets"]["session_attempts_max"], 1)
            self.assertLessEqual(len(packet["read_set"]), 8)
            self.assertEqual(spec["frame_count"], cfg["frames"])
            self.assertEqual(spec["source_fps"], cfg["fps"])
            self.assertEqual(spec["session_id"], cfg["session"])
            self.assertFalse(any("/processed/" in ref["path"] for ref in packet["inputs"]["source_files"].values()))
            attempts.add(packet["write_set"][0])
            specs.append(spec)
        self.assertEqual(len(attempts), 4)
        self.assertEqual(specs[-1]["source_index"], 1)
        self.assertTrue(specs[-1]["source_video"]["path"].endswith("/raw/vst.h264"))
        self.assertTrue(all("/source_stereo/" in spec["source_video"]["path"] for spec in specs[:3]))

    def test_execute_cannot_register_without_staged_route(self):
        with patch.object(dev, "readiness", return_value=(False, "V5_CANDIDATE_ROUTE_NOT_STAGED")), \
             patch.object(dev, "register") as register, \
             patch.object(dev, "run_worker") as worker, \
             patch.object(sys, "argv", ["executor", "--session", "0902_cards042", "--execute"]):
            with self.assertRaisesRegex(RuntimeError, "V5_CANDIDATE_ROUTE_NOT_STAGED"):
                dev.main()
        register.assert_not_called()
        worker.assert_not_called()

    def test_route_requires_exact_candidate_code_closure(self):
        paths = (dev.V5_PATH,
                 dev.REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v3.py",
                 dev.REPO_ROOT / "src/chaoyang/ops/run_exact78_fresh_frontend_v2.py",
                 dev.REPO_ROOT / "src/chaoyang/pipeline/propainter_mp4_acceptance_v2.py")
        route = {"candidate_operation": dev.OPERATION, "task_packet_required": True,
                 "current_runnable_operation": None,
                 "output_schema": "exact78-fresh-frontend-stage-seal-v2",
                 "model_calls_authorized": "ONE_DEV_SESSION_SAM31_AND_PROPAINTER_ONLY_WHEN_EXACT_PACKET_ACTIVE",
                 "training_steps_authorized": 0,
                 "code_closure": [artifact_ref(path) for path in paths]}
        self.assertTrue(dev.route_valid({"special_status_contracts": {dev.CONTRACT_KEY: route}}))
        route["code_closure"] = route["code_closure"][:-1]
        self.assertFalse(dev.route_valid({"special_status_contracts": {dev.CONTRACT_KEY: route}}))

    def test_runtime_sha_drift_stops_before_packet(self):
        ctx = dev.context("0902_cards042")
        with patch.object(dev, "V5_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "V5_ALGORITHM_SHA_DRIFT"):
                dev.runtime_refs(ctx, {})


if __name__ == "__main__":
    unittest.main()
