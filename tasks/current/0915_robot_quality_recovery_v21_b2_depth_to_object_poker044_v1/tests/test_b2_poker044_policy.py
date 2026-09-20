from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "run_b2_depth_to_visible_object_poker044_v1.py"
CONFIG = ROOT / "contracts/B2_DEPTH_TO_VISIBLE_OBJECT_POKER044_V1.json"


def load_runner():
    chaoyang = types.ModuleType("chaoyang")
    ops = types.ModuleType("chaoyang.ops")
    pipeline = types.ModuleType("chaoyang.pipeline")
    upstream = types.ModuleType("chaoyang.ops.run_0915_robot15h_object6d_wave0_v1")
    upstream.DEPTH_REFERENCE = "PHYSICAL_LEFT_ENCODED_RESIZE_ONLY_OPTICAL_Z"
    upstream.DEPTH_SHAPE = (480, 640)
    upstream.IMAGE_DOMAIN = "PHYSICAL_LEFT_SOURCEINDEX1_CROP_RESIZE_ONLY"
    upstream.INSTANCE_IDS = ("playing_card_00", "playing_card_01", "playing_card_02")
    upstream.MASK_SHAPE = (960, 1280)
    upstream.PIXEL_MAP = {"type": "ANALYTIC_PIXEL_CENTER_2X"}

    class PackedMask:
        pass

    upstream.PackedMask = PackedMask
    upstream.depth_to_mask_map = lambda: None
    upstream.finite_patch_record = lambda *args, **kwargs: None
    upstream.load_depth_frame = lambda *args, **kwargs: None
    upstream.registered_support = lambda *args, **kwargs: None
    upstream.temporally_unify_axes = lambda *args, **kwargs: None

    planar = types.ModuleType("chaoyang.pipeline.object6d_planar_observability_v2")
    planar.DIRECT_VISIBILITY = "DIRECT_VISIBLE_MASK_AND_VALID_DEPTH_ONLY"
    planar.MASK_REGISTRATION_AUTHORITY = "PINNED_ANALYTIC_PIXEL_CENTER_MAP"

    class PlanarFrameInput:
        pass

    planar.PlanarFrameInput = PlanarFrameInput
    planar.estimate_planar_frame = lambda *args, **kwargs: None

    sys.modules.update(
        {
            "chaoyang": chaoyang,
            "chaoyang.ops": ops,
            "chaoyang.pipeline": pipeline,
            "chaoyang.ops.run_0915_robot15h_object6d_wave0_v1": upstream,
            "chaoyang.pipeline.object6d_planar_observability_v2": planar,
        }
    )
    spec = importlib.util.spec_from_file_location("b2_runner_under_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Poker044PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_passes_policy_validation(self):
        self.runner.validate_config(copy.deepcopy(self.config))

    def test_every_authority_upgrade_fails_closed(self):
        forbidden = (
            "gpu_allowed",
            "foundationstereo_rerun_allowed",
            "hidden_geometry_inferred",
            "external_metric_accuracy_claimed",
            "contact_authority",
            "control_ground_truth",
            "training_eligible",
            "physical_deployment_authorized",
        )
        for field in forbidden:
            with self.subTest(field=field):
                value = copy.deepcopy(self.config)
                value[field] = True
                with self.assertRaises(RuntimeError):
                    self.runner.validate_config(value)

    def test_consumer_or_identity_upgrade_fails_closed(self):
        mutations = (
            ("consumer_allowed", True),
            ("review_only", False),
            ("physical_card_identity", "playing_card_17"),
            ("face_identity", "BACK"),
            ("depth_execution", "RERUN"),
            ("mask_scope", "PHYSICAL_IDENTITY"),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                value = copy.deepcopy(self.config)
                value[field] = replacement
                with self.assertRaises(RuntimeError):
                    self.runner.validate_config(value)

    def test_regression_pair_is_exact(self):
        for missing in ("play_cards_0915_031", "play_cards_0915_119"):
            value = copy.deepcopy(self.config)
            del value["regressions"][missing]
            with self.assertRaises(RuntimeError):
                self.runner.validate_config(value)

    def test_reference_verifier_detects_content_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"pinned")
            reference = self.runner.make_ref(path)
            self.assertEqual(
                self.runner.verify_ref(reference, label="fixture"), path.resolve()
            )
            path.write_bytes(b"drifted")
            with self.assertRaises(RuntimeError):
                self.runner.verify_ref(reference, label="fixture")


if __name__ == "__main__":
    unittest.main()
