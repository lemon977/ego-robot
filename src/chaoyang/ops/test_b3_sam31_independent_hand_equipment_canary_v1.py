from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "B3_MASK_SUCCESSOR_CANARY_V1.json"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = HERE.parents[2] / "contracts/robot_recovery/B3_MASK_SUCCESSOR_CANARY_V1.json"
SPEC = importlib.util.spec_from_file_location(
    "b3_canary", HERE / "run_b3_sam31_independent_hand_equipment_canary_v1.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RENDERER_SPEC = importlib.util.spec_from_file_location(
    "b3_renderer", HERE / "publish_b3_shallow_reviews_v1.py"
)
assert RENDERER_SPEC and RENDERER_SPEC.loader
RENDERER = importlib.util.module_from_spec(RENDERER_SPEC)
RENDERER_SPEC.loader.exec_module(RENDERER)


class B3CanaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text())

    def test_frozen_config_is_valid_and_full_coverage(self) -> None:
        MODULE.validate_frozen_config(self.config)
        for session in self.config["sessions"]:
            rows = MODULE.validate_windows(session["frame_count"], session["windows"])
            self.assertEqual(sum(row["end_exclusive"] - row["start"] for row in rows), session["frame_count"])

    def test_session_contract_has_only_rgb_as_semantic_input(self) -> None:
        for session in self.config["sessions"]:
            self.assertEqual(
                set(session), {"session_id", "task", "frame_count", "prepared_rgb", "windows"}
            )
        policy = self.config["fixed_runtime_policy"]
        self.assertEqual(policy["left_right_identity"], "UNKNOWN")
        self.assertEqual(policy["cross_chunk_identity"], "NOT_STITCHED")
        self.assertFalse(policy["consumer_allowed"])

    def test_geometry_selection_is_deterministic_and_capped(self) -> None:
        masks = np.zeros((3, 20, 20), bool)
        masks[0, 1:11, 1:11] = True
        masks[1, 2:12, 2:12] = True
        masks[2, :, :] = True
        ids = np.asarray([20, 10, 30])
        scores = np.asarray([0.8, 0.9, 0.99])
        selected, rows = MODULE.select_geometry_candidates(
            masks, scores, ids,
            minimum_area_pixels=20,
            maximum_area_fraction=0.5,
            maximum_components=2,
            maximum_candidates=1,
        )
        self.assertEqual(selected, [10])
        self.assertIn("AREA_ABOVE_MAXIMUM", rows[2]["reasons"])
        self.assertIn("ABOVE_FROZEN_CANDIDATE_CAP", rows[0]["reasons"])

    def test_overlap_is_hard_rejection_without_subtraction(self) -> None:
        hand = np.zeros((10, 10), bool)
        equipment = np.zeros((10, 10), bool)
        hand[2:6, 2:6] = True
        equipment[5:8, 5:8] = True
        gate = MODULE.separation_gate([hand], [equipment])
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["state"], "SEPARATION_REJECTED_OVERLAP")
        self.assertEqual(gate["total_overlap_pixels"], 1)
        self.assertEqual(gate["hard_max_overlap_pixels"], 0)
        np.testing.assert_array_equal(hand[2:6, 2:6], np.ones((4, 4), bool))

    def test_disjoint_candidates_pass_geometry_only_gate(self) -> None:
        hand = np.zeros((10, 10), bool)
        equipment = np.zeros((10, 10), bool)
        hand[1:3, 1:3] = True
        equipment[7:9, 7:9] = True
        gate = MODULE.separation_gate([hand], [equipment])
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["state"], "SEPARATION_PASS_GEOMETRY_ONLY")
        self.assertEqual(gate["total_overlap_pixels"], 0)

    def test_missing_role_is_unknown_not_vacuous_pass(self) -> None:
        hand = np.zeros((10, 10), bool)
        hand[1:3, 1:3] = True
        gate = MODULE.separation_gate([hand], [])
        self.assertFalse(gate["pass"])
        self.assertEqual(gate["state"], "SEPARATION_UNKNOWN_NO_EQUIPMENT_CANDIDATE")
        self.assertEqual(
            MODULE.aggregate_separation_state([gate["state"]]),
            "COMPLETED_SEPARATION_UNKNOWN",
        )

    def test_session_aggregation_prefers_rejection_then_unknown(self) -> None:
        self.assertEqual(
            MODULE.aggregate_separation_state([
                "SEPARATION_PASS_GEOMETRY_ONLY", "SEPARATION_REJECTED_OVERLAP"
            ]),
            "COMPLETED_SEPARATION_REJECTED",
        )
        self.assertEqual(
            MODULE.aggregate_separation_state([
                "SEPARATION_PASS_GEOMETRY_ONLY", "SEPARATION_UNKNOWN_NO_HAND_CANDIDATE"
            ]),
            "COMPLETED_SEPARATION_UNKNOWN",
        )

    def test_same_parent_staging_commit_and_path_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            final = parent / "final"
            staging = MODULE.begin_staging(final)
            artifact = staging / "nested" / "mask.npz"
            artifact.parent.mkdir()
            artifact.write_bytes(b"candidate")
            ref = MODULE.published_ref(artifact, staging, final)
            self.assertEqual(Path(ref["path"]), final / "nested" / "mask.npz")
            self.assertNotIn(".staging-", ref["path"])
            MODULE.commit_staging(staging, final)
            self.assertTrue((final / "nested" / "mask.npz").is_file())
            self.assertFalse(staging.exists())

    def test_aborted_staging_leaves_no_partial_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "final"
            staging = MODULE.begin_staging(final)
            (staging / "partial").write_text("partial")
            MODULE.abort_staging(staging)
            self.assertFalse(staging.exists())
            self.assertFalse(final.exists())

    def test_preflight_does_not_enter_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            final = Path(directory) / "preflight"
            with mock.patch.object(MODULE, "validate_all_pins", return_value={"pins": "verified"}), \
                 mock.patch.object(MODULE, "_run_model_into", side_effect=AssertionError("model entered")):
                result = MODULE.run_preflight(
                    CONFIG_PATH, Path(directory), final
                )
            self.assertEqual(result["status"], "PREFLIGHT_PASSED")
            self.assertFalse(result["model_loaded"])
            self.assertTrue((final / "PREFLIGHT_RECEIPT.json").is_file())

    def test_shallow_renderer_is_fresh_atomic_and_projects_final_ref(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deep = root / "deep"
            deep.mkdir()
            video = deep / "review.mp4"
            video.write_bytes(b"review-video")
            deep_result = {
                "status": "COMPLETED_DIAGNOSTIC_SEPARATION_UNKNOWN",
                "consumer_allowed": False,
                "mask_accuracy_claimed": False,
                "sessions": [{
                    "session_id": "play_cards_0915_044",
                    "status": "COMPLETED_SEPARATION_UNKNOWN",
                    "frame_count": 166,
                    "review": {"path": str(video), "bytes": video.stat().st_size,
                               "sha256": RENDERER.sha256(video)},
                }],
            }
            (deep / "RESULT.json").write_text(json.dumps(deep_result))
            final = root / "visual"
            fake_decode = {"full_decode": True, "frames": 166, "fps": "30/1",
                           "width": 1920, "height": 480}
            with mock.patch.object(RENDERER, "decode_gate", return_value=fake_decode):
                receipt = RENDERER.publish(deep, final)
            self.assertEqual(receipt["status"], "COMMITTED")
            published = Path(receipt["sessions"][0]["published"]["path"])
            self.assertEqual(published, final / "review.mp4")
            self.assertNotIn(".staging-", str(published))
            self.assertTrue((final / "SHALLOW_PUBLICATION_RECEIPT.json").is_file())

    def test_policy_drift_fails_closed(self) -> None:
        altered = json.loads(json.dumps(self.config))
        altered["fixed_runtime_policy"]["consumer_allowed"] = True
        with self.assertRaises(ValueError):
            MODULE.validate_frozen_config(altered)

    def test_left_right_or_extra_input_is_forbidden(self) -> None:
        altered = json.loads(json.dumps(self.config))
        altered["sessions"][0]["hawor"] = {"path": "forbidden"}
        with self.assertRaises(ValueError):
            MODULE.validate_frozen_config(altered)


if __name__ == "__main__":
    unittest.main()
