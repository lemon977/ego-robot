"""CPU-only tests: one tracker equipment prompt union, no model invocation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.ops import run_exact78_fresh_frontend_v2 as v2
from chaoyang.ops import run_exact78_fresh_frontend_v3 as v3
from chaoyang.ops import run_exact78_fresh_frontend_v4_tracker_equipment as v4


class TrackerEquipmentCandidateTests(unittest.TestCase):
    def test_pinned_v3_closure_is_exact(self):
        v4.verify_pinned_closure()

    def test_tracker_prompt_pair_is_generic_and_identical_across_0901_0902(self):
        chips = v4.prompt_plan({"source_family": "TRACKER_0901", "task": "chips"})
        cards = v4.prompt_plan({"source_family": "TRACKER_0902", "task": "poker"})
        self.assertEqual({key: chips[key] for key in v4.TRACKER_EQUIPMENT_PROMPTS},
                         {key: cards[key] for key in v4.TRACKER_EQUIPMENT_PROMPTS})
        self.assertEqual(cards, {
            "human": "a person's hand and forearm", "object": "a playing card",
            "equipment_tracker": "a wrist worn tracking device",
            "equipment_wristband": "a wristband",
        })
        self.assertEqual(chips["object"], "a potato chip")

    def test_0915_plan_and_mask_stage_remain_original(self):
        spec = {"source_family": "VST_0915", "task": "poker"}
        self.assertEqual(v4.prompt_plan(spec), v2._prompts(spec))
        self.assertNotIn("wristband", " ".join(v4.prompt_plan(spec).values()))
        with patch.object(v4, "ORIGINAL_MASKS") as original:
            v4.masks(spec, Path("unused"))
            original.assert_called_once_with(spec, Path("unused"))

    def test_synonym_union_provenance_and_object_protection(self):
        blank = np.zeros((3, 4), dtype=bool)
        human = blank.copy(); human[0, 0] = True
        object_mask = blank.copy(); object_mask[1, 1] = True
        tracker = blank.copy(); tracker[0, 1] = True; tracker[1, 1] = True
        band = blank.copy(); band[1, 1] = True; band[2, 2] = True
        selected = {"human": human, "object": object_mask,
                    "equipment_tracker": tracker, "equipment_wristband": band}
        images, pixels = v4.compose_tracker_roles(selected)
        self.assertEqual(pixels, {"equipment_tracker": 2, "equipment_wristband": 2,
                                  "equipment_union": 3, "equipment_wristband_only": 1})
        self.assertEqual(v4.equipment_provenance(pixels),
                         (["equipment_tracker", "equipment_wristband"],
                          "MODEL_OUTPUT_UNREVIEWED"))
        np.testing.assert_array_equal(images["object"], object_mask)
        np.testing.assert_array_equal(images["unknown"], images["human_equipment"] & object_mask)
        np.testing.assert_array_equal(images["protected"], object_mask & ~images["human_equipment"])
        np.testing.assert_array_equal(images["removal"], images["human_equipment"] & ~object_mask)
        self.assertFalse(np.any(images["removal"] & object_mask))

    def test_empty_device_stays_unverified_not_a_quality_pass(self):
        blank = np.zeros((2, 2), dtype=bool)
        human = blank.copy(); human[0, 0] = True
        images, pixels = v4.compose_tracker_roles({
            "human": human, "object": blank.copy(),
            "equipment_tracker": blank.copy(), "equipment_wristband": blank.copy(),
        })
        self.assertEqual(pixels["equipment_union"], 0)
        self.assertEqual(v4.equipment_provenance(pixels), ([], "MASK_DEVICE_UNVERIFIED"))
        np.testing.assert_array_equal(images["removal"], human)

    def test_invalid_role_or_shape_rejected_before_stage_output(self):
        blank = np.zeros((2, 2), dtype=bool)
        selected = {"human": blank, "object": blank,
                    "equipment_tracker": blank, "equipment_wristband": blank}
        with self.assertRaisesRegex(ValueError, "TRACKER_ROLE_SET_DRIFT"):
            v4.compose_tracker_roles({k: v for k, v in selected.items() if k != "object"})
        with self.assertRaisesRegex(ValueError, "TRACKER_ROLE_MASK_DOMAIN"):
            v4.compose_tracker_roles({**selected, "equipment_wristband": np.zeros((1, 2), bool)})
        with self.assertRaisesRegex(ValueError, "TRACKER_ROLE_MASK_DOMAIN"):
            v4.compose_tracker_roles({**selected, "equipment_wristband": np.zeros((2, 2), np.uint8)})

    def test_v3_pin_drift_rejected_before_any_model_or_packet(self):
        with patch.object(v4, "PINNED_V3_SHA256", "0" * 64), \
             patch.object(v3, "main") as main:
            with self.assertRaisesRegex(RuntimeError, "V4_PINNED_V3_SHA_DRIFT"):
                v4.main()
            main.assert_not_called()

    def test_mask_patch_only_process_local_and_restored_on_failure(self):
        original = v2.masks
        original_map = v2.map_stage
        original_clean = v2.clean
        seen = []

        def fail_after_patch():
            seen.append(v2.masks)
            self.assertIs(v2.map_stage, original_map)
            self.assertIs(v2.clean, original_clean)
            raise RuntimeError("packet gate rejected")

        with patch.object(v4, "verify_pinned_closure"), \
             patch.object(v3, "main", side_effect=fail_after_patch):
            with self.assertRaisesRegex(RuntimeError, "packet gate rejected"):
                v4.main()
        self.assertEqual(seen, [v4.masks])
        self.assertIs(v2.masks, original)

    def test_preexisting_mask_patch_rejected(self):
        with patch.object(v4, "verify_pinned_closure"), \
             patch.object(v2, "masks", v4.masks), \
             patch.object(v3, "main") as main:
            with self.assertRaisesRegex(RuntimeError, "V4_V2_MASK_ALREADY_PATCHED"):
                v4.main()
            main.assert_not_called()

    def test_cli_without_spec_rejected_without_model(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m",
             "chaoyang.ops.run_exact78_fresh_frontend_v4_tracker_equipment",
             "--phase", "all"], capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--spec", result.stderr)


if __name__ == "__main__":
    unittest.main()
