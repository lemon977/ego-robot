"""CPU-only negative and algebra tests for three-source prompt candidate."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.ops import run_exact78_fresh_frontend_v2 as v2
from chaoyang.ops import run_exact78_fresh_frontend_v3 as v3
from chaoyang.ops import run_exact78_fresh_frontend_v5_white_device as v5


class WhiteDevicePromptTests(unittest.TestCase):
    def test_pinned_closure(self):
        v5.verify_pinned_closure()

    def test_one_identical_prompt_across_all_three_sources(self):
        for family, task in (("TRACKER_0901", "chips"), ("TRACKER_0902", "poker"),
                             ("VST_0915", "poker")):
            spec = {"source_family": family, "task": task}
            old = v2._prompts(spec)
            new = v5.prompt_plan(spec)
            self.assertEqual(new["equipment_white_device"],
                             "a white device attached to a hand or wrist")
            self.assertEqual(new["equipment_original"], old["equipment"])
            self.assertEqual(new["human"], old["human"])
            self.assertEqual(new["object"], old["object"])
            for role in set(old) - {"equipment"}:
                self.assertEqual(new[role], old[role])
            self.assertEqual(len(new), len(old) + 1)

    def test_tracker_union_and_protected_pixels_unchanged(self):
        blank = np.zeros((3, 4), bool)
        human = blank.copy(); human[0, 0] = True
        obj = blank.copy(); obj[2, 2] = True
        white = blank.copy(); white[0, 1] = True
        images, counts = v5.compose_roles({
            "human": human, "object": obj, "equipment_original": blank.copy(),
            "equipment_white_device": white,
        }, "TRACKER_0902")
        self.assertEqual(counts["equipment_white_device_new_pixels"], 1)
        self.assertEqual(counts["protected_lost_pixels"], 0)
        np.testing.assert_array_equal(images["protected"], obj)
        np.testing.assert_array_equal(images["removal"], human | white)
        self.assertFalse(np.any(images["removal"] & obj))

    def test_vst_sleeve_and_cable_baseline_rule_unchanged(self):
        blank = np.zeros((3, 4), bool)
        sleeve = blank.copy(); sleeve[0, 0] = True
        cable = blank.copy(); cable[0, 1] = True
        white = blank.copy(); white[0, 2] = True
        images, counts = v5.compose_roles({
            "human": blank.copy(), "object": blank.copy(),
            "equipment_original": blank.copy(), "equipment_white_device": white,
            "finger_sleeve": sleeve, "yellow_cable": cable,
        }, "VST_0915")
        self.assertEqual(counts["equipment_union_pixels"], 1)
        np.testing.assert_array_equal(images["removal"], sleeve | cable | white)
        np.testing.assert_array_equal(images["finger_sleeve"], sleeve)
        np.testing.assert_array_equal(images["yellow_cable"], cable)

    def test_new_prompt_cannot_erase_baseline_protected_object(self):
        blank = np.zeros((2, 2), bool)
        obj = blank.copy(); obj[1, 1] = True
        with self.assertRaisesRegex(RuntimeError, "VISIBLE_OBJECT_PROTECTION_REGRESSION"):
            v5.compose_roles({"human": blank.copy(), "object": obj,
                              "equipment_original": blank.copy(),
                              "equipment_white_device": obj.copy()}, "TRACKER_0901")

    def test_gross_expansion_is_reported_not_new_numerical_stop(self):
        blank = np.zeros((2, 2), bool)
        white = blank.copy(); white[0, 0] = True
        _, counts = v5.compose_roles({"human": blank.copy(), "object": blank.copy(),
                                      "equipment_original": blank.copy(),
                                      "equipment_white_device": white}, "TRACKER_0901")
        self.assertEqual(counts["equipment_white_device_new_pixels"], 1)
        self.assertEqual(counts["baseline_human_equipment_pixels"], 0)

    def test_malformed_role_or_shape_rejected(self):
        blank = np.zeros((2, 2), bool)
        roles = {"human": blank, "object": blank, "equipment_original": blank,
                 "equipment_white_device": blank}
        with self.assertRaisesRegex(ValueError, "WHITE_DEVICE_ROLE_SET_DRIFT"):
            v5.compose_roles({k: v for k, v in roles.items() if k != "human"}, "TRACKER_0902")
        with self.assertRaisesRegex(ValueError, "WHITE_DEVICE_ROLE_MASK_DOMAIN"):
            v5.compose_roles({**roles, "equipment_white_device": np.zeros((1, 2), bool)},
                             "TRACKER_0902")
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_SOURCE_FAMILY"):
            v5.compose_roles(roles, "OTHER")

    def test_patch_restored_after_packet_gate_failure(self):
        original = v2.masks
        original_clean = v2.clean
        def reject():
            self.assertIs(v2.masks, v5.masks)
            self.assertIs(v2.clean, original_clean)
            raise RuntimeError("packet gate rejected")
        with patch.object(v5, "verify_pinned_closure"), patch.object(v3, "main", side_effect=reject):
            with self.assertRaisesRegex(RuntimeError, "packet gate rejected"):
                v5.main()
        self.assertIs(v2.masks, original)

    def test_no_spec_or_route_cannot_run_model(self):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "chaoyang.ops.run_exact78_fresh_frontend_v5_white_device",
             "--phase", "all"], capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--spec", result.stderr)


if __name__ == "__main__":
    unittest.main()
