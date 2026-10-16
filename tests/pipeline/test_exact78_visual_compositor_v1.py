import unittest

import numpy as np

from chaoyang.pipeline.exact78_visual_compositor_v1 import (
    VisualOwnershipConfig, compose_visual_frame,
)


class VisualCompositorTests(unittest.TestCase):
    def arrays(self, height=48, width=64):
        clean = np.full((height, width, 3), 37, np.uint8)
        robot = np.full_like(clean, 201)
        alpha = np.ones((height, width), bool)
        empty = np.zeros_like(alpha)
        return clean, robot, alpha, empty.copy(), empty.copy(), empty.copy()

    def test_observed_empty_objects(self):
        a = self.arrays()
        result = compose_visual_frame(*a, robot_observed=True, masks_available=True)
        np.testing.assert_array_equal(result["image"], a[1])

    def test_unknown_motion_is_not_fabricated(self):
        a = self.arrays()
        r = compose_visual_frame(*a, robot_observed=False, masks_available=True)
        np.testing.assert_array_equal(r["image"], a[0])
        self.assertFalse(r["draw"].any())

    def test_far_visible_object_preserved(self):
        a = self.arrays()
        a[3][20:25, 40:45] = True
        a[4][:, 0] = True
        r = compose_visual_frame(*a, robot_observed=True, masks_available=True)
        np.testing.assert_array_equal(r["image"][a[3]], a[0][a[3]])

    def test_near_object_rule_is_visual_only(self):
        a = self.arrays(480, 640)
        a[3][10, 10] = True
        a[4][10, 9] = True
        r = compose_visual_frame(*a, robot_observed=True, masks_available=True)
        self.assertTrue(r["robot_front"][10, 10])
        self.assertFalse(r["physical_occlusion_authority"])
        self.assertFalse(r["training_eligible"])

    def test_unknown_object_not_observed(self):
        a = self.arrays()
        a[3][10, 10] = a[5][10, 10] = True
        r = compose_visual_frame(*a, robot_observed=True, masks_available=True)
        self.assertFalse(r["object_front"][10, 10])
        self.assertTrue(r["object_unknown_overlap"][10, 10])

    def test_missing_masks_explicit(self):
        r = compose_visual_frame(*self.arrays(), robot_observed=True, masks_available=False)
        self.assertEqual(r["mask_coverage"], "UNAVAILABLE")

    def test_inputs_not_mutated(self):
        a = self.arrays()
        before = [v.copy() for v in a]
        compose_visual_frame(*a, robot_observed=False, masks_available=True)
        for x, y in zip(a, before):
            np.testing.assert_array_equal(x, y)

    def test_uint16_role_labels_cannot_silently_be_masks(self):
        a = list(self.arrays())
        a[4] = a[4].astype(np.uint16)
        with self.assertRaisesRegex(ValueError, "MASK_DOMAIN_OR_TYPE"):
            compose_visual_frame(*a, robot_observed=True, masks_available=True)

    def test_missing_masks_cannot_have_observations(self):
        a = self.arrays()
        a[3][0, 0] = True
        with self.assertRaisesRegex(ValueError, "UNAVAILABLE_MASKS"):
            compose_visual_frame(*a, robot_observed=True, masks_available=False)

    def test_wrong_domain_rejected(self):
        a = list(self.arrays())
        a[2] = np.zeros((20, 20), bool)
        with self.assertRaises(ValueError):
            compose_visual_frame(*a, robot_observed=True, masks_available=True)

    def test_config_nan_rejected(self):
        with self.assertRaises(ValueError):
            VisualOwnershipConfig(near_px=float("nan"))


if __name__ == "__main__":
    unittest.main()
