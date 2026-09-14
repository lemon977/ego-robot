import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_local_registered_stereo_clean_successor import (
    LocalRegistration,
    fit_local_table_registration,
    load_stereo_authority_depth,
    robust_colour_gate,
    sample_raw_from_rectified,
    transform_xy,
)


class LocalStereoCleanTest(unittest.TestCase):
    def test_registration_acceptance_property(self):
        passed = LocalRegistration(np.eye(3), "PASS", 40, 35, 28, 7, 0.7, 0.8, 1.0)
        failed = LocalRegistration(None, "HELDOUT_GATE_FAIL", 40, 35, 28, 7, 2.1, 1.0, 0.85)
        self.assertTrue(passed.accepted)
        self.assertFalse(failed.accepted)

    def test_flat_images_fail_closed(self):
        image = np.zeros((960, 1280, 3), np.uint8)
        depth = np.ones((480, 640), np.float32)
        table = np.ones((480, 640), bool)
        result = fit_local_table_registration(
            image, image, depth, table,
            np.array([[0.5, 0.0, -0.25], [0.0, 0.5, -0.25], [0.0, 0.0, 1.0]]),
            np.array([[320.0, 0.0, 319.5], [0.0, 320.0, 239.5], [0.0, 0.0, 1.0]]),
            0.0637716504026918,
        )
        self.assertFalse(result.accepted)

    def test_transform_xy(self):
        x, y, valid = transform_xy(
            np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]]),
            np.array([1.0, 5.0]), np.array([2.0, 8.0]),
        )
        np.testing.assert_allclose(x, [4.0, 8.0])
        np.testing.assert_allclose(y, [0.0, 6.0])
        self.assertTrue(valid.all())

    def test_integer_raw_sampling_and_bounds(self):
        map_x, map_y = np.meshgrid(
            np.arange(1280, dtype=np.float32), np.arange(960, dtype=np.float32)
        )
        x, y, valid = sample_raw_from_rectified(
            np.array([2.2, 1278.6, -1.0]), np.array([3.4, 958.6, 4.0]), map_x, map_y
        )
        self.assertEqual((int(x[0]), int(y[0])), (2, 3))
        self.assertTrue(bool(valid[0]))
        self.assertFalse(bool(valid[2]))

    def test_robust_colour_gate(self):
        right = np.tile(np.arange(100)[:, None], (1, 3)).astype(np.float64)
        left = right * 1.1 + 4.0
        coefficients, q99 = robust_colour_gate(left, right)
        prediction = np.column_stack((right, np.ones(len(right)))) @ coefficients
        self.assertLess(float(np.max(np.abs(prediction - left))), 1e-8)
        self.assertLess(q99, 1e-8)

    def test_depth_loader_rejects_wrong_frame_before_io(self):
        with self.assertRaisesRegex(Exception, "identity mismatch"):
            load_stereo_authority_depth(Path("/does/not/matter"), {"frame_id": 7}, 8)


if __name__ == "__main__":
    unittest.main()
