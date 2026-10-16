import os
import unittest
from unittest.mock import patch

import numpy as np

from chaoyang.exact78_baseline.native_input import nearest, matrices, head_to_optical, MANUS21
from chaoyang.exact78_baseline import pico_source


class NativeInputTests(unittest.TestCase):
    def test_nearest_is_deterministic_and_does_not_drop_video_frames(self):
        got = nearest([100, 200, 300], [110, 120, 290, 310])
        np.testing.assert_array_equal(got, [0, 0, 2, 2])

    def test_backward_clock_rejected(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OR_BACKWARD"):
            nearest([100, 100, 200], [100, 150])

    def test_pose_quaternion_not_silently_repaired(self):
        with self.assertRaisesRegex(ValueError, "QUATERNION"):
            matrices([0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(matrices([0, 0, 0, 0, 0, 0, 1]), np.eye(4))

    def test_invalid_physical_eye_rejected(self):
        with self.assertRaisesRegex(ValueError, "PHYSICAL_LEFT"):
            head_to_optical({"extrinsicConvention": "head_to_camera_3x4_row_major", "left": {"sourceIndex": 0}})

    def test_no_native_manifest_preserves_legacy_lookup(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(pico_source.glob, "glob", return_value=[]):
            with self.assertRaisesRegex(pico_source.PicoSourceError, "SESSION_LOOKUP"):
                pico_source.find_session("synthetic_missing_session")

    def test_mano_order_has_wrist_and_all_five_fingertips(self):
        self.assertEqual(list(MANUS21[[0, 4, 8, 12, 16, 20]]), [0, 4, 9, 14, 19, 24])


if __name__ == "__main__":
    unittest.main()
