import unittest

import numpy as np

from tools.evaluate_h50 import block_fde_mm, rotation_error_deg, rotation_matrix
from utils.utils_math import rotmat_to_o6d


class H50RotationEncodingTests(unittest.TestCase):
    def test_interleaved_o6d_round_trip(self) -> None:
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        encoded = rotmat_to_o6d(rotation)[None]
        np.testing.assert_allclose(rotation_matrix(encoded)[0], rotation, atol=1e-7)

    def test_known_quarter_turn_error(self) -> None:
        identity = rotmat_to_o6d(np.eye(3))[None]
        quarter_turn = rotmat_to_o6d(
            np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
        )[None]
        self.assertAlmostEqual(
            float(rotation_error_deg(identity, quarter_turn)[0]), 90.0, places=5
        )

    def test_block_fde_is_reported_in_millimetres(self) -> None:
        self.assertAlmostEqual(block_fde_mm([0.1, 0.2]), 150.0)
        self.assertEqual(block_fde_mm([]), 0.0)


if __name__ == "__main__":
    unittest.main()
