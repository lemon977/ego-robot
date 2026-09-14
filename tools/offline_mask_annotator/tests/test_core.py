from __future__ import annotations

import unittest
from decimal import Decimal

from PIL import Image

from mask_annotator.media import ProbeFrame, evenly_spaced_indices, resolve_timestamps
from mask_annotator.raster import binary_mask, rasterize


class SamplingTests(unittest.TestCase):
    def test_even_sampling_includes_ends_and_is_unique(self):
        indices = evenly_spaced_indices(100, 24)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertEqual(len(indices), 24)
        self.assertEqual(len(set(indices)), 24)

    def test_single_sample_is_middle(self):
        self.assertEqual(evenly_spaced_indices(11, 1), (5,))

    def test_explicit_vfr_timestamps_resolve_to_nearest_unique_pts(self):
        frames = tuple(
            ProbeFrame(index, index * 10, Decimal(value), None, False)
            for index, value in enumerate(("0.00", "0.07", "0.21", "0.49", "0.95"))
        )
        self.assertEqual(
            resolve_timestamps(frames, (Decimal("0.06"), Decimal("0.50"), Decimal("0.90"))),
            (1, 3, 4),
        )


class RasterTests(unittest.TestCase):
    def test_later_operation_overwrites_and_binary_is_exact(self):
        mask = rasterize([
            {"op_id": "a", "kind": "polygon", "class_id": 1, "points": [[1, 1], [8, 1], [8, 8], [1, 8]]},
            {"op_id": "b", "kind": "polygon", "class_id": 5, "points": [[4, 4], [9, 4], [9, 9], [4, 9]]},
        ], (12, 12))
        self.assertEqual(mask.getpixel((2, 2)), 1)
        self.assertEqual(mask.getpixel((5, 5)), 5)
        self.assertEqual(mask.getpixel((0, 0)), 0)
        binary = binary_mask(mask, 5)
        self.assertEqual(set(binary.getdata()), {0, 255})

    def test_background_operation_erases(self):
        mask = rasterize([
            {"op_id": "a", "kind": "brush", "class_id": 2, "radius": 3, "points": [[5, 5], [10, 5]]},
            {"op_id": "b", "kind": "brush", "class_id": 0, "radius": 1, "points": [[7, 5]]},
        ], (16, 12))
        self.assertEqual(mask.getpixel((7, 5)), 0)
        self.assertEqual(mask.getpixel((5, 5)), 2)


if __name__ == "__main__":
    unittest.main()
