"""CPU-only tests for robust depth selection."""

from __future__ import annotations

import unittest

import numpy as np

from gs_env.geometry.unproject import robust_bbox_depth


class RobustBBoxDepthTests(unittest.TestCase):
    def test_bbox_uses_median_reliable_depth_and_centre_pixel(self) -> None:
        depth = np.array([[1.0, 2.0, 100.0], [3.0, 4.0, 200.0]], dtype=np.float32)
        alpha = np.array([[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]], dtype=np.float32)
        pixel, selected_depth = robust_bbox_depth(depth, alpha, (0, 0, 3, 2))
        self.assertEqual(pixel, (1, 0))
        self.assertAlmostEqual(selected_depth, 2.5)

    def test_bbox_rejects_invalid_or_unreliable_regions(self) -> None:
        depth = np.ones((2, 2), dtype=np.float32)
        alpha = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            robust_bbox_depth(depth, alpha, (0, 0, 2, 2))
        with self.assertRaises(ValueError):
            robust_bbox_depth(depth, np.ones((2, 2), dtype=np.float32), (1, 0, 1, 2))

    def test_default_depth_confidence_threshold_is_point_seven(self) -> None:
        depth = np.array([[1.0, 9.0], [3.0, 11.0]], dtype=np.float32)
        alpha = np.array([[0.7, 0.69], [0.8, 0.2]], dtype=np.float32)
        _, selected_depth = robust_bbox_depth(depth, alpha, (0, 0, 2, 2))
        self.assertAlmostEqual(selected_depth, 2.0)
