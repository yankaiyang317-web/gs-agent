"""Tests for pixel-aligned visual observation encodings."""

from __future__ import annotations

import unittest
import numpy as np

from gs_env import CameraIntrinsics, Observation
from gs_mcp.observation_images import depth_preview


class ObservationImageTests(unittest.TestCase):
    def test_depth_preview_is_rgb_aligned_and_masks_invalid_alpha(self) -> None:
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        alpha = np.array([[1.0, 0.7], [0.69, 0.0]], dtype=np.float32)
        observation = Observation(np.zeros((2, 2, 3), dtype=np.float32), depth, alpha, np.eye(4), CameraIntrinsics(1, 1, 1, 1, 2, 2))
        preview = depth_preview(observation)
        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertFalse(np.array_equal(preview[0, 1], [0, 0, 0]))
        self.assertTrue(np.array_equal(preview[1, 0], [0, 0, 0]))
        self.assertTrue(np.array_equal(preview[1, 1], [0, 0, 0]))
        self.assertFalse(np.array_equal(preview[0, 0], preview[0, 1]))
