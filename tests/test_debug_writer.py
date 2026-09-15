"""Tests for opt-in local observation saving."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gs_env import CameraIntrinsics, Observation
from gs_mcp.debug_writer import DebugObservationWriter


class DebugWriterTests(unittest.TestCase):
    def test_save_creates_rgb_arrays_and_pose_record(self) -> None:
        observation = Observation(np.ones((2, 3, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32), np.eye(4, dtype=np.float32), CameraIntrinsics(1, 1, 1, 1, 3, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = DebugObservationWriter(directory).save(observation)
            self.assertTrue(path.is_file())
            self.assertTrue((Path(directory) / "depth" / "000000.npy").is_file())
            self.assertTrue((Path(directory) / "trajectory.jsonl").is_file())

    def test_create_session_places_outputs_under_unique_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = DebugObservationWriter(directory, create_session=True)
            self.assertEqual(writer.output_dir.parent, Path(directory))
            self.assertTrue(writer.output_dir.name.startswith("session_"))
