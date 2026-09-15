"""CPU-only tests for isolated per-start exploration run allocation."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from gs_env.types import CameraPose
from gs_mcp.campaign import ExplorationCampaign
from gs_mcp.server import create_exploration_run


class ServerRunTests(unittest.TestCase):
    def test_each_start_allocates_a_distinct_run_directory(self) -> None:
        fixed = datetime(2026, 8, 6, 16, 0, 0, 123456)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "guju"
            first = create_exploration_run(root, fixed)
            second = create_exploration_run(root, fixed)
            self.assertEqual(first.name, "run_20260806_160000_123456")
            self.assertEqual(second.name, "run_20260806_160000_123456_001")
            self.assertNotEqual(first, second)

    def test_runs_do_not_share_campaign_state(self) -> None:
        pose = CameraPose(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float64))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "guju"
            first_dir = create_exploration_run(root)
            second_dir = create_exploration_run(root)
            first = ExplorationCampaign(first_dir, pose, camera_radius=0.25, target_video_clips=2)
            second = ExplorationCampaign(second_dir, pose, camera_radius=0.25, target_video_clips=5)
            first.state["frames_recorded"] = 81
            first._save()
            self.assertEqual(first.state["target_video_clips"], 2)
            self.assertEqual(second.state["target_video_clips"], 5)
            self.assertEqual(second.state["frames_recorded"], 0)
            self.assertTrue((first_dir / "campaign.json").is_file())
            self.assertTrue((second_dir / "campaign.json").is_file())


if __name__ == "__main__":
    unittest.main()
