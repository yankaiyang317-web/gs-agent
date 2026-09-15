"""Tests for debug-video grouping without invoking an encoder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gs_mcp.video import export_completed_clip_in_background, frame_groups


class ExportDebugVideoTests(unittest.TestCase):
    def test_groups_numeric_frames_into_81_frame_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rgb = Path(directory)
            for index in range(83):
                rgb.joinpath(f"{index:06d}.png").touch()
            groups = frame_groups(rgb, 81)
            self.assertEqual([len(group) for group in groups], [81, 2])
            self.assertEqual(groups[0][0].name, "000000.png")
            self.assertEqual(groups[1][0].name, "000081.png")

    @patch("gs_mcp.video.shutil.which", return_value="ffmpeg")
    @patch("gs_mcp.video.subprocess.Popen")
    def test_completed_clip_starts_encoder_without_waiting(self, popen, _which) -> None:
        with tempfile.TemporaryDirectory() as directory:
            popen.return_value.wait.return_value = 1
            self.assertIsNotNone(export_completed_clip_in_background(Path(directory), 1))
            command = popen.call_args.args[0]
            self.assertIn("81", command)
            self.assertIn("-nostdin", command)
            self.assertTrue(str(command[-1]).endswith("clip_001.encoding.mp4"))
            self.assertIs(popen.call_args.kwargs["stdin"], __import__("subprocess").DEVNULL)

    @patch("gs_mcp.video.shutil.which", return_value="ffmpeg")
    @patch("gs_mcp.video.subprocess.Popen")
    def test_custom_fps_and_segment_size_reach_encoder(self, popen, _which) -> None:
        with tempfile.TemporaryDirectory() as directory:
            popen.return_value.wait.return_value = 1
            export_completed_clip_in_background(Path(directory), 1, fps=12.5, max_frames=270)
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("-framerate") + 1], "12.5")
            self.assertEqual(command[command.index("-start_number") + 1], "270")
            self.assertEqual(command[command.index("-frames:v") + 1], "270")
