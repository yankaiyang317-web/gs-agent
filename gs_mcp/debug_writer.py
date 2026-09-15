"""Optional local-only observation writer for inspecting an MCP session."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from gs_env.types import Observation
from gs_mcp.observation_images import depth_preview
from gs_mcp.campaign import VIDEO_FPS, VIDEO_SEGMENT_FRAMES

if TYPE_CHECKING:
    from gs_mcp.campaign import ExplorationCampaign


class DebugObservationWriter:
    """Append Agent-visible observations to a directory; disabled unless constructed."""

    def __init__(self, output_dir: str | Path, create_session: bool = False, campaign: "ExplorationCampaign | None" = None, video_segment_frames: int = VIDEO_SEGMENT_FRAMES, video_fps: float = VIDEO_FPS) -> None:
        base_dir = Path(output_dir)
        self.output_dir = base_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}" if create_session else base_dir
        self.output_dir.joinpath("rgb").mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath("depth").mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath("depth_preview").mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath("alpha").mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.campaign = campaign
        self.video_segment_frames = video_segment_frames
        self.video_fps = video_fps

    def save(self, observation: Observation, event: dict | None = None) -> Path:
        stem = f"{self.index:06d}"
        rgb_path = self.output_dir / "rgb" / f"{stem}.png"
        Image.fromarray((np.clip(observation.rgb, 0, 1) * 255).round().astype(np.uint8)).save(rgb_path)
        np.save(self.output_dir / "depth" / f"{stem}.npy", observation.depth)
        Image.fromarray(depth_preview(observation)).save(self.output_dir / "depth_preview" / f"{stem}.png")
        np.save(self.output_dir / "alpha" / f"{stem}.npy", observation.alpha)
        record = {"frame": self.index, "camera_to_world": observation.camera_to_world.tolist(), "intrinsics": vars(observation.intrinsics)}
        if event is not None:
            record["event"] = event
        with (self.output_dir / "trajectory.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.index += 1
        if self.campaign is not None:
            self.campaign.record_frame(rgb_path, observation, event)
        # Completed clips become available during a long active task. Encoding
        # must stay outside the MCP tool-call path; otherwise ffmpeg can consume
        # the 300-second tool timeout. The final partial clip is exported by the
        # server's normal-shutdown handler.
        if self.campaign is None and self.index % self.video_segment_frames == 0:
            try:
                from gs_mcp.video import export_completed_clip_in_background
                export_completed_clip_in_background(self.output_dir, self.index // self.video_segment_frames - 1, fps=self.video_fps, max_frames=self.video_segment_frames)
            except Exception:
                # Recording must never fail merely because a local preview video
                # cannot be encoded; raw PNG frames remain the source of truth.
                pass
        return rgb_path
