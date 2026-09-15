"""Contract tests for the unified HTTP Campaign profiles."""

import tempfile
from pathlib import Path

from gs_mcp.runtime import SceneRuntime


def test_development_and_demo_profiles_share_one_runtime_entry() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scenes = root / "scenes"
        scenes.mkdir()
        runtime = SceneRuntime(scenes, root / "exploration", demo_runs_root=root / "demo", device="cpu")
        development = runtime._campaign_settings("development", None, None, None, None)
        demo = runtime._campaign_settings("demo", None, None, None, None)
        assert (development["width"], development["height"]) == (960, 720)
        assert (development["video_segment_frames"], development["video_fps"]) == (81, 9.0)
        assert development["runs_root"] == (root / "exploration").resolve()
        assert (demo["width"], demo["height"]) == (2560, 1920)
        assert (demo["video_segment_frames"], demo["video_fps"]) == (270, 9.0)
        assert demo["runs_root"] == (root / "demo").resolve()


def test_http_campaign_profile_allows_explicit_render_overrides() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scenes = root / "scenes"
        scenes.mkdir()
        runtime = SceneRuntime(scenes, root / "exploration", device="cpu")
        settings = runtime._campaign_settings("demo", 1920, 1440, 270, 12)
        assert (settings["width"], settings["height"]) == (1920, 1440)
        assert settings["video_segment_frames"] == 270
        assert settings["video_fps"] == 12.0
