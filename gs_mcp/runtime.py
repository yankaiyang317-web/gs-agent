"""Runtime scene selection for a long-lived MCP process."""

from __future__ import annotations

import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from gs_env import CameraIntrinsics, GSEnvironment, load_scene_manifest
from gs_env.collision import create_collision_backend
from gs_env.rendering import GSplatRenderer
from gs_mcp.campaign import ExplorationCampaign, VIDEO_FPS, VIDEO_SEGMENT_FRAMES
from gs_mcp.debug_writer import DebugObservationWriter
from gs_mcp.server import create_exploration_run
from gs_mcp.tools import EnvironmentTools
from gs_mcp.video import export_groups, wait_for_all_exports


_SCENE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
CAMPAIGN_PROFILES = {
    "development": {"width": 960, "height": 720, "video_segment_frames": VIDEO_SEGMENT_FRAMES, "video_fps": VIDEO_FPS},
    "demo": {"width": 2560, "height": 1920, "video_segment_frames": 270, "video_fps": VIDEO_FPS},
}


class EnvironmentToolsProxy:
    """Keep MCP tool registrations stable while replacing their scene backend."""

    def __init__(self, lock: RLock | None = None) -> None:
        self._current: EnvironmentTools | None = None
        self._lock = lock or RLock()

    def bind(self, tools: EnvironmentTools | None) -> None:
        with self._lock:
            self._current = tools

    @property
    def is_loaded(self) -> bool:
        return self._current is not None

    def __getattr__(self, name: str):
        def invoke(*args, **kwargs):
            with self._lock:
                current = self._current
                if current is None:
                    raise RuntimeError("no scene is loaded; call gs_configure_campaign with a server-side scene first")
                return getattr(current, name)(*args, **kwargs)

        return invoke


@dataclass
class LoadedScene:
    key: str
    manifest_path: Path
    tools: EnvironmentTools
    environment: GSEnvironment
    campaign: ExplorationCampaign
    writer: DebugObservationWriter
    gaussian_count: int
    load_seconds: float
    profile: str
    render_width: int
    render_height: int


def scaled_intrinsics(source: CameraIntrinsics, width: int | None, height: int | None) -> CameraIntrinsics:
    """Scale image size and pinhole intrinsics without changing field of view."""
    if width is None and height is None:
        return source
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("render width and height must both be positive")
    sx, sy = width / source.width, height / source.height
    return CameraIntrinsics(source.fx * sx, source.fy * sy, source.cx * sx, source.cy * sy, width, height)


class SceneRuntime:
    """Discover and load one server-owned scene for the lifetime of a task."""

    def __init__(
        self,
        scenes_root: str | Path,
        runs_root: str | Path,
        *,
        demo_runs_root: str | Path | None = None,
        device: str = "cuda",
        debug_distance_interval: float = 0.25,
        debug_rotation_interval_deg: float = 15.0,
        max_navigation_actions: int = 0,
        blocked_retry_limit: int = 2,
    ) -> None:
        self.scenes_root = Path(scenes_root).resolve()
        self.runs_root = Path(runs_root).resolve()
        self.demo_runs_root = Path(demo_runs_root).resolve() if demo_runs_root is not None else self.runs_root.parent / "demo_runs"
        if not self.scenes_root.is_dir():
            raise ValueError(f"scenes root does not exist: {self.scenes_root}")
        self.device = device
        self.debug_distance_interval = debug_distance_interval
        self.debug_rotation_interval_deg = debug_rotation_interval_deg
        self.max_navigation_actions = max_navigation_actions
        self.blocked_retry_limit = blocked_retry_limit
        self._lock = RLock()
        self.proxy = EnvironmentToolsProxy(self._lock)
        self.current: LoadedScene | None = None

    def list_scenes(self) -> dict[str, Any]:
        scenes: list[dict[str, Any]] = []
        for path in sorted(self.scenes_root.glob("*.json")):
            item: dict[str, Any] = {"scene": path.stem, "manifest": str(path)}
            try:
                manifest = load_scene_manifest(path)
                item.update({"available": True, "name": manifest.name})
            except Exception as error:
                item.update({"available": False, "error": str(error)})
            scenes.append(item)
        return {"scenes": scenes, "loaded_scene": None if self.current is None else self.current.key}

    def describe_scene(self, scene: str) -> dict[str, Any]:
        """Validate one registered manifest without allocating GPU resources."""
        key = scene.strip()
        if not _SCENE_NAME.fullmatch(key):
            raise ValueError("scene must contain only letters, numbers, underscore, or hyphen")
        manifest_path = (self.scenes_root / f"{key}.json").resolve()
        if manifest_path.parent != self.scenes_root or not manifest_path.is_file():
            raise ValueError(f"scene manifest not found: {key}")
        manifest = load_scene_manifest(manifest_path)
        return {
            "scene": key,
            "name": manifest.name,
            "available": True,
            "has_collision_mesh": manifest.collision_mesh_path is not None,
            "camera": {"width": manifest.intrinsics.width, "height": manifest.intrinsics.height},
            "profiles": sorted(CAMPAIGN_PROFILES),
        }

    def load_scene(
        self,
        scene: str,
        *,
        profile: str = "development",
        width: int | None = None,
        height: int | None = None,
        video_segment_frames: int | None = None,
        video_fps: float | None = None,
    ) -> dict[str, Any]:
        key = scene.strip()
        if not _SCENE_NAME.fullmatch(key):
            raise ValueError("scene must contain only letters, numbers, underscore, or hyphen")
        settings = self.campaign_settings(profile, width, height, video_segment_frames, video_fps)
        manifest_path = (self.scenes_root / f"{key}.json").resolve()
        if manifest_path.parent != self.scenes_root or not manifest_path.is_file():
            raise ValueError(f"scene manifest not found: {key}")
        with self._lock:
            if self.current is not None:
                same_configuration = (
                    self.current.key == key
                    and self.current.profile == settings["profile"]
                    and self.current.render_width == settings["width"]
                    and self.current.render_height == settings["height"]
                    and self.current.campaign.video_segment_frames == settings["video_segment_frames"]
                    and self.current.campaign.video_fps == settings["video_fps"]
                )
                if same_configuration and not self.current.campaign.is_complete:
                    return {"loaded": True, "already_loaded": True, **self.scene_status()}
                if not self.current.campaign.is_complete:
                    raise RuntimeError(
                        f"cannot replace scene {self.current.key} or its runtime profile while its campaign is incomplete; "
                        "finish the configured clip target first"
                    )
                self._unload_current(export_partial=False)
            manifest = load_scene_manifest(manifest_path)
            intrinsics = scaled_intrinsics(manifest.intrinsics, settings["width"], settings["height"])
            collision = create_collision_backend(
                mesh_path=manifest.collision_mesh_path,
                world_to_asset=manifest.collision_world_to_asset,
            )
            environment = GSEnvironment(
                GSplatRenderer(device=self.device),
                initial_pose=manifest.initial_pose,
                collision_backend=collision,
                camera_radius=manifest.camera_radius,
                camera_body_height=manifest.camera_body_height,
                collision_step=manifest.collision_step,
                world_up=manifest.world_up,
            )
            gaussian_scene = environment.load_scene(manifest.ply_path)
            run_path = create_exploration_run(settings["runs_root"] / manifest.name)
            campaign = ExplorationCampaign(
                run_path,
                environment.get_pose(),
                manifest.camera_radius,
                None,
                world_up=manifest.world_up,
                video_segment_frames=settings["video_segment_frames"],
                video_fps=settings["video_fps"],
            )
            environment.restore_world_up(campaign.world_up)
            writer = DebugObservationWriter(campaign.root / "debug", create_session=False, campaign=campaign)
            tools = EnvironmentTools(
                environment,
                intrinsics,
                debug_writer=writer,
                debug_distance_interval=self.debug_distance_interval,
                debug_rotation_interval_deg=self.debug_rotation_interval_deg,
                max_navigation_actions=self.max_navigation_actions,
                blocked_retry_limit=self.blocked_retry_limit,
                campaign=campaign,
            )
            self.current = LoadedScene(
                key=key,
                manifest_path=manifest_path,
                tools=tools,
                environment=environment,
                campaign=campaign,
                writer=writer,
                gaussian_count=int(gaussian_scene.count),
                load_seconds=float(gaussian_scene.load_seconds),
                profile=settings["profile"],
                render_width=settings["width"],
                render_height=settings["height"],
            )
            self.proxy.bind(tools)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"loaded": True, "already_loaded": False, **self.scene_status()}

    def configure_campaign(
        self,
        scene: str,
        video_clips: int,
        objective: str,
        profile: str = "development",
        width: int | None = None,
        height: int | None = None,
        video_segment_frames: int | None = None,
        video_fps: float | None = None,
    ) -> dict[str, Any]:
        """Atomically load/switch a scene and configure its exact campaign target."""
        with self._lock:
            loaded = self.load_scene(
                scene,
                profile=profile,
                width=width,
                height=height,
                video_segment_frames=video_segment_frames,
                video_fps=video_fps,
            )
            campaign = self.proxy.configure_campaign(video_clips, objective)
            return {"scene": loaded, "campaign": campaign}

    def scene_status(self) -> dict[str, Any]:
        current = self.current
        if current is None:
            return {"scene": None, "loaded": False}
        return {
            "scene": current.key,
            "loaded": True,
            "manifest": str(current.manifest_path),
            "gaussian_count": current.gaussian_count,
            "load_seconds": current.load_seconds,
            "profile": current.profile,
            "render_width": current.render_width,
            "render_height": current.render_height,
            "intrinsics": vars(current.tools.intrinsics),
            "world_up": current.environment.get_world_up().tolist(),
            "collision_backend": None if current.environment.collision_backend is None else current.environment.collision_backend.backend_name,
            "collision_step": current.environment.collision_step,
            "capsule": {
                "radius": current.environment.camera_radius,
                "body_height": current.environment.camera_body_height,
                "total_height": 2 * current.environment.camera_radius + current.environment.camera_body_height,
            },
            "campaign": current.campaign.status(),
        }

    def campaign_settings(
        self,
        profile: str,
        width: int | None,
        height: int | None,
        video_segment_frames: int | None,
        video_fps: float | None,
    ) -> dict[str, Any]:
        key = profile.strip().lower()
        if key not in CAMPAIGN_PROFILES:
            raise ValueError(f"profile must be one of {sorted(CAMPAIGN_PROFILES)}")
        defaults = CAMPAIGN_PROFILES[key]
        resolved_width = int(defaults["width"] if width is None else width)
        resolved_height = int(defaults["height"] if height is None else height)
        resolved_frames = int(defaults["video_segment_frames"] if video_segment_frames is None else video_segment_frames)
        resolved_fps = float(defaults["video_fps"] if video_fps is None else video_fps)
        if resolved_width <= 0 or resolved_height <= 0:
            raise ValueError("width and height must be positive")
        if (width is None) != (height is None):
            raise ValueError("width and height must be supplied together")
        if resolved_frames <= 0 or resolved_fps <= 0:
            raise ValueError("video segment frames and FPS must be positive")
        return {
            "profile": key,
            "width": resolved_width,
            "height": resolved_height,
            "video_segment_frames": resolved_frames,
            "video_fps": resolved_fps,
            "runs_root": self.runs_root if key == "development" else self.demo_runs_root,
        }

    def close(self) -> None:
        with self._lock:
            self._unload_current(export_partial=True)

    def _unload_current(self, *, export_partial: bool) -> None:
        scene = self.current
        if scene is None:
            return
        self.proxy.bind(None)
        self.current = None
        self._finalize(scene, export_partial=export_partial)
        scene.tools.debug_writer = None
        scene.tools.campaign = None
        scene.environment.renderer.scene = None
        del scene
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    @staticmethod
    def _finalize(scene: LoadedScene, *, export_partial: bool) -> None:
        wait_for_all_exports()
        frames = int(scene.campaign.state["frames_recorded"])
        if export_partial and frames % scene.campaign.video_segment_frames:
            try:
                for output in export_groups(scene.campaign.root, scene.campaign.root / "video", fps=scene.campaign.video_fps, max_frames=scene.campaign.video_segment_frames):
                    print(f"Saved debug video: {output}", file=sys.stderr)
            except Exception as error:
                print(f"Debug video export failed: {error}", file=sys.stderr)
