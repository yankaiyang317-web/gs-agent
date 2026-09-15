"""Stdio MCP server for the minimal Agent-to-image MVP."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from gs_env import CameraIntrinsics, CameraPose, GSEnvironment, load_scene_manifest
from gs_env.geometry.colmap import load_colmap_initial_view, transform_colmap_pose_to_gs
from gs_env.collision import create_collision_backend
from gs_env.rendering import GSplatRenderer
from gs_mcp.debug_writer import DebugObservationWriter
from gs_mcp.campaign import ExplorationCampaign, VIDEO_FPS, VIDEO_SEGMENT_FRAMES
from gs_mcp.observation_images import depth_preview
from gs_mcp.tools import EnvironmentTools
from gs_mcp.video import export_groups, wait_for_all_exports


DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_CAMERA_RADIUS = 0.25


def create_exploration_run(root: str | Path, now: datetime | None = None) -> Path:
    """Create one unique run directory; callers never share campaign.json."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    for suffix in range(1000):
        name = f"run_{stamp}" if suffix == 0 else f"run_{stamp}_{suffix:03d}"
        candidate = base / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate a unique exploration run under {base}")


def _intrinsics_from_args(args: argparse.Namespace, fallback: CameraIntrinsics | None = None) -> CameraIntrinsics:
    """Apply optional CLI camera overrides to a scene preset or portable defaults."""
    has_override = any(value is not None for value in (args.width, args.height, args.fx, args.fy))
    if fallback is not None and not has_override:
        return fallback
    width = args.width if args.width is not None else DEFAULT_WIDTH
    height = args.height if args.height is not None else DEFAULT_HEIGHT
    fx = args.fx if args.fx is not None else width / 2
    fy = args.fy if args.fy is not None else height / 2
    cx = width / 2
    cy = height / 2
    return CameraIntrinsics(fx, fy, cx, cy, width, height)


def _pose_from_args(args: argparse.Namespace, fallback: CameraPose | None = None) -> CameraPose | None:
    """Use a complete explicit pose, or leave the configured/default pose unchanged."""
    if args.position is None and args.quaternion_wxyz is None:
        return fallback
    if args.position is None or args.quaternion_wxyz is None:
        raise ValueError("--position and --quaternion-wxyz must be supplied together")
    return CameraPose(np.asarray(args.position, dtype=np.float64), np.asarray(args.quaternion_wxyz, dtype=np.float64))


def _collision_defaults(collision: Any | None, radius: float | None, step: float | None) -> tuple[float, float]:
    """Use portable defaults when no scene manifest supplies collision values."""
    if collision is None:
        return (0.0 if radius is None else radius, 0.05 if step is None else step)
    return (DEFAULT_CAMERA_RADIUS if radius is None else radius, 0.05 if step is None else step)


def _observation_result(observation: Any, extra: dict[str, Any] | None = None):
    """Build text plus image MCP content; importing the SDK only at runtime."""
    from mcp.types import CallToolResult, ImageContent, TextContent

    image = PILImage.fromarray((np.clip(observation.rgb, 0, 1) * 255).round().astype(np.uint8))
    depth_image = PILImage.fromarray(depth_preview(observation))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    depth_buffer = io.BytesIO()
    depth_image.save(depth_buffer, format="PNG")
    metadata = {
        "camera_to_world": observation.camera_to_world.tolist(),
        "intrinsics": vars(observation.intrinsics),
        "depth_shape": list(observation.depth.shape),
        "alpha_shape": list(observation.alpha.shape),
        "depth_definition": "accumulated camera-space z-depth; alpha below 0.7 is invalid",
        "images": {"rgb": "RGB image", "depth_preview": "same-resolution depth: warm=near, cool=far, black=invalid"},
    }
    if extra is not None:
        metadata["action"] = extra
    return CallToolResult(content=[
        TextContent(type="text", text=json.dumps(metadata)),
        ImageContent(type="image", data=base64.b64encode(buffer.getvalue()).decode("ascii"), mimeType="image/png"),
        ImageContent(type="image", data=base64.b64encode(depth_buffer.getvalue()).decode("ascii"), mimeType="image/png"),
    ])


def create_server(tools: EnvironmentTools, scene_runtime: Any | None = None, fastmcp_options: dict[str, Any] | None = None):
    """Wrap one already-loaded environment with the official FastMCP server."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("gs-agent", **(fastmcp_options or {}))

    @mcp.tool()
    def gs_observe():
        """Render and return current RGB plus depth/alpha metadata and C2W pose."""
        return _observation_result(tools.observe())

    @mcp.tool()
    def gs_get_pose() -> dict[str, Any]:
        """Return pose, C2W matrix, and the session's fixed world-up axis."""
        return tools.get_pose()

    @mcp.tool()
    def gs_set_pose(position: list[float], quaternion_wxyz: list[float]) -> dict[str, Any]:
        """Set the MVP pose directly. This is intentionally not collision checked."""
        return tools.set_pose(position, quaternion_wxyz)

    @mcp.tool()
    def gs_move(direction: str, distance: float) -> dict[str, Any]:
        """Move in scene units: view-forward, level-sideways, or world-vertical."""
        return tools.move(direction, distance)

    @mcp.tool()
    def gs_approach_target(pixel: list[int] | None = None, stop_distance: float = 0.0, bbox: list[int] | None = None) -> Any:
        """Safely approach one visible pixel or one XYXY bbox; supply exactly one target."""
        result, observation = tools.approach_target(pixel, stop_distance, bbox)
        return _observation_result(observation, result)

    @mcp.tool()
    def gs_rotate(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> dict[str, Any]:
        """Turn on a stable horizon: positive yaw right and pitch up."""
        return tools.rotate(yaw_deg, pitch_deg)

    @mcp.tool()
    def gs_record_finding(description: str = "", boundary_candidate: bool = False) -> dict[str, Any]:
        """Record one low-quality region; set boundary_candidate only for a visually likely scene-ending direction."""
        return tools.record_finding(description, boundary_candidate)

    if scene_runtime is None:
        @mcp.tool()
        def gs_configure_campaign(video_clips: int, objective: str = "coverage") -> dict[str, Any]:
            """Set this run's exact configured clip total and exploration objective."""
            return tools.configure_campaign(video_clips, objective)
    else:
        @mcp.tool()
        def gs_configure_campaign(
            scene: str,
            video_clips: int,
            objective: str = "coverage",
            profile: str = "development",
            width: int | None = None,
            height: int | None = None,
            video_segment_frames: int | None = None,
            video_fps: float | None = None,
        ) -> dict[str, Any]:
            """Start one HTTP Campaign with a server-side scene and runtime profile.

            Development defaults to 960x720, 81 frames, and 9 FPS under
            outputs/exploration_runs. Demo defaults to 2560x1920, 270 frames,
            and 9 FPS under outputs/demo_runs. Explicit numeric overrides apply
            only to the new Campaign and do not change navigation behavior.
            """
            if not isinstance(video_clips, int) or isinstance(video_clips, bool) or video_clips <= 0:
                raise ValueError("video_clips must be a positive integer")
            if objective not in {"coverage", "reconstruction_quality"}:
                raise ValueError("objective must be coverage or reconstruction_quality")
            return scene_runtime.configure_campaign(
                scene,
                video_clips,
                objective,
                profile=profile,
                width=width,
                height=height,
                video_segment_frames=video_segment_frames,
                video_fps=video_fps,
            )

    @mcp.tool()
    def gs_report_exploration_candidates(
        route_frontiers: list[dict[str, Any]] | None = None,
        viewpoint_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Report finite routes or useful 6DoF viewpoints by pixel.

        Routes include doors and corridors, but also macro travel axes across
        open rooms/halls, meaningful far ends, open functional zones, and
        distinct sides of large occluders. Visibility from afar is not spatial
        coverage. Report one candidate per distinct connected direction or
        semantic destination, not every open ray.
        """
        return tools.report_exploration_candidates(route_frontiers, viewpoint_candidates)

    @mcp.tool()
    def gs_get_exploration_status() -> dict[str, Any]:
        """Return topology coverage, repetition detection, and the next recommendation."""
        return tools.get_exploration_status()

    @mcp.tool()
    def gs_restore_checkpoint(checkpoint_id: int, reason: str = "") -> dict[str, Any]:
        """Restore a persisted safe checkpoint only at a clip boundary or for recovery."""
        return tools.restore_checkpoint(checkpoint_id, reason)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gs-agent stdio MCP server")
    parser.add_argument("--scene-manifest", type=Path, default=None, help="JSON scene configuration; paths are relative to it")
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--position", nargs=3, type=float, default=None)
    parser.add_argument("--quaternion-wxyz", nargs=4, type=float, default=None)
    parser.add_argument("--world-up", nargs=3, type=float, default=None, help="Optional fixed scene-up vector; default derives from initial image-up")
    parser.add_argument("--colmap-images", type=Path, default=None)
    parser.add_argument("--colmap-cameras", type=Path, default=None)
    parser.add_argument("--colmap-image", default=None, help="Image name; default is the first COLMAP image")
    parser.add_argument("--colmap-similarity", type=Path, default=None, help="Optional COLMAP-world to GS-world similarity JSON")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--collision-mesh", type=Path, default=None, help="Static GLB triangle mesh for Coal analytic-capsule collision")
    parser.add_argument("--camera-radius", type=float, default=None)
    parser.add_argument("--camera-body-height", type=float, default=None, help="Vertical capsule length below the camera in scene units")
    parser.add_argument("--collision-step", type=float, default=None)
    parser.add_argument("--save-observations", type=Path, default=None, help="Standalone debug-session root when no campaign is active")
    campaign_group = parser.add_mutually_exclusive_group()
    campaign_group.add_argument("--exploration-campaign", type=Path, default=None, help="Run root; every server start creates a new run_* directory below it")
    campaign_group.add_argument("--resume-campaign", type=Path, default=None, help="Explicit existing run directory containing campaign.json")
    parser.add_argument("--target-video-clips", type=int, default=0, help="Exact number of video clips for this run; 0 defers configuration to MCP")
    parser.add_argument("--video-segment-frames", type=int, default=VIDEO_SEGMENT_FRAMES, help="Frames per video clip")
    parser.add_argument("--video-fps", type=float, default=VIDEO_FPS, help="Video encoding frame rate")
    parser.add_argument("--debug-distance-interval", type=float, default=0.25, help="Scene-unit spacing of real debug renders; ignored unless saving observations")
    parser.add_argument("--debug-rotation-interval-deg", type=float, default=15.0, help="Angular spacing of real debug renders; ignored unless saving observations")
    parser.add_argument("--max-navigation-actions", type=int, default=0, help="Maximum move/rotate/approach actions; 0 disables the task action cap")
    parser.add_argument("--blocked-retry-limit", type=int, default=2, help="Repeated near-zero-progress attempts allowed per unchanged direction")
    args = parser.parse_args()
    requested_world_up = None if args.world_up is None else np.asarray(args.world_up, dtype=np.float64)
    if args.target_video_clips < 0:
        parser.error("--target-video-clips must be non-negative")
    if args.video_segment_frames <= 0:
        parser.error("--video-segment-frames must be positive")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if args.target_video_clips and args.exploration_campaign is None and args.resume_campaign is None:
        parser.error("--target-video-clips requires --exploration-campaign or --resume-campaign")
    if args.resume_campaign is not None and not (args.resume_campaign / "campaign.json").is_file():
        parser.error("--resume-campaign must contain an existing campaign.json")
    if (args.position is None) != (args.quaternion_wxyz is None):
        parser.error("--position and --quaternion-wxyz must be supplied together")
    if args.scene_manifest is not None:
        conflicting = [args.ply, args.colmap_images, args.colmap_cameras]
        if any(value is not None for value in conflicting):
            parser.error("--scene-manifest cannot be combined with --ply or COLMAP arguments")
        manifest = load_scene_manifest(args.scene_manifest)
        ply_path = manifest.ply_path
        initial_pose = _pose_from_args(args, manifest.initial_pose)
        intrinsics = _intrinsics_from_args(args, manifest.intrinsics)
        collision_mesh_path = args.collision_mesh or manifest.collision_mesh_path
        collision_world_to_asset = manifest.collision_world_to_asset
        navigation_world_up = manifest.world_up if requested_world_up is None else requested_world_up
    else:
        if args.ply is None:
            parser.error("supply --ply or --scene-manifest")
        ply_path = args.ply
        collision_mesh_path = args.collision_mesh
        collision_world_to_asset = np.eye(4)
    if (args.colmap_images is None) != (args.colmap_cameras is None):
        parser.error("--colmap-images and --colmap-cameras must be supplied together")
    if args.colmap_images is not None and args.position is not None:
        parser.error("use either explicit --position/--quaternion-wxyz or COLMAP inputs")
    if args.scene_manifest is None and args.colmap_images is not None:
        width = DEFAULT_WIDTH if args.width is None else args.width
        height = DEFAULT_HEIGHT if args.height is None else args.height
        initial_pose, intrinsics, name = load_colmap_initial_view(args.colmap_images, args.colmap_cameras, width, height, args.colmap_image)
        if args.colmap_similarity is not None:
            initial_pose = transform_colmap_pose_to_gs(initial_pose, args.colmap_similarity)
        print(f"Using COLMAP initial image: {name}", file=sys.stderr)
    elif args.scene_manifest is None:
        initial_pose = _pose_from_args(args)
        intrinsics = _intrinsics_from_args(args)
    if args.scene_manifest is None:
        navigation_world_up = requested_world_up
    collision = create_collision_backend(
        mesh_path=collision_mesh_path,
        world_to_asset=collision_world_to_asset,
    )
    if args.scene_manifest is not None:
        default_radius, default_step = manifest.camera_radius, manifest.collision_step
        camera_radius = default_radius if args.camera_radius is None else args.camera_radius
        camera_body_height = manifest.camera_body_height if args.camera_body_height is None else args.camera_body_height
        collision_step = default_step if args.collision_step is None else args.collision_step
    else:
        camera_radius, collision_step = _collision_defaults(collision, args.camera_radius, args.collision_step)
        camera_body_height = 0.0 if args.camera_body_height is None else args.camera_body_height
    if camera_body_height < 0:
        parser.error("--camera-body-height must be non-negative")
    environment = GSEnvironment(GSplatRenderer(device=args.device), initial_pose=initial_pose, collision_backend=collision, camera_radius=camera_radius, camera_body_height=camera_body_height, collision_step=collision_step, world_up=navigation_world_up)
    campaign = None
    campaign_path = create_exploration_run(args.exploration_campaign) if args.exploration_campaign is not None else args.resume_campaign
    if campaign_path is not None:
        campaign = ExplorationCampaign(campaign_path, environment.get_pose(), camera_radius, None if args.target_video_clips == 0 else args.target_video_clips, world_up=environment.get_world_up(), video_segment_frames=args.video_segment_frames, video_fps=args.video_fps)
        environment.restore_world_up(campaign.world_up)
        if args.resume_campaign is not None:
            environment.set_pose(campaign.choose_start_pose(environment.get_pose()))
    environment.load_scene(ply_path)
    if campaign is not None:
        writer = DebugObservationWriter(campaign.root / "debug", create_session=False, campaign=campaign)
    else:
        writer = None if args.save_observations is None else DebugObservationWriter(args.save_observations, create_session=True, video_segment_frames=args.video_segment_frames, video_fps=args.video_fps)
    try:
        create_server(EnvironmentTools(environment, intrinsics, debug_writer=writer, debug_distance_interval=args.debug_distance_interval, debug_rotation_interval_deg=args.debug_rotation_interval_deg, max_navigation_actions=args.max_navigation_actions, blocked_retry_limit=args.blocked_retry_limit, campaign=campaign)).run()
    finally:
        if writer is not None:
            try:
                wait_for_all_exports()
                source = campaign.root if campaign is not None else writer.output_dir
                segment_frames = campaign.video_segment_frames if campaign is not None else args.video_segment_frames
                video_fps = campaign.video_fps if campaign is not None else args.video_fps
                for output in export_groups(source, source / "video", fps=video_fps, max_frames=segment_frames):
                    print(f"Saved debug video: {output}", file=sys.stderr)
            except Exception as error:
                print(f"Debug video export failed: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
