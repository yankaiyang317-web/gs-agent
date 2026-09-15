"""Minimal agent-to-image environment, deliberately without collision in MVP."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from gs_env.camera import CameraController
from gs_env.types import CameraIntrinsics, CameraPose, Observation
from gs_env.types import MoveResult
from gs_env.geometry.unproject import camera_point_to_world, pixel_to_camera_point, robust_bbox_depth, robust_pixel_depth


class Renderer(Protocol):
    """The small renderer surface the environment needs."""

    @property
    def is_loaded(self) -> bool: ...

    def load_scene(self, ply_path: str | Path): ...

    def render(self, camera_to_world: np.ndarray, intrinsics: CameraIntrinsics) -> Observation: ...


class GSEnvironment:
    """Own one scene and one controllable camera.

    This MVP intentionally permits direct pose setting.  Every pose-changing
    method is collision-free for now; later collision checks will be inserted
    before the controller state is committed, without changing this API.
    """

    def __init__(self, renderer: Renderer, initial_pose: CameraPose | None = None, collision_backend=None, camera_radius: float = 0.0, camera_body_height: float = 0.0, collision_step: float = 0.05, world_up: np.ndarray | list[float] | None = None) -> None:
        self.renderer = renderer
        self.camera = CameraController(initial_pose=initial_pose, world_up=world_up)
        self.collision_backend = collision_backend
        self.camera_radius = camera_radius
        self.camera_body_height = camera_body_height
        self.collision_step = collision_step

    @property
    def is_loaded(self) -> bool:
        return self.renderer.is_loaded

    def load_scene(self, ply_path: str | Path):
        return self.renderer.load_scene(ply_path)

    def get_pose(self) -> CameraPose:
        return self.camera.get_pose()

    def get_camera_to_world(self) -> np.ndarray:
        return self.camera.get_camera_to_world()

    def get_world_up(self) -> np.ndarray:
        """Return the fixed vertical direction inferred at initialization."""
        return self.camera.world_up

    def restore_world_up(self, world_up: np.ndarray | list[float]) -> None:
        """Restore a campaign vertical axis during process startup."""
        self.camera.restore_world_up(world_up)

    def set_pose(self, pose: CameraPose) -> CameraPose:
        self.camera.set_pose(pose)
        return self.get_pose()

    def set_camera_to_world(self, camera_to_world: np.ndarray) -> CameraPose:
        self.camera.set_camera_to_world(camera_to_world)
        return self.get_pose()

    def translate_local(self, *, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> CameraPose:
        return self.camera.translate_local(forward=forward, right=right, up=up)

    def move_local(self, *, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> MoveResult:
        """Move on public navigation axes, stopping before collision."""
        requested = self.camera.navigation_displacement(forward=forward, right=right, up=up)
        distance = float(np.linalg.norm(requested))
        if distance == 0:
            return MoveResult(0.0, 0.0, False, self.get_pose())
        if self.collision_backend is None:
            pose = self.camera.translate_world(requested)
            return MoveResult(distance, distance, False, pose)
        world_direction = requested / distance
        start = self.get_pose().position
        executed = self._max_free_body_distance(start, world_direction, distance)
        pose = self.camera.translate_world(world_direction * executed)
        return MoveResult(distance, executed, executed + 1e-9 < distance, pose)

    def move_world(self, displacement: np.ndarray | list[float]) -> MoveResult:
        """Collision-check and execute one explicit straight world displacement."""
        vector = np.asarray(displacement, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("world displacement must be a finite 3D vector")
        distance = float(np.linalg.norm(vector))
        if distance == 0:
            return MoveResult(0.0, 0.0, False, self.get_pose())
        return self._move_world(vector / distance, distance)

    def rotate_local(self, *, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> CameraPose:
        return self.camera.rotate_local(yaw_deg=yaw_deg, pitch_deg=pitch_deg)

    def observe(self, intrinsics: CameraIntrinsics) -> Observation:
        """Render the scene from the current pose without changing it."""
        if not self.is_loaded:
            raise RuntimeError("No scene loaded. Call load_scene() before observe().")
        return self.renderer.render(self.get_camera_to_world(), intrinsics)

    def approach_pixel(self, pixel: tuple[int, int], stop_distance: float, intrinsics: CameraIntrinsics) -> tuple[MoveResult, Observation, np.ndarray]:
        """Use current depth to move safely toward an image target and look at it."""
        if not np.isfinite(stop_distance) or stop_distance < 0:
            raise ValueError("stop_distance must be finite and non-negative")
        before = self.observe(intrinsics)
        depth = robust_pixel_depth(before.depth, before.alpha, pixel)
        return self._approach_depth_target(pixel, depth, stop_distance, intrinsics, before)

    def approach_bbox(self, bbox: tuple[int, int, int, int], stop_distance: float, intrinsics: CameraIntrinsics) -> tuple[MoveResult, Observation, np.ndarray]:
        """Approach a box-selected target using the box's median reliable depth."""
        if not np.isfinite(stop_distance) or stop_distance < 0:
            raise ValueError("stop_distance must be finite and non-negative")
        before = self.observe(intrinsics)
        pixel, depth = robust_bbox_depth(before.depth, before.alpha, bbox)
        return self._approach_depth_target(pixel, depth, stop_distance, intrinsics, before)

    def _approach_depth_target(self, pixel: tuple[int, int], depth: float, stop_distance: float, intrinsics: CameraIntrinsics, before: Observation) -> tuple[MoveResult, Observation, np.ndarray]:
        target = camera_point_to_world(pixel_to_camera_point(pixel[0], pixel[1], depth, intrinsics), before.camera_to_world)
        start = self.get_pose().position
        ray = target - start
        target_distance = float(np.linalg.norm(ray))
        requested = max(0.0, target_distance - stop_distance)
        if requested == 0:
            move = MoveResult(0.0, 0.0, False, self.get_pose())
        else:
            move = self._move_world(ray / target_distance, requested)
        # Approaching a visible target is a translation operation.  Preserve the
        # camera orientation so a low/overhead target cannot inject an implicit
        # 180-degree roll through a world-up look-at reconstruction.  Callers
        # that want to reframe the target can do so explicitly with rotate().
        return move, self.observe(intrinsics), target

    def _move_world(self, world_direction: np.ndarray, distance: float) -> MoveResult:
        """Move along an explicit world ray for depth-target approaches."""
        direction = np.asarray(world_direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if distance == 0 or norm == 0:
            return MoveResult(distance, 0.0, False, self.get_pose())
        direction = direction / norm
        if self.collision_backend is None:
            pose = self.camera.translate_world(direction * distance)
            return MoveResult(distance, distance, False, pose)
        start = self.get_pose().position
        executed = self._max_free_body_distance(start, direction, distance)
        pose = self.camera.translate_world(direction * executed)
        return MoveResult(distance, executed, executed + 1e-9 < distance, pose)

    def _max_free_body_distance(self, camera_position: np.ndarray, direction: np.ndarray, distance: float) -> float:
        """Return free travel for a vertical capsule whose top sphere is the camera.

        The mesh backend checks one analytic capsule extending down from the
        camera along the session's fixed vertical axis.
        """
        return self.collision_backend.max_free_body_distance(
            camera_position,
            direction,
            distance,
            self.camera_radius,
            self.camera_body_height,
            self.get_world_up(),
            self.collision_step,
        )
