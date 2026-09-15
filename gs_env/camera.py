"""Stateful, horizon-stable camera controller."""

from __future__ import annotations

import numpy as np

from gs_env.geometry.transforms import (
    axis_angle_rotation,
    camera_to_world_from_pose,
    pose_from_camera_to_world,
    rotation_matrix_to_quaternion,
)
from gs_env.types import CameraPose


# C2W at the documented zero CLI pose: forward is world -Z and image-up is +Y.
DEFAULT_C2W_ROTATION = np.diag([1.0, -1.0, -1.0])


class CameraController:
    """Maintain pose with world-stable navigation controls.

    The initial image-up direction defines ``world_up`` for the session.
    Positive yaw turns right around that fixed axis, positive pitch looks up,
    and roll is deliberately unavailable to agent-facing navigation.
    """

    MAX_PITCH_DEG = 85.0

    def __init__(self, initial_pose: CameraPose | None = None, world_up: np.ndarray | list[float] | None = None) -> None:
        self._pose = initial_pose or CameraPose(
            position=np.zeros(3, dtype=np.float64),
            quaternion_wxyz=rotation_matrix_to_quaternion(DEFAULT_C2W_ROTATION),
        )
        self.set_pose(self._pose)
        initial_rotation = camera_to_world_from_pose(self._pose)[:3, :3].astype(np.float64)
        up_source = -initial_rotation[:, 1] if world_up is None else np.asarray(world_up, dtype=np.float64)
        self._world_up = _normalize(up_source, "initial world-up")

    def get_pose(self) -> CameraPose:
        """Return a defensive copy of the current canonical pose."""
        return CameraPose(self._pose.position.copy(), self._pose.quaternion_wxyz.copy())

    def get_camera_to_world(self) -> np.ndarray:
        return camera_to_world_from_pose(self._pose)

    @property
    def world_up(self) -> np.ndarray:
        """Return the fixed scene-up estimate derived from the initial image."""
        return self._world_up.copy()

    def restore_world_up(self, world_up: np.ndarray | list[float]) -> None:
        """Restore a persisted session axis before navigation begins."""
        self._world_up = _normalize(np.asarray(world_up, dtype=np.float64), "persisted world-up")

    def set_pose(self, pose: CameraPose) -> None:
        """Set a finite pose after normalizing through the canonical C2W path."""
        self._pose = pose_from_camera_to_world(camera_to_world_from_pose(pose))

    def set_camera_to_world(self, camera_to_world: np.ndarray) -> None:
        self._pose = pose_from_camera_to_world(camera_to_world)

    def translate_local(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> CameraPose:
        """Translate using view-forward, level-right, and fixed world-up axes."""
        amounts = np.asarray([forward, right, up], dtype=np.float64)
        if not np.isfinite(amounts).all():
            raise ValueError("translation distances must be finite")
        rotation = camera_to_world_from_pose(self._pose)[:3, :3].astype(np.float64)
        view_forward = rotation[:, 2]
        level_right = self._level_right(view_forward, rotation[:, 0])
        displacement = forward * view_forward + right * level_right + up * self._world_up
        self._pose = CameraPose(
            self._pose.position + displacement,
            self._pose.quaternion_wxyz.copy(),
        )
        return self.get_pose()

    def rotate_local(self, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> CameraPose:
        """Apply horizon-stable yaw and clamped pitch without accumulating roll."""
        angles = np.asarray([yaw_deg, pitch_deg], dtype=np.float64)
        if not np.isfinite(angles).all():
            raise ValueError("rotation angles must be finite")
        current_rotation = camera_to_world_from_pose(self._pose)[:3, :3].astype(np.float64)
        forward = _normalize(current_rotation[:, 2], "camera forward")
        # Right turns are negative right-handed rotations around image-up.
        forward = axis_angle_rotation(self._world_up, np.deg2rad(-yaw_deg)) @ forward

        current_pitch = np.degrees(np.arcsin(np.clip(np.dot(forward, self._world_up), -1.0, 1.0)))
        target_pitch = float(np.clip(current_pitch + pitch_deg, -self.MAX_PITCH_DEG, self.MAX_PITCH_DEG))
        pitch_delta = np.deg2rad(target_pitch - current_pitch)
        right = self._level_right(forward, current_rotation[:, 0])
        forward = _normalize(axis_angle_rotation(right, pitch_delta) @ forward, "camera forward")

        # Rebuild the basis from fixed up after every action. This explicitly
        # removes any roll present in the previous pose instead of merely
        # omitting a roll command while local-axis rotations keep coupling.
        right = self._level_right(forward, right)
        image_up = _normalize(np.cross(right, forward), "camera image-up")
        stable_rotation = np.column_stack((right, -image_up, forward))
        self._pose = CameraPose(
            self._pose.position.copy(),
            rotation_matrix_to_quaternion(stable_rotation),
        )
        return self.get_pose()

    def navigation_displacement(self, *, forward: float = 0.0, right: float = 0.0, up: float = 0.0) -> np.ndarray:
        """Return a world displacement for the public navigation axes."""
        rotation = camera_to_world_from_pose(self._pose)[:3, :3].astype(np.float64)
        return forward * rotation[:, 2] + right * self._level_right(rotation[:, 2], rotation[:, 0]) + up * self._world_up

    def translate_world(self, displacement: np.ndarray | list[float]) -> CameraPose:
        """Translate by an explicit world-space vector without changing orientation."""
        vector = np.asarray(displacement, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("world displacement must be a finite 3D vector")
        self._pose = CameraPose(self._pose.position + vector, self._pose.quaternion_wxyz.copy())
        return self.get_pose()

    def _level_right(self, forward: np.ndarray, fallback_right: np.ndarray) -> np.ndarray:
        candidate = np.cross(forward, self._world_up)
        if np.linalg.norm(candidate) < 1e-8:
            candidate = fallback_right - np.dot(fallback_right, self._world_up) * self._world_up
        return _normalize(candidate, "level camera-right")


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-12:
        raise ValueError(f"{name} must be a finite non-zero 3D vector")
    return value / norm
