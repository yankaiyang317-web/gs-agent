"""Analytic capsule collision against a static GLB triangle-mesh BVH."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _rotation_from_z(axis: np.ndarray) -> np.ndarray:
    """Return a proper rotation whose local Z axis is ``axis``."""
    z = np.asarray(axis, dtype=np.float64)
    z /= np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


class MeshCapsuleCollisionBackend:
    """Use Coal's C++ BVH to query one true vertical capsule.

    The mesh is loaded and its BVH is built once when a scene is loaded. Each
    path sample checks one true capsule rather than an approximation.
    """

    supports_analytic_capsule = True
    backend_name = "mesh_capsule"

    def __init__(self, mesh_path: str | Path, world_to_asset: np.ndarray | None = None) -> None:
        try:
            import coal
        except ImportError as error:
            raise RuntimeError(
                "GLB capsule collision requires coal>=3.0.3; install the Linux runtime "
                "dependencies"
            ) from error
        self._coal = coal
        self.mesh_path = Path(mesh_path)
        transform = np.eye(4) if world_to_asset is None else np.asarray(world_to_asset, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("world_to_asset must be a finite 4x4 transform")
        self.world_to_asset = transform
        self._linear = transform[:3, :3]
        self._translation = transform[:3, 3]
        self.mesh = coal.MeshLoader().load(str(self.mesh_path))
        self._mesh_transform = coal.Transform3s()
        self._request = coal.CollisionRequest()
        self._request.num_max_contacts = 1
        self._request.enable_contact = False
        self._capsules: dict[tuple[float, float], object] = {}

    def _asset_point(self, world: np.ndarray) -> np.ndarray:
        return self._linear @ np.asarray(world, dtype=np.float64) + self._translation

    def _capsule(self, radius: float, body_height: float):
        key = (float(radius), float(body_height))
        capsule = self._capsules.get(key)
        if capsule is None:
            capsule = self._coal.Capsule(radius, body_height / 2.0)
            self._capsules[key] = capsule
        return capsule

    def body_is_free(self, camera_position: np.ndarray, radius: float, body_height: float, world_up: np.ndarray) -> bool:
        if radius < 0 or body_height < 0:
            raise ValueError("capsule radius and body height must be non-negative")
        up = np.asarray(world_up, dtype=np.float64)
        up_norm = float(np.linalg.norm(up))
        if up.shape != (3,) or not np.isfinite(up).all() or up_norm < 1e-12:
            raise ValueError("world_up must be a finite non-zero 3D vector")
        up /= up_norm
        center_world = np.asarray(camera_position, dtype=np.float64) - up * (body_height / 2.0)
        axis_asset = self._linear @ (-up)
        axis_asset /= np.linalg.norm(axis_asset)
        capsule_transform = self._coal.Transform3s(_rotation_from_z(axis_asset), self._asset_point(center_world))
        result = self._coal.CollisionResult()
        self._coal.collide(
            self.mesh,
            self._mesh_transform,
            self._capsule(radius, body_height),
            capsule_transform,
            self._request,
            result,
        )
        return not bool(result.isCollision())

    def max_free_body_distance(
        self,
        camera_position: np.ndarray,
        direction_world: np.ndarray,
        max_distance: float,
        radius: float,
        body_height: float,
        world_up: np.ndarray,
        step: float,
    ) -> float:
        """Sample one analytic capsule along a straight path."""
        direction = np.asarray(direction_world, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not np.isfinite(direction).all() or norm < 1e-12:
            raise ValueError("direction_world must be a non-zero finite vector with shape (3,)")
        if not math.isfinite(max_distance) or max_distance < 0 or not math.isfinite(step) or step <= 0:
            raise ValueError("max_distance must be non-negative and step must be positive")
        unit = direction / norm
        start = np.asarray(camera_position, dtype=np.float64)
        if not self.body_is_free(start, radius, body_height, world_up):
            return 0.0
        previous = 0.0
        distance = min(step, max_distance)
        while distance <= max_distance:
            if not self.body_is_free(start + unit * distance, radius, body_height, world_up):
                return previous
            previous = distance
            if distance == max_distance:
                break
            distance = min(distance + step, max_distance)
        return previous
