"""Portable, explicit scene configuration for the Agent environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gs_env.types import CameraIntrinsics, CameraPose


@dataclass(frozen=True)
class SceneManifest:
    """All scene-specific parameters that must not leak across datasets."""

    name: str
    ply_path: Path
    collision_mesh_path: Path | None
    collision_world_to_asset: np.ndarray
    initial_pose: CameraPose
    intrinsics: CameraIntrinsics
    camera_radius: float
    camera_body_height: float
    collision_step: float
    world_up: np.ndarray | None


def _resolved_path(manifest_path: Path, value: str, field: str) -> Path:
    path = (manifest_path.parent / value).resolve()
    if not path.is_file():
        raise ValueError(f"scene manifest {field} does not exist: {path}")
    return path


def load_scene_manifest(path: str | Path) -> SceneManifest:
    """Load one JSON manifest, resolving asset paths relative to the JSON file."""
    manifest_path = Path(path).resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "collision_voxel" in raw:
            raise ValueError(
                "collision_voxel is no longer supported; configure collision_mesh with a GLB"
            )
        camera = raw["camera"]
        pose = raw["initial_pose"]
        collision = raw.get("collision", {})
        world_up_value = raw.get("world_up")
        position = np.asarray(pose["position"], dtype=np.float64)
        quaternion = np.asarray(pose["quaternion_wxyz"], dtype=np.float64)
        intrinsics = CameraIntrinsics(
            float(camera["fx"]), float(camera["fy"]), float(camera["cx"]), float(camera["cy"]),
            int(camera["width"]), int(camera["height"]),
        )
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("initial_pose position/quaternion_wxyz must contain 3/4 values")
        radius = float(collision.get("camera_radius", 0.0))
        body_height = float(collision.get("camera_body_height", 0.0))
        step = float(collision.get("step", 0.05))
        if radius < 0 or body_height < 0 or step <= 0:
            raise ValueError(
                "collision camera_radius/camera_body_height must be non-negative and "
                "step positive"
            )
        world_up = None if world_up_value is None else np.asarray(world_up_value, dtype=np.float64)
        if world_up is not None and (world_up.shape != (3,) or not np.isfinite(world_up).all() or np.linalg.norm(world_up) < 1e-12):
            raise ValueError("world_up must be a finite non-zero 3D vector")
        collision_mesh_path = raw.get("collision_mesh")
        world_to_asset = np.asarray(collision.get("world_to_asset", np.eye(4)), dtype=np.float64)
        if world_to_asset.shape != (4, 4) or not np.isfinite(world_to_asset).all():
            raise ValueError("collision.world_to_asset must be a finite 4x4 matrix")
        rotation = world_to_asset[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("collision.world_to_asset must contain a rigid rotation without scale or reflection")
        if not np.allclose(world_to_asset[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("collision.world_to_asset must use homogeneous final row [0,0,0,1]")
        return SceneManifest(
            name=str(raw["name"]),
            ply_path=_resolved_path(manifest_path, str(raw["ply"]), "ply"),
            collision_mesh_path=None if collision_mesh_path is None else _resolved_path(manifest_path, str(collision_mesh_path), "collision_mesh"),
            collision_world_to_asset=world_to_asset,
            initial_pose=CameraPose(position, quaternion), intrinsics=intrinsics,
            camera_radius=radius, camera_body_height=body_height,
            collision_step=step, world_up=world_up,
        )
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid scene manifest: {manifest_path}") from error
