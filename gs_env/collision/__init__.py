"""Analytic mesh-capsule collision for the flying camera."""

from __future__ import annotations

from .mesh import MeshCapsuleCollisionBackend


def create_collision_backend(*, mesh_path=None, world_to_asset=None):
    """Create the one supported collision backend, or no collision if omitted."""
    return None if mesh_path is None else MeshCapsuleCollisionBackend(mesh_path, world_to_asset)


__all__ = ["MeshCapsuleCollisionBackend", "create_collision_backend"]
