"""Dependency-light tests for the mesh-capsule backend."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from gs_env.collision import create_collision_backend
from gs_env.collision.mesh import _rotation_from_z


class MeshCollisionTests(unittest.TestCase):
    def test_rotation_maps_local_z_to_requested_axis(self) -> None:
        for axis in ([0, -1, 0], [0, 0, 1], [1, 2, 3]):
            normalized = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
            rotation = _rotation_from_z(np.asarray(axis, dtype=float))
            np.testing.assert_allclose(rotation[:, 2], normalized, atol=1e-9)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_missing_coal_is_not_silently_replaced(self) -> None:
        with patch("gs_env.collision.MeshCapsuleCollisionBackend", side_effect=RuntimeError("coal missing")):
            with self.assertRaisesRegex(RuntimeError, "coal missing"):
                create_collision_backend(mesh_path="scene.glb")

    def test_omitted_mesh_disables_collision_explicitly(self) -> None:
        self.assertIsNone(create_collision_backend())


if __name__ == "__main__":
    unittest.main()
