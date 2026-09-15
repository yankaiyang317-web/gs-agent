"""CPU-only tests for portable server defaults and explicit overrides."""

from __future__ import annotations

import argparse
import unittest

import numpy as np

from gs_mcp.server import _collision_defaults, _intrinsics_from_args, _pose_from_args


class _Collision:
    pass


class ServerDefaultTests(unittest.TestCase):
    def _args(self, **overrides):
        values = {"width": None, "height": None, "fx": None, "fy": None, "position": None, "quaternion_wxyz": None}
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_new_scene_has_default_camera_and_origin_pose(self) -> None:
        intrinsics = _intrinsics_from_args(self._args())
        self.assertEqual((intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy), (320, 240, 160, 120))
        self.assertIsNone(_pose_from_args(self._args()))
        with self.assertRaises(ValueError):
            _pose_from_args(self._args(position=[1, 2, 3]))
        pose = _pose_from_args(self._args(position=[1, 2, 3], quaternion_wxyz=[0, 1, 0, 0]))
        np.testing.assert_allclose(pose.position, [1, 2, 3])
        np.testing.assert_allclose(pose.quaternion_wxyz, [0, 1, 0, 0])

    def test_explicit_values_override_mesh_defaults(self) -> None:
        self.assertEqual(_collision_defaults(_Collision(), None, None), (0.25, 0.05))
        self.assertEqual(_collision_defaults(_Collision(), 0.3, 0.01), (0.3, 0.01))
        intrinsics = _intrinsics_from_args(self._args(width=640, height=480))
        self.assertEqual((intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy), (640, 480, 320, 240))
