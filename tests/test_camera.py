"""CPU-only P1 camera convention tests."""

from __future__ import annotations

import unittest

import numpy as np

from gs_env.camera import CameraController
from gs_env.geometry.transforms import pose_from_camera_to_world, world_to_camera_from_pose


class CameraControllerTests(unittest.TestCase):
    def test_default_axes_match_renderer_convention(self) -> None:
        camera = CameraController()
        camera.translate_local(forward=1.0, right=2.0, up=3.0)
        np.testing.assert_allclose(camera.get_pose().position, [2.0, 3.0, -1.0], atol=1e-6)

    def test_yaw_right_changes_forward_heading(self) -> None:
        camera = CameraController()
        camera.rotate_local(yaw_deg=90.0)
        camera.translate_local(forward=1.0)
        np.testing.assert_allclose(camera.get_pose().position, [1.0, 0.0, 0.0], atol=1e-6)

    def test_pitch_up_changes_forward_heading(self) -> None:
        camera = CameraController()
        camera.rotate_local(pitch_deg=30.0)
        camera.translate_local(forward=1.0)
        np.testing.assert_allclose(camera.get_pose().position, [0.0, 0.5, -np.sqrt(3) / 2], atol=1e-6)

    def test_yaw_after_pitch_keeps_horizon_level(self) -> None:
        camera = CameraController()
        camera.rotate_local(pitch_deg=60.0)
        camera.rotate_local(yaw_deg=90.0)
        rotation = camera.get_camera_to_world()[:3, :3]
        np.testing.assert_allclose(rotation[:, 0], [0.0, 0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(float(np.dot(rotation[:, 0], camera.world_up)), 0.0, places=6)
        self.assertGreater(float(np.dot(-rotation[:, 1], camera.world_up)), 0.0)

    def test_alternating_yaw_and_pitch_does_not_accumulate_roll(self) -> None:
        camera = CameraController()
        for yaw, pitch in [(35, 40), (-20, -15), (75, 30), (-110, -45)]:
            camera.rotate_local(yaw_deg=yaw, pitch_deg=pitch)
            rotation = camera.get_camera_to_world()[:3, :3]
            self.assertAlmostEqual(float(np.dot(rotation[:, 0], camera.world_up)), 0.0, places=6)
            self.assertGreater(float(np.dot(-rotation[:, 1], camera.world_up)), 0.0)

    def test_pitch_is_clamped_before_the_camera_can_flip(self) -> None:
        camera = CameraController()
        camera.rotate_local(pitch_deg=120.0)
        forward = camera.get_camera_to_world()[:3, 2]
        pitch = np.degrees(np.arcsin(np.dot(forward, camera.world_up)))
        self.assertAlmostEqual(pitch, 85.0, places=4)

    def test_up_is_world_vertical_and_right_stays_level_after_pitch(self) -> None:
        camera = CameraController()
        camera.rotate_local(pitch_deg=60.0)
        camera.translate_local(up=2.0)
        np.testing.assert_allclose(camera.get_pose().position, [0.0, 2.0, 0.0], atol=1e-6)
        camera.translate_local(right=3.0)
        np.testing.assert_allclose(camera.get_pose().position, [3.0, 2.0, 0.0], atol=1e-6)

    def test_explicit_world_up_overrides_initial_image_up(self) -> None:
        camera = CameraController(world_up=[0, 0, 2])
        np.testing.assert_allclose(camera.world_up, [0, 0, 1], atol=1e-6)
        camera.translate_local(up=1.5)
        np.testing.assert_allclose(camera.get_pose().position, [0, 0, 1.5], atol=1e-6)

    def test_c2w_w2c_round_trip(self) -> None:
        camera = CameraController()
        camera.translate_local(forward=1.5, right=-0.2, up=0.7)
        camera.rotate_local(yaw_deg=31.0, pitch_deg=-12.0)
        pose = camera.get_pose()
        np.testing.assert_allclose(world_to_camera_from_pose(pose) @ camera.get_camera_to_world(), np.eye(4), atol=1e-6)
        recovered = pose_from_camera_to_world(camera.get_camera_to_world())
        np.testing.assert_allclose(recovered.position, pose.position, atol=1e-6)
        np.testing.assert_allclose(abs(np.dot(recovered.quaternion_wxyz, pose.quaternion_wxyz)), 1.0, atol=1e-6)

    def test_pose_is_defensive_copy(self) -> None:
        camera = CameraController()
        pose = camera.get_pose()
        pose.position[0] = 99
        self.assertEqual(camera.get_pose().position[0], 0.0)


if __name__ == "__main__":
    unittest.main()
