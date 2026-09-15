"""CPU-only tests for the minimal Agent-to-image environment API."""

from __future__ import annotations

import unittest

import numpy as np

from gs_env import CameraIntrinsics, CameraPose, GSEnvironment, Observation


class FakeRenderer:
    def __init__(self) -> None:
        self.loaded = False
        self.rendered_poses: list[np.ndarray] = []

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load_scene(self, _path: str) -> str:
        self.loaded = True
        return "fake-scene"

    def render(self, camera_to_world: np.ndarray, intrinsics: CameraIntrinsics) -> Observation:
        self.rendered_poses.append(camera_to_world.copy())
        zeros = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
        ones = np.ones_like(zeros)
        return Observation(np.zeros((*zeros.shape, 3), dtype=np.float32), ones, ones, camera_to_world, intrinsics)


class RecordingAnalyticCapsuleCollision:
    def __init__(self) -> None:
        self.calls = []

    def max_free_body_distance(self, *args) -> float:
        self.calls.append(args)
        return args[2]


class EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = FakeRenderer()
        self.intrinsics = CameraIntrinsics(50, 50, 2, 2, 4, 4)

    def test_initial_pose_is_optional_and_respected(self) -> None:
        pose = CameraPose(np.array([4.0, 5.0, 6.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        environment = GSEnvironment(self.renderer, initial_pose=pose)
        np.testing.assert_allclose(environment.get_pose().position, [4, 5, 6])
        np.testing.assert_allclose(environment.get_camera_to_world()[:3, :3], np.eye(3))

    def test_observe_uses_current_pose_and_requires_scene(self) -> None:
        environment = GSEnvironment(self.renderer)
        with self.assertRaisesRegex(RuntimeError, "No scene loaded"):
            environment.observe(self.intrinsics)
        environment.load_scene("fake.ply")
        environment.translate_local(forward=2.0)
        observation = environment.observe(self.intrinsics)
        self.assertEqual(observation.rgb.shape, (4, 4, 3))
        np.testing.assert_allclose(self.renderer.rendered_poses[-1], environment.get_camera_to_world())

    def test_direct_pose_and_incremental_control_share_one_camera(self) -> None:
        environment = GSEnvironment(self.renderer)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [1, 2, 3]
        environment.set_camera_to_world(c2w)
        environment.rotate_local(yaw_deg=90)
        environment.translate_local(forward=1)
        # Direct pose changes do not redefine the session's fixed up axis.
        np.testing.assert_allclose(environment.get_pose().position, [0, 2, 3], atol=1e-6)

    def test_persisted_world_up_can_be_restored_before_navigation(self) -> None:
        environment = GSEnvironment(self.renderer)
        environment.restore_world_up([0, 0, 1])
        environment.translate_local(up=2)
        np.testing.assert_allclose(environment.get_pose().position, [0, 0, 2], atol=1e-6)

    def test_approach_pixel_preserves_camera_orientation(self) -> None:
        environment = GSEnvironment(self.renderer)
        environment.load_scene("fake.ply")
        environment.rotate_local(yaw_deg=31, pitch_deg=-12)
        before = environment.get_pose()
        environment.approach_pixel((3, 3), 0.25, self.intrinsics)
        after = environment.get_pose()
        self.assertAlmostEqual(abs(float(np.dot(before.quaternion_wxyz, after.quaternion_wxyz))), 1.0, places=6)
        self.assertGreater(float(np.linalg.norm(after.position - before.position)), 0.0)

    def test_analytic_capsule_backend_receives_one_body_query(self) -> None:
        collision = RecordingAnalyticCapsuleCollision()
        environment = GSEnvironment(
            self.renderer,
            collision_backend=collision,
            camera_radius=0.25,
            camera_body_height=0.9,
            collision_step=0.02,
            world_up=[0, 1, 0],
        )
        result = environment.move_local(forward=1.0)
        self.assertEqual(result.executed_distance, 1.0)
        self.assertEqual(len(collision.calls), 1)
        start, _direction, distance, radius, body_height, world_up, step = collision.calls[0]
        np.testing.assert_allclose(start, [0, 0, 0])
        np.testing.assert_allclose(world_up, [0, 1, 0])
        self.assertEqual((distance, radius, body_height, step), (1.0, 0.25, 0.9, 0.02))



if __name__ == "__main__":
    unittest.main()
