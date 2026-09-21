"""CPU-only tests for portable scene manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gs_env.scene_manifest import load_scene_manifest
from gs_env.geometry.transforms import quaternion_to_rotation_matrix


class SceneManifestTests(unittest.TestCase):
    def test_guju_preset_has_verified_upright_navigation_frame(self) -> None:
        raw = json.loads((Path(__file__).parents[1] / "scenes" / "guju.json").read_text(encoding="utf-8"))
        rotation = quaternion_to_rotation_matrix(np.asarray(raw["initial_pose"]["quaternion_wxyz"], dtype=np.float64))
        image_up = -rotation[:, 1]

        np.testing.assert_allclose(image_up, [0.0, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(raw["world_up"], image_up, atol=1e-6)
        np.testing.assert_allclose(raw["initial_pose"]["position"], [0.0, -45.0, 0.0], atol=1e-6)
        self.assertEqual(2 * raw["collision"]["camera_radius"] + raw["collision"]["camera_body_height"], 1.4)
        self.assertEqual(Path(raw["collision_mesh"]).name, "1lpn0524_v006.collision.glb")

    def test_retained_scenes_use_only_verified_mesh_capsule_assets(self) -> None:
        scene_root = Path(__file__).parents[1] / "scenes"
        expected_meshes = {
            "jiudian": "y253p58x.collision.glb",
            "guju": "1lpn0524_v006.collision.glb",
            "laojie": "nl039zwl_v006.collision.glb",
        }
        for scene, mesh_name in expected_meshes.items():
            with self.subTest(scene=scene):
                raw = json.loads((scene_root / f"{scene}.json").read_text(encoding="utf-8"))
                self.assertEqual(Path(raw["collision_mesh"]).name, mesh_name)
                self.assertEqual(2 * raw["collision"]["camera_radius"] + raw["collision"]["camera_body_height"], 1.4)
                self.assertEqual(raw["collision"]["step"], 0.02)
                np.testing.assert_allclose(
                    raw["collision"]["world_to_asset"], np.diag([-1, -1, 1, 1])
                )

    def test_resolves_scene_relative_assets_and_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("scene.ply").touch()
            root.joinpath("scene.glb").touch()
            root.joinpath("scene.json").write_text(json.dumps({
                "name": "test", "ply": "scene.ply", "collision_mesh": "scene.glb",
                "initial_pose": {"position": [1, 2, 3], "quaternion_wxyz": [1, 0, 0, 0]},
                "camera": {"width": 20, "height": 10, "fx": 11, "fy": 12, "cx": 9, "cy": 4},
                "collision": {"camera_radius": 0.2, "camera_body_height": 0.7, "step": 0.03},
                "world_up": [0, 0, 1],
            }), encoding="utf-8")
            manifest = load_scene_manifest(root / "scene.json")
            self.assertEqual(manifest.name, "test")
            self.assertEqual(manifest.ply_path, root / "scene.ply")
            self.assertEqual(manifest.intrinsics.width, 20)
            self.assertEqual(manifest.camera_radius, 0.2)
            self.assertEqual(manifest.camera_body_height, 0.7)
            self.assertEqual(manifest.world_up.tolist(), [0.0, 0.0, 1.0])
            self.assertEqual(manifest.collision_mesh_path, root / "scene.glb")
            np.testing.assert_allclose(manifest.collision_world_to_asset, np.eye(4))

    def test_rejects_non_rigid_collision_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("scene.ply").touch()
            root.joinpath("scene.glb").touch()
            path = root / "scene.json"
            path.write_text(json.dumps({
                "name": "bad-transform", "ply": "scene.ply", "collision_mesh": "scene.glb",
                "initial_pose": {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                "camera": {"width": 1, "height": 1, "fx": 1, "fy": 1, "cx": 0, "cy": 0},
                "collision": {"world_to_asset": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rigid rotation"):
                load_scene_manifest(path)

    def test_rejects_legacy_voxel_collision_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("scene.ply").touch()
            root.joinpath("scene.voxel.json").touch()
            path = root / "scene.json"
            path.write_text(json.dumps({
                "name": "legacy", "ply": "scene.ply", "collision_voxel": "scene.voxel.json",
                "initial_pose": {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                "camera": {"width": 1, "height": 1, "fx": 1, "fy": 1, "cx": 0, "cy": 0},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no longer supported"):
                load_scene_manifest(path)

    def test_rejects_missing_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps({"name": "bad", "ply": "missing.ply", "initial_pose": {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}, "camera": {"width": 1, "height": 1, "fx": 1, "fy": 1, "cx": 0, "cy": 0}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_scene_manifest(path)

    def test_rejects_zero_world_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("scene.ply").touch()
            path = root / "scene.json"
            path.write_text(json.dumps({
                "name": "bad-up", "ply": "scene.ply", "world_up": [0, 0, 0],
                "initial_pose": {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                "camera": {"width": 1, "height": 1, "fx": 1, "fy": 1, "cx": 0, "cy": 0},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "world_up"):
                load_scene_manifest(path)

    def test_rejects_negative_camera_body_height(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.joinpath("scene.ply").touch()
            path = root / "scene.json"
            path.write_text(json.dumps({
                "name": "bad-body", "ply": "scene.ply",
                "initial_pose": {"position": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]},
                "camera": {"width": 1, "height": 1, "fx": 1, "fy": 1, "cx": 0, "cy": 0},
                "collision": {"camera_body_height": -0.1},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "camera_body_height"):
                load_scene_manifest(path)
