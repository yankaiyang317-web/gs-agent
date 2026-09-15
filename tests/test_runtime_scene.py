"""CPU-only tests for runtime-selectable server scenes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gs_env.types import CameraIntrinsics
from gs_mcp.runtime import EnvironmentToolsProxy, SceneRuntime, scaled_intrinsics


class RuntimeSceneTests(unittest.TestCase):
    def test_intrinsics_scale_without_changing_field_of_view(self) -> None:
        source = CameraIntrinsics(160, 120, 160, 120, 320, 240)
        scaled = scaled_intrinsics(source, 960, 720)
        self.assertEqual((scaled.width, scaled.height), (960, 720))
        self.assertEqual((scaled.fx, scaled.fy, scaled.cx, scaled.cy), (480, 360, 480, 360))

    def test_proxy_requires_a_loaded_scene_then_delegates(self) -> None:
        proxy = EnvironmentToolsProxy()
        with self.assertRaisesRegex(RuntimeError, "gs_configure_campaign"):
            proxy.get_pose()

        class FakeTools:
            def get_pose(self):
                return {"loaded": True}

        proxy.bind(FakeTools())  # type: ignore[arg-type]
        self.assertEqual(proxy.get_pose(), {"loaded": True})

    def test_scene_names_cannot_escape_server_scene_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes"
            scenes.mkdir()
            runtime = SceneRuntime(scenes, root / "runs", device="cpu")
            with self.assertRaisesRegex(ValueError, "letters"):
                runtime.load_scene("../secret")

    def test_runtime_profiles_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes"
            scenes.mkdir()
            runtime = SceneRuntime(scenes, root / "runs", device="cpu")
            with self.assertRaisesRegex(ValueError, "profile"):
                runtime.load_scene("laojie", profile="unknown")
            with self.assertRaisesRegex(ValueError, "supplied together"):
                runtime.load_scene("laojie", width=1920)

    def test_invalid_manifest_is_listed_but_not_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes"
            scenes.mkdir()
            (scenes / "broken.json").write_text("{}", encoding="utf-8")
            runtime = SceneRuntime(scenes, root / "runs", device="cpu")
            result = runtime.list_scenes()
            self.assertEqual(result["loaded_scene"], None)
            self.assertEqual(result["scenes"][0]["scene"], "broken")
            self.assertFalse(result["scenes"][0]["available"])

    def test_incomplete_campaign_cannot_switch_to_another_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = root / "scenes"
            scenes.mkdir()
            (scenes / "laojie.json").write_text("{}", encoding="utf-8")
            runtime = SceneRuntime(scenes, root / "runs", device="cpu")
            runtime.current = SimpleNamespace(key="guju", campaign=SimpleNamespace(is_complete=False))  # type: ignore[assignment]
            with self.assertRaisesRegex(RuntimeError, "campaign is incomplete"):
                runtime.load_scene("laojie")


if __name__ == "__main__":
    unittest.main()
