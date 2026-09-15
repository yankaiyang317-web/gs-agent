"""Render one scene manifest's actual initial pose for deployment smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs_env import load_scene_manifest
from gs_env.geometry.transforms import camera_to_world_from_pose
from gs_env.rendering import GSplatRenderer
from gs_mcp.observation_images import depth_preview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = load_scene_manifest(args.scene_manifest)
    renderer = GSplatRenderer(device=args.device)
    scene = renderer.load_scene(manifest.ply_path)
    observation = renderer.render(camera_to_world_from_pose(manifest.initial_pose), manifest.intrinsics)

    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(observation.rgb, 0, 1) * 255).round().astype(np.uint8)).save(args.out / "rgb.png")
    Image.fromarray(depth_preview(observation)).save(args.out / "depth_preview.png")
    args.out.joinpath("result.json").write_text(json.dumps({
        "scene": manifest.name,
        "gaussian_count": scene.count,
        "load_seconds": scene.load_seconds,
        "camera_to_world": observation.camera_to_world.tolist(),
        "world_up": None if manifest.world_up is None else manifest.world_up.tolist(),
    }, indent=2), encoding="utf-8")
    print(f"render_smoke=ok scene={manifest.name} gaussians={scene.count} out={args.out}")


if __name__ == "__main__":
    main()
