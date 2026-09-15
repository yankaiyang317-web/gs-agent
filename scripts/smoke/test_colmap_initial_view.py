"""Render one calibrated COLMAP training view as a P0/MCP initialization smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs_env.geometry.colmap import load_colmap_initial_view, transform_colmap_pose_to_gs
from gs_env.geometry.transforms import camera_to_world_from_pose
from gs_env.rendering import GSplatRenderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--similarity", default=None)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pose, intrinsics, image_name = load_colmap_initial_view(args.images, args.cameras, args.width, args.height, args.image)
    if args.similarity:
        pose = transform_colmap_pose_to_gs(pose, args.similarity)
    renderer = GSplatRenderer()
    renderer.load_scene(args.ply)
    observation = renderer.render(camera_to_world_from_pose(pose), intrinsics)
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((observation.rgb * 255).round().astype(np.uint8)).save(args.out / "rgb.png")
    np.save(args.out / "depth.npy", observation.depth)
    np.save(args.out / "alpha.npy", observation.alpha)
    (args.out / "colmap_pose.txt").write_text(f"image={image_name}\nposition={pose.position.tolist()}\nquaternion_wxyz={pose.quaternion_wxyz.tolist()}\n", encoding="utf-8")
    print(f"saved {args.out}; image={image_name}")


if __name__ == "__main__":
    main()
