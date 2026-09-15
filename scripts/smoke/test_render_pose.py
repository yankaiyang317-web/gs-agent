"""Render one P0 RGB/depth/alpha observation from an explicit 6DoF camera pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# Allow direct script invocation without
# requiring an editable install during local development.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs_env.rendering import GSplatRenderer
from gs_env.types import CameraIntrinsics


def camera_to_world_from_ypr(position: list[float], yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Return c2w for the convention documented in docs/p0_renderer.md."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    # At zero angles, the camera looks along world -Z and has image-up along world +Y.
    rotation = ry @ rx @ rz @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = np.asarray(position, dtype=np.float32)
    return c2w


def _save_visualization(array: np.ndarray, path: Path) -> None:
    valid = np.isfinite(array) & (array > 0)
    image = np.zeros(array.shape, dtype=np.uint8)
    if valid.any():
        low, high = np.percentile(array[valid], [1, 99])
        image[valid] = np.clip((array[valid] - low) / max(high - low, 1e-8) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--position", nargs=3, type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fx = args.fx if args.fx is not None else args.width / 2
    fy = args.fy if args.fy is not None else args.height / 2
    intrinsics = CameraIntrinsics(fx, fy, args.width / 2, args.height / 2, args.width, args.height)
    c2w = camera_to_world_from_ypr(args.position, args.yaw, args.pitch, args.roll)
    renderer = GSplatRenderer()
    scene = renderer.load_scene(args.ply)
    observation = renderer.render(c2w, intrinsics)
    Image.fromarray((observation.rgb * 255).round().astype(np.uint8)).save(args.out / "rgb.png")
    np.save(args.out / "depth.npy", observation.depth)
    np.save(args.out / "alpha.npy", observation.alpha)
    _save_visualization(observation.depth, args.out / "depth_vis.png")
    Image.fromarray((np.clip(observation.alpha, 0, 1) * 255).round().astype(np.uint8)).save(args.out / "alpha_vis.png")
    (args.out / "pose.json").write_text(json.dumps({"camera_to_world": c2w.tolist(), "intrinsics": vars(intrinsics), "gaussian_count": scene.count}, indent=2), encoding="utf-8")
    print(f"saved {args.out}; gaussians={scene.count}; load_seconds={scene.load_seconds:.3f}")


if __name__ == "__main__":
    main()
