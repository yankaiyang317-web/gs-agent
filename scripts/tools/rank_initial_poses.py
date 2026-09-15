"""Rank coarse initial-pose candidates by rendered alpha and depth quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs_env.geometry.look_at import look_at_rotation
from gs_env.geometry.transforms import camera_to_world_from_pose, rotation_matrix_to_quaternion
from gs_env.rendering import GSplatRenderer
from gs_env.scene_manifest import load_scene_manifest
from gs_env.types import CameraIntrinsics, CameraPose


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _candidate_score(alpha: np.ndarray, depth: np.ndarray) -> dict[str, float]:
    confident = alpha >= 0.7
    visible = alpha >= 0.2
    finite = np.isfinite(depth) & (depth > 0)
    useful = confident & finite
    confident_fraction = float(np.mean(confident))
    visible_fraction = float(np.mean(visible))
    if np.any(useful):
        useful_depth = depth[useful]
        median_depth = float(np.median(useful_depth))
        near_fraction = float(np.mean(useful_depth < 0.5))
        depth_spread = float(np.percentile(useful_depth, 90) - np.percentile(useful_depth, 10))
    else:
        median_depth = 0.0
        near_fraction = 1.0
        depth_spread = 0.0
    score = confident_fraction + 0.2 * visible_fraction + 0.05 * min(depth_spread, 20.0) - 0.5 * near_fraction
    return {
        "score": score,
        "confident_fraction": confident_fraction,
        "visible_fraction": visible_fraction,
        "median_depth": median_depth,
        "near_fraction": near_fraction,
        "depth_spread": depth_spread,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--x", required=True, help="comma-separated world X values")
    parser.add_argument("--y", required=True, help="comma-separated world Y values")
    parser.add_argument("--z", required=True, help="comma-separated world Z values")
    parser.add_argument("--yaw", default="0,90,180,270", help="degrees around world_up; 0 faces +Z")
    parser.add_argument("--pitch-down", type=float, default=10.0)
    parser.add_argument(
        "--world-up",
        help="optional comma-separated override; defaults to the scene manifest",
    )
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--keep", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = load_scene_manifest(args.scene_manifest)
    up = np.asarray(_floats(args.world_up) if args.world_up else manifest.world_up, dtype=np.float64)
    if up.shape != (3,):
        raise ValueError("--world-up must contain exactly three comma-separated values")
    up /= np.linalg.norm(up)
    renderer = GSplatRenderer(device=args.device)
    renderer.load_scene(manifest.ply_path)
    intrinsics = CameraIntrinsics(
        fx=args.width / 2.0,
        fy=args.height / 2.0,
        cx=args.width / 2.0,
        cy=args.height / 2.0,
        width=args.width,
        height=args.height,
    )
    pitch = np.deg2rad(args.pitch_down)
    results: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for x in _floats(args.x):
        for y in _floats(args.y):
            for z in _floats(args.z):
                position = np.array([x, y, z], dtype=np.float64)
                for yaw in _floats(args.yaw):
                    angle = np.deg2rad(yaw)
                    horizontal = np.array([np.sin(angle), 0.0, np.cos(angle)], dtype=np.float64)
                    forward = np.cos(pitch) * horizontal - np.sin(pitch) * up
                    rotation = look_at_rotation(position, position + forward, up)
                    pose = CameraPose(position, rotation_matrix_to_quaternion(rotation))
                    observation = renderer.render(camera_to_world_from_pose(pose), intrinsics)
                    metrics = _candidate_score(observation.alpha, observation.depth)
                    metrics.update({"position": position.tolist(), "yaw_deg": yaw, "pitch_down_deg": args.pitch_down, "quaternion_wxyz": pose.quaternion_wxyz.tolist()})
                    results.append((metrics, observation.rgb, observation.alpha))

    results.sort(key=lambda item: item[0]["score"], reverse=True)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, (metrics, rgb, alpha) in enumerate(results[: args.keep]):
        stem = f"rank_{index:02d}"
        Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(args.out / f"{stem}.png")
        Image.fromarray((np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)).save(args.out / f"{stem}_alpha.png")
        summary.append({"rank": index, **metrics, "rgb": f"{stem}.png", "alpha": f"{stem}_alpha.png"})
    args.out.joinpath("ranking.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
