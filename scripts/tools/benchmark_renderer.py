"""Benchmark P0 renderer latency with one already-loaded scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gs_env.rendering import GSplatRenderer
from gs_env.types import CameraIntrinsics
from scripts.smoke.test_render_pose import camera_to_world_from_ypr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--position", nargs=3, type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark.json"))
    args = parser.parse_args()
    if args.frames < 100:
        raise ValueError("P0 benchmark requires at least 100 frames")
    intrinsics = CameraIntrinsics(args.width / 2, args.height / 2, args.width / 2, args.height / 2, args.width, args.height)
    c2w = camera_to_world_from_ypr(args.position, args.yaw, args.pitch, args.roll)
    renderer = GSplatRenderer()
    scene = renderer.load_scene(args.ply)
    torch.cuda.reset_peak_memory_stats(renderer.device)
    torch.cuda.synchronize(renderer.device)
    started = perf_counter()
    renderer.render(c2w, intrinsics)
    torch.cuda.synchronize(renderer.device)
    first_ms = (perf_counter() - started) * 1e3
    timings = []
    for _ in range(args.frames):
        torch.cuda.synchronize(renderer.device)
        started = perf_counter()
        renderer.render(c2w, intrinsics)
        torch.cuda.synchronize(renderer.device)
        timings.append((perf_counter() - started) * 1e3)
    report = {"ply": str(args.ply), "resolution": [args.width, args.height], "gaussian_count": scene.count, "ply_load_seconds": scene.load_seconds, "first_frame_ms": first_ms, "mean_frame_ms": float(np.mean(timings)), "p50_frame_ms": float(np.percentile(timings, 50)), "p95_frame_ms": float(np.percentile(timings, 95)), "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(renderer.device) / 1024**2, "frames": args.frames}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
