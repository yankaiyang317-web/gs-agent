"""Cold-load and smoke-test one scene's analytic capsule collision mesh."""

from __future__ import annotations

import argparse
import json
import resource
import struct
import time
from pathlib import Path

import numpy as np

from gs_env.collision.mesh import MeshCapsuleCollisionBackend
from gs_env.geometry.transforms import quaternion_to_rotation_matrix
from gs_env.scene_manifest import load_scene_manifest


def _glb_triangle_count(path: Path) -> int:
    with path.open("rb") as stream:
        magic, version, _length = struct.unpack("<4sII", stream.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"not a glTF 2.0 binary: {path}")
        chunk_length, chunk_type = struct.unpack("<II", stream.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError(f"GLB does not start with a JSON chunk: {path}")
        document = json.loads(stream.read(chunk_length).decode("utf-8"))

    accessors = document.get("accessors", [])
    triangles = 0
    for mesh in document.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if primitive.get("mode", 4) != 4:
                continue
            index_accessor = primitive.get("indices")
            if index_accessor is not None:
                triangles += int(accessors[index_accessor]["count"]) // 3
            else:
                position_accessor = primitive["attributes"]["POSITION"]
                triangles += int(accessors[position_accessor]["count"]) // 3
    return triangles


def _ply_stats(path: Path) -> dict:
    from plyfile import PlyData

    vertices = PlyData.read(path)["vertex"].data
    return {
        "path": str(path),
        "gaussians": len(vertices),
        "bounds": {
            axis: [float(np.min(vertices[axis])), float(np.max(vertices[axis]))]
            for axis in ("x", "y", "z")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_manifest", type=Path)
    parser.add_argument("--mesh", type=Path, help="validate this candidate instead of the manifest mesh")
    parser.add_argument("--clustered-ply", type=Path, help="also report retained Gaussian count and XYZ bounds")
    parser.add_argument("--bounds-only", action="store_true", help="skip Coal and report only --clustered-ply")
    parser.add_argument("--probe-distance", type=float, default=0.1)
    args = parser.parse_args()

    manifest = load_scene_manifest(args.scene_manifest)
    if args.bounds_only:
        if args.clustered_ply is None:
            parser.error("--bounds-only requires --clustered-ply")
        print(json.dumps({"scene": manifest.name, "clustered_ply": _ply_stats(args.clustered_ply)}, indent=2))
        return
    mesh_path = args.mesh.resolve() if args.mesh else manifest.collision_mesh_path
    if mesh_path is None or not mesh_path.is_file():
        raise ValueError(f"collision mesh does not exist: {mesh_path}")

    rotation = quaternion_to_rotation_matrix(manifest.initial_pose.quaternion_wxyz)
    world_up = manifest.world_up if manifest.world_up is not None else -rotation[:, 1]
    world_up = world_up / np.linalg.norm(world_up)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    backend = MeshCapsuleCollisionBackend(mesh_path, manifest.collision_world_to_asset)
    load_seconds = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    position = manifest.initial_pose.position
    initial_free = backend.body_is_free(
        position, manifest.camera_radius, manifest.camera_body_height, world_up
    )
    probes = {}
    for label, direction in (
        ("world_x+", [1, 0, 0]),
        ("world_x-", [-1, 0, 0]),
        ("world_y+", [0, 1, 0]),
        ("world_y-", [0, -1, 0]),
        ("world_z+", [0, 0, 1]),
        ("world_z-", [0, 0, -1]),
    ):
        probes[label] = backend.max_free_body_distance(
            position,
            np.asarray(direction, dtype=np.float64),
            args.probe_distance,
            manifest.camera_radius,
            manifest.camera_body_height,
            world_up,
            manifest.collision_step,
        )

    result = {
        "scene": manifest.name,
        "mesh": str(mesh_path),
        "bytes": mesh_path.stat().st_size,
        "triangles": _glb_triangle_count(mesh_path),
        "load_seconds": round(load_seconds, 3),
        "peak_rss_delta_mib": round(max(0, rss_after - rss_before) / 1024, 1),
        "backend": backend.backend_name,
        "capsule_total_height": manifest.camera_body_height + 2 * manifest.camera_radius,
        "collision_step": manifest.collision_step,
        "initial_capsule_free": initial_free,
        "short_path_free_distance": probes,
    }
    if args.clustered_ply:
        result["clustered_ply"] = _ply_stats(args.clustered_ply)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
