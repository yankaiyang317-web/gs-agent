"""Read a calibrated COLMAP text-camera pose as this project's C2W pose."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from gs_env.geometry.transforms import camera_to_world_from_pose, rotation_matrix_to_quaternion
from gs_env.types import CameraIntrinsics, CameraPose


def _data_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def load_colmap_initial_view(images_path: str | Path, cameras_path: str | Path, width: int, height: int, image_name: str | None = None) -> tuple[CameraPose, CameraIntrinsics, str]:
    """Convert a COLMAP image record (world-to-camera) to project C2W pose."""
    image_lines = _data_lines(Path(images_path))
    records = image_lines[::2]  # COLMAP stores each image header then its 2D-points line.
    selected = next((line for line in records if image_name and line.split()[-1] == image_name), None) if image_name else records[0]
    if selected is None:
        raise ValueError(f"COLMAP image {image_name!r} was not found")
    fields = selected.split()
    if len(fields) < 10:
        raise ValueError("invalid COLMAP images.txt image header")
    qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
    camera_id, selected_name = int(fields[8]), fields[9]
    camera_fields = next((line.split() for line in _data_lines(Path(cameras_path)) if int(line.split()[0]) == camera_id), None)
    if camera_fields is None or camera_fields[1] not in {"SIMPLE_PINHOLE", "PINHOLE"}:
        raise ValueError("only COLMAP SIMPLE_PINHOLE and PINHOLE cameras are supported")
    source_width, source_height = map(int, camera_fields[2:4])
    params = list(map(float, camera_fields[4:]))
    if camera_fields[1] == "SIMPLE_PINHOLE":
        fx_source = fy_source = params[0]
        cx_source, cy_source = params[1:3]
    else:
        fx_source, fy_source, cx_source, cy_source = params[:4]
    sx, sy = width / source_width, height / source_height
    intrinsics = CameraIntrinsics(fx_source * sx, fy_source * sy, cx_source * sx, cy_source * sy, width, height)
    # COLMAP's q,t map world to camera: x_cam = R_wc x_world + t_wc.
    q_wc = np.array([qw, qx, qy, qz], dtype=np.float64)
    from gs_env.geometry.transforms import quaternion_to_rotation_matrix
    rotation_wc = quaternion_to_rotation_matrix(q_wc)
    rotation_cw = rotation_wc.T
    position = -rotation_cw @ np.array([tx, ty, tz], dtype=np.float64)
    return CameraPose(position, rotation_matrix_to_quaternion(rotation_cw)), intrinsics, selected_name


def transform_colmap_pose_to_gs(pose: CameraPose, similarity_path: str | Path) -> CameraPose:
    """Apply a scene-specific COLMAP-world -> Gaussian-world similarity transform."""
    data = json.loads(Path(similarity_path).read_text(encoding="utf-8"))
    scale = float(data["scale"])
    rotation = np.asarray(data["rotation_colmap_to_gs"], dtype=np.float64)
    translation = np.asarray(data["translation_colmap_to_gs"], dtype=np.float64)
    if not np.isfinite(scale) or scale <= 0 or translation.shape != (3,):
        raise ValueError("invalid similarity scale or translation")
    # Validate the rotation through the canonical converter.
    rotation_matrix_to_quaternion(rotation)
    c2w = camera_to_world_from_pose(pose)
    return CameraPose(scale * rotation @ pose.position + translation, rotation_matrix_to_quaternion(rotation @ c2w[:3, :3]))
