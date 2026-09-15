"""Depth unprojection utilities for image-targeted movement."""

from __future__ import annotations

import numpy as np

from gs_env.types import CameraIntrinsics

DEPTH_MIN_ALPHA = 0.7


def robust_pixel_depth(
    depth: np.ndarray,
    alpha: np.ndarray,
    pixel: tuple[int, int],
    radius: int = 2,
    min_alpha: float = DEPTH_MIN_ALPHA,
) -> float:
    """Return median valid depth in a small pixel neighbourhood."""
    u, v = pixel
    if radius < 0 or not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]) or depth.shape != alpha.shape:
        raise ValueError("pixel must lie within matching depth and alpha images")
    y0, y1 = max(0, v - radius), min(depth.shape[0], v + radius + 1)
    x0, x1 = max(0, u - radius), min(depth.shape[1], u + radius + 1)
    values = depth[y0:y1, x0:x1]
    valid = np.isfinite(values) & (values > 0) & (alpha[y0:y1, x0:x1] >= min_alpha)
    if not valid.any():
        raise ValueError("target has no valid high-alpha depth in its neighbourhood")
    return float(np.median(values[valid]))


def robust_bbox_depth(
    depth: np.ndarray,
    alpha: np.ndarray,
    bbox: tuple[int, int, int, int],
    min_alpha: float = DEPTH_MIN_ALPHA,
) -> tuple[tuple[int, int], float]:
    """Return the bbox centre pixel and median reliable depth inside an XYXY box.

    Bounding boxes are ``[x0, y0, x1, y1]`` with an exclusive lower-right
    corner, matching common image-processing conventions.
    """
    x0, y0, x1, y1 = bbox
    height, width = depth.shape
    if depth.shape != alpha.shape or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("bbox must be [x0, y0, x1, y1] within matching depth and alpha images")
    values = depth[y0:y1, x0:x1]
    valid = np.isfinite(values) & (values > 0) & (alpha[y0:y1, x0:x1] >= min_alpha)
    if not valid.any():
        raise ValueError("target bbox has no valid high-alpha depth")
    pixel = ((x0 + x1 - 1) // 2, (y0 + y1 - 1) // 2)
    return pixel, float(np.median(values[valid]))


def pixel_to_camera_point(u: float, v: float, depth: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Unproject a pinhole pixel and camera-space Z depth to OpenCV camera XYZ."""
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("depth must be finite and positive")
    return np.array([(u - intrinsics.cx) * depth / intrinsics.fx, (v - intrinsics.cy) * depth / intrinsics.fy, depth], dtype=np.float64)


def camera_point_to_world(point_camera: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    point = np.asarray(point_camera, dtype=np.float64)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if point.shape != (3,) or c2w.shape != (4, 4):
        raise ValueError("point_camera must be 3D and camera_to_world must be 4x4")
    return c2w[:3, :3] @ point + c2w[:3, 3]
