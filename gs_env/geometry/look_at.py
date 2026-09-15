"""Construct an OpenCV-camera C2W orientation that looks at a world point."""

from __future__ import annotations

import numpy as np


def look_at_rotation(camera_position: np.ndarray, target_position: np.ndarray, world_up: np.ndarray | None = None) -> np.ndarray:
    """Return C2W rotation whose camera +Z points at target and image-up is stabilized."""
    position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target_position, dtype=np.float64)
    forward = target - position
    norm = np.linalg.norm(forward)
    if position.shape != (3,) or target.shape != (3,) or norm < 1e-9:
        raise ValueError("camera and target must be distinct 3D points")
    forward /= norm
    up = np.array([0.0, 1.0, 0.0]) if world_up is None else np.asarray(world_up, dtype=np.float64)
    if abs(np.dot(forward, up) / np.linalg.norm(up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    image_down = np.cross(forward, right)
    return np.column_stack([right, image_down, forward])
