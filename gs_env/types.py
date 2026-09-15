"""Data types shared by the P0 headless renderer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class CameraPose:
    """A camera pose with a world position and a C2W quaternion in wxyz order.

    ``quaternion_wxyz`` maps gsplat/OpenCV camera coordinates (+X right,
    +Y image-down, +Z forward) into world coordinates.  It is normalized by
    :class:`gs_env.camera.CameraController` and by ``from_camera_to_world``.
    """

    position: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class MoveResult:
    """Result of one collision-aware navigation translation."""

    requested_distance: float
    executed_distance: float
    collided: bool
    pose: CameraPose


@dataclass(frozen=True)
class Observation:
    """One headless render; RGB is float32 [0, 1], depth/alpha are HxW."""

    rgb: np.ndarray
    depth: np.ndarray
    alpha: np.ndarray
    camera_to_world: np.ndarray
    intrinsics: CameraIntrinsics
