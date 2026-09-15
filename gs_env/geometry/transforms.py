"""Tested rigid-transform helpers for the single project-wide camera convention."""

from __future__ import annotations

import numpy as np

from gs_env.types import CameraPose


def _vector(value: np.ndarray | list[float] | tuple[float, ...], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result


def normalize_quaternion(quaternion_wxyz: np.ndarray | list[float]) -> np.ndarray:
    """Return a unit wxyz quaternion, rejecting zero or non-finite input."""
    quaternion = _vector(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("quaternion_wxyz must not be zero")
    return quaternion / norm


def quaternion_to_rotation_matrix(quaternion_wxyz: np.ndarray | list[float]) -> np.ndarray:
    """Convert a unit-or-not wxyz quaternion to a 3x3 active rotation matrix."""
    w, x, y, z = normalize_quaternion(quaternion_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to its normalized wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite matrix with shape (3, 3)")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError("rotation must be orthonormal with determinant +1")
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale, (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array([(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale])
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array([(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale])
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array([(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale])
    return normalize_quaternion(quaternion)


def camera_to_world_from_pose(pose: CameraPose) -> np.ndarray:
    """Build the 4x4 C2W matrix used directly by the renderer."""
    position = _vector(pose.position, 3, "position")
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = quaternion_to_rotation_matrix(pose.quaternion_wxyz).astype(np.float32)
    c2w[:3, 3] = position.astype(np.float32)
    return c2w


def pose_from_camera_to_world(camera_to_world: np.ndarray) -> CameraPose:
    """Parse a rigid 4x4 C2W transform into the canonical pose representation."""
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all() or not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("camera_to_world must be a finite rigid 4x4 homogeneous matrix")
    return CameraPose(c2w[:3, 3].copy(), rotation_matrix_to_quaternion(c2w[:3, :3]))


def world_to_camera_from_pose(pose: CameraPose) -> np.ndarray:
    """Build W2C, the inverse of a camera pose's C2W transform."""
    rotation = quaternion_to_rotation_matrix(pose.quaternion_wxyz)
    position = _vector(pose.position, 3, "position")
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation.T.astype(np.float32)
    w2c[:3, 3] = (-rotation.T @ position).astype(np.float32)
    return w2c


def local_to_world_vector(pose: CameraPose, vector_camera: np.ndarray | list[float]) -> np.ndarray:
    """Rotate a direction/vector from camera-local to world coordinates."""
    return quaternion_to_rotation_matrix(pose.quaternion_wxyz) @ _vector(vector_camera, 3, "vector_camera")


def axis_angle_rotation(axis: np.ndarray | list[float], angle_rad: float) -> np.ndarray:
    """Return the right-handed active rotation around a local coordinate axis."""
    unit_axis = _vector(axis, 3, "axis")
    norm = np.linalg.norm(unit_axis)
    if norm < 1e-12 or not np.isfinite(angle_rad):
        raise ValueError("axis must be non-zero and angle_rad finite")
    x, y, z = unit_axis / norm
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return cosine * np.eye(3) + sine * cross + (1 - cosine) * np.outer([x, y, z], [x, y, z])
