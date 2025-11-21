"""Utility helpers for pose math."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

VectorLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class EcefVector:
    """Represents a vector anchored at an origin with an associated orientation."""

    origin: np.ndarray
    orientation: np.ndarray
    # XYZ components in the local frame defined by origin + orientation
    # Coordinates are forward, right, down
    components: np.ndarray


def _as_vec3(vector: VectorLike, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


def _as_quaternion(quaternion: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(quaternion, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"orientation must have shape (4,), got {arr.shape}")
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("orientation quaternion must be non-zero")
    return arr / norm


def _quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    return np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])


def _quaternion_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def ecef_vector_from_pose(
    origin_position: VectorLike,
    origin_orientation: Sequence[float] | np.ndarray,
    target_position: VectorLike,
) -> EcefVector:
    """Return the vector from origin -> target relative to the origin pose.

    The origin pose defines the coordinate frame (via its quaternion).  The returned
    vector stores the original inputs as well as the translation expressed in the
    origin-centric local frame.
    """

    origin = _as_vec3(origin_position, "origin_position")
    target = _as_vec3(target_position, "target_position")
    orientation = _as_quaternion(origin_orientation)

    delta = target - origin
    rotation = _quaternion_to_rotation_matrix(orientation)
    components = rotation.T @ delta
    return EcefVector(origin=origin, orientation=orientation, components=components)


def quaternion_difference(
    prev_orientation: Sequence[float] | np.ndarray,
    next_orientation: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the quaternion rotating from prev -> next in [w, x, y, z] order."""

    prev_quat = _as_quaternion(prev_orientation)
    next_quat = _as_quaternion(next_orientation)
    delta = _quaternion_multiply(next_quat, _quaternion_conjugate(prev_quat))
    return delta / np.linalg.norm(delta)


def quaternion_to_yaw(quaternion: Sequence[float] | np.ndarray) -> float:
    """Compute the yaw (rotation around Z) from a quaternion in radians."""

    quat = _as_quaternion(quaternion)
    w, x, y, z = quat
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


__all__ = [
    "EcefVector",
    "ecef_vector_from_pose",
    "quaternion_difference",
    "quaternion_to_yaw",
]
