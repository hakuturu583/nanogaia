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


__all__ = ["EcefVector", "ecef_vector_from_pose"]
