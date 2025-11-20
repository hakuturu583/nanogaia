"""Utility helpers for working with Comma2k19 pose data."""

from __future__ import annotations

from typing import Sequence

import numpy as np

VectorLike = Sequence[float] | np.ndarray


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
    """Return the rotation matrix for a quaternion with [w, x, y, z] order."""

    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def relative_position_from_previous_pose(
    prev_position: VectorLike,
    prev_orientation: Sequence[float] | np.ndarray,
    next_position: VectorLike,
) -> np.ndarray:
    """Express the next pose's position in the coordinate frame of the previous pose.

    The dataset stores global positions in the ECEF frame and quaternions in the
    ``[w, x, y, z]`` order.  This helper computes the translation to the next pose
    and rotates it into the previous pose's coordinate frame so that the caller can
    reason in a local, car-centric space.
    """

    prev_pos = _as_vec3(prev_position, "prev_position")
    next_pos = _as_vec3(next_position, "next_position")
    quat = _as_quaternion(prev_orientation)

    delta_global = next_pos - prev_pos
    rotation = _quaternion_to_rotation_matrix(quat)
    return rotation.T @ delta_global


__all__ = ["relative_position_from_previous_pose"]
