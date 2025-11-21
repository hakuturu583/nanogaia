import numpy as np
import pytest

from nanogaia.comma2k19_dataset import CommaDataset


def test_get_diff_2d_pose_returns_relative_xy_and_yaw():
    current_position = np.zeros(3)
    next_position = np.array([1.0, 0.0, 0.0])
    current_orientation = np.array([1.0, 0.0, 0.0, 0.0])
    next_orientation = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])

    diff = CommaDataset._get_diff_2d_pose(
        current_orientation, current_position, next_orientation, next_position
    )

    assert diff.shape == (3,)
    np.testing.assert_allclose(diff[:2], np.array([1.0, 0.0]), atol=1e-6)
    assert diff[2] == pytest.approx(np.pi / 2)
