import numpy as np

from nanogaia.util import relative_position_from_previous_pose


def test_relative_position_identity_orientation_matches_difference():
    prev_pos = np.array([1.0, 2.0, 3.0])
    next_pos = np.array([2.5, 1.0, 4.0])
    orientation = np.array([1.0, 0.0, 0.0, 0.0])

    result = relative_position_from_previous_pose(prev_pos, orientation, next_pos)

    np.testing.assert_allclose(result, next_pos - prev_pos)


def test_relative_position_accounts_for_previous_orientation():
    prev_pos = np.zeros(3)
    next_pos = np.array([0.0, 1.0, 0.0])
    # 90 degree rotation around Z in [w, x, y, z] format
    orientation = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])

    result = relative_position_from_previous_pose(prev_pos, orientation, next_pos)

    np.testing.assert_allclose(result, np.array([1.0, 0.0, 0.0]), atol=1e-6)
