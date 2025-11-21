import numpy as np

from nanogaia.util import ecef_vector_from_pose
import pytest


def test_ecef_vector_components_match_difference_for_identity_orientation():
    origin = np.array([-2713211.5650, -4266767.1871, 3874714.6275])
    target = np.array([-2713211.4180, -4266768.3242, 3874713.3811])
    orientation = np.array([1.0, 0.0, 0.0, 0.0])

    vector = ecef_vector_from_pose(origin, orientation, target)

    np.testing.assert_allclose(vector.components, target - origin)
    np.testing.assert_allclose(vector.origin, origin)
    np.testing.assert_allclose(vector.orientation, orientation)


def test_ecef_vector_rotates_components_into_local_frame():
    origin = np.zeros(3)
    target = np.array([0.0, 1.0, 0.0])
    orientation = np.array([np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2])

    vector = ecef_vector_from_pose(origin, orientation, target)

    np.testing.assert_allclose(vector.components, np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_ecef_vector_validates_shapes():
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    target = np.array([1.0, 2.0, 3.0])
    orientation = np.array([1.0, 0.0, 0.0])

    with np.testing.assert_raises(ValueError):
        ecef_vector_from_pose(origin, np.array([1.0, 0.0, 0.0, 0.0]), target)

    with np.testing.assert_raises(ValueError):
        ecef_vector_from_pose(np.zeros(3), orientation, target)


def test_eccf_to_local_coordinate():
    origin = np.array([-2713211.5650, -4266767.1871, 3874714.6275])
    target = np.array([-2713211.4180, -4266768.3242, 3874713.3811])
    orientation = np.array([0.4482, -0.5959, 0.6500, 0.1466])
    vector = ecef_vector_from_pose(origin, orientation, target).components
    assert vector.shape == (3,)
    assert vector[0] == pytest.approx(1.69195453)
    assert vector[1] == pytest.approx(0.01441992)
    assert vector[2] == pytest.approx(-0.07211311)
