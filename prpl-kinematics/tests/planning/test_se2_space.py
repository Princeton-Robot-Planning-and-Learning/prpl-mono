"""Unit tests for the SE2Space configuration space."""

import math

import numpy as np
import pytest

from prpl_kinematics.planning import SE2Space


def test_se2_space_sampling_and_distance():
    """SE2Space samples within the box and measures yaw the short way around."""
    space = SE2Space("base", (-2.0, 2.0), (-1.0, 1.0))
    assert space.dimension == 3
    rng = np.random.default_rng(0)
    for _ in range(50):
        x, y, yaw = space.sample(rng)
        assert -2.0 <= x <= 2.0 and -1.0 <= y <= 1.0 and -math.pi <= yaw <= math.pi
    config = {"base": [1.0, 0.5, 0.3]}
    assert space.to_configuration(space.to_vector(config)) == config
    # Pure translation is Euclidean; pure yaw wraps the short way.
    assert space.distance(
        np.array([0, 0, 0.0]), np.array([3, 4, 0.0])
    ) == pytest.approx(5.0)
    assert space.distance(
        np.array([0, 0, 3.0]), np.array([0, 0, -3.0])
    ) == pytest.approx(2 * math.pi - 6.0)
    assert np.allclose(space.clamp(np.array([5.0, -5.0, 0.0]))[:2], [2.0, -1.0])


def test_se2_space_interpolates_yaw_short_way():
    """SE2 interpolation crosses the +-pi seam in yaw."""
    space = SE2Space("base", (-5.0, 5.0), (-5.0, 5.0))
    yaws = [
        w[2]
        for w in space.interpolate(np.array([0, 0, 3.0]), np.array([0, 0, -3.0]), 0.1)
    ]
    assert np.all(np.abs(np.diff([3.0] + yaws)) <= 0.1 + 1e-9)
    assert (yaws[-1] - (-3.0)) % (2 * math.pi) == pytest.approx(0.0, abs=1e-9)
