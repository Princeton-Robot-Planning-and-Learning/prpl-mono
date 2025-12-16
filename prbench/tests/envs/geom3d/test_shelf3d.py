"""Tests for ground3d.py."""

import numpy as np
import pytest

from prbench.envs.geom3d.shelf3d import (
    Shelf3DEnv,
)


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Shelf3DEnv(num_cubes=2, use_gui=False, render_mode="rgb_array")
    yield environment
    environment.close()


def test_shelf3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in shelf env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env._object_centric_env.physics_client_id)
