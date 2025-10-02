"""Tests for dyn_obstruction2d.py."""

# import imageio.v2 as iio
from gymnasium.spaces import Box

import prbench


def test_dyn_obstruction2d_observation_random_actions():
    """Tests that observations are vectors with fixed dimensionality.

    Also tests env creation and random actions.
    """
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushPullHook2D-o10-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)
    env.close()
