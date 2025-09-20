"""Tests for obstruction2d.py."""

from gymnasium.spaces import Box

import prbench


def test_obstruction2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o2-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_obstruction2d_action_space():
    """Tests that the actions are valid and the step functions works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/Obstruction2D-o4-v0")
    obs, _ = env.reset()
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
