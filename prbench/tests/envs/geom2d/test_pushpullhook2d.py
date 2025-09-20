"""Tests for pushpullhook2d.py."""

from conftest import MAKE_VIDEOS
from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo

import prbench
from prbench.envs.geom2d.pushpullhook2d import ObjectCentricPushPullHook2DEnv


def test_object_centric_pushpullhook2d_env():
    """Tests for ObjectCentricPushPullHook2DEnv()."""
    env = ObjectCentricPushPullHook2DEnv()
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")
    env.reset(seed=123)
    env.action_space.seed(123)
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
    env.close()


def test_pushpullhook2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/PushPullHook2D-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_pushpullhook2d_action_space():
    """Tests that actions are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/PushPullHook2D-v0")
    assert isinstance(env.action_space, Box)
    for _ in range(5):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
