"""Tests for dyn_pushtee.py."""

import numpy as np
from gymnasium.spaces import Box

import prbench


def test_dyn_pusht_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_dyn_pusht_action_space():
    """Tests that the actions are valid and the step function works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-v0")
    obs, _ = env.reset(seed=0)

    # Test that robot moves with delta actions
    for s in range(3):
        obs, _ = env.reset(seed=s)
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        robot_object = name_to_object["robot"]
        robot_x = state.get(robot_object, "x")
        robot_y = state.get(robot_object, "y")

        # Command robot to move with delta action
        delta_action = np.array([0.05, 0.05], dtype=np.float32)

        # After one step, robot should move in positive x and y
        obs_, _, _, _, _ = env.step(delta_action)
        state_ = env.observation_space.devectorize(obs_)
        robot_x_ = state_.get(robot_object, "x")
        robot_y_ = state_.get(robot_object, "y")

        # Robot should have moved in positive x and y directions
        assert np.isclose(robot_x_, robot_x + 0.05, atol=1e-5)
        assert np.isclose(robot_y_, robot_y + 0.05, atol=1e-5)


def test_dyn_pusht_random_actions():
    """Tests that observations are valid with random actions."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(3):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            assert 0.0 <= reward <= 1.0  # Reward should be clipped between 0 and 1
            if terminated or truncated:
                break
    env.close()
