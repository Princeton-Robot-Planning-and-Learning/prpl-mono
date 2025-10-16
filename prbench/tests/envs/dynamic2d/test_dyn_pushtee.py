"""Tests for dyn_pushtee.py."""

import numpy as np
from gymnasium.spaces import Box
import pytest
import prbench
from prbench.utils import load_demo

def test_dyn_pusht_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-t1-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_dyn_pusht_action_space():
    """Tests that the actions are valid and the step function works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-t1-v0")
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
    env = prbench.make("prbench/DynPushT-t1-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(3):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            if terminated or truncated:
                break
    env.close()


def test_dyn_pusht_goal_achievement():
    """Tests that the goal can be achieved by moving the robot to the goal."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-t1-v0")
    obs, _ = env.reset(seed=42)
    state = env.observation_space.devectorize(obs)
    name_to_object = {obj.name: obj for obj in state.data}
    tblock_object = name_to_object["tblock"]
    goal_tblock_object = name_to_object["goal_tblock"]

    zero_action = np.array([0.0, 0.0], dtype=np.float32)
    _, _, terminated, _, _ = env.step(zero_action)
    assert not terminated

    # Move tblock to goal position
    new_state = state.copy()
    new_state.set(tblock_object, "x", state.get(goal_tblock_object, "x"))
    new_state.set(tblock_object, "y", state.get(goal_tblock_object, "y"))
    new_state.set(tblock_object, "theta", state.get(goal_tblock_object, "theta"))
    obs, _ = env.reset(options={"init_state": new_state})
    _, _, terminated, _, _ = env.step(zero_action)
    assert terminated


def test_dyn_pusht_replayable():
    """Tests that reset with options works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-t1-v0")

    # Extract demo information
    demo_path = 'prbench/demos/DynPushT-t1/0/1760636935.p'
    demo_data = load_demo(demo_path)
    env_id = demo_data["env_id"]
    actions = demo_data["actions"]
    expected_observations = demo_data["observations"]
    expected_rewards = demo_data.get("rewards", None)
    seed = demo_data["seed"]

    # Skip if no actions to replay
    if len(actions) == 0:
        pytest.skip(f"Demo {demo_path} contains no actions")

    # Create environment
    env = prbench.make(env_id, render_mode="rgb_array")

    # Test reproducibility: reset with seed and replay actions
    obs, _ = env.reset(seed=seed)

    # Check initial observation matches
    assert np.allclose(
        obs, expected_observations[0], atol=1e-5
    ), f"Initial observation mismatch in {demo_path}"

    # Replay all actions and verify observations/rewards
    for i, _ in enumerate(expected_observations):
        action = actions[i]
        obs_next, reward, terminated, truncated, _ = env.step(action)
        # img = env.render()  # type: ignore[no-untyped-call]
        # iio.imwrite(f"debug/replay_pickle/dyn_obstruction2d_{i:03d}.png", img)

        # Check observation matches
        expected_obs = expected_observations[i + 1]
        assert np.allclose(
            obs_next, expected_obs, atol=1e-5
        ), f"Step {i} observation mismatch in {demo_path}"

        # Check reward matches (if available)
        if expected_rewards is not None and i < len(expected_rewards):
            expected_reward = expected_rewards[i]
            assert reward == expected_reward, (
                f"Reward mismatch at step {i} in {demo_path}: "
                f"got {reward}, expected {expected_reward}"
            )
        # Stop if episode ended early
        if terminated or truncated:
            break
    env.close()  # type: ignore[no-untyped-call]


def test_dyn_pusht_resetable():
    """Tests that reset with options works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-t1-v0")

    # Extract demo information
    demo_path = 'prbench/demos/DynPushT-t1/0/1760636935.p'
    demo_data = load_demo(demo_path)
    env_id = demo_data["env_id"]
    actions = demo_data["actions"]
    expected_observations = demo_data["observations"]
    expected_rewards = demo_data.get("rewards", None)

    # Skip if no actions to replay
    if len(actions) == 0:
        pytest.skip(f"Demo {demo_path} contains no actions")

    # Create environment
    env = prbench.make(env_id, render_mode="rgb_array")

    # Test reproducibility: reset with seed and replay actions
    # Replay all actions and verify observations/rewards
    for i, prev_obs in enumerate(expected_observations):
        reset_options = {"init_state": prev_obs}
        obs, _ = env.reset(options=reset_options)

        action = actions[i]
        obs_next, reward, terminated, truncated, _ = env.step(action)
        # img = env.render()  # type: ignore[no-untyped-call]
        # iio.imwrite(f"debug/replay_pickle/dyn_obstruction2d_{i:03d}.png", img)

        # # Check observation matches
        expected_obs = expected_observations[i + 1]
        assert np.allclose(
            obs_next, expected_obs, atol=1e-5
        ), f"Step {i} observation mismatch in {demo_path}"

        # Check reward matches (if available)
        if expected_rewards is not None and i < len(expected_rewards):
            expected_reward = expected_rewards[i]
            assert reward == expected_reward, (
                f"Reward mismatch at step {i} in {demo_path}: "
                f"got {reward}, expected {expected_reward}"
            )
        # Stop if episode ended early
        if terminated or truncated:
            break
    env.close()  # type: ignore[no-untyped-call]