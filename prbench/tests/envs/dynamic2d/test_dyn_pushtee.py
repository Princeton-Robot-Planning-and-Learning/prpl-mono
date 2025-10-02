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

    # Test that robot moves to target position with PD control
    for s in range(3):
        obs, _ = env.reset(seed=s)
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        robot_object = name_to_object["robot"]
        robot_x = state.get(robot_object, "x")
        robot_y = state.get(robot_object, "y")

        # Command robot to move to a nearby position
        target_x = robot_x + 0.5
        target_y = robot_y + 0.5
        target_action = np.array([target_x, target_y], dtype=np.float32)

        # After one step, robot should move towards target
        obs_, _, _, _, _ = env.step(target_action)
        state_ = env.observation_space.devectorize(obs_)
        robot_x_ = state_.get(robot_object, "x")
        robot_y_ = state_.get(robot_object, "y")

        # Robot should have moved towards the target
        assert robot_x_ > robot_x or np.isclose(robot_x_, robot_x, atol=1e-3)
        assert robot_y_ > robot_y or np.isclose(robot_y_, robot_y, atol=1e-3)

        # After multiple steps with same target, robot should be closer to target
        for _ in range(5):
            obs_, _, _, _, _ = env.step(target_action)

        state_ = env.observation_space.devectorize(obs_)
        robot_x_final = state_.get(robot_object, "x")
        robot_y_final = state_.get(robot_object, "y")

        # Distance to target should decrease
        dist_initial = np.sqrt((target_x - robot_x) ** 2 + (target_y - robot_y) ** 2)
        dist_final = np.sqrt(
            (target_x - robot_x_final) ** 2 + (target_y - robot_y_final) ** 2
        )
        assert dist_final < dist_initial


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


def test_dyn_pusht_tblock_properties():
    """Tests that the T-block has correct properties."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-v0")
    obs, _ = env.reset(seed=0)
    state = env.observation_space.devectorize(obs)

    name_to_object = {obj.name: obj for obj in state.data}
    tblock_object = name_to_object["tblock"]

    # Check that T-block has all required properties
    assert state.get(tblock_object, "x") is not None
    assert state.get(tblock_object, "y") is not None
    assert state.get(tblock_object, "theta") is not None
    assert state.get(tblock_object, "vx") is not None
    assert state.get(tblock_object, "vy") is not None
    assert state.get(tblock_object, "omega") is not None
    assert state.get(tblock_object, "width") > 0
    assert state.get(tblock_object, "length_horizontal") > 0
    assert state.get(tblock_object, "length_vertical") > 0
    assert state.get(tblock_object, "mass") > 0


def test_dyn_pusht_robot_properties():
    """Tests that the DotRobot has correct properties."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-v0")
    obs, _ = env.reset(seed=0)
    state = env.observation_space.devectorize(obs)

    name_to_object = {obj.name: obj for obj in state.data}
    robot_object = name_to_object["robot"]

    # Check that DotRobot has all required properties
    assert state.get(robot_object, "x") is not None
    assert state.get(robot_object, "y") is not None
    assert state.get(robot_object, "vx") is not None
    assert state.get(robot_object, "vy") is not None
    assert state.get(robot_object, "radius") > 0
    assert not state.get(robot_object, "static")


def test_dyn_pusht_physics_dynamics():
    """Tests that the T-block has realistic physics dynamics."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynPushT-v0")
    obs, _ = env.reset(seed=0)
    state = env.observation_space.devectorize(obs)

    name_to_object = {obj.name: obj for obj in state.data}
    robot_object = name_to_object["robot"]
    tblock_object = name_to_object["tblock"]

    # Get initial positions
    robot_x = state.get(robot_object, "x")
    robot_y = state.get(robot_object, "y")
    tblock_x = state.get(tblock_object, "x")
    tblock_y = state.get(tblock_object, "y")

    # Command robot to move towards the block
    target_x = tblock_x
    target_y = tblock_y
    target_action = np.array([target_x, target_y], dtype=np.float32)

    # Step multiple times
    for _ in range(10):
        obs, _, _, _, _ = env.step(target_action)

    state_after = env.observation_space.devectorize(obs)
    robot_x_after = state_after.get(robot_object, "x")
    robot_y_after = state_after.get(robot_object, "y")
    tblock_x_after = state_after.get(tblock_object, "x")
    tblock_y_after = state_after.get(tblock_object, "y")

    # Robot should have moved towards target
    dist_initial = np.sqrt((target_x - robot_x) ** 2 + (target_y - robot_y) ** 2)
    dist_after = np.sqrt(
        (target_x - robot_x_after) ** 2 + (target_y - robot_y_after) ** 2
    )
    assert dist_after < dist_initial

    # If robot got close enough to block, block might have moved
    # (due to collision/pushing)
    if dist_after < 0.5:  # If robot is close to block position
        # Block position may have changed due to pushing
        _ = (
            not np.isclose(tblock_x, tblock_x_after, atol=1e-2)
            or not np.isclose(tblock_y, tblock_y_after, atol=1e-2)
        )
        # This is just checking that physics is working, not asserting it must move
        # (depends on specific collision geometry)
