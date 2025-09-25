"""Tests for dyn_obstruction2d.py."""

# import imageio.v2 as iio
import numpy as np
from gymnasium.spaces import Box

import prbench
from prbench.envs.dynamic2d.dyn_obstruction2d import DynObstruction2DEnvConfig


def test_config_cant_subclass():
    """Tests that DynObstruction2DEnvConfig cannot be subclassed but can be
    instantiated."""
    # Test that the class can be instantiated
    config = DynObstruction2DEnvConfig()
    assert config is not None

    # Test that subclassing raises TypeError
    with np.testing.assert_raises(TypeError):

        class SubConfig(DynObstruction2DEnvConfig):  # pylint: disable=unused-variable
            """This should raise a TypeError because DynObstruction2DEnvConfig is
            final."""


def test_dyn_obstruction2d_observation_space():
    """Tests that observations are vectors with fixed dimensionality."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o2-v0")
    assert isinstance(env.observation_space, Box)
    for _ in range(5):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)


def test_dyn_obstruction2d_action_space():
    """Tests that the actions are valid and the step function works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o0-v0")
    obs, _ = env.reset(seed=0)
    stable_move = np.array([0.05, 0.05, np.pi / 16, 0.0, 0.0], dtype=np.float32)
    # Check the control precision of base movements
    for s in range(5):
        obs, _ = env.reset(seed=s)
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        robot_object = name_to_object["robot"]
        robot_x = state.get(robot_object, "x")
        robot_y = state.get(robot_object, "y")
        robot_theta = state.get(robot_object, "theta")
        robot_arm_length = state.get(robot_object, "arm_length")
        robot_finger_gap = state.get(robot_object, "finger_gap")
        obs_, _, _, _, _ = env.step(stable_move)
        state_ = env.observation_space.devectorize(obs_)
        robot_x_ = state_.get(robot_object, "x")
        robot_y_ = state_.get(robot_object, "y")
        robot_theta_ = state_.get(robot_object, "theta")
        robot_arm_length_ = state_.get(robot_object, "arm_length")
        robot_finger_gap_ = state_.get(robot_object, "finger_gap")
        assert np.isclose(robot_x + stable_move[0], robot_x_, atol=1e-3)
        assert np.isclose(robot_y + stable_move[1], robot_y_, atol=1e-3)
        assert np.isclose(
            robot_theta + stable_move[2], robot_theta_, atol=1e-2
        )  # 0.5 degree
        assert np.isclose(
            robot_arm_length + stable_move[3], robot_arm_length_, atol=1e-3
        )
        assert np.isclose(
            robot_finger_gap - stable_move[4], robot_finger_gap_, atol=1e-3
        )

    obs, _ = env.reset(seed=0)
    stable_move = np.array([0.0, 0.0, 0.0, 0.05, -0.02], dtype=np.float32)
    # Check the control precision of base movements
    for s in [1, 2]:
        obs, _ = env.reset(seed=s)
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        robot_object = name_to_object["robot"]
        robot_x = state.get(robot_object, "x")
        robot_y = state.get(robot_object, "y")
        robot_theta = state.get(robot_object, "theta")
        robot_arm_length = state.get(robot_object, "arm_joint")
        robot_finger_gap = state.get(robot_object, "finger_gap")
        obs_, _, _, _, _ = env.step(stable_move)
        state_ = env.observation_space.devectorize(obs_)
        robot_x_ = state_.get(robot_object, "x")
        robot_y_ = state_.get(robot_object, "y")
        robot_theta_ = state_.get(robot_object, "theta")
        robot_arm_length_ = state_.get(robot_object, "arm_joint")
        robot_finger_gap_ = state_.get(robot_object, "finger_gap")
        assert np.isclose(robot_x + stable_move[0], robot_x_, atol=1e-3)
        assert np.isclose(robot_y + stable_move[1], robot_y_, atol=1e-3)
        assert np.isclose(
            robot_theta + stable_move[2], robot_theta_, atol=1e-2
        )  # 0.5 degree
        assert np.isclose(
            robot_arm_length + stable_move[3], robot_arm_length_, atol=1e-3
        )
        assert np.isclose(
            robot_finger_gap + stable_move[4], robot_finger_gap_, atol=1e-3
        )


def test_dyn_obstruction2d_grasping_droppping():
    """Tests that the actions are valid and the step function works."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o0-v0")
    obs, _ = env.reset(seed=0)
    state = env.observation_space.devectorize(obs)
    reset_state = state.copy()
    name_to_object = {obj.name: obj for obj in state.data}
    robot_object = name_to_object["robot"]
    target_block_object = name_to_object["target_block"]
    reset_state.set(robot_object, "x", 1.6)
    reset_state.set(robot_object, "y", 0.6)
    reset_state.set(robot_object, "theta", -np.pi / 2)
    reset_state.set(robot_object, "arm_joint", 0.24)
    reset_state.set(robot_object, "finger_gap", 0.32)
    reset_state.set(target_block_object, "x", 1.6)
    reset_state.set(target_block_object, "y", 0.2)
    reset_state.set(target_block_object, "theta", 0.0)
    reset_state.set(target_block_object, "width", 0.2)
    reset_state.set(target_block_object, "height", 0.2)
    obs, _ = env.reset(options={"init_state": reset_state})
    stable_move = np.array([0.0, 0.0, 0.0, 0.0, -0.01], dtype=np.float32)
    # Check the grasping behavior
    _, _, _, _, _ = env.step(stable_move)
    # Should not hold the object yet
    obj_centric_env = (
        env.unwrapped._object_centric_env  # pylint: disable=protected-access
    )
    assert len(obj_centric_env.robot.held_objects) == 0
    for _ in range(6):
        obs, _, _, _, _ = env.step(stable_move)
    # Should hold the object now
    obj_centric_env = (
        env.unwrapped._object_centric_env  # pylint: disable=protected-access
    )
    assert len(obj_centric_env.robot.held_objects) == 1
    # Check move the object with the robot
    move_with_object = np.array([0.0, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(3):
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        target_block_x = state.get(target_block_object, "x")
        target_block_y = state.get(target_block_object, "y")
        target_block_theta = state.get(target_block_object, "theta")
        obs_, _, _, _, _ = env.step(move_with_object)
        state_ = env.observation_space.devectorize(obs_)
        target_block_x_ = state_.get(target_block_object, "x")
        target_block_y_ = state_.get(target_block_object, "y")
        target_block_theta_ = state_.get(target_block_object, "theta")
        assert np.isclose(
            target_block_x + move_with_object[0], target_block_x_, atol=1e-3
        )
        assert np.isclose(
            target_block_y + move_with_object[1], target_block_y_, atol=1e-2
        )
        assert np.isclose(
            target_block_theta + move_with_object[2], target_block_theta_, atol=1e-2
        )
        obs = obs_

    move_with_object = np.array([0.0, 0.05, np.pi / 16, 0.0, 0.0], dtype=np.float32)
    for _ in range(5):
        state = env.observation_space.devectorize(obs)
        name_to_object = {obj.name: obj for obj in state.data}
        target_block_x = state.get(target_block_object, "x")
        target_block_y = state.get(target_block_object, "y")
        target_block_theta = state.get(target_block_object, "theta")
        obs_, _, _, _, _ = env.step(move_with_object)
        state_ = env.observation_space.devectorize(obs_)
        target_block_x_ = state_.get(target_block_object, "x")
        target_block_y_ = state_.get(target_block_object, "y")
        target_block_theta_ = state_.get(target_block_object, "theta")
        # Note that the rotation of the base cases a larger movement of the object
        assert target_block_x_ != (target_block_x + move_with_object[0])
        assert target_block_y_ != (target_block_y + move_with_object[1])
        # But the relative angle should be preserved
        assert np.isclose(
            target_block_theta + move_with_object[2], target_block_theta_, atol=1e-2
        )
        obs = obs_

    # Drop the object to the ground
    stable_move = np.array([0.0, 0.0, 0.0, 0.0, 0.01], dtype=np.float32)
    # Check the dropping behavior
    obs, _, _, _, _ = env.step(stable_move)
    state = env.observation_space.devectorize(obs)
    curr_y = state.get(target_block_object, "y")
    # Should not hold the object
    obj_centric_env = (
        env.unwrapped._object_centric_env  # pylint: disable=protected-access
    )
    assert len(obj_centric_env.robot.held_objects) == 0
    # The dropping behavior happends in the next step
    obs, _, _, _, _ = env.step(stable_move)
    state = env.observation_space.devectorize(obs)
    new_y = state.get(target_block_object, "y")
    assert new_y < curr_y  # The object should fall down due to gravity


def test_dyn_obstruction2d_different_obstruction_counts():
    """Tests that different numbers of obstructions work."""
    prbench.register_all_environments()

    for num_obs in [0, 1, 2, 3]:
        env = prbench.make(f"prbench/DynObstruction2D-o{num_obs}-v0")
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

        # Take a few steps to ensure environment works
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, _truncated, _info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, (int, float))
            if terminated:
                break


def test_dyn_obstruction2d_reset_consistency():
    """Tests that reset produces consistent observations."""
    prbench.register_all_environments()
    env = prbench.make("prbench/DynObstruction2D-o2-v0")

    # Test multiple resets
    for _ in range(3):
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)

        # Environment should not be terminated at start
        action = env.action_space.sample()
        _obs, reward, _terminated, _truncated, _info = env.step(action)
        # First step should give -1 reward (goal not satisfied immediately)
        assert reward == -1.0
