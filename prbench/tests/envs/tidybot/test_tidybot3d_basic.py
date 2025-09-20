"""Basic tests for the TidyBot3D environment observation and action space validity,
step, and reset."""

import numpy as np

from prbench.envs.tidybot.mujoco_utils import MjAct
from prbench.envs.tidybot.tidybot3d import TidyBot3DEnv


def test_tidybot3d_observation_space():
    """Test that the observation returned by TidyBot3DEnv.reset() is within the
    observation space."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    obs = env.reset()[0]
    assert env.observation_space.contains(obs), "Observation not in observation space"
    env.close()


def test_tidybot3d_action_space():
    """Test that a sampled action is within the TidyBot3DEnv action space."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    action = env.action_space.sample()
    assert env.action_space.contains(action), "Action not in action space"
    assert isinstance(action, MjAct)
    env.close()


def test_tidybot3d_step():
    """Test that stepping the environment leads to some nontrivial change."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, _, _, _, _ = env.step(action)
    assert not np.allclose(obs["vec"], next_obs["vec"])
    env.close()


def test_tidybot3d_reset_returns_valid_observation():
    """Test that reset() returns an observation in the observation space."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    obs, info = env.reset()
    assert env.observation_space.contains(
        obs
    ), "Reset observation not in observation space"
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_reset_returns_valid_observation_with_rendering():
    """Test that reset() returns an observation in the observation space when rendering
    is enabled."""
    env = TidyBot3DEnv(num_objects=3, render_images=True)
    obs, info = env.reset()
    assert env.observation_space.contains(
        obs
    ), "Reset observation not in observation space"
    assert isinstance(info, dict)
    env.close()


def test_tidybot3d_step_returns_valid_outputs():
    """Test that step() returns valid outputs: obs in space, reward is float, done flags
    are bools."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(
        obs
    ), "Step observation not in observation space"
    assert isinstance(reward, float), "Reward is not a float"
    assert isinstance(terminated, bool), "Terminated is not a bool"
    assert isinstance(truncated, bool), "Truncated is not a bool"
    assert isinstance(info, dict), "Info is not a dict"
    env.close()


def test_tidybot3d_get_object_pos_quat():
    """Test that get_object_pos_quat() returns valid position and orientation."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    env.reset()
    for name in env._object_names:  # pylint: disable=protected-access
        pos, quat = env.get_object_pos_quat(name)
        assert len(pos) == 3, "Position should have 3 elements"
        assert len(quat) == 4, "Quaternion should have 4 elements"
    env.close()


def test_tidybot3d_set_get_object_pos_quat_consistency():
    """Test that setting and then getting an object's position and orientation is
    consistent."""
    env = TidyBot3DEnv(num_objects=3, render_images=False)
    env.reset()
    for name in env._object_names:  # pylint: disable=protected-access
        original_pos, original_quat = env.get_object_pos_quat(name)
        new_pos = [p + 0.1 for p in original_pos]
        new_quat = original_quat  # Keep orientation the same for simplicity
        env._set_object_pos_quat(  # pylint: disable=protected-access
            name, new_pos, new_quat
        )
        updated_pos, updated_quat = env.get_object_pos_quat(name)
        assert all(
            abs(o - u) < 1e-5 for o, u in zip(new_pos, updated_pos)
        ), "Position not set correctly"
        assert all(
            abs(o - u) < 1e-5 for o, u in zip(new_quat, updated_quat)
        ), "Orientation not set correctly"
    env.close()
