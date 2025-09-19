"""Tests for gym utilities."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from prpl_utils.gym_utils import MultiEnvWrapper


def test_multi_env_wrapper():
    """Test basic functionality of MultiEnvWrapper with CartPole."""

    # Create environment factory function
    env_fn = lambda: gym.make("CartPole-v1")

    # Create multi-environment wrapper with 3 environments
    num_envs = 3
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

    assert multi_env.num_envs == num_envs
    assert hasattr(multi_env, "action_space")
    assert hasattr(multi_env, "observation_space")

    # Test reset
    obs_batch, info_batch = multi_env.reset(seed=123)
    assert obs_batch.shape[0] == num_envs
    assert obs_batch.shape[1] == 4  # CartPole observation space
    assert isinstance(info_batch, dict)

    # Test step with random actions
    actions = multi_env.action_space.sample()
    assert actions.shape[0] == num_envs

    obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)
    assert obs_batch.shape[0] == num_envs
    assert obs_batch.shape[1] == 4  # CartPole observation space
    assert rewards.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)
    assert isinstance(info_batch, dict)

    # Test a few more steps to verify functionality
    for _ in range(3):
        actions = multi_env.action_space.sample()
        obs_batch, rewards, terminated, truncated, info_batch = multi_env.step(actions)
        assert obs_batch.shape[0] == num_envs

    # Close environments
    multi_env.close()


def test_multi_env_wrapper_mountaincar():
    """Test MultiEnvWrapper with MountainCar environment."""

    env_fn = lambda: gym.make("MountainCar-v0")
    num_envs = 2
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

    # Test spaces
    assert multi_env.num_envs == num_envs
    assert multi_env.single_action_space.n == 3  # MountainCar has 3 actions
    assert multi_env.single_observation_space.shape == (2,)  # Position and velocity

    # Test reset
    obs_batch, _ = multi_env.reset(seed=42)
    assert obs_batch.shape == (num_envs, 2)

    # Test step
    actions = np.array([0, 1])  # Specific actions for each environment
    obs_batch, rewards, _, _, _ = multi_env.step(actions)
    assert obs_batch.shape == (num_envs, 2)
    assert rewards.shape == (num_envs,)

    multi_env.close()


def test_multi_env_wrapper_auto_reset():
    """Test auto_reset functionality."""

    # Use CartPole which has a relatively short episode length
    env_fn = lambda: gym.make("CartPole-v1", max_episode_steps=10)
    num_envs = 2
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs, auto_reset=True)

    _obs_batch, _ = multi_env.reset(seed=123)

    # Run enough steps to likely trigger episode termination
    terminated_count = 0
    for _ in range(50):
        actions = multi_env.action_space.sample()
        _obs_batch, _rewards, terminated, truncated, _info_batch = multi_env.step(
            actions
        )
        if np.any(terminated) or np.any(truncated):
            terminated_count += 1

    # Should have seen some terminations with auto-reset
    assert terminated_count > 0, "Expected some episodes to terminate and auto-reset"

    multi_env.close()


def test_multi_env_wrapper_seeding():
    """Test seeding behavior."""

    env_fn = lambda: gym.make("CartPole-v1")
    num_envs = 3

    # Test with single seed
    multi_env1 = MultiEnvWrapper(env_fn, num_envs=num_envs)
    obs1, _ = multi_env1.reset(seed=42)

    multi_env2 = MultiEnvWrapper(env_fn, num_envs=num_envs)
    obs2, _ = multi_env2.reset(seed=42)

    # Should get same initial observations with same seed
    np.testing.assert_array_equal(obs1, obs2)

    # Test with different seeds
    obs3, _ = multi_env1.reset(seed=123)
    assert not np.array_equal(obs1, obs3)

    # Test with seed list
    seed_list = [10, 20, 30]
    obs4, _ = multi_env1.reset(seed=seed_list)
    assert obs4.shape == (num_envs, 4)

    multi_env1.close()
    multi_env2.close()


def test_multi_env_wrapper_different_action_spaces():
    """Test with environment that has continuous action space."""

    try:
        env_fn = lambda: gym.make("Pendulum-v1")
        num_envs = 2
        multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs)

        # Test spaces
        assert multi_env.num_envs == num_envs
        assert multi_env.single_observation_space.shape == (3,)
        assert multi_env.single_action_space.shape == (1,)

        # Test reset and step
        obs_batch, _ = multi_env.reset(seed=42)
        assert obs_batch.shape == (num_envs, 3)

        actions = multi_env.action_space.sample()
        assert actions.shape == (num_envs, 1)

        obs_batch, rewards, _terminated, _truncated, _info_batch = multi_env.step(
            actions
        )
        assert obs_batch.shape == (num_envs, 3)
        assert rewards.shape == (num_envs,)

        multi_env.close()

    except gym.error.UnregisteredEnv:
        # Skip test if Pendulum is not available
        pytest.skip("Pendulum-v1 environment not available")


def test_multi_env_wrapper_no_auto_reset():
    """Test behavior with auto_reset=False."""

    env_fn = lambda: gym.make("CartPole-v1", max_episode_steps=5)
    num_envs = 2
    multi_env = MultiEnvWrapper(env_fn, num_envs=num_envs, auto_reset=False)

    _obs_batch, _ = multi_env.reset(seed=123)

    # Run enough steps to trigger termination
    done_envs = set()
    for step in range(20):
        actions = multi_env.action_space.sample()
        _obs_batch, _rewards, terminated, truncated, _info_batch = multi_env.step(
            actions
        )

        # Track which environments are done
        for i in range(num_envs):
            if terminated[i] or truncated[i]:
                done_envs.add(i)

        # Once we have terminated environments, verify they stay terminated
        if done_envs and step > 10:
            break

    # Should have some terminated environments
    assert len(done_envs) > 0, "Expected some environments to terminate"

    multi_env.close()


def test_multi_env_wrapper_max_steps():
    """Test that environments are truncated at the same max step."""

    env_fn = lambda: gym.make("CartPole-v1")
    num_envs = 3
    max_episode_steps = 10
    multi_env = MultiEnvWrapper(
        env_fn,
        num_envs=num_envs,
        max_episode_steps=max_episode_steps,
        auto_reset=False,  # Disable auto-reset to test truncation behavior
    )

    _obs_batch, _ = multi_env.reset(seed=123)

    # Step through the environment until we hit the max step limit
    for step in range(max_episode_steps + 2):
        actions = multi_env.action_space.sample()
        _obs_batch, _rewards, _terminated, truncated, _info_batch = multi_env.step(
            actions
        )

        if step < max_episode_steps - 1:
            # Before max steps, no environments should be truncated
            assert not np.any(
                truncated
            ), f"Step {step}: Unexpected truncation before max_episode_steps"
        elif step == max_episode_steps - 1:
            # At max steps, all environments should be truncated
            assert np.all(
                truncated
            ), f"Step {step}: All environments should be truncated at max_episode_steps"
            # Verify all environments are marked as needing reset
            assert np.all(
                multi_env._env_needs_reset  # pylint: disable=protected-access
            ), "All environments should need reset after truncation"
            break

    multi_env.close()


def test_multi_env_wrapper_to_tensor():
    """Test that to_tensor flag converts observations to PyTorch tensors."""

    env_fn = lambda: gym.make("CartPole-v1")
    num_envs = 3

    # Test with to_tensor=False (default) - should return numpy arrays
    multi_env_numpy = MultiEnvWrapper(env_fn, num_envs=num_envs, to_tensor=False)
    obs_numpy, _ = multi_env_numpy.reset(seed=123)

    # Observations should be numpy arrays
    assert isinstance(
        obs_numpy, np.ndarray
    ), f"Expected numpy array, got {type(obs_numpy)}"
    assert obs_numpy.shape == (num_envs, 4)  # CartPole observation space

    # Step and check return types
    actions = multi_env_numpy.action_space.sample()
    obs_numpy, rewards_numpy, terminated_numpy, truncated_numpy, _info = (
        multi_env_numpy.step(actions)
    )

    assert isinstance(obs_numpy, np.ndarray), "Observations should be numpy arrays"
    assert isinstance(rewards_numpy, np.ndarray), "Rewards should be numpy arrays"
    assert isinstance(terminated_numpy, np.ndarray), "Terminated should be numpy arrays"
    assert isinstance(truncated_numpy, np.ndarray), "Truncated should be numpy arrays"

    multi_env_numpy.close()

    # Test with to_tensor=True - should return PyTorch tensors
    multi_env_tensor = MultiEnvWrapper(
        env_fn,
        num_envs=num_envs,
        to_tensor=True,
        device="cpu",  # Use CPU for consistent testing
    )
    obs_tensor, _ = multi_env_tensor.reset(seed=123)

    # Observations should be PyTorch tensors
    assert isinstance(
        obs_tensor, torch.Tensor
    ), f"Expected torch.Tensor, got {type(obs_tensor)}"
    assert obs_tensor.shape == (num_envs, 4)  # CartPole observation space
    assert obs_tensor.device.type == "cpu", "Tensor should be on CPU device"

    # Step and check return types
    actions = multi_env_tensor.action_space.sample()
    obs_tensor, rewards_tensor, terminated_tensor, truncated_tensor, _info = (
        multi_env_tensor.step(actions)
    )

    assert isinstance(obs_tensor, torch.Tensor), "Observations should be torch tensors"
    assert isinstance(rewards_tensor, torch.Tensor), "Rewards should be torch tensors"
    assert isinstance(
        terminated_tensor, torch.Tensor
    ), "Terminated should be torch tensors"
    assert isinstance(
        truncated_tensor, torch.Tensor
    ), "Truncated should be torch tensors"

    # Verify device placement
    assert obs_tensor.device.type == "cpu", "Observation tensors should be on CPU"
    assert rewards_tensor.device.type == "cpu", "Reward tensors should be on CPU"

    # Verify tensor dtypes are appropriate
    assert obs_tensor.dtype == torch.float32, "Observation tensors should be float32"
    assert rewards_tensor.dtype == torch.float32, "Reward tensors should be float32"
    assert terminated_tensor.dtype == torch.bool, "Terminated tensors should be bool"
    assert truncated_tensor.dtype == torch.bool, "Truncated tensors should be bool"

    # Test tensor actions work correctly
    tensor_actions = torch.tensor([0, 1, 0], dtype=torch.int64, device="cpu")
    obs_from_tensor_actions, _rewards, _terminated, _truncated, _info = (
        multi_env_tensor.step(tensor_actions)
    )
    assert isinstance(
        obs_from_tensor_actions, torch.Tensor
    ), "Should handle tensor actions and return tensors"

    multi_env_tensor.close()

    # Verify that numpy and tensor results have same numerical values (with same seed)
    # Reset both environments with same seed to compare values
    multi_env_numpy = MultiEnvWrapper(env_fn, num_envs=num_envs, to_tensor=False)
    multi_env_tensor = MultiEnvWrapper(
        env_fn, num_envs=num_envs, to_tensor=True, device="cpu"
    )

    obs_numpy, _ = multi_env_numpy.reset(seed=42)
    obs_tensor, _ = multi_env_tensor.reset(seed=42)

    # Convert tensor to numpy for comparison
    obs_tensor_as_numpy = obs_tensor.detach().cpu().numpy()
    np.testing.assert_array_almost_equal(
        obs_numpy,
        obs_tensor_as_numpy,
        decimal=6,
        err_msg="Tensor and numpy modes should produce same values with same seed",
    )

    multi_env_numpy.close()
    multi_env_tensor.close()
