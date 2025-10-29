"""Tests for the SAC agent."""

import gymnasium
import numpy as np
import prbench
from gymnasium import spaces
from omegaconf import DictConfig

from prbench_rl.sac_agent import SACAgent
from prbench_rl.gym_utils import make_fixed_env


def test_sac_agent_with_prbench_environment():
    """Test SAC agent interaction with PRBench environment (no training)."""
    prbench.register_all_environments()
    env = prbench.make("prbench/StickButton2D-b1-v0")

    # Ensure we have continuous action space
    assert isinstance(env.action_space, spaces.Box)
    assert isinstance(env.observation_space, spaces.Box)

    # Create SAC agent with minimal config for testing
    cfg = DictConfig(
        {
            "total_timesteps": 1000,
            "policy_lr": 3e-4,
            "q_lr": 1e-3,
            "num_envs": 1,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,
            "learning_starts": 100,
            "buffer_size": 10000,
            "policy_frequency": 2,
            "target_network_frequency": 1,
            "alpha": 0.2,
            "autotune": True,
            "hidden_size": 64,
            "torch_deterministic": True,
            "cuda": False,
            "tf_log": False,
        }
    )

    agent = SACAgent(
        seed=456,
        observation_space=env.observation_space,
        action_space=env.action_space,
        cfg=cfg,
    )

    # Test agent in eval mode (no training)
    agent.eval()  # type: ignore[no-untyped-call]

    obs, info = env.reset(seed=456)
    agent.reset(obs, info)

    # Test agent interaction with environment
    for _ in range(20):
        assert env.observation_space.contains(obs)

        action = agent.step()
        assert env.action_space.contains(action)
        assert isinstance(action, np.ndarray)

        obs, reward, terminated, truncated, info = env.step(action)

        # Test transition learning (should not raise errors)
        agent.update(
            obs=obs,
            reward=reward,
            done=terminated or truncated,
            info=info,
        )

        if terminated or truncated:
            break

    env.close()
    agent.close()

def test_sac_agent_training_with_fixed_environment():
    """Test SAC agent can overfit on fixed environment setup."""
    prbench.register_all_environments()

    # Register with gymnasium
    gymnasium.register(
        id="StickButton2D-SAC-Fixed-v0",
        entry_point=make_fixed_env,
    )

    # Create SAC agent with config for quick overfitting
    cfg = DictConfig(
        {
            "total_timesteps": 10000,  # Fewer timesteps for SAC
            "policy_lr": 3e-4,
            "q_lr": 1e-3,
            "num_envs": 1,
            "gamma": 0.99,
            "tau": 0.005,
            "batch_size": 256,  # Smaller batch for testing
            "learning_starts": 1000,  # Start learning after some exploration
            "buffer_size": 10000,
            "policy_frequency": 1,  # Update policy every step
            "target_network_frequency": 1,
            "alpha": 0.2,
            "autotune": True,
            "hidden_size": 256,
            "torch_deterministic": True,
            "cuda": False,
            "eval_freq": 0,  # Disable eval during training for speed
            "tf_log_dir": "unit_test_exp",
            "exp_name": "sac_fixed_env_test",
        }
    )

    agent = SACAgent(
        seed=123,
        cfg=cfg,
        env_id="StickButton2D-SAC-Fixed-v0",  # Use the registered wrapper ID
        max_episode_steps=100,
    )

    before_train_eval = agent.evaluate(5)
    mean_r_before = np.mean(before_train_eval["episodic_return"])

    # Test training
    train_metric = agent.train()

    # should have episodic_return in train_metric
    assert "episodic_return" in train_metric
    episodic_returns = train_metric["episodic_return"]
    assert len(episodic_returns) > 10
    mean_r_after = np.mean(episodic_returns[-5:])  # Mean of last 5 episodes
    assert (
        mean_r_after > mean_r_before
    ), f"Agent did not improve: before={mean_r_before}, after={mean_r_after}"
    agent.close()
