"""Tests for the PPO agent."""

import gymnasium
import numpy as np
import prbench
from gymnasium import spaces
from omegaconf import DictConfig

from prbench_rl.ppo_agent import PPOAgent
from prbench_rl.gym_utils import make_fixed_env


def test_ppo_agent_with_prbench_environment():
    """Test PPO agent interaction with PRBench environment (no training)."""
    prbench.register_all_environments()
    env = prbench.make("prbench/StickButton2D-b1-v0")

    # Ensure we have continuous action space
    assert isinstance(env.action_space, spaces.Box)
    assert isinstance(env.observation_space, spaces.Box)

    # Create PPO agent with minimal config for testing
    cfg = DictConfig(
        {
            "total_timesteps": 1000,
            "learning_rate": 3e-4,
            "num_envs": 1,
            "num_steps": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": 2,
            "update_epochs": 2,
            "norm_adv": True,
            "clip_coef": 0.2,
            "clip_vloss": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
            "hidden_size": 32,
            "torch_deterministic": True,
            "cuda": False,
            "tf_log": False,
        }
    )

    agent = PPOAgent(
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


def test_ppo_agent_training_with_fixed_environment():
    """Test PPO agent can overfit on fixed environment setup."""
    prbench.register_all_environments()

    # Register trivial env with gymnasium
    gymnasium.register(
        id="StickButton2D-Fixed-v0",
        entry_point=make_fixed_env,
    )

    # Create PPO agent with small config for quick overfitting
    cfg = DictConfig(
        {
            "total_timesteps": 3000,  # Use > 3000 to ensure overfitting
            "learning_rate": 3e-3,  # Higher learning rate for faster learning
            "num_envs": 1,
            "num_steps": 256,  # Small rollout for quick updates
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": 32,
            "update_epochs": 10,
            "norm_adv": True,
            "clip_coef": 0.2,
            "clip_vloss": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
            "hidden_size": 128,  # Small network for faster training
            "torch_deterministic": True,
            "cuda": False,
            "anneal_lr": False,
            "tf_log_dir": "unit_test_exp",
            "exp_name": "ppo_fixed_env_test",
        }
    )

    agent = PPOAgent(
        seed=123,
        cfg=cfg,
        env_id="StickButton2D-Fixed-v0",  # Use the registered wrapper ID
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
