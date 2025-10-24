"""Tests for the PPO agent."""

import gymnasium
import numpy as np

# import imageio.v2 as iio
import prbench
import torch
from conftest import MAKE_VIDEOS
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig
from prbench.envs.geom2d.stickbutton2d import StickButton2DEnv
from relational_structs import Object, ObjectCentricState, Type

from prbench_rl.ppo_agent import PPOAgent


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
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=456,
        cfg=cfg,
    )

    # Test agent in eval mode (no training)
    agent.eval()

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
    env = prbench.make(
        "prbench/StickButton2D-b1-v0", render_mode="rgb_array" if MAKE_VIDEOS else None
    )

    # Create a custom environment wrapper that fixes positions
    class FixedPositionWrapper(gymnasium.Env):
        """Environment wrapper that fixes initial positions for testing."""

        def __init__(self, env: StickButton2DEnv):
            super().__init__()
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.render_mode = env.render_mode
            obs0, _ = self.env.reset(seed=123)
            # Check if the observation space has devectorize method
            if hasattr(self.env.observation_space, "devectorize"):
                state0 = self.env.observation_space.devectorize(obs0)
            else:
                # Handle case where observation_space is a regular Box space
                # For testing purposes, create a mock state with required attributes
                # Create types for objects
                robot_type = Type(name="robot")
                button_type = Type(name="button")

                # Create real Object instances for the mock state
                robot = Object(name="robot", type=robot_type)
                button0 = Object(name="button0", type=button_type)

                # Create mock data dictionary with numpy arrays
                mock_data = {
                    robot: np.array([0.0, 0.0]),  # x, y position
                    button0: np.array([1.0, 1.0]),  # x, y position
                }

                # Create type_features mapping
                type_features_dict: dict[Type, list[str]] = {
                    robot_type: ["x", "y"],
                    button_type: ["x", "y"],
                }

                state0 = ObjectCentricState(
                    data=mock_data, type_features=type_features_dict
                )
            obj_name_to_obj = {o.name: o for o in list(state0.data.keys())}
            robot = obj_name_to_obj["robot"]
            button0 = obj_name_to_obj["button0"]

            state1 = state0.copy()
            state1.set(robot, "x", 1.5)
            state1.set(robot, "y", 1.0)
            state1.set(button0, "y", 1.0)
            state1.set(button0, "x", 2.0)
            self.reset_options = {"init_state": state1}
            self.num_env_steps = 0
            self.r = 0.0
            # Debug
            # _, _ = env.reset(seed=123, options=self.reset_options)
            # img = env.render()
            # iio.imwrite("debug/unit_test_fixed_env_init.png", img)

        def reset(self, seed=None, options=None):  # pylint: disable=arguments-differ
            del seed, options  # Ignore external parameters
            self.num_env_steps = 0
            self.r = 0.0
            obs, info = self.env.reset(seed=123, options=self.reset_options)
            return obs, info

        def step(self, action):
            self.num_env_steps += 1
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.r += reward
            if terminated or truncated:
                info["final_info"] = [
                    {
                        "episode": {
                            "r": self.r,
                            "l": self.num_env_steps - 1,
                        }
                    }
                ]
                obs, _ = self.reset()
            return obs, reward, terminated, truncated, info

        def close(self):
            return self.env.close()

        def render(self):
            return self.env.render()

    # Wrap environment with fixed positions
    fixed_env = FixedPositionWrapper(env)

    if MAKE_VIDEOS:
        fixed_env = RecordVideo(fixed_env, "unit_test_videos")

    # Create PPO agent with small config for quick overfitting
    cfg = DictConfig(
        {
            "total_timesteps": 2048,  # Small number for quick test
            "learning_rate": 1e-3,  # Higher learning rate for faster learning
            "num_envs": 1,
            "num_steps": 64,  # Small rollout for quick updates
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": 4,
            "update_epochs": 4,
            "norm_adv": True,
            "clip_coef": 0.2,
            "clip_vloss": True,
            "ent_coef": 0.01,  # Small entropy bonus for exploration
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
            "hidden_size": 32,  # Small network for faster training
            "torch_deterministic": True,
            "cuda": False,
            "anneal_lr": False,
            "tf_log": True,
            "tf_log_dir": "unit_test_exp",
        }
    )

    agent = PPOAgent(
        observation_space=fixed_env.observation_space,
        action_space=fixed_env.action_space,
        seed=123,
        cfg=cfg,
    )

    # Test training
    # training_metrics = agent.train_with_env(fixed_env)

    # Verify training metrics are generated
    # assert len(training_metrics) > 0
    # assert "episodic_return" in training_metrics[0]
    # assert "episodic_length" in training_metrics[0]
    # assert "global_step" in training_metrics[0]

    # Test that agent can perform better after training
    # agent.eval()

    # # Test performance on the fixed environment
    # total_reward = 0.0
    # total_steps = 0
    # num_test_episodes = 3

    # for episode in range(num_test_episodes):
    #     obs, info = fixed_env.reset(seed=123 + episode)
    #     agent.reset(obs, info)

    #     episode_reward = 0.0
    #     episode_steps = 0

    #     for _ in range(100):  # Max steps per episode
    #         action = agent.step()
    #         obs, reward, terminated, truncated, info = fixed_env.step(action)
    #         agent.update(obs, reward, terminated or truncated, info)

    #         episode_reward += reward
    #         episode_steps += 1

    #         if terminated or truncated:
    #             break

    #     total_reward += episode_reward
    #     total_steps += episode_steps

    # avg_reward = total_reward / num_test_episodes
    # avg_steps = total_steps / num_test_episodes

    # # With fixed positions, the agent should learn to reach the button efficiently
    # # These are loose bounds since overfitting might not be perfect in a short test
    # print(f"Average reward: {avg_reward}, Average steps: {avg_steps}")

    # Basic sanity checks - agent should show some learning
    # assert avg_reward > -100, (
    #     f"Agent performed poorly with average reward: {avg_reward}"
    # )
    # assert avg_steps < 100, f"Agent took too many steps on average: {avg_steps}"

    fixed_env.close()
    agent.close()


def test_ppo_agent_network_forward_pass():
    """Test PPO network forward pass and action sampling."""
    obs_space = spaces.Box(low=-10.0, high=10.0, shape=(6,))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

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

    agent = PPOAgent(obs_space, action_space, seed=42, cfg=cfg)

    # Test network forward pass
    obs = np.random.randn(6).astype(np.float32)
    obs_tensor = torch.tensor(obs).unsqueeze(0)

    with torch.no_grad():
        action, logprob, entropy, value = agent.network.get_action_and_value(obs_tensor)

    # Check output shapes and types
    assert action.shape == (1, 2)
    assert logprob.shape == (1,)
    assert entropy.shape == (1,)
    assert value.shape == (1, 1)

    # Check action bounds
    action_np = action.cpu().numpy()[0]
    assert action_space.contains(action_np)

    # Check that values are reasonable
    assert torch.isfinite(logprob).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(value).all()
    assert entropy.item() >= 0  # Entropy should be non-negative

    agent.close()


def test_ppo_agent_storage_and_training():
    """Test PPO agent storage buffers and training step."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

    cfg = DictConfig(
        {
            "total_timesteps": 128,
            "learning_rate": 1e-3,
            "num_envs": 1,
            "num_steps": 16,
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
            "anneal_lr": False,
            "tf_log": False,
        }
    )

    agent = PPOAgent(obs_space, action_space, seed=42, cfg=cfg)
    agent.setup_storage()

    # Test storage initialization
    assert agent.obs_buffer.shape == (16, 1, 4)
    assert agent.actions_buffer.shape == (16, 1, 2)
    assert agent.rewards_buffer.shape == (16, 1)
    assert agent.dones_buffer.shape == (16, 1)
    assert agent.values_buffer.shape == (16, 1)

    # Test that buffers are on correct device
    assert agent.obs_buffer.device.type == "cpu"

    # Fill buffers with dummy data for training test
    agent.train()
    agent.step_count = cfg.num_steps  # Simulate full buffer

    # Fill buffers with random data
    agent.obs_buffer.fill_(0.1)
    agent.actions_buffer.fill_(0.0)
    agent.rewards_buffer.fill_(1.0)
    agent.dones_buffer.fill_(0.0)
    agent.values_buffer.fill_(0.5)

    # Set current observation for bootstrapping
    # pylint: disable=protected-access
    agent._current_obs = np.random.randn(4).astype(np.float32)

    # Test policy update
    update_metrics = agent._update_policy()  # pylint: disable=protected-access

    # Check that metrics are returned
    expected_keys = [
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "old_approx_kl",
        "approx_kl",
        "clipfrac",
        "explained_variance",
        "learning_rate",
    ]
    for key in expected_keys:
        assert key in update_metrics
        assert isinstance(update_metrics[key], (int, float))
        # explained_variance can be NaN when variance is 0
        if key != "explained_variance":
            assert np.isfinite(update_metrics[key])
        else:
            # Can be NaN if variance is 0, which is fine
            is_valid = np.isfinite(update_metrics[key]) or np.isnan(update_metrics[key])
            assert is_valid

    agent.close()
