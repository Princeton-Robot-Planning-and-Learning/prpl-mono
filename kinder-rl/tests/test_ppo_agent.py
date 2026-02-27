"""Tests for the PPO agent."""

import gymnasium
import imageio.v2 as iio
import kinder
import numpy as np
import pytest
from gymnasium import spaces
from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnv
from omegaconf import DictConfig

from kinder_rl.ppo_agent import PPOAgent


def test_ppo_agent_with_kinder_environment():
    """Test PPO agent interaction with KinDER environment (no training)."""
    kinder.register_all_environments()
    env = kinder.make("kinder/StickButton2D-b1-v0")

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
    kinder.register_all_environments()

    # Create a custom environment wrapper that fixes positions
    # NOTE: This env will by default truncate after 100 steps
    # so it is not registered with "kinder", but with gymnasium directly.
    class FixedPositionWrapper(gymnasium.Env):
        """Environment wrapper that fixes initial positions for testing."""

        def __init__(self, env: StickButton2DEnv):
            super().__init__()
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.render_mode = env.render_mode
            self.metadata = env.metadata
            obs0, _ = self.env.reset(seed=123)
            # Check if the observation space has devectorize method
            assert hasattr(self.env.observation_space, "devectorize")
            state0 = self.env.observation_space.devectorize(obs0)

            obj_name_to_obj = {o.name: o for o in list(state0.data.keys())}
            robot = obj_name_to_obj["robot"]
            button0 = obj_name_to_obj["button0"]

            state1 = state0.copy()
            state1.set(robot, "x", 1.8)
            state1.set(robot, "y", 1.0)
            state1.set(button0, "y", 1.0)
            state1.set(button0, "x", 2.0)
            self.reset_options = {"init_state": state1}
            self.num_env_steps = 0
            self.max_episode_steps = 100
            self.r = 0.0
            # Debug rendering only if render_mode is set
            # if self.render_mode is not None:
            #     _, _ = env.reset(seed=123, options=self.reset_options)
            #     img = env.render()
            #     os.makedirs("debug", exist_ok=True)
            #     iio.imwrite("debug/unit_test_fixed_env_init.png", img)

        def reset(self, seed=None, options=None):  # pylint: disable=arguments-differ
            del seed, options  # Ignore external parameters
            self.num_env_steps = 0
            self.r = 0.0
            obs, info = self.env.reset(seed=123, options=self.reset_options)
            return obs, info

        def step(self, action):
            self.num_env_steps += 1
            obs, reward, terminated, _, info = self.env.step(action)
            truncated = self.num_env_steps >= self.max_episode_steps
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

    # Register the wrapped environment with a custom ID so PPO can create it
    def make_fixed_env(render_mode=None):
        """Factory function to create the fixed environment."""
        base_env = kinder.make(
            "kinder/StickButton2D-b1-v0",
            render_mode=render_mode,
        )
        return FixedPositionWrapper(base_env)

    # Register with gymnasium
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

    # Test training
    train_metric = agent.train()

    # should have episodic_return in train_metric
    assert "episodic_return" in train_metric["eval"]
    episodic_returns = train_metric["eval"]["episodic_return"]
    assert len(episodic_returns) > 5
    mean_r_after = np.mean(episodic_returns[-5:])  # Mean of last 5 episodes
    assert mean_r_after > -300.0, f"Agent did not improve: mean return {mean_r_after}"
    agent.close()


@pytest.mark.skip(reason="The script takes too long to run in CI.")
def test_ppo_agent_training_with_fixed_environment_basemotion3d():
    """Test PPO agent can overfit on fixed BaseMotion3D environment."""
    kinder.register_all_environments()

    # Create a custom environment wrapper that fixes positions
    class FixedPositionWrapper(gymnasium.Env):
        """Environment wrapper that fixes initial positions for testing."""

        def __init__(self, env):
            super().__init__()
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.render_mode = env.render_mode
            self.metadata = env.metadata

            # Reset once to get an initial state
            obs0, _ = self.env.reset(seed=123)
            assert hasattr(self.env.observation_space, "devectorize")
            state0 = self.env.observation_space.devectorize(obs0)

            obj_name_to_obj = {o.name: o for o in list(state0.data.keys())}
            robot = obj_name_to_obj["robot"]
            target = obj_name_to_obj["target"]

            # Create a fixed initial state with robot and target nearby
            # Robot starts at origin, target is at (0.5, 0.5) - close enough to reach
            state1 = state0.copy()
            state1.set(robot, "pos_base_x", 0.0)
            state1.set(robot, "pos_base_y", 0.0)
            state1.set(robot, "pos_base_rot", 0.0)
            state1.set(target, "x", 0.1)
            state1.set(target, "y", 0.1)
            state1.set(target, "z", 0.2)  # default target_z from config

            self.reset_options = {"init_state": state1}

            self.reset_state = state1
            self.num_env_steps = 0
            self.max_episode_steps = 100
            self.r = 0.0
            self.curr_distance = 0.0
            # Debug rendering only if render_mode is set
            if self.render_mode is not None:
                _, _ = env.reset(seed=123, options=self.reset_options)
                img = env.render()
                iio.imwrite("debug/unit_test_fixed_env_init.png", img)

        def reset(self, seed=None, options=None):  # pylint: disable=arguments-differ
            del seed, options  # Ignore external parameters
            self.num_env_steps = 0
            self.r = 0.0
            obs, info = self.env.reset(seed=123, options=self.reset_options)
            self.curr_distance = self.compute_distance(obs)
            return obs, info

        def compute_distance(self, obs):
            """Compute distance from robot to target."""
            state = self.env.observation_space.devectorize(obs)
            obj_map = {o.name: o for o in state.data.keys()}

            robot = obj_map.get("robot")
            target = obj_map.get("target")
            if robot is None or target is None:
                return float("inf")

            # Robot base position
            robot_x = state.get(robot, "pos_base_x")
            robot_y = state.get(robot, "pos_base_y")

            # Target position
            target_x = state.get(target, "x")
            target_y = state.get(target, "y")

            distance = np.sqrt((robot_x - target_x) ** 2 + (robot_y - target_y) ** 2)
            return distance

        def step(self, action):
            self.num_env_steps += 1
            obs, _, terminated, _, info = self.env.step(action)
            reward = self.compute_reward(obs, terminated)
            truncated = self.num_env_steps >= self.max_episode_steps
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

        def compute_reward(self, obs, terminated):
            """Compute shaped reward based on distance to goal."""
            # 1. Terminal Bonus
            if terminated:
                return 100.0

            current_distance = self.compute_distance(obs)

            # 2. Distance Shaping (The "Guide")
            # Formula: (Old - New)
            # If we get closer, (Old > New), result is Positive.
            raw_shaping = self.curr_distance - current_distance

            # Scale this up!
            # Since your world is 0.1 units wide, a step might be 0.001.
            # Multiply by 100 or 1000 so the gradient is felt by the network.
            shaping_reward = raw_shaping * 100.0

            # 3. Time Penalty (The "Clock")
            # Forces the agent to not loiter.
            # Must be small enough that moving closer (shaping) > penalty.
            time_penalty = -0.1

            # Update state
            self.curr_distance = current_distance

            return shaping_reward + time_penalty

    # Register the wrapped environment with a custom ID
    def make_fixed_env(render_mode=None):
        """Factory function to create the fixed environment."""
        base_env = kinder.make(
            "kinder/BaseMotion3D-v0",
            render_mode=render_mode,
        )
        return FixedPositionWrapper(base_env)

    # Register with gymnasium
    gymnasium.register(
        id="BaseMotion3D-Fixed-v0",
        entry_point=make_fixed_env,
    )

    # Create PPO agent with config for quick overfitting
    cfg = DictConfig(
        {
            "total_timesteps": 500000,  # More timesteps for 3D env
            "learning_rate": 3e-4,  # Higher learning rate for faster learning
            "num_envs": 16,
            "num_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": 4,
            "update_epochs": 10,
            "norm_adv": True,
            "clip_coef": 0.1,
            "clip_vloss": True,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
            "hidden_size": 128,
            "torch_deterministic": True,
            "cuda": False,
            "anneal_lr": True,
            "tf_log_dir": "unit_test_exp",
            "exp_name": "ppo_basemotion3d_fixed_test3",
        }
    )

    agent = PPOAgent(
        seed=123,
        cfg=cfg,
        env_id="BaseMotion3D-Fixed-v0",
        max_episode_steps=100,
    )

    # Test training
    train_metric = agent.train()

    # Should have episodic_return in train_metric
    assert "episodic_return" in train_metric["eval"]
    episodic_returns = train_metric["eval"]["episodic_return"]
    assert len(episodic_returns) > 5
    mean_r_after = np.mean(episodic_returns[-5:])  # Mean of last 5 episodes
    assert mean_r_after > -100.0, f"Agent did not improve: mean return {mean_r_after}"
    agent.close()


def test_dense_reward_wrapper_basemotion3d():
    """Test that dense reward wrapper works for BaseMotion3D."""
    # Import here to avoid importing in tests that skip dense reward functionality
    from kinder_rl.dense_rewards import (  # pylint: disable=import-outside-toplevel
        wrap_with_dense_reward,
    )

    kinder.register_all_environments()

    # Test that BaseMotion3D has dense reward implemented
    env = kinder.make("kinder/BaseMotion3D-v0")
    wrapped_env = wrap_with_dense_reward(
        env, "kinder/BaseMotion3D-v0", reward_scale=0.1
    )

    wrapped_env.reset(seed=42)
    action = wrapped_env.action_space.sample()
    _, _, _, _, info = wrapped_env.step(action)

    # Check that dense reward info is present
    assert "sparse_reward" in info
    assert "dense_reward" in info
    assert np.isfinite(info["dense_reward"])
    # Dense reward can be positive (moved closer) or negative (moved away/time penalty)

    wrapped_env.close()


def test_dense_reward_not_implemented_raises():
    """Test that NotImplementedError is raised for unsupported environments."""
    # Import here to avoid importing in tests that skip dense reward functionality
    from kinder_rl.dense_rewards import (  # pylint: disable=import-outside-toplevel
        wrap_with_dense_reward,
    )

    kinder.register_all_environments()

    # DynObstruction2D doesn't have dense reward implemented
    env = kinder.make("kinder/DynObstruction2D-o1-v0")

    with pytest.raises(NotImplementedError) as exc_info:
        wrap_with_dense_reward(env, "kinder/DynObstruction2D-o1-v0")

    assert "Dense reward not implemented" in str(exc_info.value)
    env.close()


def test_ppo_with_dense_reward_basemotion3d():
    """Test PPO agent with dense reward on BaseMotion3D."""
    kinder.register_all_environments()

    # Create PPO agent with dense reward enabled
    cfg = DictConfig(
        {
            "total_timesteps": 2000,
            "learning_rate": 3e-4,
            "num_envs": 1,
            "num_steps": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "num_minibatches": 4,
            "update_epochs": 4,
            "norm_adv": True,
            "clip_coef": 0.2,
            "clip_vloss": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "target_kl": None,
            "hidden_size": 64,
            "torch_deterministic": True,
            "cuda": False,
            "anneal_lr": False,
            "tf_log": False,
            "dense_reward": True,
            "dense_reward_scale": 0.1,
        }
    )

    agent = PPOAgent(
        seed=123,
        cfg=cfg,
        env_id="kinder/BaseMotion3D-v0",
        max_episode_steps=50,
    )

    # Training should complete without errors
    train_metric = agent.train(eval_episodes=3)

    assert "episodic_return" in train_metric["eval"]
    assert len(train_metric["eval"]["episodic_return"]) == 3
    agent.close()
