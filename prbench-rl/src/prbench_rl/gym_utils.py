"""Utilities for working with Gymnasium environments."""

import gymnasium as gym
import numpy as np
import prbench


def make_env_ppo(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
    gamma: float = 0.99,
):
    """Create a single environment instance with appropriate wrappers for ppo."""

    def thunk():
        if capture_video and idx == 0:
            if "prbench" in env_id:
                env = prbench.make(env_id, render_mode="rgb_array")
            else:
                env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if "prbench" in env_id:
                env = prbench.make(env_id)
            else:
                env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # NOTE: PRBench by default has infinite horizon, so we set a time limit here
        if "prbench" in env_id:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    return thunk


def make_env_sac(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
):
    """Create a single environment instance with appropriate wrappers for sac."""

    def thunk():
        if capture_video and idx == 0:
            if "prbench" in env_id:
                env = prbench.make(env_id, render_mode="rgb_array")
            else:
                env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if "prbench" in env_id:
                env = prbench.make(env_id)
            else:
                env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # NOTE: PRBench by default has infinite horizon, so we set a time limit here
        if "prbench" in env_id:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    return thunk
