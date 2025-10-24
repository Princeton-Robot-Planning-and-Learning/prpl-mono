"""Utilities for working with Gymnasium environments."""

import gymnasium as gym
import numpy as np
import prbench
import torch


def make_env(
    env_id: str,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
    gamma: float = 0.99,
):
    """Create a single environment instance with appropriate wrappers."""

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


def evaluate_ppo(
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
    max_episode_steps: int = 300,
) -> list[float]:
    """Evaluate a PPO agent.

    Args:
        env_id: Environment ID to evaluate on
        eval_episodes: Number of episodes to evaluate
        run_name: Name for the run (used for video naming)
        Model: The PyTorch model class
        device: Device to run on
        capture_video: Whether to capture video
        max_episode_steps: Maximum steps per episode

    Returns:
        List of episodic returns
    """
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, 0, capture_video, run_name, max_episode_steps)]
    )
    agent = Model(envs).to(device)
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns: list[float] = []
    while len(episodic_returns) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                episode_return = info["episode"]["r"]
                print(
                    f"eval_episode={len(episodic_returns)}, "
                    f"episodic_return={episode_return}"
                )
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns
