"""PRBench RL package."""

from gymnasium import spaces
from gymnasium.core import Env
from omegaconf import DictConfig

from prbench_rl.agent import BaseRLAgent
from prbench_rl.ppo_agent import PPOAgent
from prbench_rl.random_agent import RandomAgent

__all__ = ["create_rl_agents"]


def create_rl_agents(agent_cfg: DictConfig, env: Env, seed: int) -> BaseRLAgent:
    """Create agent based on configuration."""
    observation_space = env.observation_space
    action_space = env.action_space

    # Ensure we have Box spaces for continuous control
    if not isinstance(observation_space, spaces.Box) or not isinstance(
        action_space, spaces.Box
    ):
        raise ValueError("PPO agent requires Box observation and action spaces")

    if agent_cfg.name == "random":
        return RandomAgent(observation_space, action_space, seed, agent_cfg)
    if agent_cfg.name == "ppo":
        return PPOAgent(observation_space, action_space, seed, agent_cfg)
    raise ValueError(f"Unknown agent type: {agent_cfg.name}")
