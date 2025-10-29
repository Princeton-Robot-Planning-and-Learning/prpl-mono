"""PRBench RL package."""

from omegaconf import DictConfig

from prbench_rl.agent import BaseRLAgent
from prbench_rl.ppo_agent import PPOAgent
from prbench_rl.random_agent import RandomAgent
from prbench_rl.sac_agent import SACAgent

__all__ = ["create_rl_agents"]


def create_rl_agents(
    agent_cfg: DictConfig, env_id: str, max_episode_steps: int, seed: int
) -> BaseRLAgent:
    """Create agent based on configuration."""

    if agent_cfg.name == "random":
        return RandomAgent(seed, env_id, max_episode_steps, agent_cfg)
    if agent_cfg.name == "ppo":
        return PPOAgent(seed, env_id, max_episode_steps, agent_cfg)
    if agent_cfg.name == "sac":
        return SACAgent(seed, env_id, max_episode_steps, agent_cfg)
    raise ValueError(f"Unknown agent type: {agent_cfg.name}")
