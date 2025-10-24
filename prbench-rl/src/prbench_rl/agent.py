"""Base RL agent interface for PRBench environments."""

import abc
from typing import Any, TypeVar

# Create temporary environment to get spaces
import gymnasium as gym
import prbench
from omegaconf import DictConfig
from prpl_utils.gym_agent import Agent

_O = TypeVar("_O")
_U = TypeVar("_U")


class BaseRLAgent(Agent[_O, _U]):
    """Base class for RL agents in PRBench environments."""

    def __init__(
        self,
        seed: int,
        env_id: str | None = None,
        max_episode_steps: int | None = None,
        cfg: DictConfig | None = None,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ) -> None:
        super().__init__(seed)
        self.cfg = cfg if cfg is not None else DictConfig({})
        self.env_id = env_id if env_id is not None else ""
        self.max_episode_steps = (
            max_episode_steps if max_episode_steps is not None else 0
        )
        self.seed(seed)

        # Support two initialization patterns:
        # 1. With env_id (creates spaces from environment)
        # 2. With observation_space and action_space directly
        if observation_space is not None and action_space is not None:
            self.observation_space = observation_space
            self.action_space = action_space
            # Seed the action space for reproducibility
            self.action_space.seed(seed)
        elif env_id is not None:
            if "prbench" in env_id:
                temp_env = prbench.make(env_id)
            else:
                temp_env = gym.make(env_id)
            # Apply FlattenObservation wrapper like in make_env
            temp_env = gym.wrappers.FlattenObservation(temp_env)
            self.observation_space = temp_env.observation_space
            self.action_space = temp_env.action_space
            # Seed the action space for reproducibility
            self.action_space.seed(seed)
            temp_env.close()  # type: ignore
        else:
            raise ValueError(
                "Must provide either (env_id) or (observation_space, action_space)"
            )

    @abc.abstractmethod
    def _get_action(self) -> _U:
        """Produce an action to execute now."""

    def train(self) -> dict[str, Any]:  # type: ignore
        """Switch to train mode."""
        self._train_or_eval = "train"
        return {}

    def evaluate(self, eval_episodes: int) -> dict[str, Any]:
        """Switch to evaluation mode."""
        del eval_episodes
        self._train_or_eval = "eval"
        return {}

    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        # Base implementation does nothing
        del filepath

    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        # Base implementation does nothing
        del filepath

    def close(self) -> None:
        """Clean up resources."""
        # Base implementation does nothing
