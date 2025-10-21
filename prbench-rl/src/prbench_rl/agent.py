"""Base RL agent interface for PRBench environments."""

import abc
from typing import Any, TypeVar

from gymnasium import spaces
from gymnasium.core import Env
from omegaconf import DictConfig
from prpl_utils.gym_agent import Agent

_O = TypeVar("_O")
_U = TypeVar("_U")


class BaseRLAgent(Agent[_O, _U]):
    """Base class for RL agents in PRBench environments."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        seed: int,
        cfg: DictConfig,
    ) -> None:
        super().__init__(seed)
        self.observation_space = observation_space
        self.action_space = action_space
        self.cfg = cfg
        self.action_space.seed(seed)

    @abc.abstractmethod
    def _get_action(self) -> _U:
        """Produce an action to execute now."""

    def train(self) -> None:
        """Switch to train mode."""
        self._train_or_eval = "train"

    @abc.abstractmethod
    def train_with_env(
        self,
        env: Env,
    ) -> list[dict[str, Any]]:
        """Training the agent with an interactive environment."""
        del env  # Unused
        self.train()
        return []

    def save(self, filepath: str) -> None:
        """Save agent parameters."""
        # Base implementation does nothing
        del filepath

    def load(self, filepath: str) -> None:
        """Load agent parameters."""
        # Base implementation does nothing
        del filepath
