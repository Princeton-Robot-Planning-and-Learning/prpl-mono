"""Base RL agent interface for PRBench environments."""

import abc
from typing import Any, Dict, TypeVar

from omegaconf import DictConfig
from prpl_utils.gym_agent import Agent
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

# Create temporary environment to get spaces
import gymnasium as gym
import prbench

_O = TypeVar("_O")
_U = TypeVar("_U")


class Logger:
    """Logger for RL training and evaluation.

    Logs to TensorBoard and optionally to Weights & Biases.
    """

    def __init__(self, tensorboard: SummaryWriter) -> None:
        """Initialize the logger with TensorBoard and optional Weights & Biases."""
        self.writer = tensorboard

    def add_scalar(self, tag: str, scalar_value: float | Tensor, step: int = 0) -> None:
        """Log a scalar value to TensorBoard and optionally to Weights & Biases."""
        self.writer.add_scalar(tag, scalar_value, step)  # type: ignore

    def close(self) -> None:
        """Close the logger."""
        self.writer.close()  # type: ignore


class BaseRLAgent(Agent[_O, _U]):
    """Base class for RL agents in PRBench environments."""

    def __init__(
        self,
        seed: int,
        env_id: str,
        max_episode_steps: int,
        cfg: DictConfig,
    ) -> None:
        super().__init__(seed)
        self.cfg = cfg
        self.env_id = env_id
        self.max_episode_steps = max_episode_steps
        self.seed(seed)

        if "prbench" in env_id:
            temp_env = prbench.make(env_id)
        else:
            temp_env = gym.make(env_id)
        # Apply FlattenObservation wrapper like in make_env
        temp_env = gym.wrappers.FlattenObservation(temp_env)
        self.observation_space = temp_env.observation_space
        self.action_space = temp_env.action_space
        temp_env.close()  # type: ignore

    @abc.abstractmethod
    def _get_action(self) -> _U:
        """Produce an action to execute now."""

    def train(self) -> Dict[str, Any]:  # type: ignore
        """Switch to train mode."""
        self._train_or_eval = "train"
        return {}

    def evaluate(self, eval_episodes: int) -> Dict[str, Any]:
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
