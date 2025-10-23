"""Base RL agent interface for PRBench environments."""

import abc
from typing import Any, TypeVar, Dict

from torch import Tensor
from gymnasium import spaces
from omegaconf import DictConfig
from prpl_utils.gym_agent import Agent
from torch.utils.tensorboard import SummaryWriter


from prbench_rl.gym_utils import MultiEnvWrapper

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

    @abc.abstractmethod
    def _get_action(self) -> _U:
        """Produce an action to execute now."""

    @abc.abstractmethod
    def train(self) -> Dict[str, Any]:
        """Switch to train mode."""
        self._train_or_eval = "train"
        return {}

    @abc.abstractmethod
    def evaluate(self, eval_episodes: int) -> Dict[str, Any]:
        """Switch to evaluation mode."""
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
