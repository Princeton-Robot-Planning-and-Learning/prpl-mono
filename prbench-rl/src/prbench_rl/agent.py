"""Base RL agent interface for PRBench environments."""

import abc
from typing import Any, TypeVar, Optional

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
        env: MultiEnvWrapper,
        eval_env: Optional[MultiEnvWrapper] = None,
    ) -> list[dict[str, Any]]:
        """Training the agent with an interactive batched environment.
        Note that evaluation env is seperated because we might want to render
        during evaluation with fewer environments.
        """
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
