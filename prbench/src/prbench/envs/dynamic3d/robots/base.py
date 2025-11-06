"""Base robot class for dynamic3d environments."""

import abc
from typing import Any

from relational_structs import Array

from prbench.envs.dynamic3d.mujoco_utils import MjObs, MujocoEnv


class Robot(MujocoEnv, abc.ABC):
    """Abstract base class for robots in dynamic3d environments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the robot environment.

        Args:
            *args: Positional arguments passed to MujocoEnv.
            **kwargs: Keyword arguments passed to MujocoEnv.
        """
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[MjObs, dict[str, Any]]:
        """Reset the robot environment.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options for resetting the environment.

        Returns:
            A tuple containing the observation and info dict.
        """
        return super().reset(seed=seed, options=options)

    @abc.abstractmethod
    def step(self, action: Array) -> tuple[MjObs, float, bool, bool, dict[str, Any]]:
        """Step the robot environment with the given action.

        Args:
            action: The action to take in the environment.

        Returns:
            A tuple containing (observation, reward, terminated, truncated, info).
        """
        super().step(action)

    @abc.abstractmethod
    def reward(self, obs: MjObs) -> float:
        """Compute the reward from an observation.

        Args:
            obs: The observation to compute reward from.

        Returns:
            The computed reward value.
        """
