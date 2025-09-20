"""Base class for approaches."""

import abc
from typing import Generic, TypeVar

from gymnasium.spaces import Space
from prpl_utils.gym_agent import Agent

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


class BaseApproach(Agent[_ObsType, _ActType], Generic[_ObsType, _ActType], abc.ABC):
    """Base class for approaches."""

    def __init__(
        self,
        environment_description: str,
        observation_space: Space[_ObsType],
        action_space: Space[_ActType],
        seed: int,
    ) -> None:
        super().__init__(seed)
        self._environment_description = environment_description
        self._action_space = action_space
        self._action_space.seed(seed)
        self._observation_space = observation_space
        self._observation_space.seed(seed)
