"""Base class for Dynamic3D robot environments."""

import abc
from typing import Any

import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray
from relational_structs import Array, ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.utils import create_state_from_dict

from prbench.core import ObjectCentricPRBenchEnv, _ConfigType
from prbench.envs.dynamic3d.object_types import MujocoObjectTypeFeatures


class ObjectCentricDynamic3DRobotEnv(
    ObjectCentricPRBenchEnv[ObjectCentricState, Array, _ConfigType]  # type: ignore
):
    """Base class for Dynamic3D robot environments."""

    def _create_constant_initial_state(self) -> ObjectCentricState:
        """Create the constant initial state (static objects that never change)."""
        # For TidyBot, we don't have static objects that persist across resets
        # All objects are created dynamically in each episode
        return create_state_from_dict({}, MujocoObjectTypeFeatures)

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        """Create observation space based on TidyBot's object types."""
        types = set(self.type_features.keys())
        return ObjectCentricStateSpace(types)

    @abc.abstractmethod
    def _create_action_space(self, config: _ConfigType) -> Space[Array]:  # type: ignore
        """Create action space for robot's control interface."""

    @abc.abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        """Subclasses must implement."""

    @abc.abstractmethod
    def step(
        self, action: Array
    ) -> tuple[ObjectCentricState, float, bool, bool, dict[str, Any]]:
        """Subclasses must implement."""

    @abc.abstractmethod
    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Subclasses must implement."""

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return MujocoObjectTypeFeatures
