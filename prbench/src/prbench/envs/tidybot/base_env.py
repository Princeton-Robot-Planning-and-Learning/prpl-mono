"""Base class for Dynamic3D robot environments."""

import abc
from typing import Any

import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prpl_utils.spaces import FunctionalSpace
from relational_structs import ObjectCentricState, ObjectCentricStateSpace, Type
from relational_structs.utils import create_state_from_dict

from prbench.core import ObjectCentricPRBenchEnv, _ConfigType
from prbench.envs.tidybot.mujoco_utils import MjAct
from prbench.envs.tidybot.object_types import MujocoObjectTypeFeatures


class ObjectCentricDynamic3DRobotEnv(
    ObjectCentricPRBenchEnv[ObjectCentricState, MjAct, _ConfigType]  # type: ignore
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

    def _create_action_space(self, config: _ConfigType) -> Space[MjAct]:  # type: ignore
        """Create action space for TidyBot's control interface."""
        # TidyBot actions: base_pose (3), arm_pos (3), arm_quat (4), gripper_pos (1)
        low = np.array(
            [-1.0, -1.0, -np.pi, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
        )
        high = np.array([1.0, 1.0, np.pi, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        def _contains_fn(x: Any) -> bool:
            return isinstance(x, MjAct)

        def _sample_fn(rng: np.random.Generator) -> MjAct:
            ctrl = rng.uniform(low, high)
            return MjAct(position_ctrl=ctrl)

        return FunctionalSpace(contains_fn=_contains_fn, sample_fn=_sample_fn)

    @abc.abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObjectCentricState, dict[str, Any]]:
        """Subclasses must implement."""

    @abc.abstractmethod
    def step(
        self, action: MjAct
    ) -> tuple[ObjectCentricState, float, bool, bool, dict[str, Any]]:
        """Subclasses must implement."""

    @abc.abstractmethod
    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Subclasses must implement."""

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return MujocoObjectTypeFeatures
