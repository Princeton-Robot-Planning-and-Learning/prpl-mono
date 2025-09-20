"""Base classes for all PRBench environments."""

import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium
import numpy as np
from numpy.typing import NDArray
from relational_structs import (
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.utils import RobotActionSpace


@dataclass(frozen=True)
class PRBenchEnvConfig:
    """Scene configuration for a PRBench environment."""

    render_fps: int = 20


# All object-centric PRBench environments have object-centric states.
_ObsType = TypeVar("_ObsType", bound=ObjectCentricState)
# All PRBench environments have array actions.
_ActType = TypeVar("_ActType", bound=NDArray[Any])
# All PRBench environments have an environment config.
_ConfigType = TypeVar("_ConfigType", bound=PRBenchEnvConfig)


class ObjectCentricPRBenchEnv(
    gymnasium.Env[_ObsType, _ActType], Generic[_ObsType, _ActType, _ConfigType]
):
    """Base class for object-centric PRBench environments."""

    # Only RGB rendering is implemented.
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, config: _ConfigType, render_mode: str | None = "rgb_array"
    ) -> None:
        self.config = config
        self.render_mode = render_mode
        self.observation_space = self._create_observation_space(config)
        # Set up metadata for rendering. Subclasses will add to the metadata.
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self.config.render_fps,
        }
        # I'm not completely sure why this type: ignore is necessary. I tried to fix it
        # for a while and gave up.
        self.action_space = self._create_action_space(config)  # type: ignore

        # Maintain an independent initial_constant_state, including static objects
        # that never change throughout the lifetime of the environment.
        # NOTE: we defer the creation of the initial constant state because subclasses
        # may create objects inside their __init__() after this parent method is called.
        self._initial_constant_state: _ObsType | None = None

        super().__init__()

    @abc.abstractmethod
    def _create_constant_initial_state(self) -> _ObsType:
        """Create the constant initial state."""

    @abc.abstractmethod
    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        """Create the observation space given the config."""

    @abc.abstractmethod
    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        """Create the action space given the config."""

    @abc.abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[_ObsType, dict]:
        """Subclasses must implement."""

    @abc.abstractmethod
    def step(self, action: _ActType) -> tuple[_ObsType, float, bool, bool, dict]:
        """Subclasses must implement."""

    @abc.abstractmethod
    def render(self) -> NDArray[np.uint8]:  # type: ignore
        """Subclasses must implement."""

    @property
    @abc.abstractmethod
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""

    @property
    def initial_constant_state(self) -> _ObsType:
        """Get the initial constant state, which includes static objects."""
        if self._initial_constant_state is None:
            self._initial_constant_state = self._create_constant_initial_state()
        return self._initial_constant_state.copy()

    def get_state_with_constant_objects(self, state: _ObsType) -> _ObsType:
        """Get the full state, which includes both dynamic and static objects."""
        # Merge the initial constant state with the current state.
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        return full_state

    def get_action_from_gui_input(self, gui_input: dict[str, Any]) -> _ActType:
        """Optionally implement a GUI interface, e.g., for demo collection."""
        del gui_input
        return np.array([])  # type: ignore


class ConstantObjectPRBenchEnv(gymnasium.Env[NDArray[Any], NDArray[Any]]):
    """Defined by an object-centric PRBench environment and a constant object set.

    The point of this pattern is to allow implementing object-centric environments with
    variable numbers of objects, but then also create versions of the environment with a
    constant number of objects so it is easy to apply, e.g., RL approaches that use
    fixed-dimensional observation and action spaces.
    """

    # NOTE: we need to define render_modes in the class instead of the instance because
    # gym.make extracts render_modes from the class (entry_point) before instantiation.
    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}

    def __init__(self, *args, render_mode: str | None = None, **kwargs) -> None:
        super().__init__()
        self._object_centric_env = self._create_object_centric_env(*args, **kwargs)
        # Create a Box version of the observation space by extracting the constant
        # objects from an exemplar state.
        exemplar_object_centric_state, _ = self._object_centric_env.reset()
        obj_name_to_obj = {o.name: o for o in exemplar_object_centric_state}
        obj_names = self._get_constant_object_names(exemplar_object_centric_state)
        self._constant_objects: list[Object] = [obj_name_to_obj[o] for o in obj_names]
        # This is a Box space with some extra functionality to allow easy vectorizing.
        assert isinstance(
            self._object_centric_env.observation_space, ObjectCentricStateSpace
        )
        self.observation_space = self._object_centric_env.observation_space.to_box(
            self._constant_objects, self._object_centric_env.type_features
        )
        self.action_space = self._object_centric_env.action_space
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        # The action space already inherits from Box, so we don't need to change it.
        assert isinstance(self.action_space, RobotActionSpace)
        # Add descriptions to metadata for doc generation.
        obs_md = self.observation_space.create_markdown_description()
        act_md = self.action_space.create_markdown_description()
        env_md = self._create_env_markdown_description()
        reward_md = self._create_reward_markdown_description()
        references_md = self._create_references_markdown_description()
        # Update the metadata. Note that we need to define the render_modes in the class
        # rather than in the instance because gym.make() extracts render_modes from cls.
        self.metadata = self.metadata.copy()
        self.metadata.update(
            {
                "description": env_md,
                "observation_space_description": obs_md,
                "action_space_description": act_md,
                "reward_description": reward_md,
                "references": references_md,
                "render_fps": self._object_centric_env.metadata.get("render_fps", 20),
            }
        )
        self.render_mode = render_mode

    @abc.abstractmethod
    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricPRBenchEnv:
        """Create the underlying object-centric environment."""

    @abc.abstractmethod
    def _get_constant_object_names(
        self, exemplar_state: ObjectCentricState
    ) -> list[str]:
        """The ordered names of the constant objects extracted from the observations."""

    @abc.abstractmethod
    def _create_env_markdown_description(self) -> str:
        """Create a markdown description of the overall environment."""

    @abc.abstractmethod
    def _create_reward_markdown_description(self) -> str:
        """Create a markdown description of the environment rewards."""

    @abc.abstractmethod
    def _create_references_markdown_description(self) -> str:
        """Create a markdown description of the reference (e.g. papers) for this env."""

    def reset(self, *args, **kwargs) -> tuple[NDArray[np.float32], dict]:
        super().reset(*args, **kwargs)  # necessary to reset RNG if seed is given
        if (kwargs.get("options") is not None) and (
            "init_state" in kwargs.get("options", {})
        ):
            # NOTE: From user perspective, they might just pass in a state
            # that is similar to the observation array for resetting,
            # not an ObjectCentricState.
            if not isinstance(kwargs["options"]["init_state"], ObjectCentricState):
                assert isinstance(kwargs["options"]["init_state"], np.ndarray)
                assert isinstance(self.observation_space, ObjectCentricBoxSpace)
                obj_centric_state = self.observation_space.devectorize(
                    kwargs["options"]["init_state"]
                )
                kwargs["options"]["init_state"] = obj_centric_state
        obs, info = self._object_centric_env.reset(*args, **kwargs)
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, info

    def step(
        self, *args, **kwargs
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict]:
        obs, reward, terminated, truncated, done = self._object_centric_env.step(
            *args, **kwargs
        )
        assert isinstance(self.observation_space, ObjectCentricBoxSpace)
        vec_obs = self.observation_space.vectorize(obs)
        return vec_obs, reward, terminated, truncated, done

    def render(self):
        return self._object_centric_env.render()

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Get the mapping from human inputs to actions."""
        return self._object_centric_env.get_action_from_gui_input(gui_input)
