"""Base class for Geom2D robot environments."""

import abc
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import gymnasium
import numpy as np
from numpy.typing import NDArray
from prpl_utils.utils import wrap_angle
from relational_structs import (
    Array,
    Object,
    ObjectCentricState,
    ObjectCentricStateSpace,
    Type,
)
from relational_structs.utils import create_state_from_dict

from prbench.core import ObjectCentricPRBenchEnv, PRBenchEnvConfig, RobotActionSpace
from prbench.envs.geom2d.object_types import (
    CRVRobotType,
    Geom2DRobotEnvTypeFeatures,
)
from prbench.envs.geom2d.structs import MultiBody2D, SE2Pose
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_geom2d_crv_robot_action_from_gui_input,
    get_suctioned_objects,
    snap_suctioned_objects,
)
from prbench.envs.utils import render_2dstate, state_2d_has_collision


@dataclass(frozen=True)
class Geom2DRobotEnvConfig(PRBenchEnvConfig):
    """Scene configuration for a Geom2DRobotEnv."""

    # The world is oriented like a standard X/Y coordinate frame.
    world_min_x: float = 0.0
    world_max_x: float = 10.0
    world_min_y: float = 0.0
    world_max_y: float = 10.0

    # Action space parameters.
    min_dx: float = -5e-1
    max_dx: float = 5e-1
    min_dy: float = -5e-1
    max_dy: float = 5e-1
    min_dtheta: float = -np.pi / 16
    max_dtheta: float = np.pi / 16
    min_darm: float = -1e-1
    max_darm: float = 1e-1
    min_vac: float = 0.0
    max_vac: float = 1.0

    # For rendering.
    render_dpi: int = 50


_ConfigType = TypeVar("_ConfigType", bound=Geom2DRobotEnvConfig)


class ObjectCentricGeom2DRobotEnv(
    ObjectCentricPRBenchEnv[ObjectCentricState, Array, _ConfigType],
    Generic[_ConfigType],
):
    """Base class for Geom2D robot environments.

    NOTE: this implementation currently assumes we are using CRVRobotType.
    If we add other robot types in the future, we will need to refactor a bit.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Initialized by reset().
        self._current_state: ObjectCentricState | None = None

        # Used for collision checking.
        self._static_object_body_cache: dict[Object, MultiBody2D] = {}

    @abc.abstractmethod
    def _create_constant_initial_state_dict(self) -> dict[Object, dict[str, float]]:
        """Create the constant initial state dict."""

    @abc.abstractmethod
    def _sample_initial_state(self) -> ObjectCentricState:
        """Use self.np_random to sample an initial state."""

    @abc.abstractmethod
    def _get_reward_and_done(self) -> tuple[float, bool]:
        """Calculate reward and termination based on self._current_state."""

    def _create_observation_space(self, config: _ConfigType) -> ObjectCentricStateSpace:
        types = set(self.type_features)
        return ObjectCentricStateSpace(types)

    def _create_action_space(self, config: _ConfigType) -> RobotActionSpace:
        return CRVRobotActionSpace(
            min_dx=config.min_dx,
            max_dx=config.max_dx,
            min_dy=config.min_dy,
            max_dy=config.max_dy,
            min_dtheta=config.min_dtheta,
            max_dtheta=config.max_dtheta,
            min_darm=config.min_darm,
            max_darm=config.max_darm,
            min_vac=config.min_vac,
            max_vac=config.max_vac,
        )

    def _create_constant_initial_state(self) -> ObjectCentricState:
        initial_state_dict = self._create_constant_initial_state_dict()
        return create_state_from_dict(initial_state_dict, Geom2DRobotEnvTypeFeatures)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObjectCentricState, dict]:
        # Reset random seeding.
        gymnasium.Env.reset(self, seed=seed)

        # Need to flush the cache in case static objects move.
        self._static_object_body_cache = {}

        # For testing purposes only, the options may specify an initial scene.
        if options is not None and "init_state" in options:
            self._current_state = options["init_state"].copy()

        # Otherwise, set up the initial scene here.
        else:
            self._current_state = self._sample_initial_state()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: Array) -> tuple[ObjectCentricState, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        dx, dy, dtheta, darm, vac = action
        assert self._current_state is not None, "Need to call reset()"
        state = self._current_state.copy()
        robots = [o for o in state if o.is_instance(CRVRobotType)]
        assert len(robots) == 1, "Multi-robot not yet supported"
        robot = robots[0]

        # NOTE: xy clipping is not needed because world boundaries are assumed
        # handled by collision detection with walls.
        new_x = state.get(robot, "x") + dx
        new_y = state.get(robot, "y") + dy
        new_theta = wrap_angle(state.get(robot, "theta") + dtheta)
        min_arm = state.get(robot, "base_radius")
        max_arm = state.get(robot, "arm_length")
        new_arm = np.clip(state.get(robot, "arm_joint") + darm, min_arm, max_arm)
        state.set(robot, "x", new_x)
        state.set(robot, "y", new_y)
        state.set(robot, "arm_joint", new_arm)
        state.set(robot, "theta", new_theta)
        state.set(robot, "vacuum", vac)

        # The order here is subtle and important:
        # 1) Look at which objects were suctioned in the *previous* time step.
        # 2) Get the transform between gripper and object in the *previous*.
        # 3) Update the position of the object to snap to the robot *now*.
        # 4) When checking collisions, make sure to include all objects that
        #    may have moved. This cannot be derived from `state` alone!
        # The last point was previously overlook and led to bugs where the held
        # objects could come into collision with other objects if the suction is
        # disabled at the right time.

        # Update the state of any objects that are currently suctioned.
        # NOTE: this is both objects and their SE2 transforms.
        suctioned_objs = get_suctioned_objects(self._current_state, robot)
        snap_suctioned_objects(state, robot, suctioned_objs)

        # Update non-static objects if contact is detected between them
        # and the suctioned objects.
        state, moved_objects = self.get_objects_to_move(state, suctioned_objs)

        # Check for collisions, and only update the state if none exist.
        moving_objects = (
            {robot} | {o for o, _ in suctioned_objs} | {o for o, _ in moved_objects}
        )
        full_state = state.copy()
        full_state.data.update(self.initial_constant_state.data)
        obstacles = set(full_state) - moving_objects
        if not state_2d_has_collision(
            full_state, moving_objects, obstacles, self._static_object_body_cache
        ):
            self._current_state = state

        reward, terminated = self._get_reward_and_done()
        truncated = False  # no maximum horizon, by default
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    @property
    def type_features(self) -> dict[Type, list[str]]:
        """The types and features for this environment."""
        return Geom2DRobotEnvTypeFeatures

    def _get_obs(self) -> ObjectCentricState:
        assert self._current_state is not None, "Need to call reset()"
        # NOTE: Based on the discussion, we commit to providing
        # only the changeable objects in the state.
        # A learning-based algorithm has no access to the
        # initial constant state, as the algorithm should learn
        # to handle them if they affect decision making.

        # That being said, we still want to provide an interface
        # for accessing the static objects, as some baselines
        # (planner model) requires such information.
        return self._current_state.copy()

    def _get_info(self) -> dict:
        return {}  # no extra info provided right now

    @property
    def full_state(self) -> ObjectCentricState:
        """Get the full state, which includes both dynamic and static objects."""
        assert self._current_state is not None
        return self.get_state_with_constant_objects(self._current_state)

    def get_objects_to_move(
        self,
        state: ObjectCentricState,
        suctioned_objs: list[tuple[Object, SE2Pose]],
    ) -> tuple[ObjectCentricState, set[tuple[Object, SE2Pose]]]:
        """Get the set of objects that should be moved based on the current state and
        robot actions.

        Implement this in the derived class.
        """
        del suctioned_objs  # not used, but subclasses may use
        # Explicitly type the set to ensure it's set[Object]
        moved_objects: set[tuple[Object, SE2Pose]] = set()
        return state, moved_objects

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        assert self.render_mode == "rgb_array"
        assert self._current_state is not None, "Need to call reset()"
        render_input_state = self._current_state.copy()
        render_input_state.data.update(self.initial_constant_state.data)
        return render_2dstate(
            render_input_state,
            self._static_object_body_cache,
            self.config.world_min_x,
            self.config.world_max_x,
            self.config.world_min_y,
            self.config.world_max_y,
            self.config.render_dpi,
        )

    def get_action_from_gui_input(
        self, gui_input: dict[str, Any]
    ) -> NDArray[np.float32]:
        """Get the mapping from human inputs to actions."""
        assert isinstance(self.action_space, CRVRobotActionSpace)
        return get_geom2d_crv_robot_action_from_gui_input(self.action_space, gui_input)
