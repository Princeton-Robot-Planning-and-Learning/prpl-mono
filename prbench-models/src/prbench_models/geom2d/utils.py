"""Utilities for 2D geometry robot manipulation tasks."""

import abc
from typing import Optional, Sequence, Union

import numpy as np
from bilevel_planning.structs import GroundParameterizedController
from numpy.typing import NDArray
from prbench.envs.geom2d.object_types import CRVRobotType
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.geom2d.utils import CRVRobotActionSpace
from relational_structs import Object, ObjectCentricState


class Geom2dRobotController(GroundParameterizedController, abc.ABC):
    """General controller for 2D robot manipulation tasks using SE2 waypoints."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
        safe_y: float = 0.8,
    ) -> None:
        self._robot = objects[0]
        assert self._robot.is_instance(CRVRobotType)
        super().__init__(objects)
        self._current_params: Union[tuple[float, ...], float] = 0.0
        self._current_plan: Union[list[NDArray[np.float32]], None] = None
        self._current_state: Union[ObjectCentricState, None] = None
        self._safe_y = safe_y
        self._init_constant_state = init_constant_state
        # Extract max deltas from action space bounds
        self._max_delta_x = action_space.high[0]
        self._max_delta_y = action_space.high[1]
        self._max_delta_theta = action_space.high[2]
        self._max_delta_arm = action_space.high[3]

    @abc.abstractmethod
    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate a waypoint plan with SE2 pose and arm length values."""

    @abc.abstractmethod
    def _get_vacuum_actions(self) -> tuple[float, float]:
        """Get vacuum actions for during and after waypoint movement."""

    def _waypoints_to_plan(
        self,
        state: ObjectCentricState,
        waypoints: list[tuple[SE2Pose, float]],
        vacuum_during_plan: float,
    ) -> list[NDArray[np.float32]]:
        curr_x = state.get(self._robot, "x")
        curr_y = state.get(self._robot, "y")
        curr_theta = state.get(self._robot, "theta")
        curr_arm = state.get(self._robot, "arm_joint")
        current_pos = (SE2Pose(curr_x, curr_y, curr_theta), curr_arm)
        waypoints = [current_pos] + waypoints
        plan: list[NDArray[np.float32]] = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            start_pose = np.array([start[0].x, start[0].y, start[0].theta, start[1]])
            end_pose = np.array([end[0].x, end[0].y, end[0].theta, end[1]])
            if np.allclose(start_pose, end_pose):
                continue
            total_dx = end[0].x - start[0].x
            total_dy = end[0].y - start[0].y
            total_dtheta = end[0].theta - start[0].theta
            total_darm = end[1] - start[1]
            num_steps = int(
                max(
                    np.ceil(abs(total_dx) / self._max_delta_x),
                    np.ceil(abs(total_dy) / self._max_delta_y),
                    np.ceil(abs(total_dtheta) / self._max_delta_theta),
                    np.ceil(abs(total_darm) / self._max_delta_arm),
                )
            )
            dx = total_dx / num_steps
            dy = total_dy / num_steps
            dtheta = total_dtheta / num_steps
            darm = total_darm / num_steps
            action = np.array(
                [dx, dy, dtheta, darm, vacuum_during_plan], dtype=np.float32
            )
            for _ in range(num_steps):
                plan.append(action)

        return plan

    def reset(
        self, x: ObjectCentricState, params: Union[tuple[float, ...], float]
    ) -> None:
        """Reset the controller with new state and parameters."""
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        """Check if the controller has finished executing its plan."""
        return self._current_plan is not None and len(self._current_plan) == 0

    def step(self) -> NDArray[np.float32]:
        """Execute the next action in the controller's plan."""
        assert self._current_state is not None
        if self._current_plan is None:
            self._current_plan = self._generate_plan(self._current_state)
        return self._current_plan.pop(0)

    def observe(self, x: ObjectCentricState) -> None:
        """Update the controller with a new observed state."""
        self._current_state = x

    def _generate_plan(self, x: ObjectCentricState) -> list[NDArray[np.float32]]:
        waypoints = self._generate_waypoints(x)
        vacuum_during_plan, vacuum_after_plan = self._get_vacuum_actions()
        waypoint_plan = self._waypoints_to_plan(x, waypoints, vacuum_during_plan)
        plan_suffix: list[NDArray[np.float32]] = [
            # Change the vacuum.
            np.array([0, 0, 0, 0, vacuum_after_plan], dtype=np.float32),
        ]
        return waypoint_plan + plan_suffix
