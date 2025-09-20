"""Parameterized skills for the Motion2D environment."""

from typing import Optional, Sequence, cast

import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    run_motion_planning_for_crv_robot,
    state_2d_has_collision,
)
from relational_structs import (
    Object,
    ObjectCentricState,
)

from prbench_models.geom2d.utils import Geom2dRobotController


# Controllers.
class GroundMoveToTgtController(Geom2dRobotController):
    """Controller for moving the robot to the target region."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._target = objects[1]
        self._action_space = action_space

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        # Sample a point within the target region and a random orientation
        target_x = x.get(self._target, "x")
        target_y = x.get(self._target, "y")
        target_width = x.get(self._target, "width")
        target_height = x.get(self._target, "height")
        full_state = x.copy()
        if self._init_constant_state is not None:
            full_state.data.update(self._init_constant_state.data)
        while True:
            # Sample relative position within the target region
            rel_x = rng.uniform(0.1, 0.9)
            rel_y = rng.uniform(0.1, 0.9)
            # Sample random orientation
            abs_theta = rng.uniform(-np.pi, np.pi)

            # Convert to absolute coordinates within target bounds
            abs_x = target_x + rel_x * target_width
            abs_y = target_y + rel_y * target_height
            full_state.set(self._robot, "x", abs_x)
            full_state.set(self._robot, "y", abs_y)
            full_state.set(self._robot, "theta", abs_theta)
            # Check collision
            moving_objects = {self._robot}
            static_objects = set(full_state) - moving_objects
            if not state_2d_has_collision(
                full_state, moving_objects, static_objects, {}
            ):
                break
        # Relative orientation
        rel_theta = (abs_theta + np.pi) / (2 * np.pi)

        return (rel_x, rel_y, rel_theta)

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 0.0  # No vacuum needed for motion

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_radius = state.get(self._robot, "base_radius")

        # Calculate target position from parameters
        params = cast(tuple[float, float, float], self._current_params)
        target_x = state.get(self._target, "x")
        target_y = state.get(self._target, "y")
        target_width = state.get(self._target, "width")
        target_height = state.get(self._target, "height")

        final_x = target_x + params[0] * target_width
        final_y = target_y + params[1] * target_height
        # Convert to absolute angle
        final_theta = params[2] * 2 * np.pi - np.pi
        final_pose = SE2Pose(final_x, final_y, final_theta)

        # Use motion planning to find collision-free path
        assert isinstance(self._action_space, CRVRobotActionSpace)
        collision_free_waypoints = run_motion_planning_for_crv_robot(
            state, self._robot, final_pose, self._action_space
        )

        final_waypoints: list[tuple[SE2Pose, float]] = []

        if collision_free_waypoints is not None:
            for wp in collision_free_waypoints:
                final_waypoints.append((wp, robot_radius))
            return final_waypoints
        # If motion planning fails, raise failure
        raise TrajectorySamplingFailure(
            "Failed to find a collision-free path to target."
        )


class GroundMoveToPassageController(GroundMoveToTgtController):
    """Controller for moving the robot to a passage."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._obstacle1 = objects[1]
        self._obstacle2 = objects[2]

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        # Sample a point between the two obstacles
        obstacle1_x = x.get(self._obstacle1, "x")
        obstacle1_width = x.get(self._obstacle1, "width")
        obstacle1_y = x.get(self._obstacle1, "y")
        obstacle2_y = x.get(self._obstacle2, "y")
        obstacle2_height = x.get(self._obstacle2, "height")
        robot_radius = x.get(self._robot, "base_radius")
        full_state = x.copy()
        if self._init_constant_state is not None:
            full_state.data.update(self._init_constant_state.data)
        while True:
            rel_x = rng.uniform(0.1, 0.9)
            rel_y = rng.uniform(0.1, 0.9)

            abs_x = (
                obstacle1_x
                + obstacle1_width / 2
                - robot_radius
                + 2 * robot_radius * rel_x
            )
            abs_y = (
                obstacle2_y
                + obstacle2_height
                + robot_radius
                + (obstacle1_y - (obstacle2_y + obstacle2_height + 2 * robot_radius))
                * rel_y
            )
            abs_theta = rng.uniform(-np.pi, np.pi)

            full_state.set(self._robot, "theta", abs_theta)
            full_state.set(self._robot, "x", abs_x)
            full_state.set(self._robot, "y", abs_y)

            # Check collision
            moving_objects = {self._robot}
            static_objects = set(full_state) - moving_objects
            if not state_2d_has_collision(
                full_state, moving_objects, static_objects, {}
            ):
                break
        rel_theta = (abs_theta + np.pi) / (2 * np.pi)
        return (rel_x, rel_y, rel_theta)

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_radius = state.get(self._robot, "base_radius")
        obstacle1_x = state.get(self._obstacle1, "x")
        obstacle1_width = state.get(self._obstacle1, "width")
        obstacle1_y = state.get(self._obstacle1, "y")
        obstacle2_y = state.get(self._obstacle2, "y")
        obstacle2_height = state.get(self._obstacle2, "height")

        # Calculate target position from parameters
        params = cast(tuple[float, float, float], self._current_params)
        abs_x = (
            obstacle1_x
            + obstacle1_width / 2
            - robot_radius
            + 2 * robot_radius * params[0]
        )
        abs_y = (
            obstacle2_y
            + obstacle2_height
            + robot_radius
            + (obstacle1_y - (obstacle2_y + obstacle2_height + 2 * robot_radius))
            * params[1]
        )
        abs_theta = params[2] * 2 * np.pi - np.pi

        final_pose = SE2Pose(abs_x, abs_y, abs_theta)

        # Use motion planning to find collision-free path
        assert isinstance(self._action_space, CRVRobotActionSpace)
        mp_state = state.copy()
        if self._init_constant_state is not None:
            mp_state.data.update(self._init_constant_state.data)
        collision_free_waypoints = run_motion_planning_for_crv_robot(
            mp_state, self._robot, final_pose, self._action_space
        )

        final_waypoints: list[tuple[SE2Pose, float]] = []

        if collision_free_waypoints is not None:
            for wp in collision_free_waypoints:
                final_waypoints.append((wp, robot_radius))
            return final_waypoints

        # If motion planning fails, raise failure
        raise TrajectorySamplingFailure(
            "Failed to find a collision-free path to target."
        )
