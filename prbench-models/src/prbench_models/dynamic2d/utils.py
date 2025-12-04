"""Utilities for 2D dynamic robot manipulation tasks."""

import abc
from typing import Iterable, Optional, Sequence, Union

import numpy as np
from bilevel_planning.structs import GroundParameterizedController
from numpy.typing import NDArray
from prbench.envs.dynamic2d.dyn_obstruction2d import (
    DynObstruction2DEnvConfig,
)
from prbench.envs.dynamic2d.object_types import KinRobotType
from prbench.envs.dynamic2d.utils import KinRobotActionSpace
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.utils import state_2d_has_collision
from prpl_utils.motion_planning import BiRRT
from prpl_utils.utils import get_signed_angle_distance, wrap_angle
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.objects import Object


class Dynamic2dRobotController(GroundParameterizedController, abc.ABC):
    """General controller for 2D dynamic robot manipulation tasks using SE2
    waypoints."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: KinRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        self._robot = objects[0]
        assert self._robot.is_instance(KinRobotType)
        super().__init__(objects)
        self._current_params: Union[tuple[float, ...], float] = 0.0
        self._current_plan: Union[list[NDArray[np.float32]], None] = None
        self._current_state: Union[ObjectCentricState, None] = None
        self._init_constant_state = init_constant_state
        # Extract max deltas from action space bounds
        self._max_delta_x = action_space.high[0]
        self._max_delta_y = action_space.high[1]
        self._max_delta_theta = action_space.high[2]
        self._max_delta_arm = action_space.high[3]
        self._max_delta_gripper = action_space.high[4]

        env_config = DynObstruction2DEnvConfig()
        self.world_x_min = env_config.world_min_x + env_config.robot_base_radius
        self.world_x_max = env_config.world_max_x - env_config.robot_base_radius
        self.world_y_min = env_config.world_min_y + env_config.robot_base_radius
        self.world_y_max = env_config.world_max_y - env_config.robot_base_radius
        self.finger_gap_max = env_config.gripper_base_height

    @abc.abstractmethod
    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate a waypoint plan with SE2 pose and arm length values."""

    @abc.abstractmethod
    def _get_gripper_actions(self, state: ObjectCentricState) -> tuple[float, float]:
        """Get gripper actions (deltas) for during and after waypoint movement.

        Args:
            state: Current state to calculate gripper delta from.

        Returns:
            Tuple of (delta_during_plan, delta_after_plan) where values are:
            - Positive values mean opening the gripper (increasing finger_gap)
            - Negative values mean closing the gripper (decreasing finger_gap)
            - 0.0 means no change
            These are changes (deltas) in finger_gap, not absolute values.
        """

    def _requires_multi_phase_gripper(self) -> bool:
        """Check if this controller requires multi-phase gripper execution.

        Override this method to force multi-phase execution (e.g., for pick controllers
        that need to move to target first, then close gripper).

        Args:
            state: Current state.

        Returns:
            True if multi-phase execution is required, False otherwise.
        """
        return False

    def _waypoints_to_plan(
        self,
        state: ObjectCentricState,
        waypoints: list[tuple[SE2Pose, float]],
        gripper_during_plan: float,
    ) -> list[NDArray[np.float32]]:
        """Convert waypoints to action plan using BiRRT motion planning."""
        curr_x = state.get(self._robot, "x")
        curr_y = state.get(self._robot, "y")
        curr_theta = state.get(self._robot, "theta")
        curr_arm = state.get(self._robot, "arm_joint")
        current_pos: tuple[SE2Pose, float] = (
            SE2Pose(curr_x, curr_y, curr_theta),
            curr_arm,
        )
        waypoints = [current_pos] + waypoints

        # Create a static state copy for collision checking
        full_state = state.copy()
        if self._init_constant_state is not None:
            full_state.data.update(self._init_constant_state.data)

        # Get robot arm length bounds
        max_arm_length = state.get(self._robot, "arm_length")
        min_arm_length = (
            state.get(self._robot, "base_radius")
            + state.get(self._robot, "gripper_base_height") / 2
            + 1e-4
        )

        rng = np.random.default_rng(0)

        # Define motion planning functions
        def sample_fn(_: tuple[SE2Pose, float]) -> tuple[SE2Pose, float]:
            """Sample a robot configuration (pose + arm length)."""
            x = rng.uniform(self.world_x_min, self.world_x_max)
            y = rng.uniform(self.world_y_min, self.world_y_max)
            theta = rng.uniform(-np.pi, np.pi)
            arm = rng.uniform(min_arm_length, max_arm_length)
            return (SE2Pose(x, y, theta), arm)

        def extend_fn(
            pt1: tuple[SE2Pose, float], pt2: tuple[SE2Pose, float]
        ) -> Iterable[tuple[SE2Pose, float]]:
            """Interpolate between two configurations respecting action space bounds."""
            pose1, arm1 = pt1
            pose2, arm2 = pt2

            dx = pose2.x - pose1.x
            dy = pose2.y - pose1.y
            dtheta = get_signed_angle_distance(pose2.theta, pose1.theta)
            darm = arm2 - arm1

            # Calculate number of steps needed for each dimension
            abs_x = self._max_delta_x if dx > 0 else abs(self._max_delta_x)
            abs_y = self._max_delta_y if dy > 0 else abs(self._max_delta_y)
            abs_theta = (
                self._max_delta_theta if dtheta > 0 else abs(self._max_delta_theta)
            )
            abs_arm = self._max_delta_arm if darm > 0 else abs(self._max_delta_arm)

            x_num_steps = max(1, int(np.ceil(abs(dx) / abs_x)) if abs_x > 0 else 1)
            y_num_steps = max(1, int(np.ceil(abs(dy) / abs_y)) if abs_y > 0 else 1)
            theta_num_steps = max(
                1, int(np.ceil(abs(dtheta) / abs_theta)) if abs_theta > 0 else 1
            )
            arm_num_steps = max(
                1, int(np.ceil(abs(darm) / abs_arm)) if abs_arm > 0 else 1
            )

            num_steps = max(x_num_steps, y_num_steps, theta_num_steps, arm_num_steps)

            path: list[tuple[SE2Pose, float]] = []
            for i in range(num_steps + 1):
                alpha = i / num_steps if num_steps > 0 else 1.0
                x = pose1.x + alpha * dx
                y = pose1.y + alpha * dy
                theta = wrap_angle(pose1.theta + alpha * dtheta)
                arm = arm1 + alpha * darm
                path.append((SE2Pose(x, y, theta), arm))

            return path

        def collision_fn(pt: tuple[SE2Pose, float]) -> bool:
            """Check if a configuration is collision-free."""
            pose, arm = pt

            # Update state with robot configuration
            test_state = full_state.copy()
            test_state.set(self._robot, "x", pose.x)
            test_state.set(self._robot, "y", pose.y)
            test_state.set(self._robot, "theta", pose.theta)
            test_state.set(self._robot, "arm_joint", arm)

            # Check collisions
            moving_objects = {self._robot}
            static_objects = set(test_state) - moving_objects

            return state_2d_has_collision(
                test_state, moving_objects, static_objects, {}
            )

        def distance_fn(
            pt1: tuple[SE2Pose, float], pt2: tuple[SE2Pose, float]
        ) -> float:
            """Calculate distance between two configurations."""
            pose1, arm1 = pt1
            pose2, arm2 = pt2

            dx = pose2.x - pose1.x
            dy = pose2.y - pose1.y
            dtheta = get_signed_angle_distance(pose2.theta, pose1.theta)
            darm = arm2 - arm1

            # Weighted distance: position + orientation + arm
            return np.sqrt(dx**2 + dy**2) + abs(dtheta) + abs(darm)

        # Create BiRRT planner
        birrt = BiRRT(
            sample_fn=sample_fn,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            distance_fn=distance_fn,
            rng=rng,
            num_attempts=10,
            num_iters=100,
            smooth_amt=50,
        )

        # Plan path between waypoints
        plan: list[NDArray[np.float32]] = []
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            # Check if start and end are the same
            if np.allclose(
                [start[0].x, start[0].y, start[0].theta, start[1]],
                [end[0].x, end[0].y, end[0].theta, end[1]],
            ):
                continue

            # Try direct path first
            direct_path = birrt.try_direct_path(start, end)
            if direct_path is not None:
                path = direct_path
            else:
                # Use BiRRT to plan
                birrt_path = birrt.query(start, end)

                if birrt_path is None:
                    # If planning fails, fall back to direct interpolation
                    path = list(extend_fn(start, end))
                else:
                    path = birrt_path

            # Convert path to actions
            for pt1, pt2 in zip(path[:-1], path[1:]):
                pose1, arm1 = pt1
                pose2, arm2 = pt2

                dx = pose2.x - pose1.x
                dy = pose2.y - pose1.y
                dtheta = get_signed_angle_distance(pose2.theta, pose1.theta)
                darm = arm2 - arm1

                action = np.array(
                    [dx, dy, dtheta, darm, gripper_during_plan], dtype=np.float32
                )
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
        gripper_delta_during_plan, gripper_delta_after_plan = self._get_gripper_actions(
            x
        )

        max_gripper_delta = abs(self._max_delta_gripper)

        # Check if we need multi-phase execution
        # Either explicitly requested or if gripper_after_plan requires multiple steps
        requires_multi_phase = (
            self._requires_multi_phase_gripper()
            or abs(gripper_delta_after_plan) > max_gripper_delta
        )

        if requires_multi_phase:
            # Multi-phase: move to waypoint, then adjust gripper
            # Phase 1: Move to final waypoint with gripper_delta_during_plan
            # (typically 0.0 for pick)
            waypoint_plan = self._waypoints_to_plan(
                x, waypoints, gripper_delta_during_plan
            )

            # Phase 2: Adjust gripper gradually
            gripper_plan: list[NDArray[np.float32]] = []
            remaining_delta = gripper_delta_after_plan
            while abs(remaining_delta) > 1e-6:
                step_delta = np.clip(
                    remaining_delta, -max_gripper_delta, max_gripper_delta
                )
                gripper_plan.append(
                    np.array([0, 0, 0, 0, step_delta], dtype=np.float32)
                )
                remaining_delta -= step_delta

            return waypoint_plan + gripper_plan

        # Single phase: move with gripper action without final gripper adjustment
        waypoint_plan = self._waypoints_to_plan(x, waypoints, gripper_delta_during_plan)

        return waypoint_plan
