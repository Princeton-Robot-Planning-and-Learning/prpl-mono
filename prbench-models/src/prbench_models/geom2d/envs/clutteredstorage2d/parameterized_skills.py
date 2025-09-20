"""Parameterized skills for the ClutteredStorage2D environment."""

from typing import Optional, Sequence, cast

import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from prbench.envs.geom2d.clutteredstorage2d import ClutteredStorage2DEnvConfig
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    get_tool_tip_position,
    run_motion_planning_for_crv_robot,
    snap_suctioned_objects,
    state_2d_has_collision,
)
from relational_structs import (
    Object,
    ObjectCentricState,
)

from prbench_models.geom2d.utils import Geom2dRobotController


# Controllers.
class GroundPickBlockNotOnShelfController(Geom2dRobotController):
    """Controller for picking a block that is initially not on the shelf."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._shelf = objects[2]
        self._action_space = action_space

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float]:
        # Sample grasp ratio on the height of the block
        # <0.0: custom frame dx/dy < 0
        # >0.0: custom frame dx/dy > 0
        while True:
            grasp_ratio = rng.uniform(-1.0, 1.0)
            if grasp_ratio != 0.0:
                break
        max_arm_length = x.get(self._robot, "arm_length")
        min_arm_length = (
            x.get(self._robot, "base_radius")
            + x.get(self._robot, "gripper_width") / 2
            + 1e-4
        )
        arm_length = rng.uniform(min_arm_length, max_arm_length)
        return (grasp_ratio, arm_length)

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 1.0

    def _calculate_grasp_robot_pose(self, state: ObjectCentricState) -> SE2Pose:
        """Calculate the actual grasp point based on ratio parameter."""
        if isinstance(self._current_params, tuple):
            grasp_ratio, arm_length = self._current_params
        else:
            raise ValueError("Expected tuple parameters for grasp ratio and arm length")

        # Get block properties and grasp frame
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_theta = state.get(self._block, "theta")
        rel_point_dx = state.get(self._block, "width") / 2
        rel_point = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
            rel_point_dx, 0.0, 0.0
        )

        # Relative SE2 pose w.r.t the grasp frame
        custom_dx = (
            state.get(self._block, "width") / 2
            + arm_length
            + state.get(self._robot, "gripper_width")
        )
        custom_dx *= -1 if grasp_ratio < 0 else 1  # Right or left side grasp
        # Custom dy is always positive.
        custom_dy = abs(grasp_ratio) * state.get(self._block, "height")
        custom_dtheta = 0.0 if grasp_ratio < 0 else np.pi
        custom_pose = SE2Pose(custom_dx, custom_dy, custom_dtheta)

        target_se2_pose = rel_point * custom_pose
        return target_se2_pose

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")

        # Calculate grasp point and robot target position
        target_se2_pose = self._calculate_grasp_robot_pose(state)
        if isinstance(self._current_params, tuple):
            _, desired_arm_length = self._current_params
        else:
            raise ValueError("Expected tuple parameters for grasp ratio and arm length")

        # Plan collision-free waypoints to the target pose
        # We set the arm to be the shortest length during motion planning
        mp_state = state.copy()
        mp_state.set(self._robot, "arm_joint", robot_radius)
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            mp_state.data.update(init_constant_state.data)
        assert isinstance(self._action_space, CRVRobotActionSpace)
        collision_free_waypoints = run_motion_planning_for_crv_robot(
            mp_state, self._robot, target_se2_pose, self._action_space
        )
        # Always first make arm shortest to avoid collisions
        final_waypoints: list[tuple[SE2Pose, float]] = [
            (SE2Pose(robot_x, robot_y, robot_theta), robot_radius)
        ]

        if collision_free_waypoints is not None:
            for wp in collision_free_waypoints:
                final_waypoints.append((wp, robot_radius))
            final_waypoints.append((target_se2_pose, desired_arm_length))
            return final_waypoints
        # If motion planning fails, raise failure
        raise TrajectorySamplingFailure(
            "Failed to find a collision-free path to target."
        )


class GroundPlaceBlockOnShelfController(Geom2dRobotController):
    """Controller for placing a block onto the shelf."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._shelf = objects[2]
        self._action_space = action_space

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float]:
        del x  # Unused
        # Sample place ratio
        # w.r.t (shelf_width - block_width)
        # and (shelf_height - block_height)
        relative_dx = rng.uniform(0.01, 0.99)
        # Bias towards inside the shelf
        relative_dy = rng.uniform(0.1, 0.95)
        return (relative_dx, relative_dy)

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 1.0, 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")
        robot_arm_length = state.get(self._robot, "arm_length")
        gripper_height = state.get(self._robot, "gripper_height")
        gripper_width = state.get(self._robot, "gripper_width")
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_theta = state.get(self._block, "theta")
        block_width = state.get(self._block, "width")
        block_height = state.get(self._block, "height")
        block_curr_center = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
            block_width / 2, block_height / 2, 0.0
        )
        _, gripper_to_block = get_suctioned_objects(state, self._robot)[0]

        # Calculate pre-place position based on Shelf position
        shelf_x = state.get(self._shelf, "x1")
        shelf_width = state.get(self._shelf, "width1")
        shelf_y = state.get(self._shelf, "y1")
        shelf_height = state.get(self._shelf, "height1")
        assert isinstance(
            self._current_params, tuple
        ), "PlaceBlock expects tuple params"
        # x can be anywhere in the shelf width (no collision)
        x_min = shelf_x + gripper_height / 2
        x_max = shelf_x + shelf_width - gripper_height / 2
        x_min = min(x_min, x_max)
        x_max = max(x_min, x_max)
        block_desired_x_center = x_min + (x_max - x_min) * self._current_params[0]
        # y is confined to inside the shelf height (no collision)
        y_min = min(shelf_y + block_width / 2, shelf_y + shelf_height - block_width / 2)
        y_max = max(shelf_y + block_width / 2, shelf_y + shelf_height - block_width / 2)
        block_desired_y_center = y_min + (y_max - y_min) * self._current_params[1]
        # Note: The desired orientation depends on how is the blocked grasped.
        # If grasping from the left side, the block should be placed with
        # theta = np.pi / 2
        # If grasping from the right side, the block should be placed with
        # theta = -np.pi / 2
        gripper_x, gripper_y = get_tool_tip_position(state, self._robot)
        gripper_frame = SE2Pose(gripper_x, gripper_y, block_theta)
        relative_frame = block_curr_center.inverse * gripper_frame
        if relative_frame.x < 0:
            # Left side grasp
            block_desired_center = SE2Pose(
                block_desired_x_center, block_desired_y_center, np.pi / 2
            )
        else:
            # Right side grasp
            block_desired_center = SE2Pose(
                block_desired_x_center, block_desired_y_center, -np.pi / 2
            )
        gripper_final_desired_pose = (
            block_desired_center
            * SE2Pose(-block_width / 2, -block_height / 2, 0.0)
            * gripper_to_block.inverse
        )

        final_robot_y = gripper_final_desired_pose.y - robot_arm_length - gripper_width

        pre_place_robot_x = gripper_final_desired_pose.x
        pre_place_robot_y = final_robot_y - shelf_height
        pre_place_pose_0 = SE2Pose(pre_place_robot_x, pre_place_robot_y, np.pi / 2)

        current_wp = (
            SE2Pose(robot_x, robot_y, robot_theta),
            state.get(self._robot, "arm_joint"),
        )
        # Plan collision-free waypoints to the target pose
        # We set the arm to be the longest during motion planning
        final_waypoints: list[tuple[SE2Pose, float]] = []
        mp_state = state.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            mp_state.data.update(init_constant_state.data)
        assert isinstance(self._action_space, CRVRobotActionSpace)
        collision_free_waypoints_0 = run_motion_planning_for_crv_robot(
            mp_state, self._robot, pre_place_pose_0, self._action_space
        )
        if collision_free_waypoints_0 is None:
            # Stay static
            return [current_wp]
        for wp in collision_free_waypoints_0:
            final_waypoints.append((wp, robot_radius))

        # Stretch the arm to the desired position
        if collision_free_waypoints_0:
            last_wp = collision_free_waypoints_0[-1]
            final_waypoints.append((last_wp, robot_arm_length))

        mp_state.set(self._robot, "x", pre_place_robot_x)
        mp_state.set(self._robot, "y", pre_place_robot_y)
        mp_state.set(self._robot, "theta", np.pi / 2)
        mp_state.set(self._robot, "arm_joint", robot_arm_length)
        if init_constant_state is not None:
            mp_state.data.update(init_constant_state.data)
        pre_place_pose_1 = SE2Pose(pre_place_robot_x, final_robot_y, np.pi / 2)
        collision_free_waypoints_1 = run_motion_planning_for_crv_robot(
            mp_state, self._robot, pre_place_pose_1, self._action_space
        )
        if collision_free_waypoints_1 is None:
            # Stay static
            return [current_wp]

        for wp in collision_free_waypoints_1:
            final_waypoints.append((wp, robot_arm_length))

        return final_waypoints


class GroundPickBlockOnShelfController(GroundPickBlockNotOnShelfController):
    """Controller for grasping the block that is not on the shelf yet.

    The grasping point is either on the up or bottom side of the block.
    """

    def _calculate_grasp_robot_pose(self, state: ObjectCentricState) -> SE2Pose:
        """Calculate the actual grasp point based on ratio parameter."""
        if isinstance(self._current_params, tuple):
            grasp_ratio, arm_length = self._current_params
        else:
            raise ValueError("Expected tuple parameters for grasp ratio and arm length")

        # Get block properties and grasp frame
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_theta = state.get(self._block, "theta")
        rel_point_dy = state.get(self._block, "height") / 2
        rel_point = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
            0.0, rel_point_dy, 0.0
        )

        # Relative SE2 pose w.r.t the grasp frame
        custom_dy = (
            state.get(self._block, "height") / 2
            + arm_length
            + state.get(self._robot, "gripper_width")
        )
        custom_dy *= -1 if grasp_ratio < 0 else 1  # top or bottom side grasp
        # Custom dx is always positive.
        custom_dx = abs(grasp_ratio) * state.get(self._block, "width")
        custom_dtheta = np.pi / 2 if grasp_ratio < 0 else -np.pi / 2
        custom_pose = SE2Pose(custom_dx, custom_dy, custom_dtheta)

        target_se2_pose = rel_point * custom_pose
        return target_se2_pose


class GroundPlaceBlockNotOnShelfController(Geom2dRobotController):
    """Controller for placing the block not on the shelf."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._shelf = objects[2]
        self._action_space = action_space
        env_config = ClutteredStorage2DEnvConfig()
        self.world_x_min = env_config.world_min_x + env_config.robot_base_radius
        self.world_x_max = env_config.world_max_x - env_config.robot_base_radius
        self.world_y_min = env_config.world_min_y + env_config.robot_base_radius
        self.world_y_max = (
            env_config.world_max_y
            - env_config.shelf_height
            - env_config.robot_base_radius
        )

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 1.0, 0.0

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        # Sample place ratio
        # w.r.t (shelf_width - block_width)
        # and (shelf_height - block_height)
        full_state = x.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            full_state.data.update(init_constant_state.data)
        while True:
            abs_x = rng.uniform(self.world_x_min, self.world_x_max)
            abs_y = rng.uniform(self.world_y_min, self.world_y_max)
            abs_theta = rng.uniform(-np.pi, np.pi)
            full_state.set(self._robot, "x", abs_x)
            full_state.set(self._robot, "y", abs_y)
            full_state.set(self._robot, "theta", abs_theta)
            suctioned_objects = get_suctioned_objects(x, self._robot)
            snap_suctioned_objects(full_state, self._robot, suctioned_objects)
            # Check collision
            moving_objects = {self._robot} | {o for o, _ in suctioned_objects}
            static_objects = set(full_state) - moving_objects
            if not state_2d_has_collision(
                full_state, moving_objects, static_objects, {}
            ):
                break
        rel_x = (abs_x - self.world_x_min) / (self.world_x_max - self.world_x_min)
        rel_y = (abs_y - self.world_y_min) / (self.world_y_max - self.world_y_min)
        rel_theta = (abs_theta + np.pi) / (2 * np.pi)

        return (rel_x, rel_y, rel_theta)

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")
        # Calculate place position
        params = cast(tuple[float, ...], self._current_params)
        final_robot_x = (
            self.world_x_min + (self.world_x_max - self.world_x_min) * params[0]
        )
        final_robot_y = (
            self.world_y_min + (self.world_y_max - self.world_y_min) * params[1]
        )
        final_robot_theta = -np.pi + (2 * np.pi) * params[2]
        final_robot_pose = SE2Pose(final_robot_x, final_robot_y, final_robot_theta)

        current_wp = (
            SE2Pose(robot_x, robot_y, robot_theta),
            robot_radius,
        )
        # Plan collision-free waypoints to the target pose
        # We set the arm to be the longest during motion planning
        final_waypoints: list[tuple[SE2Pose, float]] = [current_wp]
        mp_state = state.copy()
        mp_state.set(self._robot, "arm_joint", robot_radius)
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            mp_state.data.update(init_constant_state.data)
        assert isinstance(self._action_space, CRVRobotActionSpace)
        collision_free_waypoints_0 = run_motion_planning_for_crv_robot(
            mp_state, self._robot, final_robot_pose, self._action_space
        )
        if collision_free_waypoints_0 is None:
            # Stay static
            return final_waypoints
        for wp in collision_free_waypoints_0:
            final_waypoints.append((wp, robot_radius))

        return final_waypoints
