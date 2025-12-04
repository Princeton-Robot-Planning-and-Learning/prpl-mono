"""Parameterized skills for the DynObstruction2D environment."""

from typing import Optional, Sequence, cast

import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from prbench.envs.dynamic2d.dyn_obstruction2d import (
    DynObstruction2DEnvConfig,
    TargetBlockType,
    TargetSurfaceType,
)
from prbench.envs.dynamic2d.object_types import DynRectangleType, KinRobotType
from prbench.envs.dynamic2d.utils import KinRobotActionSpace
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.utils import state_2d_has_collision
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.objects import Object, Variable

from prbench_models.dynamic2d.utils import Dynamic2dRobotController


# Controllers.
class GroundPickController(Dynamic2dRobotController):
    """Controller for picking the target block or obstruction."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: KinRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._action_space = action_space

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        # Sample grasp ratio and side
        # grasp_ratio: determines position along the side ([0.0, 1.0])
        # we assume we will always pick from the top side
        grasp_ratio = rng.uniform(0.0, 0.1)
        side = rng.uniform(0.5, 0.75)
        max_arm_length = x.get(self._robot, "arm_length")
        min_arm_length = (
            x.get(self._robot, "base_radius")
            + x.get(self._robot, "gripper_base_height") / 2
            + 1e-4
        )
        arm_length = rng.uniform(min_arm_length, max_arm_length)

        # Pack parameters: side determines grasp approach, ratio determines position
        return (grasp_ratio, side, arm_length)

    def _requires_multi_phase_gripper(self) -> bool:
        """Pick controller always uses two phases: move to block, then close gripper."""
        return True

    def _get_gripper_actions(self, state: ObjectCentricState) -> tuple[float, float]:
        """Get gripper actions for pick: keep open during movement,
        close to block width after reaching.

        Returns:
            (delta_during, delta_after) where:
            - delta_during: 0.0 (keep current gap during movement)
            - delta_after: change needed to close gripper to block_width
        """
        curr_finger_gap = state.get(self._robot, "finger_gap")
        finger_width = state.get(self._robot, "finger_width")
        block_width = state.get(self._block, "width")

        # Desired finger gap is slightly smaller than block width for grasping
        desired_finger_gap = max(0.0, block_width + finger_width - 0.175)

        # Calculate delta needed (negative means closing)
        delta_needed = desired_finger_gap - curr_finger_gap

        # During movement: keep gripper open (no change)
        # After reaching: close gripper to desired gap
        return 0.0, delta_needed

    def _calculate_grasp_robot_pose(
        self,
        state: ObjectCentricState,
        ratio: float,
        side: float,
        arm_length: float,
    ) -> SE2Pose:
        """Calculate the grasp point based on side and ratio parameters."""
        # Get block properties
        block_x = state.get(self._block, "x")
        block_y = state.get(self._block, "y")
        block_theta = state.get(self._block, "theta")
        block_width = state.get(self._block, "width")
        block_height = state.get(self._block, "height")

        # Calculate reference point and approach direction based on side
        gripper_height = state.get(self._robot, "gripper_base_height")
        if side < 0.25:  # left side
            custom_dx = -(arm_length + gripper_height)
            custom_dy = ratio * block_height
            custom_dtheta = 0.0
        elif 0.25 <= side < 0.5:  # right side
            custom_dx = arm_length + gripper_height + block_width
            custom_dy = ratio * block_height
            custom_dtheta = np.pi
        elif 0.5 <= side < 0.75:  # top side
            custom_dx = ratio * block_width
            # import ipdb

            # ipdb.set_trace()
            custom_dy = arm_length + gripper_height - 0.05
            custom_dtheta = -np.pi / 2
        else:  # bottom side
            custom_dx = ratio * block_width
            custom_dy = -(arm_length + gripper_height)
            custom_dtheta = np.pi / 2

        target_se2_pose = SE2Pose(block_x, block_y, block_theta) * SE2Pose(
            custom_dx, custom_dy, custom_dtheta
        )
        return target_se2_pose

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Generate waypoints to the grasp point."""
        params = cast(tuple[float, ...], self._current_params)
        grasp_ratio = params[0]
        side = params[1]
        desired_arm_length = params[2]
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")
        # Calculate grasp point and robot target position
        target_se2_pose = self._calculate_grasp_robot_pose(
            state, grasp_ratio, side, desired_arm_length
        )

        full_state = state.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            full_state.data.update(init_constant_state.data)

        # Check if the target pose is collision-free
        full_state.set(self._robot, "x", target_se2_pose.x)
        full_state.set(self._robot, "y", target_se2_pose.y)
        full_state.set(self._robot, "theta", target_se2_pose.theta)
        full_state.set(self._robot, "arm_joint", desired_arm_length)

        # Check target state collision
        moving_objects = {self._robot}
        static_objects = set(full_state) - moving_objects

        if state_2d_has_collision(full_state, moving_objects, static_objects, {}):
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free path to target."
            )

        # Simple waypoint generation: go directly to target
        # In a full implementation, we could use motion planning here
        final_waypoints: list[tuple[SE2Pose, float]] = [
            (SE2Pose(robot_x, robot_y, robot_theta), robot_radius)
        ]
        final_waypoints.append((target_se2_pose, desired_arm_length))
        return final_waypoints


class GroundPlaceController(Dynamic2dRobotController):
    """Controller for placing rectangular objects (target blocks or obstructions) in a
    collision-free location."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: KinRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._action_space = action_space
        env_config = DynObstruction2DEnvConfig()
        self.world_x_min = env_config.world_min_x + env_config.robot_base_radius
        self.world_x_max = env_config.world_max_x - env_config.robot_base_radius
        self.world_y_min = env_config.world_min_y + env_config.robot_base_radius
        self.world_y_max = env_config.world_max_y - env_config.robot_base_radius

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        # Sample robot pose
        abs_x = rng.uniform(self.world_x_min, self.world_x_max)
        abs_y = rng.uniform(self.world_y_min, self.world_y_max)
        abs_theta = rng.uniform(-np.pi, np.pi)

        rel_x = (abs_x - self.world_x_min) / (self.world_x_max - self.world_x_min)
        rel_y = (abs_y - self.world_y_min) / (self.world_y_max - self.world_y_min)
        rel_theta = (abs_theta + np.pi) / (2 * np.pi)

        return (rel_x, rel_y, rel_theta)

    def _get_gripper_actions(self, state: ObjectCentricState) -> tuple[float, float]:
        """Get gripper actions for place: keep closed during movement,
        open after placing.

        Returns:
            (delta_during, delta_after) where:
            - delta_during: 0.0 (keep current gap during movement)
            - delta_after: change needed to open gripper (positive)
        """
        curr_finger_gap = state.get(self._robot, "finger_gap")
        # Open gripper after placing (increase finger_gap)
        desired_finger_gap = self.finger_gap_max
        delta_to_open = desired_finger_gap - curr_finger_gap

        return 0.0, delta_to_open

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_arm_joint = state.get(self._robot, "arm_joint")
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
            robot_arm_joint,
        )

        # Check if the target pose is collision-free
        full_state = state.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            full_state.data.update(init_constant_state.data)

        full_state.set(self._robot, "x", final_robot_x)
        full_state.set(self._robot, "y", final_robot_y)
        full_state.set(self._robot, "theta", final_robot_theta)

        # Check if block is held
        held_objects = []
        for obj in full_state:
            if obj != self._robot:
                try:
                    held = full_state.get(obj, "held")
                    if held > 0.5:
                        held_objects.append(obj)
                except KeyError:
                    pass

        # Check collision
        moving_objects = {self._robot} | set(held_objects)
        static_objects = set(full_state) - moving_objects
        if state_2d_has_collision(
            full_state, moving_objects, static_objects, {}, ignore_z_orders=True
        ):
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free path to target."
            )

        # Simple waypoint generation
        final_waypoints: list[tuple[SE2Pose, float]] = [current_wp]
        final_waypoints.append((final_robot_pose, robot_arm_joint))
        return final_waypoints


class GroundMoveToController(Dynamic2dRobotController):
    """Controller for moving the robot to the target region."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: KinRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._robot = objects[0]
        self._tgt_block = objects[1]
        self._tgt_surface = objects[2]
        self._action_space = action_space

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        # Sample a random orientation
        abs_theta = rng.uniform(-np.pi, np.pi)

        # Relative orientation
        rel_theta = (abs_theta + np.pi) / (2 * np.pi)

        return rel_theta

    def _get_gripper_actions(self, state: ObjectCentricState) -> tuple[float, float]:
        """Get gripper actions for move-to: keep current gap during movement,
        no change after.

        Returns:
            (delta_during, delta_after) where:
            - delta_during: 0.0 (keep current gap during movement)
            - delta_after: 0.0 (no change after moving)
        """
        # Keep gripper at current gap during movement, no change after
        return 0.0, 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_arm_joint = state.get(self._robot, "arm_joint")
        gripper_height = state.get(self._robot, "gripper_base_height")
        tgt_x = state.get(self._tgt_surface, "x")
        tgt_y = state.get(self._tgt_surface, "y")
        tgt_theta = state.get(self._tgt_surface, "theta")
        tgt_width = state.get(self._tgt_surface, "width")
        tgt_height = state.get(self._tgt_surface, "height")
        block_width = state.get(self._tgt_block, "width")
        block_height = state.get(self._tgt_block, "height")

        print(tgt_x, tgt_y, block_height)

        target_region_pose = SE2Pose(tgt_x, tgt_y, tgt_theta) * SE2Pose(
            tgt_width / 2, tgt_height / 2, 0.0
        )

        print(f"target region pose {target_region_pose}")

        # Calculate target position from parameters
        params = cast(float, self._current_params)
        target_theta = params * 2 * np.pi - np.pi
        tgt_pose_center = SE2Pose(
            target_region_pose.x, target_region_pose.y, target_theta
        )

        print(f"tgt pose center {tgt_pose_center}")
        bottom2center = SE2Pose(block_width / 2, block_height / 2, 0.0)
        tgt_pose_bottom = tgt_pose_center * bottom2center.inverse

        print(f"tgt pose bottom {tgt_pose_bottom}")

        # Calculate robot pose to place block on surface
        surface_top_y = tgt_pose_bottom.y
        print(f"surface top y {surface_top_y}")
        block_center_y = (
            surface_top_y + block_height
        )  # Center of block when placed on surface
        print(block_center_y)
        # Calculate gripper target position (above the block center for placement)
        # Gripper should be positioned above the block with some clearance
        gripper_clearance = gripper_height
        gripper_target_y = block_center_y + gripper_clearance

        # Gripper target pose (above the block center, at same x as block center)
        print(f"gripper target y  {gripper_target_y} ")
        gripper_target_pose = SE2Pose(
            tgt_pose_bottom.x, tgt_pose_bottom.y + gripper_target_y, target_theta
        )

        robot_pose = gripper_target_pose

        # Check if the target pose is collision-free
        full_state = state.copy()
        init_constant_state = self._init_constant_state
        if init_constant_state is not None:
            full_state.data.update(init_constant_state.data)

        # Convert to absolute coordinates within target bounds
        full_state.set(self._tgt_block, "x", tgt_pose_bottom.x)
        full_state.set(self._tgt_block, "y", tgt_pose_bottom.y)
        full_state.set(self._tgt_block, "theta", target_theta)

        full_state.set(self._robot, "x", robot_pose.x)
        full_state.set(self._robot, "y", robot_pose.y)
        full_state.set(self._robot, "theta", robot_pose.theta)

        # Check collision
        moving_objects = {self._robot, self._tgt_block, self._tgt_surface}
        static_objects = set(full_state) - moving_objects
        collision = state_2d_has_collision(
            full_state, moving_objects, static_objects, {}
        )
        if collision:
            raise TrajectorySamplingFailure(
                "Failed to find a collision-free path to target."
            )

        # Simple waypoint generation
        final_waypoints: list[tuple[SE2Pose, float]] = []
        final_waypoints.append((robot_pose, robot_arm_joint))
        return final_waypoints


class GroundPushController(Dynamic2dRobotController):
    """Controller for moving the robot to a desired location while pushing objects along
    the way."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: KinRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._block = objects[1]
        self._action_space = action_space
        env_config = DynObstruction2DEnvConfig()
        self.world_x_min = env_config.world_min_x + env_config.robot_base_radius
        self.world_x_max = env_config.world_max_x - env_config.robot_base_radius
        self.world_y_min = env_config.world_min_y + env_config.robot_base_radius
        self.world_y_max = env_config.world_max_y - env_config.robot_base_radius

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float, float]:
        # Sample robot pose
        abs_x = rng.uniform(self.world_x_min, self.world_x_max)
        abs_y = rng.uniform(self.world_y_min, self.world_y_max)
        abs_theta = rng.uniform(-np.pi, np.pi)

        rel_x = (abs_x - self.world_x_min) / (self.world_x_max - self.world_x_min)
        rel_y = (abs_y - self.world_y_min) / (self.world_y_max - self.world_y_min)
        rel_theta = (abs_theta + np.pi) / (2 * np.pi)

        return (rel_x, rel_y, rel_theta)

    def _get_gripper_actions(self, state: ObjectCentricState) -> tuple[float, float]:
        """Get gripper actions for pushing: keep closed during and after movement.

        Returns:
            (delta_during, delta_after) where:
            - delta_during: 0.0 (keep current gap during movement)
            - delta_after: 0.0 (keep current gap after movement)
        """
        return 0.0, 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_arm_joint = state.get(self._robot, "arm_joint")
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
            robot_arm_joint,
        )

        # IMPORTANT - Do not check if target pose is collision-free
        # Simple waypoint generation
        final_waypoints: list[tuple[SE2Pose, float]] = [current_wp]
        final_waypoints.append((final_robot_pose, robot_arm_joint))
        return final_waypoints


def create_lifted_controllers(
    action_space: KinRobotActionSpace,
    init_constant_state: Optional[ObjectCentricState] = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for DynObstruction2D.

    Args:
        action_space: The action space for the KinRobot.
        init_constant_state: Optional initial constant state.

    Returns:
        Dictionary mapping controller names to LiftedParameterizedController instances.
    """

    # Define params_space for each controller type
    pick_params_space = Box(
        low=np.array([0.0, 0.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32,
    )
    place_params_space = Box(
        low=np.array([0.0, 0.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32,
    )
    move_to_params_space = Box(
        low=np.array([0.0]),
        high=np.array([1.0]),
        dtype=np.float32,
    )
    push_params_space = Box(
        low=np.array([0.0, 0.0, 0.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32,
    )

    # Create partial controller classes that include the action_space
    class PickController(GroundPickController):
        """Controller for picking the target block or obstruction."""

        def __init__(self, objects):
            super().__init__(objects, action_space, init_constant_state)

    class PlaceController(GroundPlaceController):
        """Controller for placing the obstruction."""

        def __init__(self, objects):
            super().__init__(objects, action_space, init_constant_state)

    class MoveToTgtController(GroundMoveToController):
        """Controller for moving the robot to the target region."""

        def __init__(self, objects):
            super().__init__(objects, action_space, init_constant_state)

    class PushController(GroundPushController):
        """Controller for robot to push objects on the way to the target region."""

        def __init__(self, objects):
            super().__init__(objects, action_space, init_constant_state)

    # Create variables for lifted controllers
    robot = Variable("?robot", KinRobotType)
    target_block = Variable("?target_block", TargetBlockType)
    target_surface = Variable("?target_surface", TargetSurfaceType)
    obstruction = Variable("?obstruction", DynRectangleType)

    # Lifted controllers
    pick_tgt_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block],
        PickController,
        pick_params_space,
    )

    place_tgt_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block], PlaceController, place_params_space
    )

    pick_obstruction_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstruction],
            PickController,
            pick_params_space,
        )
    )

    place_obstruction_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstruction],
            PlaceController,
            place_params_space,
        )
    )

    move_to_tgt_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target_block, target_surface],
            MoveToTgtController,
            move_to_params_space,
        )
    )

    push_tgt_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block],
        PushController,
        push_params_space,
    )

    push_obstruction_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstruction],
            PushController,
            push_params_space,
        )
    )

    return {
        "pick_tgt": pick_tgt_controller,
        "place_tgt": place_tgt_controller,
        "pick_obstruction": pick_obstruction_controller,
        "place_obstruction": place_obstruction_controller,
        "move_to_tgt": move_to_tgt_controller,
        "push_tgt": push_tgt_controller,
        "push_obstruction": push_obstruction_controller,
    }
