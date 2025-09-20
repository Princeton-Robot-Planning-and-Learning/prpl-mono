"""Parameterized skills for the StickButton2D environment."""

from typing import Optional, Sequence

import numpy as np
from prbench.envs.geom2d.structs import SE2Pose
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
)
from relational_structs import (
    Object,
    ObjectCentricState,
)

from prbench_models.geom2d.utils import Geom2dRobotController


# Controllers.
class GroundPickStickController(Geom2dRobotController):
    """Controller for moving the robot to pick the stick."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._stick = objects[1]

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[float, float]:
        # Sample grasp ratio along the stick [0,1] and desired arm length
        grasp_ratio = rng.uniform(0.0, 1.0)
        max_arm_length = x.get(self._robot, "arm_length")
        min_arm_length = x.get(self._robot, "base_radius")
        arm_length = rng.uniform(min_arm_length, max_arm_length)
        return (grasp_ratio, arm_length)

    def _calculate_grasp_point(self, state: ObjectCentricState) -> tuple[float, float]:
        """Calculate the actual grasp point based on ratio parameter."""
        if isinstance(self._current_params, tuple):
            grasp_ratio, _ = self._current_params
        else:
            raise ValueError("GroundPickStickController requires tuple parameters")

        # Get stick properties
        stick_x = state.get(self._stick, "x")
        stick_y = state.get(self._stick, "y")
        stick_width = state.get(self._stick, "width")

        # Get robot gripper properties
        gripper_height = state.get(self._robot, "gripper_height")

        full_line_length = stick_width + 2 * gripper_height
        line_length = full_line_length * grasp_ratio
        side_ratio = gripper_height / full_line_length
        bottom_ratio = stick_width / full_line_length

        # Define the grasping line from left bottom to right bottom of stick
        # Line starts at left edge and extends by gripper width on each side
        left_x = stick_x
        right_x = stick_x + stick_width
        bottom_y = stick_y
        grasp_x: float = 0.0
        grasp_y: float = 0.0

        if grasp_ratio < side_ratio:  # Grasping from left side
            grasp_x = left_x
            grasp_y = bottom_y + (gripper_height - line_length)
        elif (
            side_ratio <= grasp_ratio < side_ratio + bottom_ratio
        ):  # Grasping from bottom
            grasp_x = left_x + (grasp_ratio - side_ratio) * full_line_length
            grasp_y = bottom_y
        else:  # Grasping from right side
            grasp_x = right_x
            grasp_y = bottom_y + (line_length - gripper_height - stick_width)

        return grasp_x, grasp_y

    def _calculate_robot_position(
        self, state: ObjectCentricState, grasp_x: float, grasp_y: float
    ) -> tuple[float, float, float]:
        """Calculate robot position and orientation to reach grasp point."""
        if isinstance(self._current_params, tuple):
            _, desired_arm_length = self._current_params
        else:
            raise ValueError("GroundPickStickController requires tuple parameters")

        # Get stick properties
        stick_x = state.get(self._stick, "x")
        stick_width = state.get(self._stick, "width")

        # Get robot properties
        gripper_width = state.get(self._robot, "gripper_width")

        # Determine which side of the stick we're grasping from
        stick_left = stick_x
        stick_right = stick_x + stick_width

        robot_x: float = 0.0
        robot_y: float = 0.0
        robot_theta: float = 0.0

        if grasp_x < stick_left + stick_width * 0.01:  # Left side
            robot_x = grasp_x - desired_arm_length - gripper_width
            robot_y = grasp_y
            robot_theta = 0.0  # Facing right
        elif grasp_x > stick_right - stick_width * 0.01:  # Right side
            robot_x = grasp_x + desired_arm_length + gripper_width
            robot_y = grasp_y
            robot_theta = np.pi  # Facing left
        else:  # Bottom side
            robot_x = grasp_x
            robot_y = grasp_y - desired_arm_length - gripper_width
            robot_theta = np.pi / 2  # Facing up

        return robot_x, robot_y, robot_theta

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")
        robot_gripper_width = state.get(self._robot, "gripper_width")
        safe_y = robot_radius + robot_gripper_width * 2

        # Calculate grasp point and robot target position
        grasp_x, grasp_y = self._calculate_grasp_point(state)
        target_x, target_y, target_theta = self._calculate_robot_position(
            state, grasp_x, grasp_y
        )
        if isinstance(self._current_params, tuple):
            _, desired_arm_length = self._current_params
        else:
            raise ValueError("GroundPickStickController requires tuple parameters")

        return [
            # Start by moving the arm inside the robot's base
            (SE2Pose(robot_x, safe_y, robot_theta), robot_radius),
            # Start by moving to safe height with current orientation
            (SE2Pose(robot_x, safe_y, robot_theta), robot_radius),
            # Move to target x position at safe height
            (SE2Pose(target_x, safe_y, robot_theta), robot_radius),
            # Orient towards the stick
            (SE2Pose(target_x, safe_y, target_theta), robot_radius),
            # Move down to grasp position
            (SE2Pose(target_x, target_y, target_theta), robot_radius),
            # Extend arm to desired length
            (SE2Pose(target_x, target_y, target_theta), desired_arm_length),
        ]

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 1.0


class GroundPlaceStickController(Geom2dRobotController):
    """Controller for moving the robot to place the stick down."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._stick = objects[1]

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        # Parameter represents absolute x position where to release the stick
        del x, rng  # not used in this controller
        return 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_base_radius = state.get(self._robot, "base_radius")
        robot_theta = state.get(self._robot, "theta")

        return [
            # Just move the arm back
            (SE2Pose(robot_x, robot_y, robot_theta), robot_base_radius),
        ]

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 0.0


class GroundRobotPressButtonController(Geom2dRobotController):
    """Controller for moving the robot to press a button."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._button = objects[1]

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        del x, rng  # not used in this controller
        return 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        robot_theta = state.get(self._robot, "theta")
        robot_radius = state.get(self._robot, "base_radius")
        button_x = state.get(self._button, "x")
        button_y = state.get(self._button, "y")

        # Position robot base to intersect with button
        # For intersection, robot base should be close to button center
        target_x = button_x
        target_y = button_y  # Put robot base at button level for intersection

        return [
            # Move down so robot base overlaps with button
            (SE2Pose(target_x, target_y, robot_theta), robot_radius),
        ]

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 0.0, 0.0


class GroundStickPressButtonController(Geom2dRobotController):
    """Controller for moving the robot to use the stick to press a button."""

    def __init__(
        self,
        objects: Sequence[Object],
        action_space: CRVRobotActionSpace,
        init_constant_state: Optional[ObjectCentricState] = None,
    ) -> None:
        super().__init__(objects, action_space, init_constant_state)
        self._stick = objects[1]
        self._button = objects[2]

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        del x, rng  # not used in this controller
        return 0.0

    def _generate_waypoints(
        self, state: ObjectCentricState
    ) -> list[tuple[SE2Pose, float]]:
        """Assume we always use the stick far end to press the button."""
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_theta = state.get(self._robot, "theta")
        robot_arm_joint = state.get(self._robot, "arm_joint")
        button_x = state.get(self._button, "x")
        button_y = state.get(self._button, "y")
        stick_far_x = state.get(self._stick, "x") + state.get(self._stick, "width")
        stick_far_y = state.get(self._stick, "y") + state.get(self._stick, "height")

        dx = button_x - stick_far_x
        dy = button_y - stick_far_y

        # Position robot so stick can reach button
        # Account for stick length and robot arm extension
        target_x = robot_x + dx
        target_y = robot_y + dy

        return [
            # Move down to press button with stick
            (SE2Pose(target_x, target_y, robot_theta), robot_arm_joint),
        ]

    def _get_vacuum_actions(self) -> tuple[float, float]:
        return 1.0, 1.0  # Keep holding stick
