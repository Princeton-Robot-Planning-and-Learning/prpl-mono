"""Parameterized skills for the Ground3D environment."""

from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from kinder.envs.kinematic3d.ground3d import (
    Ground3DObjectCentricState,
    Kinematic3DRobotType,
    ObjectCentricGround3DEnv,
)
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from pybullet_helpers.geometry import Pose, SE2Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
)
from relational_structs import (
    Object,
    ObjectCentricState,
    Variable,
)

from kinder_models.kinematic3d.constants import (
    GRASP_TRANSFORM_TO_OBJECT,
    GRIPPER_CLOSE_THRESHOLD,
    GRIPPER_OPEN_THRESHOLD,
    HOME_JOINT_POSITIONS,
)

# constants
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.45, 0.6)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi / 2, np.pi / 2)
PLACE_X_OFFSET_BOUNDS = (-0.1, 0.1)
PLACE_Y_OFFSET_BOUNDS = (-0.1, 0.1)


# Utility functions.
def get_target_robot_pose_from_parameters(
    target_object_pose: SE2Pose, target_distance: float, target_rot: float
) -> SE2Pose:
    """Determine the pose for the robot given the state and parameters.

    The robot will be facing the target_object_pose position while being target_distance
    away, and rotated w.r.t. the target_object_pose rotation by target_rot.
    """
    # Absolute angle of the line from the robot to the target.
    ang = target_object_pose.rot + target_rot

    # Place the robot `target_distance` away from the target along -ang
    tx, ty = target_object_pose.x, target_object_pose.y  # target translation (x, y).
    rx = tx - target_distance * np.cos(ang)
    ry = ty - target_distance * np.sin(ang)

    # Robot faces the target: heading points along +ang (toward the target).
    return SE2Pose(rx, ry, ang)


# Controllers.
class GroundPickController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for picking up an object."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricGround3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target = objects
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_plan: list[SE2Pose] | None = None
        self._current_state: ObjectCentricState | None = None
        self._navigated: bool = False
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """No parameters needed for base motion - just move to target."""
        assert isinstance(x, Ground3DObjectCentricState)
        distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)
        rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([distance, rot])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Ground3DObjectCentricState)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            target_pose = self._current_state.get_object_pose(
                self.objects[1].name
            ).to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                target_pose, self._current_params[0], self._current_params[1]
            )

            # Run base motion planning to the target pose.
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=self._sim._get_collision_object_ids(),  # pylint: disable=protected-access
                seed=0,  # for determinism
            )

            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")

            # Store the plan (excluding the first state which is the current state).
            self._current_plan = base_plan[1:]

        if not self._navigated:
            # Pop the next target base pose from the plan.
            assert self._current_plan is not None
            target_base_pose = self._current_plan.pop(0)
            if len(self._current_plan) == 0:
                self._navigated = True

            # Compute delta base pose.
            current_base_pose = self._current_state.base_pose
            delta = target_base_pose - current_base_pose
            delta_lst = [delta.x, delta.y, delta.rot]

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = delta_lst + [0.0] * 7 + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._navigated and not self._pre_grasp:
            # Generate the motion plan if it doesn't exist yet.
            if self._current_arm_joint_plan is None:
                self._sim.set_state(self._current_state)
                # Create target pose from target position and sampled orientation.
                target_grasp_pose_world = self._current_state.get_object_pose(
                    self.objects[1].name
                )

                target_end_effector_pose = multiply_poses(
                    target_grasp_pose_world,
                    GRASP_TRANSFORM_TO_OBJECT,
                )

                # Run inverse kinematics to get joint positions.
                try:
                    joint_positions = inverse_kinematics(
                        self._sim.robot.arm,
                        target_end_effector_pose,
                        validate=True,
                        set_joints=False,
                    )
                except InverseKinematicsError as e:
                    raise TrajectorySamplingFailure(
                        f"IK failed for target pose {target_end_effector_pose}"
                    ) from e

                # Run motion planning to the target joint positions.
                joint_plan = run_motion_planning(
                    self._sim.robot.arm,
                    initial_positions=self._sim.robot.arm.get_joint_positions(),
                    target_positions=joint_positions,
                    collision_bodies=self._sim._get_collision_object_ids(),  # pylint: disable=protected-access
                    seed=0,  # for determinism
                    physics_client_id=self._sim.physics_client_id,
                )

                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_arm_joint_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_arm_joint_plan is not None
            target_joints = self._current_arm_joint_plan.pop(0)
            if len(self._current_arm_joint_plan) == 0:
                self._pre_grasp = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._pre_grasp and not self._closed_gripper:
            if (
                self._get_current_robot_gripper_pose() > GRIPPER_CLOSE_THRESHOLD
                and np.isclose(
                    self._get_current_robot_gripper_pose(),
                    self._last_gripper_state,
                    atol=0.02,
                )
            ):
                self._closed_gripper = True
            action_lst = [0.0] * 10 + [-1.0]
            action = np.array(action_lst, dtype=np.float32)
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action

        if self._closed_gripper and not self._lifted:
            # Generate the motion plan if it doesn't exist yet.
            if self._current_retract_plan is None:

                self._sim.set_state(self._current_state)

                grasped_object_id = (
                    self._sim._grasped_object_id  # pylint: disable=protected-access
                )
                grasped_object_transform = (
                    self._sim._grasped_object_transform  # pylint: disable=protected-access
                )
                all_collision_ids = (
                    self._sim._get_collision_object_ids()  # pylint: disable=protected-access
                )
                # Run motion planning to the target joint positions.
                joint_plan = run_motion_planning(
                    self._sim.robot.arm,
                    initial_positions=self._sim.robot.arm.get_joint_positions(),
                    target_positions=HOME_JOINT_POSITIONS.tolist(),
                    collision_bodies=all_collision_ids - {grasped_object_id},
                    seed=0,  # for determinism
                    physics_client_id=self._sim.physics_client_id,
                    held_object=grasped_object_id,
                    base_link_to_held_obj=grasped_object_transform,
                )

                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_retract_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_retract_plan is not None
            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._current_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return x.get(robot_obj, "finger_state")


class GroundPlaceController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for placing an object."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricGround3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target = objects
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_plan: list[SE2Pose] | None = None
        self._current_state: ObjectCentricState | None = None
        self._navigated: bool = False
        self._pre_place: bool = False
        self._opened_gripper: bool = False
        self._lifted: bool = False
        self._target_place_pose_se2: SE2Pose | None = None
        self._target_place_pose_world: Pose | None = None

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """No parameters needed for base motion - just move to target."""
        assert isinstance(x, Ground3DObjectCentricState)
        place_x_offset = rng.uniform(*PLACE_X_OFFSET_BOUNDS)  # type: ignore
        place_y_offset = rng.uniform(*PLACE_Y_OFFSET_BOUNDS)  # type: ignore
        return np.array([place_x_offset, place_y_offset])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Ground3DObjectCentricState)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            target_pose = self._current_state.get_object_pose(self.objects[1].name)
            self._target_place_pose_world = Pose(
                (
                    target_pose.position[0] + self._current_params[0],
                    target_pose.position[1] + self._current_params[1],
                    self._current_state.get_object_half_extents(self.objects[1].name)[2]
                    + 0.01,
                ),
                target_pose.orientation,
            )
            self._target_place_pose_se2 = self._target_place_pose_world.to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                self._target_place_pose_se2, 0.5, 0.0
            )
            # Run base motion planning to the target pose.
            grasped_object_id = (
                self._sim._grasped_object_id  # pylint: disable=protected-access
            )
            grasped_object_transform = (
                self._sim._grasped_object_transform  # pylint: disable=protected-access
            )
            all_collision_ids = (
                self._sim._get_collision_object_ids()  # pylint: disable=protected-access
            )
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=all_collision_ids - {grasped_object_id},
                seed=0,  # for determinism
                held_object=grasped_object_id,
                base_link_to_held_obj=grasped_object_transform,
            )

            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")

            # Store the plan (excluding the first state which is the current state).
            self._current_plan = base_plan[1:]

        if not self._navigated:
            # Pop the next target base pose from the plan.
            assert self._current_plan is not None
            target_base_pose = self._current_plan.pop(0)
            if len(self._current_plan) == 0:
                self._navigated = True

            # Compute delta base pose.
            current_base_pose = self._current_state.base_pose
            delta = target_base_pose - current_base_pose
            delta_lst = [delta.x, delta.y, delta.rot]

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = delta_lst + [0.0] * 7 + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._navigated and not self._pre_place:
            # Generate the motion plan if it doesn't exist yet.
            if self._current_arm_joint_plan is None:
                self._sim.set_state(self._current_state)
                # Create target pose from target position and sampled orientation.

                assert self._target_place_pose_world is not None
                target_end_effector_pose = multiply_poses(
                    self._target_place_pose_world,
                    GRASP_TRANSFORM_TO_OBJECT,
                )

                # Run inverse kinematics to get joint positions.
                try:
                    joint_positions = inverse_kinematics(
                        self._sim.robot.arm,
                        target_end_effector_pose,
                        validate=True,
                        set_joints=False,
                    )
                except InverseKinematicsError as e:
                    raise TrajectorySamplingFailure(
                        f"IK failed for target pose {target_end_effector_pose}"
                    ) from e

                # Run motion planning to the target joint positions.
                grasped_object_id = (
                    self._sim._grasped_object_id  # pylint: disable=protected-access
                )
                grasped_object_transform = (
                    self._sim._grasped_object_transform  # pylint: disable=protected-access
                )
                all_collision_ids = (
                    self._sim._get_collision_object_ids()  # pylint: disable=protected-access
                )
                joint_plan = run_motion_planning(
                    self._sim.robot.arm,
                    initial_positions=self._sim.robot.arm.get_joint_positions(),
                    target_positions=joint_positions,
                    collision_bodies=all_collision_ids - {grasped_object_id},
                    seed=0,  # for determinism
                    physics_client_id=self._sim.physics_client_id,
                    held_object=grasped_object_id,
                    base_link_to_held_obj=grasped_object_transform,
                )

                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_arm_joint_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_arm_joint_plan is not None
            target_joints = self._current_arm_joint_plan.pop(0)
            if len(self._current_arm_joint_plan) == 0:
                self._pre_place = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        if self._pre_place and not self._opened_gripper:
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._opened_gripper = True
            action_lst = [0.0] * 10 + [1.0]
            action = np.array(action_lst, dtype=np.float32)
            return action

        if self._opened_gripper and not self._lifted:
            # Generate the motion plan if it doesn't exist yet.
            if self._current_retract_plan is None:

                self._sim.set_state(self._current_state)

                # Run motion planning to the target joint positions.
                joint_plan = run_motion_planning(
                    self._sim.robot.arm,
                    initial_positions=self._sim.robot.arm.get_joint_positions(),
                    target_positions=HOME_JOINT_POSITIONS.tolist(),
                    collision_bodies=self._sim._get_collision_object_ids(),  # pylint: disable=protected-access
                    seed=0,  # for determinism
                    physics_client_id=self._sim.physics_client_id,
                )

                if joint_plan is None:
                    raise TrajectorySamplingFailure("Motion planning failed")

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan,
                    self._sim.robot.arm,
                    max_distance=self._sim.config.max_action_mag / 2,
                )

                # Store the plan (excluding the first state which is the current state).
                self._current_retract_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._current_retract_plan is not None
            target_joints = self._current_retract_plan.pop(0)
            if len(self._current_retract_plan) == 0:
                self._lifted = True
            # Compute delta joint positions.
            delta_lst = get_jointwise_difference(
                self._joint_infos,
                target_joints[:7],
                self._current_state.joint_positions,
            )

            # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
            action_lst = [0.0] * 3 + delta_lst + [0.0]
            action = np.array(action_lst, dtype=np.float32)

            return action

        raise ValueError("Invalid state")

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._current_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return x.get(robot_obj, "finger_state")


def create_lifted_controllers(
    action_space: Kinematic3DRobotActionSpace,
    sim: ObjectCentricGround3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for Ground3D."""
    del action_space

    # Create partial controller classes that include the sim
    class PickController(GroundPickController):
        """Controller for picking up an object."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    class PlaceController(GroundPlaceController):
        """Controller for placing an object."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    # Lifted controllers
    pick_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        PickController,
        Box(
            low=np.array(
                [
                    MOVE_TO_TARGET_DISTANCE_BOUNDS[0],
                    MOVE_TO_TARGET_ROT_BOUNDS[0],
                ]
            ),
            high=np.array(
                [
                    MOVE_TO_TARGET_DISTANCE_BOUNDS[1],
                    MOVE_TO_TARGET_ROT_BOUNDS[1],
                ]
            ),
        ),
    )

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    # lifted place controller
    place_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target],
        PlaceController,
        Box(
            low=np.array([PLACE_X_OFFSET_BOUNDS[0], PLACE_Y_OFFSET_BOUNDS[0]]),
            high=np.array([PLACE_X_OFFSET_BOUNDS[1], PLACE_Y_OFFSET_BOUNDS[1]]),
        ),
    )

    return {
        "pick": pick_controller,
        "place": place_controller,
    }
