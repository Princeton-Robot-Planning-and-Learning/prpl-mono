"""Parameterized skills for the Obstruction3D environment."""

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
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
)
from kinder.envs.kinematic3d.obstruction3d import (
    Kinematic3DRobotType,
    ObjectCentricObstruction3DEnv,
    Obstruction3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    smoothly_follow_end_effector_path,
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


# Controllers.
class GroundPickController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for picking an object."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricObstruction3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._object = objects
        self._current_params: JointPositions | None = None
        self._current_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._pre_retract_plan: list[JointPositions] | None = None
        self._current_state: ObjectCentricState | None = None
        self._pre_grasp: bool = False
        self._closed_gripper: bool = False
        self._pre_lifted: bool = False
        self._lifted: bool = False
        self._last_gripper_state: float = 0.0

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> JointPositions:
        assert isinstance(x, Obstruction3DObjectCentricState)
        self._sim.set_state(x)

        # Create target pose from target position and sampled orientation.
        target_grasp_pose_world = x.target_block_pose

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

        return joint_positions

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._pre_grasp and self._closed_gripper and self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Obstruction3DObjectCentricState)
        self._sim.set_state(self._current_state)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:

            # Run motion planning to the target joint positions.
            joint_plan = run_motion_planning(
                self._sim.robot.arm,
                initial_positions=self._sim.robot.arm.get_joint_positions(),
                target_positions=self._current_params,
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
            self._current_plan = joint_plan[1:]

        if not self._pre_grasp and not self._closed_gripper:
            # Pop the next target joint positions from the plan.
            assert self._current_plan is not None
            target_joints = self._current_plan.pop(0)
            if len(self._current_plan) == 0:
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
        if self._closed_gripper and not self._pre_lifted:
            # Generate the motion plan if it doesn't exist yet.
            if self._pre_retract_plan is None:
                current_end_effector_pose = self._sim.robot.arm.get_end_effector_pose()
                pre_retract_pose = Pose(
                    (
                        current_end_effector_pose.position[0],
                        current_end_effector_pose.position[1],
                        current_end_effector_pose.position[2] + 0.05,
                    ),
                    current_end_effector_pose.orientation,
                )
                joint_distance_fn = create_joint_distance_fn(self._sim.robot.arm)
                # Run motion planning to the target joint positions.
                joint_plan = smoothly_follow_end_effector_path(
                    self._sim.robot.arm,
                    [current_end_effector_pose, pre_retract_pose],
                    initial_joints=self._sim.robot.arm.get_joint_positions(),
                    collision_ids={},  # type: ignore
                    seed=0,  # for determinism
                    joint_distance_fn=joint_distance_fn,
                    max_smoothing_iters_per_step=1,
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
                self._pre_retract_plan = joint_plan[1:]
            # Pop the next target joint positions from the plan.
            assert self._pre_retract_plan is not None
            target_joints = self._pre_retract_plan.pop(0)
            if len(self._pre_retract_plan) == 0:
                self._pre_lifted = True
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

        if self._pre_lifted and not self._lifted:
            # Generate the motion plan if it doesn't exist yet.
            if self._current_retract_plan is None:

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
        sim: ObjectCentricObstruction3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._object = objects
        self._current_params: JointPositions | None = None
        self._current_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_state: ObjectCentricState | None = None
        self._pre_place: bool = False
        self._open_gripper: bool = False
        self._returned: bool = False
        self._last_gripper_state: float = 0.0

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> JointPositions:
        assert isinstance(x, Obstruction3DObjectCentricState)
        self._sim.set_state(x)

        # Create target pose from target position and sampled orientation.
        placement_padding = 1e-4  # leave some room to prevent collisions with surface
        target_place_pose_world = Pose(
            (
                x.target_region_pose.position[0],
                x.target_region_pose.position[1],
                x.target_region_pose.position[2]
                + x.target_region_half_extents[2]
                + x.target_block_half_extents[2]
                + placement_padding,
            ),
            x.target_region_pose.orientation,
        )

        target_end_effector_pose = multiply_poses(
            target_place_pose_world,
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

        return joint_positions

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._pre_place and self._open_gripper and self._returned

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Obstruction3DObjectCentricState)
        self._sim.set_state(self._current_state)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:

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
                target_positions=self._current_params,
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
            self._current_plan = joint_plan[1:]

        if not self._pre_place and not self._open_gripper:
            # Pop the next target joint positions from the plan.
            assert self._current_plan is not None
            target_joints = self._current_plan.pop(0)
            if len(self._current_plan) == 0:
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
        if self._pre_place and not self._open_gripper:
            if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
                self._open_gripper = True
            action_lst = [0.0] * 10 + [1.0]
            action = np.array(action_lst, dtype=np.float32)
            self._last_gripper_state = self._get_current_robot_gripper_pose()
            return action
        if self._open_gripper and not self._returned:
            # Generate the motion plan if it doesn't exist yet.
            if self._current_retract_plan is None:

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
                self._returned = True
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
    sim: ObjectCentricObstruction3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for Obstruction3D."""
    del action_space

    # Create partial controller classes that include the sim
    class PickController(GroundPickController):
        """Controller for picking an object."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target_block = Variable("?target_block", Kinematic3DCuboidType)

    # Lifted controllers
    pick_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_block],
        PickController,
        Box(-np.inf, np.inf, (7,)),
    )

    class PlaceController(GroundPlaceController):
        """Controller for placing an object."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target_region = Variable("?target_region", Kinematic3DCuboidType)

    place_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target_region],
        PlaceController,
        Box(-np.inf, np.inf, (7,)),
    )

    return {
        "pick": pick_controller,
        "place": place_controller,
    }
