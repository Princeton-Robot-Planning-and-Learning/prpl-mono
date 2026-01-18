"""Base controllers for 3D geometric environments."""

from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
)
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from prbench.envs.geom3d.base_env import ObjectCentricGeom3DRobotEnv
from prbench.envs.geom3d.utils import (
    Geom3DObjectCentricState,
)
from pybullet_helpers.geometry import Pose, SE2Pose
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from relational_structs import (
    Object,
    ObjectCentricState,
)

# constants
GRIPPER_OPEN_THRESHOLD = 0.01
HOME_JOINT_POSITIONS = np.deg2rad([0, -20, 180, -146, 0, -50, 90, 0, 0, 0, 0, 0, 0])


class BasePlaceController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Base class for place controllers with common motion planning logic."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricGeom3DRobotEnv,
    ) -> None:
        """Initialize the base place controller."""
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.joint_infos[:7]
        self._robot, self._target, self._target_table = objects
        self._current_params: np.ndarray | None = None
        self._current_arm_joint_plan: list[JointPositions] | None = None
        self._current_place_arm_joint_plan: list[JointPositions] | None = None
        self._current_retract_plan: list[JointPositions] | None = None
        self._current_plan: list[SE2Pose] | None = None
        self._current_state: ObjectCentricState | None = None
        self._navigated: bool = False
        self._pre_place: bool = False
        self._opened_gripper: bool = False
        self._lifted: bool = False
        self._target_place_pose_se2: SE2Pose | None = None
        self._target_place_pose_world: Pose | None = None
        self._pre_place_pose_world: Pose | None = None

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def navigate(self) -> np.ndarray:
        """Navigate to the next target base pose."""
        # Pop the next target base pose from the plan.
        assert self._current_plan is not None
        target_base_pose = self._current_plan.pop(0)
        if len(self._current_plan) == 0:
            self._navigated = True

        # Compute delta base pose.
        assert isinstance(self._current_state, Geom3DObjectCentricState)
        current_base_pose = self._current_state.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]

        # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)

        return action

    def pre_place(self, collision_ids: set[int] | None = None) -> np.ndarray:
        """Pre-place the object."""
        if collision_ids is None:
            collision_ids = (
                self._sim._get_collision_object_ids()  # pylint: disable=protected-access
            )
        # Generate the motion plan if it doesn't exist yet.
        if self._current_arm_joint_plan is None:
            self._sim.set_state(self._current_state)
            # Create target pose from target position and sampled orientation.

            assert self._pre_place_pose_world is not None
            assert self._target_place_pose_world is not None

            grasped_object_id = (
                self._sim._grasped_object_id  # pylint: disable=protected-access
            )
            grasped_object_transform = (
                self._sim._grasped_object_transform  # pylint: disable=protected-access
            )
            if grasped_object_id is not None:
                collision_ids = collision_ids - {grasped_object_id}

            joint_distance_fn = create_joint_distance_fn(self._sim.robot.arm)
            # Run motion planning to the target joint positions.
            try:
                joint_plan = smoothly_follow_end_effector_path(
                    self._sim.robot.arm,
                    [self._pre_place_pose_world, self._target_place_pose_world],
                    initial_joints=self._sim.robot.arm.get_joint_positions(),
                    collision_ids=collision_ids,
                    seed=0,  # for determinism
                    joint_distance_fn=joint_distance_fn,
                    max_smoothing_iters_per_step=1,
                    held_object=grasped_object_id,
                    base_link_to_held_obj=grasped_object_transform,
                )
            except InverseKinematicsError:
                joint_plan = None
                # Debugging
                import pybullet as p
                while True:
                    p.getMouseEvents(self._sim.physics_client_id)

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
        assert isinstance(self._current_state, Geom3DObjectCentricState)
        delta_lst = get_jointwise_difference(
            self._joint_infos,
            target_joints[:7],
            self._current_state.joint_positions,
        )

        # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)

        return action

    def open_gripper(self) -> np.ndarray:
        """Open the gripper."""
        if self._get_current_robot_gripper_pose() < GRIPPER_OPEN_THRESHOLD:
            self._opened_gripper = True
        action_lst = [0.0] * 10 + [1.0]
        action = np.array(action_lst, dtype=np.float32)
        return action

    def lift(self, collision_ids: set[int] | None = None) -> np.ndarray:
        """Lift the object."""
        if collision_ids is None:
            collision_ids = (
                self._sim._get_collision_object_ids()  # pylint: disable=protected-access
            )
        # Generate the motion plan if it doesn't exist yet.
        if self._current_retract_plan is None:

            self._sim.set_state(self._current_state)

            # Run motion planning to the target joint positions.
            joint_plan = run_motion_planning(  # type: ignore
                self._sim.robot.arm,
                initial_positions=self._sim.robot.arm.get_joint_positions(),
                target_positions=HOME_JOINT_POSITIONS.tolist(),
                collision_bodies=collision_ids,
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
        assert isinstance(self._current_state, Geom3DObjectCentricState)
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

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x

    def _get_current_robot_gripper_pose(self) -> float:
        x = self._current_state
        assert x is not None
        robot_obj = x.get_object_from_name("robot")
        return x.get(robot_obj, "finger_state")
