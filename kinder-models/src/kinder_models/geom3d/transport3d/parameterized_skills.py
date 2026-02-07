"""Parameterized skills for the Transport3D environment."""

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
from kinder.envs.kinematic3d.object_types import Kinematic3DCuboidType
from kinder.envs.kinematic3d.transport3d import (
    Kinematic3DRobotType,
    ObjectCentricTransport3DEnv,
    Transport3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from pybullet_helpers.geometry import Pose, SE2Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    MotionPlanningHyperparameters,
    create_joint_distance_fn,
    remap_joint_position_plan_to_constant_distance,
    remap_se2_pose_plan_to_constant_distance,
    run_motion_planning,
    run_single_arm_mobile_base_motion_planning,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from relational_structs import (
    Object,
    ObjectCentricState,
    Variable,
)

from kinder_models.kinematic3d.base_controllers import BasePlaceController
from kinder_models.kinematic3d.constants import (
    GRIPPER_CLOSE_THRESHOLD,
    HOME_JOINT_POSITIONS,
)
from kinder_models.kinematic3d.utils import get_target_robot_pose_from_parameters

# constants
GRASP_TRANSFORM_TO_OBJECT_BOX = Pose(
    (0.0, 0.15, 0.08), (0.707, 0.707, 0, 0)
)  # side grasp
GRASP_TRANSFORM_TO_OBJECT_CUBE = Pose((0.005, 0, 0.02), (0.707, -0.707, 0, 0))
SIDE_PLACE_TRANSFORM_TO_OBJECT = Pose((0.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.5, 0.6)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi / 4, np.pi / 4)
PLACE_X_OFFSET_BOUNDS_BOX = (-0.1, 0.0)
PLACE_Y_OFFSET_BOUNDS_BOX = (-0.05, 0.05)
PLACE_X_OFFSET_BOUNDS_CUBE = (-0.05, 0.05)
PLACE_Y_OFFSET_BOUNDS_CUBE = (-0.05, 0.05)
PLACE_X_OFFSET_BOUNDS_TABLE = (-0.15, 0.15)
PLACE_Y_OFFSET_BOUNDS_TABLE = (-0.25, 0.25)


# Controllers.
class GroundPickController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for picking up an object."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricTransport3DEnv,
        birrt_extend_num_interp: int = 10,
        smooth_mp_max_time: float = 120.0,
        smooth_mp_max_candidate_plans: int = 50,
        base_mp_birrt_smooth_amt: int = 100,
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
        self._target_pick_pose_world: Pose | None = None
        self._pre_pick_pose_world: Pose | None = None
        # Motion planning hyperparameters.
        self._birrt_extend_num_interp = birrt_extend_num_interp
        self._smooth_mp_max_time = smooth_mp_max_time
        self._smooth_mp_max_candidate_plans = smooth_mp_max_candidate_plans
        self._base_mp_birrt_smooth_amt = base_mp_birrt_smooth_amt

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """No parameters needed for base motion - just move to target."""
        assert isinstance(x, Transport3DObjectCentricState)
        distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)  # type: ignore
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
        assert isinstance(self._current_state, Transport3DObjectCentricState)

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
                hyperparameters=MotionPlanningHyperparameters(
                    birrt_extend_num_interp=self._birrt_extend_num_interp,
                    birrt_smooth_amt=self._base_mp_birrt_smooth_amt,
                ),
            )

            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")

            # Remap the plan to ensure we stay within action limits.
            base_plan = remap_se2_pose_plan_to_constant_distance(
                base_plan,
                max_distance=self._sim.config.max_action_mag,
            )

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

                if "box" in self.objects[1].name:
                    grasp_transform = GRASP_TRANSFORM_TO_OBJECT_BOX
                else:
                    grasp_transform = GRASP_TRANSFORM_TO_OBJECT_CUBE
                target_end_effector_pose = multiply_poses(
                    target_grasp_pose_world,
                    grasp_transform,
                )

                self._target_pick_pose_world = target_end_effector_pose
                self._pre_pick_pose_world = Pose(
                    (
                        target_end_effector_pose.position[0],
                        target_end_effector_pose.position[1],
                        target_end_effector_pose.position[2] + 0.1,
                    ),
                    target_end_effector_pose.orientation,
                )

                collision_ids = (
                    self._sim._get_collision_object_ids()  # pylint: disable=protected-access
                )
                joint_distance_fn = create_joint_distance_fn(self._sim.robot.arm)

                # First run motion planning to get to the pre-pick pose.
                smooth_mp_kwargs: dict[str, Any] = {
                    "max_time": self._smooth_mp_max_time,
                    "max_candidate_plans": self._smooth_mp_max_candidate_plans,
                    "birrt_extend_num_interp": self._birrt_extend_num_interp,
                }
                try:
                    joint_plan1 = run_smooth_motion_planning_to_pose(
                        self._pre_pick_pose_world,
                        self._sim.robot.arm,
                        collision_ids=collision_ids,
                        end_effector_frame_to_plan_frame=Pose.identity(),
                        seed=0,  # for determinism
                        **smooth_mp_kwargs,
                    )
                except InverseKinematicsError:
                    joint_plan1 = None
                    # Debugging
                    # import pybullet as p
                    # while True:
                    #     p.getMouseEvents(self._sim.physics_client_id)

                if joint_plan1 is None:
                    raise TrajectorySamplingFailure("Motion planning failed")

                # Run motion planning to the target joint positions.
                try:
                    self._sim.robot.arm.set_joints(joint_plan1[-1])
                    ee_pose = self._sim.robot.arm.get_end_effector_pose()
                    assert ee_pose.allclose(self._pre_pick_pose_world, atol=1e-4)
                    joint_plan2 = smoothly_follow_end_effector_path(
                        self._sim.robot.arm,
                        [self._pre_pick_pose_world, self._target_pick_pose_world],
                        initial_joints=self._sim.robot.arm.get_joint_positions(),
                        collision_ids=collision_ids,
                        seed=0,  # for determinism
                        joint_distance_fn=joint_distance_fn,
                        max_smoothing_iters_per_step=1,
                        include_start=False,
                    )
                except InverseKinematicsError:
                    joint_plan2 = None
                    # Debugging
                    # import pybullet as p
                    # while True:
                    #     p.getMouseEvents(self._sim.physics_client_id)

                if joint_plan2 is None:
                    raise TrajectorySamplingFailure("Motion planning failed")

                # Remap the plan to ensure we stay within action limits.
                joint_plan = remap_joint_position_plan_to_constant_distance(
                    joint_plan1 + joint_plan2,
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
                all_collision_ids -= (
                    self._sim._get_inside_object_ids()  # pylint: disable=protected-access
                )
                joint_plan = run_motion_planning(  # type: ignore
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


class GroundPlaceController(BasePlaceController):
    """Controller for placing an object."""

    def sample_parameters(self, x: ObjectCentricState, rng: np.random.Generator) -> Any:
        """No parameters needed for base motion - just move to target."""
        assert isinstance(x, Transport3DObjectCentricState)
        if "box" in self.objects[1].name:
            place_x_offset_bounds = PLACE_X_OFFSET_BOUNDS_BOX
            place_y_offset_bounds = PLACE_Y_OFFSET_BOUNDS_BOX
        elif "cube" in self.objects[1].name and "box" in self.objects[2].name:
            place_x_offset_bounds = PLACE_X_OFFSET_BOUNDS_CUBE
            place_y_offset_bounds = PLACE_Y_OFFSET_BOUNDS_CUBE
        elif "cube" in self.objects[1].name and "table" in self.objects[2].name:
            place_x_offset_bounds = PLACE_X_OFFSET_BOUNDS_TABLE
            place_y_offset_bounds = PLACE_Y_OFFSET_BOUNDS_TABLE
        else:
            raise ValueError("Invalid target object")
        place_x_offset = rng.uniform(*place_x_offset_bounds)  # type: ignore
        place_y_offset = rng.uniform(*place_y_offset_bounds)  # type: ignore
        return np.array([place_x_offset, place_y_offset])

    def terminated(self) -> bool:
        return self._lifted

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Transport3DObjectCentricState)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            # Get the grasp transform to compute EE pose from desired object pose.
            grasped_object_transform = (
                self._sim._grasped_object_transform  # pylint: disable=protected-access
            )
            assert grasped_object_transform is not None

            # Compute the desired object placement pose (where the held object
            # should end up). The object should be placed upright.
            target_surface_pose = self._current_state.get_object_pose(
                self.objects[2].name
            )
            if "box" in self.objects[1].name and "table" in self.objects[2].name:
                # Place box on table: box center should be at table surface
                # + box bottom thickness + half box height.
                desired_object_z = (
                    target_surface_pose.position[2]
                    + self._sim.config.table_half_extents[2]
                    + self._sim.config.box_wall_thickness
                    + self._sim.config.box_half_extents[2]
                )
                desired_object_pose = Pose(
                    (
                        target_surface_pose.position[0] + self._current_params[0],
                        target_surface_pose.position[1] + self._current_params[1],
                        desired_object_z,
                    ),
                    (0, 0, 0, 1),  # Upright (identity quaternion, xyzw format)
                )
            elif "cube" in self.objects[1].name and "table" in self.objects[2].name:
                # Place cube on table: cube center at table surface + half cube size.
                desired_object_z = (
                    target_surface_pose.position[2]
                    + self._sim.config.table_half_extents[2]
                    + self._sim.config.block_size / 2
                )
                desired_object_pose = Pose(
                    (
                        target_surface_pose.position[0] + self._current_params[0],
                        target_surface_pose.position[1] + self._current_params[1],
                        desired_object_z,
                    ),
                    (0, 0, 0, 1),  # Upright (identity quaternion, xyzw format)
                )
            elif "cube" in self.objects[1].name and "box" in self.objects[2].name:
                # Place cube inside box: cube center at box bottom + half cube size.
                # The x,y are relative to the box position.
                desired_object_z = (
                    self._sim.config.box_wall_thickness
                    + self._sim.config.block_size / 2
                    + target_surface_pose.position[2]
                    - self._sim.config.box_half_extents[2]
                )
                desired_object_pose = Pose(
                    (
                        target_surface_pose.position[0] + self._current_params[0],
                        target_surface_pose.position[1] + self._current_params[1],
                        desired_object_z,
                    ),
                    (0, 0, 0, 1),  # Upright (identity quaternion, xyzw format)
                )
            else:
                raise ValueError("Invalid target object")

            # Compute EE pose from desired object pose using the grasp transform.
            # object_pose = ee_pose * grasped_object_transform
            # => ee_pose = object_pose * grasped_object_transform.invert()
            self._target_place_pose_world = multiply_poses(
                desired_object_pose, grasped_object_transform.invert()
            )

            distance = 0.65
            pre_place_height = 0.03

            self._pre_place_pose_world = Pose(
                (
                    self._target_place_pose_world.position[0],
                    self._target_place_pose_world.position[1],
                    self._target_place_pose_world.position[2] + pre_place_height,
                ),
                self._target_place_pose_world.orientation,
            )

            target_pose_temp_se2 = target_surface_pose.to_se2()
            self._target_place_pose_se2 = SE2Pose(
                target_pose_temp_se2.x + self._current_params[0],
                target_pose_temp_se2.y + self._current_params[1],
                target_pose_temp_se2.rot,
            )
            target_base_pose = get_target_robot_pose_from_parameters(
                self._target_place_pose_se2, distance, 0.0
            )

            # Run base motion planning to the target pose.
            grasped_object_id = (
                self._sim._grasped_object_id  # pylint: disable=protected-access
            )
            all_collision_ids = (
                self._sim._get_collision_object_ids()  # pylint: disable=protected-access
            )
            if "box" in self.objects[1].name:
                collision_bodies = {
                    self._sim.table_id  # type: ignore # pylint: disable=protected-access
                }
            else:
                collision_bodies = all_collision_ids - {grasped_object_id}
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=collision_bodies,
                seed=0,  # for determinism
                held_object=grasped_object_id,
                base_link_to_held_obj=grasped_object_transform,
                hyperparameters=MotionPlanningHyperparameters(
                    birrt_extend_num_interp=self._birrt_extend_num_interp,
                    birrt_smooth_amt=self._base_mp_birrt_smooth_amt,
                ),
            )

            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")

            # Remap the plan to ensure we stay within action limits.
            base_plan = remap_se2_pose_plan_to_constant_distance(
                base_plan,
                max_distance=self._sim.config.max_action_mag,
            )

            # Store the plan (excluding the first state which is the current state).
            self._current_plan = base_plan[1:]

        if not self._navigated:
            return self.navigate()

        if self._navigated and not self._pre_place:
            return self.pre_place()

        if self._pre_place and not self._opened_gripper:
            return self.open_gripper()

        if self._opened_gripper and not self._lifted:
            return self.lift()

        raise ValueError("Invalid state")


def create_lifted_controllers(
    action_space: Kinematic3DRobotActionSpace,
    sim: ObjectCentricTransport3DEnv,
    birrt_extend_num_interp: int = 10,
    smooth_mp_max_time: float = 0.1,
    smooth_mp_max_candidate_plans: int = 1,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for Transport3D.

    Args:
        action_space: The action space for the controllers.
        sim: The simulation environment.
        birrt_extend_num_interp: Number of interpolation steps for BiRRT extension.
            Higher values produce smoother motion but are slower. None uses default.
        smooth_mp_max_time: Maximum time for smooth motion planning.
        smooth_mp_max_candidate_plans: Maximum candidate plans to consider
            for smooth motion planning. Higher values may produce smoother motion.
    """
    del action_space

    # Create partial controller classes that include the sim
    class PickController(GroundPickController):
        """Controller for picking up an object."""

        def __init__(self, objects):
            super().__init__(
                objects,
                sim,
                birrt_extend_num_interp=birrt_extend_num_interp,
                smooth_mp_max_time=smooth_mp_max_time,
                smooth_mp_max_candidate_plans=smooth_mp_max_candidate_plans,
            )

    class PlaceController(GroundPlaceController):
        """Controller for placing an object."""

        def __init__(self, objects):
            super().__init__(
                objects,
                sim,
                birrt_extend_num_interp=birrt_extend_num_interp,
                smooth_mp_max_time=smooth_mp_max_time,
                smooth_mp_max_candidate_plans=smooth_mp_max_candidate_plans,
            )

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
    target_table = Variable("?target_table", Kinematic3DCuboidType)

    # lifted place controller
    place_controller: LiftedParameterizedController = LiftedParameterizedController(
        [robot, target, target_table],
        PlaceController,
        Box(
            low=np.array([-0.15, -0.25]),
            high=np.array([0.15, 0.25]),
        ),
    )

    return {
        "pick": pick_controller,
        "place": place_controller,
    }
