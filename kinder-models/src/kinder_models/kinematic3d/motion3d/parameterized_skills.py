"""Parameterized skills for the Motion3D environment."""

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
from kinder.envs.kinematic3d.motion3d import (
    Kinematic3DPointType,
    Kinematic3DRobotType,
    Motion3DObjectCentricState,
    ObjectCentricMotion3DEnv,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import (
    InverseKinematicsError,
    inverse_kinematics,
)
from pybullet_helpers.joint import JointPositions, get_jointwise_difference
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_motion_planning,
)
from relational_structs import (
    Object,
    ObjectCentricState,
    Variable,
)


# Controllers.
class GroundMoveToTargetController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for moving the robot arm to the target."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricMotion3DEnv,
    ) -> None:
        super().__init__(objects)
        self._sim = sim
        self._joint_infos = sim.robot.arm.get_arm_joint_infos()[:7]
        self._robot, self._target = objects
        self._current_params: JointPositions | None = None
        self._current_plan: list[JointPositions] | None = None
        self._current_state: ObjectCentricState | None = None

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> JointPositions:
        assert isinstance(x, Motion3DObjectCentricState)
        self._sim.set_state(x)

        # Sample end effector orientation
        u1, u2, u3 = rng.random(size=3)
        quaternion = (
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        )

        # Create target pose from target position and sampled orientation.
        target_pose = Pose(x.target_position, quaternion)

        # Run inverse kinematics to get joint positions.
        try:
            joint_positions = inverse_kinematics(
                self._sim.robot.arm, target_pose, validate=True, set_joints=False
            )
        except InverseKinematicsError as e:
            raise TrajectorySamplingFailure(
                f"IK failed for target pose {target_pose}"
            ) from e

        return joint_positions

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._current_plan is not None and len(self._current_plan) == 0

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Motion3DObjectCentricState)
        self._sim.set_state(self._current_state)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:

            # Run motion planning to the target joint positions.
            joint_plan = run_motion_planning(
                self._sim.robot.arm,
                initial_positions=self._sim.robot.arm.get_joint_positions(),
                target_positions=self._current_params,
                collision_bodies=set(),
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

        # Pop the next target joint positions from the plan.
        assert self._current_plan is not None
        target_joints = self._current_plan.pop(0)

        # Compute delta joint positions.
        delta_lst = get_jointwise_difference(
            self._joint_infos, target_joints[:7], self._current_state.joint_positions
        )

        # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)

        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x


def create_lifted_controllers(
    action_space: Kinematic3DRobotActionSpace,
    sim: ObjectCentricMotion3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for Motion3D."""
    del action_space

    # Create partial controller classes that include the sim
    class MoveToTargetController(GroundMoveToTargetController):
        """Controller for moving the robot to the target."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DPointType)

    # Lifted controllers
    move_to_target_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetController,
            Box(-np.inf, np.inf, (7,)),
        )
    )
    return {
        "move_to_target": move_to_target_controller,
    }
