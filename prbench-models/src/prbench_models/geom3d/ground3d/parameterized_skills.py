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
from prbench.envs.geom3d.object_types import (
    Geom3DCuboidType,
)
from prbench.envs.geom3d.ground3d import (
    Ground3DObjectCentricState,
    Geom3DRobotType,
    ObjectCentricGround3DEnv,
)
from prbench.envs.geom3d.utils import (
    Geom3DRobotActionSpace,
)
from pybullet_helpers.geometry import SE2Pose
from pybullet_helpers.motion_planning import (
    run_single_arm_mobile_base_motion_planning,
)
from relational_structs import (
    Object,
    ObjectCentricState,
    Variable,
)
from spatialmath import SE2

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
        self._robot, self._target = objects
        self._current_params: tuple[()] | None = None
        self._current_plan: list[SE2Pose] | None = None
        self._current_state: ObjectCentricState | None = None

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> tuple[Any, ...]:
        """No parameters needed for base motion - just move to target."""
        assert isinstance(x, Ground3DObjectCentricState)
        # No parameters needed, just return empty tuple
        return tuple()

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._current_params = params
        self._current_plan = None
        self._current_state = x

    def terminated(self) -> bool:
        return self._current_plan is not None and len(self._current_plan) == 0

    def step(self) -> np.ndarray:
        assert self._current_state is not None
        assert self._current_params is not None
        assert isinstance(self._current_state, Ground3DObjectCentricState)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            target_pose = self._current_state.get_object_pose("cube0").to_se2()
            target_base_pose = get_target_robot_pose_from_parameters(
                target_pose, 0.3, 0.0
            )
            # Run base motion planning to the target pose.
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                target_base_pose,
                collision_bodies=set(),
                seed=0,  # for determinism
            )

            if base_plan is None:
                raise TrajectorySamplingFailure("Base motion planning failed")

            # Store the plan (excluding the first state which is the current state).
            self._current_plan = base_plan[1:]

        # Pop the next target base pose from the plan.
        assert self._current_plan is not None
        target_base_pose = self._current_plan.pop(0)

        # Compute delta base pose.
        current_base_pose = self._current_state.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]

        # Create action: [base_x, base_y, base_rot, joint1, ..., joint7, gripper].
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)

        return action

    def observe(self, x: ObjectCentricState) -> None:
        self._current_state = x


def create_lifted_controllers(
    action_space: Geom3DRobotActionSpace,
    sim: ObjectCentricGround3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for Ground3D."""

    # Create partial controller classes that include the sim
    class PickController(GroundPickController):
        """Controller for picking up an object."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Geom3DRobotType)
    target = Variable("?target", Geom3DCuboidType)

    # Lifted controllers
    pick_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            PickController,
            action_space,
        )
    )
    return {
        "pick": pick_controller,
    }
