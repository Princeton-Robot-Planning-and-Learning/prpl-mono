"""Parameterized skills for the BaseMotion3D environment."""

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
from kinder.envs.kinematic3d.base_motion3d import (
    BaseMotion3DObjectCentricState,
    Kinematic3DPointType,
    Kinematic3DRobotType,
    ObjectCentricBaseMotion3DEnv,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
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


# Controllers.
class GroundMoveBaseToTargetController(
    GroundParameterizedController[ObjectCentricState, np.ndarray]
):
    """Controller for moving the robot base to the target."""

    def __init__(
        self,
        objects: Sequence[Object],
        sim: ObjectCentricBaseMotion3DEnv,
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
        assert isinstance(x, BaseMotion3DObjectCentricState)
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
        assert isinstance(self._current_state, BaseMotion3DObjectCentricState)

        # Generate the motion plan if it doesn't exist yet.
        if self._current_plan is None:
            self._sim.set_state(self._current_state)

            # Run base motion planning to the target pose.
            base_plan = run_single_arm_mobile_base_motion_planning(
                self._sim.robot,
                self._sim.robot.base.get_pose(),
                self._current_state.target_base_pose,
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
    action_space: Kinematic3DRobotActionSpace,
    sim: ObjectCentricBaseMotion3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for BaseMotion3D."""
    del action_space

    # Create partial controller classes that include the sim
    class MoveBaseToTargetController(GroundMoveBaseToTargetController):
        """Controller for moving the robot base to the target."""

        def __init__(self, objects):
            super().__init__(objects, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DPointType)

    # Lifted controllers
    move_base_to_target_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveBaseToTargetController,
            Box(0.0, 1.0, (0,)),
        )
    )
    return {
        "move_base_to_target": move_base_to_target_controller,
    }
