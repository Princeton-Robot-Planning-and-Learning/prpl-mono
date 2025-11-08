"""Parameterized skills for the TidyBot3D base motion environment."""

from typing import Any, ClassVar

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from prbench.envs.dynamic3d.object_types import MujocoObjectType, MujocoRobotObjectType
from prbench.envs.dynamic3d.tidybot_robot_env import TidyBot3DRobotActionSpace
from relational_structs import (
    Array,
    ObjectCentricState,
    Variable,
)

from prbench_models.dynamic3d.base_motion.state_abstractions import goal_deriver


def create_lifted_controllers(
    action_space: TidyBot3DRobotActionSpace,
    init_constant_state: ObjectCentricState | None = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for TidyBot3D base motion."""

    del action_space, init_constant_state  # not used

    # Controllers.
    class MoveToTargetGroundController(
        GroundParameterizedController[ObjectCentricState, Array]
    ):
        """Controller for moving directly to the target."""

        max_magnitude: ClassVar[float] = 1e-1

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._last_state: ObjectCentricState | None = None

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> Any:
            # No parameters for this controller.
            return tuple()

        def reset(self, x: ObjectCentricState, params: Any) -> None:
            self._last_state = x

        def terminated(self) -> bool:
            assert self._last_state is not None
            goal = goal_deriver(self._last_state)
            return goal.check_state(self._last_state)

        def step(self) -> Array:
            # Take one step towards the target.
            state = self._last_state
            assert state is not None
            target = state.get_object_from_name("cube1")
            robot = state.get_object_from_name("robot")
            target_x = state.get(target, "x")
            target_y = state.get(target, "y")
            robot_x = state.get(robot, "pos_base_x")
            robot_y = state.get(robot, "pos_base_y")
            total_dx = target_x - robot_x
            total_dy = target_y - robot_y
            total_distance = (total_dx**2 + total_dy**2) ** 0.5
            if total_distance <= self.max_magnitude:
                distance_to_move = total_distance
            else:
                distance_to_move = self.max_magnitude
            dx = distance_to_move * total_dx / total_distance
            dy = distance_to_move * total_dy / total_distance
            act = np.array([dx, dy, 0] + [0.0] * 8)
            return act

        def observe(self, x: ObjectCentricState) -> None:
            self._last_state = x

    robot = Variable("?robot", MujocoRobotObjectType)
    target = Variable("?target", MujocoObjectType)

    LiftedMoveToTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetGroundController,
        )
    )

    return {"move_to_target": LiftedMoveToTargetController}
