"""Parameterized skills for the TidyBot3D ground environment."""

from typing import Any

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
)
from prbench.envs.dynamic3d.object_types import MujocoObjectType, MujocoRobotObjectType
from prbench.envs.dynamic3d.tidybot_robot_env import TidyBot3DRobotActionSpace
from prbench_models.dynamic3d.utils import get_overhead_object_se2_pose
from relational_structs import (
    Array,
    ObjectCentricState,
    Variable,
)

# Constants.
MAX_BASE_MOVEMENT_MAGNITUDE = 1e-1
MOVE_TO_TARGET_DISTANCE_BOUNDS = (0.1, 0.3)
MOVE_TO_TARGET_ROT_BOUNDS = (-np.pi, np.pi)


class MoveToTargetGroundController(
    GroundParameterizedController[ObjectCentricState, Array]
):
    """Controller for motion planning to reach a target.
    
    The object parameters are:
        robot: The robot itself.
        object: The target object (cube).

    The continuous parameters are:
        target_distance: float
        target_rot: float (radians)

    The controller uses motion planning to move the robot base to reach the target. The
    target base pose is computed as follows: starting with the target object pose, get
    the target _robot_ pose by applying the target distance and target rot from the
    continuous parameters. Note that the robot will always be facing directly towards
    the target object.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_state: ObjectCentricState | None = None
        self._current_params: np.ndarray | None = None

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> Any:
        distance = rng.uniform(*MOVE_TO_TARGET_DISTANCE_BOUNDS)
        rot = rng.uniform(*MOVE_TO_TARGET_ROT_BOUNDS)
        return np.array([distance, rot])

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        self._last_state = x
        assert isinstance(params, np.ndarray)
        self._current_params = params.copy()
        # Make a motion plan.
        target = x.get_object_from_name("cube1")
        target_se2 = get_overhead_object_se2_pose(x, target)
        import ipdb; ipdb.set_trace()


    def terminated(self) -> bool:
        assert self._last_state is not None
        return False

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


def create_lifted_controllers(
    action_space: TidyBot3DRobotActionSpace,
    init_constant_state: ObjectCentricState | None = None,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for the TidyBot3D ground environment."""

    del action_space, init_constant_state  # not used

    # Controllers.

    robot = Variable("?robot", MujocoRobotObjectType)
    target = Variable("?target", MujocoObjectType)

    LiftedMoveToTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetGroundController,
        )
    )

    return {"move_to_target": LiftedMoveToTargetController}
