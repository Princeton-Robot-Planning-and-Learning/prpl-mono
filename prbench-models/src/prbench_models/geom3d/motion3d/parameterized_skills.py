"""Parameterized skills for the Motion3D environment."""

from typing import Any, Sequence

import numpy as np
from bilevel_planning.structs import GroundParameterizedController, \
    LiftedParameterizedController
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from gymnasium.spaces import Box
from prbench.envs.geom3d.motion3d import (
    Geom3DPointType,
    Geom3DRobotType,
    ObjectCentricMotion3DEnv,
    Motion3DObjectCentricState,
)
from prbench.envs.geom3d.utils import (
    Geom3DRobotActionSpace,
)
from pybullet_helpers.joint import JointPositions
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
        self._robot, self._target = objects

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> JointPositions:
        assert isinstance(x, Motion3DObjectCentricState)
        self._sim.set_state(x)  # in case there is any state dependency in IK
        # Sample joint positions given the end effector target position.
        # In other words, run IK, but also sample orientations.
        # TODO sample quaternion...
        # TODO run IK...
        # TODO raise TrajectorySamplingFailure if IK fails
        import ipdb

        ipdb.set_trace()

    def reset(self, x: ObjectCentricState, params: Any) -> None:
        import ipdb

        ipdb.set_trace()

    def terminated(self) -> bool:
        import ipdb

        ipdb.set_trace()

    def step(self) -> np.ndarray:
        import ipdb

        ipdb.set_trace()

    def observe(self, x: ObjectCentricState) -> None:
        import ipdb

        ipdb.set_trace()


def create_lifted_controllers(
    action_space: Geom3DRobotActionSpace,
    sim: ObjectCentricMotion3DEnv,
) -> dict[str, LiftedParameterizedController]:
    """Create lifted parameterized controllers for Motion3D."""

    # Create partial controller classes that include the action_space
    class MoveToTargetController(GroundMoveToTargetController):
        """Controller for moving the robot to the target."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim)

    # Create variables for lifted controllers
    robot = Variable("?robot", Geom3DRobotType)
    target = Variable("?target", Geom3DPointType)

    # Lifted controllers
    move_to_target_controller: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetController,
            sim.robot.arm.action_space,
        )
    )
    return {
        "move_to_target": move_to_target_controller,
    }
