"""State abstractions for the TidyBot3D base motion environment."""

from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from prbench.envs.dynamic3d.object_types import (
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from prbench.envs.dynamic3d.tidybot_rewards import BaseMotionRewardCalculator
from relational_structs import (
    GroundAtom,
    ObjectCentricState,
    Predicate,
)

# Predicates.
AtTarget = Predicate("AtTarget", [MujocoTidyBotRobotObjectType, MujocoObjectType])


# State abstractor.
def state_abstractor(state: ObjectCentricState) -> RelationalAbstractState:
    """Get the abstract state for the current state."""
    atoms: set[GroundAtom] = set()

    target = state.get_object_from_name("cube1")
    robot = state.get_object_from_name("robot")
    target_x = state.get(target, "x")
    target_y = state.get(target, "y")
    robot_x = state.get(robot, "pos_base_x")
    robot_y = state.get(robot, "pos_base_y")
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = (dx**2 + dy**2) ** 0.5
    # Divide threshold by 2 to avoid possible numerical issues.
    if distance <= BaseMotionRewardCalculator.dist_thresh / 2:
        atoms.add(GroundAtom(AtTarget, [robot, target]))
    objects = {robot, target}
    return RelationalAbstractState(atoms, objects)


# Goal abstractor.
def goal_deriver(state: ObjectCentricState) -> RelationalAbstractGoal:
    """The goal is to have the robot on the target."""
    target = state.get_object_from_name("cube1")
    robot = state.get_object_from_name("robot")
    atoms = {GroundAtom(AtTarget, [robot, target])}
    return RelationalAbstractGoal(atoms, state_abstractor)
