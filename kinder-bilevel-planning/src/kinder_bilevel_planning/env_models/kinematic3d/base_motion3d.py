"""Bilevel planning models for the base motion 3D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.base_motion3d import (
    BaseMotion3DObjectCentricState,
    Kinematic3DPointType,
    Kinematic3DRobotType,
    ObjectCentricBaseMotion3DEnv,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from kinder_models.kinematic3d.base_motion3d.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
) -> SesameModels:
    """Create the env models for base motion 3D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)

    sim = ObjectCentricBaseMotion3DEnv(allow_state_access=True)

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        assert isinstance(state, BaseMotion3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {Kinematic3DPointType, Kinematic3DRobotType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    AtTgt = Predicate("AtTgt", [Kinematic3DRobotType, Kinematic3DPointType])
    predicates = {AtTgt}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target = x.get_objects(Kinematic3DPointType)[0]

        atoms: set[GroundAtom] = set()

        # Check if robot base is at the target.
        assert isinstance(x, BaseMotion3DObjectCentricState)
        sim.set_state(x)
        if sim.goal_reached():
            atoms.add(GroundAtom(AtTgt, [robot, target]))

        objects = {robot, target}
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the robot base at the target pose."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target = x.get_objects(Kinematic3DPointType)[0]
        atoms = {GroundAtom(AtTgt, [robot, target])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DPointType)

    MoveBaseToTargetOperator = LiftedOperator(
        "MoveBaseToTarget",
        [robot, target],
        preconditions=set(),
        add_effects={LiftedAtom(AtTgt, [robot, target])},
        delete_effects=set(),
    )

    # Get lifted controllers from kinder_models
    lifted_controllers = create_lifted_controllers(action_space, sim)
    MoveBaseToTargetController = lifted_controllers["move_base_to_target"]

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveBaseToTargetOperator, MoveBaseToTargetController),
    }

    # Finalize the models.
    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
