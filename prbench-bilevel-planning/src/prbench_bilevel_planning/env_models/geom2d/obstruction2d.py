"""Bilevel planning models for the obstruction 2D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.geom2d.object_types import CRVRobotType, RectangleType
from prbench.envs.geom2d.obstruction2d import (
    ObjectCentricObstruction2DEnv,
    TargetBlockType,
    TargetSurfaceType,
)
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    is_on,
)
from prbench_models.geom2d.envs.obstruction2d.parameterized_skills import (
    GroundPickController,
    GroundPlaceOnTableController,
    GroundPlaceOnTargetController,
)
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    Object,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace


def create_bilevel_planning_models(
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricObstruction2DEnv(num_obstructions=num_obstructions)

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
        sim.reset(options={"init_state": state})
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {CRVRobotType, RectangleType, TargetBlockType, TargetSurfaceType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    Holding = Predicate("Holding", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTable = Predicate("OnTable", [RectangleType])
    OnTarget = Predicate("OnTarget", [RectangleType])
    predicates = {Holding, HandEmpty, OnTable, OnTarget}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = Object("robot", CRVRobotType)
        target = Object("target_block", TargetBlockType)
        target_surface = Object("target_surface", TargetSurfaceType)
        obstructions: set[Object] = set()
        for i in range(num_obstructions):
            obstruction = Object(f"obstruction{i}", RectangleType)
            obstructions.add(obstruction)
        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        for obj in suctioned_objs & (obstructions | {target}):
            atoms.add(GroundAtom(Holding, [robot, obj]))
        if not suctioned_objs:
            atoms.add(GroundAtom(HandEmpty, [robot]))
        # Add "on" atoms.
        for block in obstructions | {target}:
            if is_on(x, block, target_surface, {}):
                atoms.add(GroundAtom(OnTarget, [block]))
            elif block not in suctioned_objs:
                atoms.add(GroundAtom(OnTable, [block]))
        objects = {robot, target, target_surface} | obstructions
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is always the same in this environment."""
        del x  # not needed
        target = Object("target_block", TargetBlockType)
        atoms = {GroundAtom(OnTarget, [target])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    block = Variable("?block", RectangleType)
    PickFromTableOperator = LiftedOperator(
        "PickFromTable",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
    )
    PickFromTargetOperator = LiftedOperator(
        "PickFromTarget",
        [robot, block],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
        add_effects={LiftedAtom(Holding, [robot, block])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
    )
    PlaceOnTableOperator = LiftedOperator(
        "PlaceOnTable",
        [robot, block],
        preconditions={LiftedAtom(Holding, [robot, block])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTable, [block])},
        delete_effects={LiftedAtom(Holding, [robot, block])},
    )
    PlaceOnTargetOperator = LiftedOperator(
        "PlaceOnTarget",
        [robot, block],
        preconditions={LiftedAtom(Holding, [robot, block])},
        add_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnTarget, [block])},
        delete_effects={LiftedAtom(Holding, [robot, block])},
    )

    # Create partial controller classes that include the action_space
    class PickController(GroundPickController):
        """Pick controller with pre-configured action space."""

        def __init__(self, objects):
            super().__init__(objects, action_space)

    class PlaceOnTableController(GroundPlaceOnTableController):
        """Place on table controller with pre-configured action space."""

        def __init__(self, objects):
            super().__init__(objects, action_space)

    class PlaceOnTargetController(GroundPlaceOnTargetController):
        """Place on target controller with pre-configured action space."""

        def __init__(self, objects):
            super().__init__(objects, action_space)

    PickControllerLifted: LiftedParameterizedController = LiftedParameterizedController(
        [robot, block],
        PickController,
    )

    PlaceOnTableControllerLifted: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block],
            PlaceOnTableController,
        )
    )

    PlaceOnTargetControllerLifted: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, block],
            PlaceOnTargetController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(PickFromTableOperator, PickControllerLifted),
        LiftedSkill(PickFromTargetOperator, PickControllerLifted),
        LiftedSkill(PlaceOnTableOperator, PlaceOnTableControllerLifted),
        LiftedSkill(PlaceOnTargetOperator, PlaceOnTargetControllerLifted),
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
