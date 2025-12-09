"""Bilevel planning models for the dynamic obstruction 2D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.dynamic2d.dyn_obstruction2d import (
    ObjectCentricDynObstruction2DEnv,
    TargetBlockType,
    TargetSurfaceType,
)
from prbench.envs.geom2d.object_types import CRVRobotType, RectangleType
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    get_suctioned_objects,
    is_on,
)
from prbench_models.dynamic2d.dynobstruction2d import (
    create_lifted_controllers,
)
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
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for dynamic obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricDynObstruction2DEnv(num_obstructions=num_obstructions)

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
    HoldingTgt = Predicate("HoldingTgt", [CRVRobotType, TargetBlockType])
    HoldingObstruction = Predicate("HoldingObstruction", [CRVRobotType, RectangleType])
    HandEmpty = Predicate("HandEmpty", [CRVRobotType])
    OnTgtSurface = Predicate("Inside", [TargetBlockType, TargetSurfaceType])
    predicates = {HoldingTgt, HoldingObstruction, HandEmpty, OnTgtSurface}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        target_block = x.get_objects(TargetBlockType)[0]
        target_surface = x.get_objects(TargetSurfaceType)[0]
        obstructions = x.get_objects(RectangleType)
        atoms: set[GroundAtom] = set()
        # Add holding / handempty atoms.
        suctioned_objs = {o for o, _ in get_suctioned_objects(x, robot)}
        # Check what the robot is holding
        if target_block in suctioned_objs:
            atoms.add(GroundAtom(HoldingTgt, [robot, target_block]))
        else:
            # Check if holding any obstruction
            held_obstruction = None
            for obstruction in obstructions:
                if obstruction in suctioned_objs:
                    held_obstruction = obstruction
                    break

            if held_obstruction is not None:
                atoms.add(GroundAtom(HoldingObstruction, [robot, held_obstruction]))
            else:
                atoms.add(GroundAtom(HandEmpty, [robot]))

        # Add inside atom
        if is_on(x, target_block, target_surface, {}):
            atoms.add(GroundAtom(OnTgtSurface, [target_block, target_surface]))

        objects = {robot, target_block} | set(obstructions)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to place the target block on the target surface."""
        target_block = x.get_objects(TargetBlockType)[0]
        target_surface = x.get_objects(TargetSurfaceType)[0]
        atoms = {GroundAtom(OnTgtSurface, [target_block, target_surface])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target_block = Variable("?target_block", TargetBlockType)
    target_surface = Variable("?target_surface", TargetSurfaceType)
    obstruction = Variable("?obstruction", RectangleType)

    PickTgtOperator = LiftedOperator(
        "PickTgt",
        [robot, target_block],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingTgt, [robot, target_block])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PlaceTgtOperator = LiftedOperator(
        "PlaceTgt",
        [robot, target_block, target_surface],
        preconditions={LiftedAtom(HoldingTgt, [robot, target_block])},
        add_effects={
            LiftedAtom(OnTgtSurface, [target_block, target_surface]),
            LiftedAtom(HandEmpty, [robot]),
        },
        delete_effects={LiftedAtom(HoldingTgt, [robot, target_block])},
    )

    PickObstructionOperator = LiftedOperator(
        "PickObstruction",
        [robot, obstruction],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PlaceObstructionOperator = LiftedOperator(
        "PlaceObstruction",
        [robot, obstruction],
        preconditions={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        add_effects={LiftedAtom(HandEmpty, [robot])},
        delete_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
    )

    # TODO
    PushOperator = LiftedOperator(
        "Push",
        [robot, target_block],
        preconditions={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        add_effects={LiftedAtom(HandEmpty, [robot])},
        delete_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
    )

    # TODO
    MoveToOperator = LiftedOperator(
        "MoveTo",
        [robot, target_block],
        preconditions={LiftedAtom(HoldingObstruction, [robot, obstruction])},
        add_effects={LiftedAtom(HandEmpty, [robot])},
        delete_effects={LiftedAtom(HoldingObstruction, [robot, obstruction])},
    )

    # Get lifted controllers from prbench_models
    lifted_controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state
    )
    PickTgtController = lifted_controllers["pick_tgt"]
    PickObstructionController = lifted_controllers["pick_obstruction"]
    PlaceObstructionController = lifted_controllers["place_obstruction"]
    PlaceTgtController = lifted_controllers["place_tgt"]
    MoveToTgtController = lifted_controllers["move_to_tgt"]
    PushTgtController = lifted_controllers["push_tgt"]
    PushObstructionController = lifted_controllers["push_obstruction"]

    # Finalize the skills.
    skills = {
        LiftedSkill(PickTgtOperator, PickTgtController),
        LiftedSkill(PickObstructionOperator, PickObstructionController),
        LiftedSkill(PlaceObstructionOperator, PlaceObstructionController),
        LiftedSkill(PlaceTgtOperator, PlaceTgtController),
        LiftedSkill(MoveToOperator, MoveToTgtController),
        LiftedSkill(PushOperator, PushTgtController),
        LiftedSkill(PushOperator, PushObstructionController)
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
