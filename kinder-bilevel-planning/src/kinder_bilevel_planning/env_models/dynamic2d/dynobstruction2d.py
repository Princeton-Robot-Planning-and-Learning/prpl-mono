"""Bilevel planning models for the dynamic obstruction 2D environment."""

from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.dynamic2d.dyn_obstruction2d import (
    ObjectCentricDynObstruction2DEnv,
    TargetBlockType,
    TargetSurfaceType,
)
from kinder.envs.dynamic2d.object_types import DynRectangleType, KinRobotType
from kinder.envs.dynamic2d.utils import (
    KinRobotActionSpace,
)
from kinder_models.dynamic2d.dynobstruction2d.parameterized_skills import (
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
    observation_space: Space, action_space: Space, num_obstructions: int
) -> SesameModels:
    """Create the env models for dynamic obstruction 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, KinRobotActionSpace)

    sim = ObjectCentricDynObstruction2DEnv(num_obstructions=num_obstructions)

    # Convert observations into states. The important thing is that states are hashable.
    def observation_to_state(o: NDArray) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    # Create the transition function.
    def transition_fn(
        x: ObjectCentricState,
        u: NDArray,
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        sim.reset(seed=123)
        sim._add_state_to_space(state)  # pylint: disable=protected-access
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {KinRobotType, DynRectangleType, TargetBlockType, TargetSurfaceType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    HoldingTgt = Predicate("HoldingTgt", [KinRobotType, TargetBlockType])
    HoldingObstruction = Predicate(
        "HoldingObstruction", [KinRobotType, DynRectangleType]
    )
    HandEmpty = Predicate("HandEmpty", [KinRobotType])
    OnTgtSurface = Predicate("OnTgt", [TargetBlockType, TargetSurfaceType])
    AboveTgtSurface = Predicate("AboveTgt", [KinRobotType])
    predicates = {
        HoldingTgt,
        HoldingObstruction,
        HandEmpty,
        OnTgtSurface,
        AboveTgtSurface,
    }

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(KinRobotType)[0]
        target_block = x.get_objects(TargetBlockType)[0]
        target_surface = x.get_objects(TargetSurfaceType)[0]
        obstructions = x.get_objects(DynRectangleType)

        atoms: set[GroundAtom] = set()

        # Check what the robot is holding
        if x.get(target_block, "held"):
            atoms.add(GroundAtom(HoldingTgt, [robot, target_block]))
        else:
            # Check if holding any obstruction
            held_obstruction = None
            for obstruction in obstructions:
                if x.get(obstruction, "held"):
                    held_obstruction = obstruction
                    break

            if held_obstruction is not None:
                atoms.add(GroundAtom(HoldingObstruction, [robot, held_obstruction]))
            else:
                atoms.add(GroundAtom(HandEmpty, [robot]))

        # Add on atom
        target_surface_x = x.get(target_surface, "x")
        target_surface_y = x.get(target_surface, "y")
        target_surface_height = x.get(target_surface, "height")
        target_block_x = x.get(target_block, "x")
        target_block_y = x.get(target_block, "y")
        target_block_width = x.get(target_block, "width")
        target_block_height = x.get(target_block, "height")

        if abs(target_block_x - target_surface_x) < target_block_width / 2 + 0.01:
            if (
                abs(
                    (target_block_y - target_block_height / 2)
                    - (target_surface_y + target_surface_height / 2)
                )
                <= 0.01
            ):
                atoms.add(GroundAtom(OnTgtSurface, [target_block, target_surface]))

        # Add above atom
        robot_x = x.get(robot, "x")
        if abs(robot_x - target_surface_x) < target_block_width / 2 + 0.01:
            atoms.add(GroundAtom(AboveTgtSurface, [robot]))

        objects = {robot, target_block, target_surface} | set(obstructions)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to place the target block on the target surface."""
        target_block = x.get_objects(TargetBlockType)[0]
        target_surface = x.get_objects(TargetSurfaceType)[0]
        atoms = {GroundAtom(OnTgtSurface, [target_block, target_surface])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", KinRobotType)
    target_block = Variable("?target_block", TargetBlockType)
    target_surface = Variable("?target_surface", TargetSurfaceType)
    obstruction = Variable("?obstruction", DynRectangleType)

    PickTgtOperator = LiftedOperator(
        "PickTgt",
        [robot, target_block],
        preconditions={LiftedAtom(HandEmpty, [robot])},
        add_effects={LiftedAtom(HoldingTgt, [robot, target_block])},
        delete_effects={LiftedAtom(HandEmpty, [robot])},
    )

    PlaceTgtOperator = LiftedOperator(
        "PlaceTgt",
        [robot, target_block],
        preconditions={LiftedAtom(HoldingTgt, [robot, target_block])},
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
        },
        delete_effects={LiftedAtom(HoldingTgt, [robot, target_block])},
    )

    PlaceTgtSurfaceOperator = LiftedOperator(
        "PlaceTgtSurface",
        [robot, target_block, target_surface],
        preconditions={LiftedAtom(HoldingTgt, [robot, target_block])},
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTgtSurface, [target_block, target_surface]),
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

    MoveOperator = LiftedOperator(
        "Move",
        [robot],
        preconditions=set(),
        add_effects=set(),
        delete_effects=set(),
    )

    # Get lifted controllers from kinder_models
    lifted_controllers = create_lifted_controllers(
        action_space, sim.initial_constant_state
    )
    PickTgtController = lifted_controllers["pick_tgt"]
    PickObstructionController = lifted_controllers["pick_obstruction"]
    PlaceObstructionController = lifted_controllers["place_obstruction"]
    PlaceTgtController = lifted_controllers["place_tgt"]
    PlaceTgtSurfaceController = lifted_controllers["place_tgt_surface"]
    MoveController = lifted_controllers["move"]

    # Finalize the skills.
    skills = {
        LiftedSkill(PickTgtOperator, PickTgtController),
        LiftedSkill(PickObstructionOperator, PickObstructionController),
        LiftedSkill(PlaceObstructionOperator, PlaceObstructionController),
        LiftedSkill(PlaceTgtOperator, PlaceTgtController),
        LiftedSkill(PlaceTgtSurfaceOperator, PlaceTgtSurfaceController),
        LiftedSkill(MoveOperator, MoveController),
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
