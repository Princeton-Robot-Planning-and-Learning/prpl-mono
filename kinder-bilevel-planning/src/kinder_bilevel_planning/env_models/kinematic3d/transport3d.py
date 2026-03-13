"""Bilevel planning models for the transport 3D environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
)
from kinder.envs.kinematic3d.transport3d import (
    Kinematic3DRobotType,
    ObjectCentricTransport3DEnv,
    Transport3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import (
    Kinematic3DRobotActionSpace,
)
from kinder_models.kinematic3d.transport3d.parameterized_skills import (
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

GRIPPER_OPEN_THRESHOLD = 0.01


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
) -> SesameModels:
    """Create the env models for shelf 3D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)

    sim = ObjectCentricTransport3DEnv(num_cubes=num_objects, allow_state_access=True)

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
        assert isinstance(state, Transport3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {Kinematic3DCuboidType, Kinematic3DFixtureType, Kinematic3DRobotType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    OnTable = Predicate("OnTable", [Kinematic3DCuboidType, Kinematic3DCuboidType])
    OnGround = Predicate("OnGround", [Kinematic3DCuboidType])
    Holding = Predicate("Holding", [Kinematic3DRobotType, Kinematic3DCuboidType])
    HandEmpty = Predicate("HandEmpty", [Kinematic3DRobotType])
    predicates = {OnTable, OnGround, Holding, HandEmpty}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_table = x.get_object_from_name("table")

        atoms: set[GroundAtom] = set()

        # Check if robot base is at the target.
        assert isinstance(x, Transport3DObjectCentricState)
        sim.set_state(x)

        # OnGround.
        on_ground_tol = 0.01
        for target in target_objects:
            z = x.get(target, "pose_z")
            bb_z = x.get(target, "half_extent_z")
            floor_z = sim.config.floor_z
            if np.isclose(z, bb_z, atol=floor_z + on_ground_tol):
                atoms.add(GroundAtom(OnGround, [target]))

        # HandEmpty.
        if x.grasped_object is None:
            if x.get(robot, "finger_state") < GRIPPER_OPEN_THRESHOLD:
                atoms.add(GroundAtom(HandEmpty, [robot]))

        # Holding.
        for target in target_objects:
            if (
                x.get(target, "pose_z") > 0.3
                and x.get(robot, "finger_state") > GRIPPER_OPEN_THRESHOLD
            ):
                if target.name == x.grasped_object:
                    atoms.add(GroundAtom(Holding, [robot, target]))

        # OnTable.
        for target in target_objects:
            if target.name != target_table.name:
                if x.get(target, "pose_z") > 0.3:
                    if abs(
                        x.get(target, "pose_x") - x.get(target_table, "pose_x")
                    ) < x.get(target_table, "half_extent_x") and abs(
                        x.get(target, "pose_y") - x.get(target_table, "pose_y")
                    ) < x.get(
                        target_table, "half_extent_y"
                    ):
                        atoms.add(GroundAtom(OnTable, [target, target_table]))

        objects = {robot} | set(target_objects)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the robot at the target pose."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_table = x.get_object_from_name("table")
        atoms: set[GroundAtom] = set()
        atoms.add(GroundAtom(OnGround, [target_table]))
        atoms.add(GroundAtom(HandEmpty, [robot]))
        for target in target_objects:
            if target.name != target_table.name:
                atoms.add(GroundAtom(OnTable, [target, target_table]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)

    PickOperator = LiftedOperator(
        "Pick",
        [robot, target],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnGround, [target])},
        add_effects={LiftedAtom(Holding, [robot, target])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnGround, [target])},
    )

    # Get lifted controllers from kinder_models
    lifted_controllers = create_lifted_controllers(action_space, sim)
    PickController = lifted_controllers["pick"]

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    target_table = Variable("?target_table", Kinematic3DCuboidType)

    PlaceOperator = LiftedOperator(
        "Place",
        [robot, target, target_table],
        preconditions={LiftedAtom(Holding, [robot, target])},
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnTable, [target, target_table]),
        },
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )

    PlaceController = lifted_controllers["place"]

    # Finalize the skills.
    skills = {
        LiftedSkill(PickOperator, PickController),
        LiftedSkill(PlaceOperator, PlaceController),
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
