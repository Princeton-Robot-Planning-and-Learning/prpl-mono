"""Bilevel planning models for the TidyBot3D cupboard environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.dynamic3d.object_types import (
    MujocoFixtureObjectType,
    MujocoMovableObjectType,
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from prbench.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from prbench_models.dynamic3d.cupboard_real.state_abstractions import (
    AtHome,
    AtPremanipulationTarget,
    CupboardRealStateAbstractor,
    HandEmpty,
    Holding,
    OnFixture,
    OnGround,
)
from prbench_models.dynamic3d.ground.parameterized_skills import (
    create_lifted_controllers,
)
from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    ObjectCentricState,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
    render_images: bool = False,
) -> SesameModels:
    """Create the env models for TidyBot base motion."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    sim = ObjectCentricTidyBot3DEnv(
        scene_type="cupboard_real",
        num_objects=num_objects,
        render_images=render_images,
    )

    # State and goal abstractors.
    abstractor = CupboardRealStateAbstractor(sim)
    state_abstractor = abstractor.state_abstractor
    goal_deriver = abstractor.goal_deriver_place

    # Need to call reset to initialize the qpos, qvel.
    sim.reset()

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
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    # Types.
    types = {
        MujocoTidyBotRobotObjectType,
        MujocoObjectType,
        MujocoFixtureObjectType,
        MujocoMovableObjectType,
    }  # pylint: disable=line-too-long

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    predicates = {
        AtPremanipulationTarget,
        Holding,
        HandEmpty,
        OnGround,
        OnFixture,
        AtHome,
    }

    # Move to target from home operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)

    MoveToTargetOperatorFromHome = LiftedOperator(
        "MoveToTargetFromHome",
        [robot, target],
        preconditions={LiftedAtom(AtHome, [robot])},
        add_effects={LiftedAtom(AtPremanipulationTarget, [robot, target])},
        delete_effects={LiftedAtom(AtHome, [robot])},
    )

    # Move to target from other target operator.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)
    prev_target = Variable("?prev_target", MujocoObjectType)

    MoveToTargetOperatorFromOtherTarget = LiftedOperator(
        "MoveToTargetFromOtherTarget",
        [robot, target, prev_target],
        preconditions={LiftedAtom(AtPremanipulationTarget, [robot, prev_target])},
        add_effects={LiftedAtom(AtPremanipulationTarget, [robot, target])},
        delete_effects={LiftedAtom(AtPremanipulationTarget, [robot, prev_target])},
    )

    # Pick ground controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)

    PickTargetOperator = LiftedOperator(
        "pick_ground",
        [robot, target],
        preconditions={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AtPremanipulationTarget, [robot, target]),
            LiftedAtom(OnGround, [target]),
        },
        add_effects={LiftedAtom(Holding, [robot, target])},
        delete_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnGround, [target]),
        },
    )

    # Place cupboard controller.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoMovableObjectType)
    target_place = Variable("?target_place", MujocoFixtureObjectType)

    PlaceTargetOperator = LiftedOperator(
        "place_target",
        [robot, target, target_place],
        preconditions={
            LiftedAtom(Holding, [robot, target]),
            LiftedAtom(AtPremanipulationTarget, [robot, target_place]),
        },
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnFixture, [target, target_place]),
        },
        delete_effects={
            LiftedAtom(Holding, [robot, target]),
        },
    )

    # Controllers.
    controllers = create_lifted_controllers(action_space, sim.initial_constant_state)
    LiftedMoveToTargetController = controllers["move_to_target"]
    LiftedMoveToTargetFromOtherTargetController = controllers[
        "move_to_target_from_other_target"
    ]
    LiftedPickGroundController = controllers["pick_ground"]
    LiftedPlaceGroundController = controllers["place_ground"]

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToTargetOperatorFromHome, LiftedMoveToTargetController),
        LiftedSkill(
            MoveToTargetOperatorFromOtherTarget,
            LiftedMoveToTargetFromOtherTargetController,
        ),
        LiftedSkill(PickTargetOperator, LiftedPickGroundController),
        LiftedSkill(PlaceTargetOperator, LiftedPlaceGroundController),
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
