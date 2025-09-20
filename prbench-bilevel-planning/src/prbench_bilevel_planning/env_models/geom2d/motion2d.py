"""Bilevel planning models for the motion 2D environment."""

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
from prbench.envs.geom2d.motion2d import (
    ObjectCentricMotion2DEnv,
    RectangleType,
    TargetRegionType,
)
from prbench.envs.geom2d.object_types import CRVRobotType
from prbench.envs.geom2d.utils import (
    CRVRobotActionSpace,
    rectangle_object_to_geom,
)
from prbench_models.geom2d.envs.motion2d.parameterized_skills import (
    GroundMoveToPassageController,
    GroundMoveToTgtController,
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
    observation_space: Space, action_space: Space, num_passages: int = 2
) -> SesameModels:
    """Create the env models for motion 2D."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, CRVRobotActionSpace)

    sim = ObjectCentricMotion2DEnv(num_passages=num_passages)

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
    types = {CRVRobotType, TargetRegionType, RectangleType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    AtTgt = Predicate("AtTgt", [CRVRobotType, TargetRegionType])
    NotAtTgt = Predicate("NotAtTgt", [CRVRobotType, TargetRegionType])
    AtPassage = Predicate("AtPassage", [CRVRobotType, RectangleType, RectangleType])
    NotAtPassage = Predicate(
        "NotAtPassage", [CRVRobotType, RectangleType, RectangleType]
    )
    NotAtAnyPassage = Predicate("NotAtAnyPassage", [CRVRobotType])
    predicates = {AtTgt, NotAtTgt, AtPassage, NotAtPassage, NotAtAnyPassage}

    # State abstractor.
    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]
        obstacles = x.get_objects(RectangleType)

        atoms: set[GroundAtom] = set()

        # Check if robot is in the target region
        robot_x = x.get(robot, "x")
        robot_y = x.get(robot, "y")
        target_region_geom = rectangle_object_to_geom(x, target_region, {})

        if target_region_geom.contains_point(robot_x, robot_y):
            atoms.add(GroundAtom(AtTgt, [robot, target_region]))
        else:
            atoms.add(GroundAtom(NotAtTgt, [robot, target_region]))

        robot_x = x.get(robot, "x")
        robot_radius = x.get(robot, "base_radius")
        robot_y = x.get(robot, "y")
        at_any_passage = False
        for obs1 in obstacles:
            for obs2 in obstacles:
                if obs1 != obs2:
                    obs1_x = x.get(obs1, "x")
                    obs2_x = x.get(obs2, "x")
                    if obs1_x == obs2_x:
                        obs_width = x.get(obs1, "width")
                        obs1_y = x.get(obs1, "y")
                        obs2_y = x.get(obs2, "y")
                        obs2_height = x.get(obs2, "height")
                        if obs1_y < obs2_y:
                            # Obstacle1 should be higher than obstacle2
                            continue
                        y_min = obs2_y + obs2_height
                        y_max = obs1_y
                        assert y_min <= y_max, "Obstacles should not overlap vertically"
                        x_close = abs(obs1_x + obs_width / 2 - robot_x) < robot_radius
                        y_close = y_min <= robot_y <= y_max
                        if x_close and y_close:
                            at_any_passage = True
                            atoms.add(GroundAtom(AtPassage, [robot, obs1, obs2]))
                        else:
                            atoms.add(GroundAtom(NotAtPassage, [robot, obs1, obs2]))
        if not at_any_passage:
            atoms.add(GroundAtom(NotAtAnyPassage, [robot]))

        objects = {robot, target_region} | set(obstacles)
        return RelationalAbstractState(atoms, objects)

    # Goal abstractor.
    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have the robot at the target region."""
        robot = x.get_objects(CRVRobotType)[0]
        target_region = x.get_objects(TargetRegionType)[0]
        atoms = {GroundAtom(AtTgt, [robot, target_region])}
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators.
    robot = Variable("?robot", CRVRobotType)
    target = Variable("?target", TargetRegionType)
    obstacle1 = Variable("?obstacle1", RectangleType)
    obstacle2 = Variable("?obstacle2", RectangleType)
    obstacle3 = Variable("?obstacle3", RectangleType)
    obstacle4 = Variable("?obstacle4", RectangleType)

    MoveToTgtFromNoPassageOperator = LiftedOperator(
        "MoveToTgtFromNoPassage",
        [robot, target],
        preconditions={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(NotAtAnyPassage, [robot]),
        },
        add_effects={LiftedAtom(AtTgt, [robot, target])},
        delete_effects={LiftedAtom(NotAtTgt, [robot, target])},
    )

    MoveToTgtFromPassageOperator = LiftedOperator(
        "MoveToTgtFromPassage",
        [robot, target, obstacle1, obstacle2],
        preconditions={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(AtPassage, [robot, obstacle1, obstacle2]),
        },
        add_effects={
            LiftedAtom(AtTgt, [robot, target]),
            LiftedAtom(NotAtAnyPassage, [robot]),
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
        },
        delete_effects={
            LiftedAtom(NotAtTgt, [robot, target]),
            LiftedAtom(AtPassage, [robot, obstacle1, obstacle2]),
        },
    )

    MoveToPassageFromNoPassageOperator = LiftedOperator(
        "MoveToPassageFromNoPassage",
        [robot, obstacle1, obstacle2],
        preconditions={
            LiftedAtom(NotAtAnyPassage, [robot]),
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
        },
        add_effects={LiftedAtom(AtPassage, [robot, obstacle1, obstacle2])},
        delete_effects={
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
            LiftedAtom(NotAtAnyPassage, [robot]),
        },
    )

    MoveToPassageFromPassageOperator = LiftedOperator(
        "MoveToPassageFromPassage",
        [robot, obstacle1, obstacle2, obstacle3, obstacle4],
        preconditions={
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
            LiftedAtom(AtPassage, [robot, obstacle3, obstacle4]),
        },
        add_effects={LiftedAtom(NotAtPassage, [robot, obstacle3, obstacle4])},
        delete_effects={
            LiftedAtom(AtPassage, [robot, obstacle3, obstacle4]),
            LiftedAtom(NotAtPassage, [robot, obstacle1, obstacle2]),
        },
    )

    # Create partial controller classes that include the action_space
    class MoveToTgtController(GroundMoveToTgtController):
        """Controller for moving the robot to the target region."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    class MoveToPassageController(GroundMoveToPassageController):
        """Controller for moving the robot to a passage."""

        def __init__(self, objects):
            super().__init__(objects, action_space, sim.initial_constant_state)

    # Lifted controllers.
    MoveToTgtFromNoPassageController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTgtController,
        )
    )
    MoveToTgtFromPassageController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target, obstacle1, obstacle2],
            MoveToTgtController,
        )
    )
    MoveToPassageFromNoPassageController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstacle1, obstacle2],
            MoveToPassageController,
        )
    )
    MoveToPassageFromPassageController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, obstacle1, obstacle2, obstacle3, obstacle4],
            MoveToPassageController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToTgtFromNoPassageOperator, MoveToTgtFromNoPassageController),
        LiftedSkill(MoveToTgtFromPassageOperator, MoveToTgtFromPassageController),
        LiftedSkill(
            MoveToPassageFromNoPassageOperator, MoveToPassageFromNoPassageController
        ),
        LiftedSkill(
            MoveToPassageFromPassageOperator, MoveToPassageFromPassageController
        ),
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
