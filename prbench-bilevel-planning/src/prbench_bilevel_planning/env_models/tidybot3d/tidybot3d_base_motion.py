"""Bilevel planning models for the TidyBot3D base motion environment."""

from typing import Any, ClassVar

import numpy as np
from bilevel_planning.structs import (
    GroundParameterizedController,
    LiftedParameterizedController,
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.dynamic3d.object_types import MujocoObjectType, MujocoRobotObjectType
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from prbench.envs.dynamic3d.tidybot_rewards import BaseMotionRewardCalculator
from prbench.envs.dynamic3d.tidybot_robot_env import TidyBot3DRobotActionSpace
from relational_structs import (
    Array,
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
    """Create the env models for TidyBot base motion."""
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, TidyBot3DRobotActionSpace)

    sim = ObjectCentricTidyBot3DEnv(
        scene_type="base_motion",
        num_objects=1,
        render_images=False,
    )

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
    types = {MujocoRobotObjectType, MujocoObjectType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    AtTarget = Predicate("AtTarget", [MujocoRobotObjectType, MujocoObjectType])
    predicates = {AtTarget}

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

    # Operators.
    robot = Variable("?robot", MujocoRobotObjectType)
    target = Variable("?target", MujocoObjectType)

    MoveToTargetOperator = LiftedOperator(
        "MoveToTarget",
        [robot, target],
        preconditions=set(),
        add_effects={LiftedAtom(AtTarget, [robot, target])},
        delete_effects=set(),
    )

    # Controllers (may later move into prbench_models).
    class MoveToTargetGroundController(
        GroundParameterizedController[ObjectCentricState, Array]
    ):
        """Controller for moving directly to the target."""

        max_magnitude: ClassVar[float] = 1e-1

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._last_state: ObjectCentricState | None = None

        def sample_parameters(
            self, x: ObjectCentricState, rng: np.random.Generator
        ) -> Any:
            # No parameters for this controller.
            return tuple()

        def reset(self, x: ObjectCentricState, params: Any) -> None:
            self._last_state = x

        def terminated(self) -> bool:
            assert self._last_state is not None
            goal = goal_deriver(self._last_state)
            return goal.check_state(self._last_state)

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

    LiftedMoveToTargetController: LiftedParameterizedController = (
        LiftedParameterizedController(
            [robot, target],
            MoveToTargetGroundController,
        )
    )

    # Finalize the skills.
    skills = {
        LiftedSkill(MoveToTargetOperator, LiftedMoveToTargetController),
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
