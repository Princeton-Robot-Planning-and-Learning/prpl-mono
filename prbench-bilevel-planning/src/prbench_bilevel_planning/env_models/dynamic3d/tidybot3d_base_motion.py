"""Bilevel planning models for the TidyBot3D base motion environment."""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    SesameModels,
)
from gymnasium.spaces import Space
from numpy.typing import NDArray
from prbench.envs.dynamic3d.object_types import (
    MujocoObjectType,
    MujocoTidyBotRobotObjectType,
)
from prbench.envs.dynamic3d.robots.tidybot_robot_env import TidyBot3DRobotActionSpace
from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv
from prbench_models.dynamic3d.base_motion.parameterized_skills import (
    create_lifted_controllers,
)
from prbench_models.dynamic3d.base_motion.state_abstractions import (
    AtTarget,
    goal_deriver,
    state_abstractor,
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
    types = {MujocoTidyBotRobotObjectType, MujocoObjectType}

    # Create the state space.
    state_space = ObjectCentricStateSpace(types)

    # Predicates.
    predicates = {AtTarget}

    # Operators.
    robot = Variable("?robot", MujocoTidyBotRobotObjectType)
    target = Variable("?target", MujocoObjectType)

    MoveToTargetOperator = LiftedOperator(
        "MoveToTarget",
        [robot, target],
        preconditions=set(),
        add_effects={LiftedAtom(AtTarget, [robot, target])},
        delete_effects=set(),
    )

    # Controllers.
    controllers = create_lifted_controllers(action_space, sim.initial_constant_state)
    LiftedMoveToTargetController = controllers["move_to_target"]

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
