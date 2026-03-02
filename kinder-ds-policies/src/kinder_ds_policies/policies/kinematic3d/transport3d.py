"""Domain-specific policy for Transport3D environment.

This policy implements a scripted sequence of pick and place operations using the
parameterized skills from kinder-models. The logic mirrors the test in kinder-
models/tests/kinematic3d/transport3d.
"""

from typing import Any

import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySamplingFailure,
)
from kinder.envs.kinematic3d.transport3d import ObjectCentricTransport3DEnv
from kinder.envs.kinematic3d.utils import (
    Kinematic3DObjectCentricState,
    Kinematic3DRobotActionSpace,
)
from kinder_models.kinematic3d.transport3d.parameterized_skills import (
    create_lifted_controllers,
)
from numpy.typing import NDArray
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder_ds_policies.policies.base import PolicyFailure, StatefulPolicy

__all__ = ["create_domain_specific_policy"]


class Transport3DScriptedPolicy(StatefulPolicy):
    """A stateful scripted policy for Transport3D.

    This policy maintains state between calls to track which controller is currently
    executing and the overall progress through the task.
    """

    def __init__(
        self,
        observation_space: ObjectCentricBoxSpace,
        action_space: Kinematic3DRobotActionSpace,
        num_cubes: int,
        seed: int = 123,
        birrt_extend_num_interp: int = 25,
        smooth_mp_max_time: float = 120.0,
        smooth_mp_max_candidate_plans: int = 20,
    ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._num_cubes = num_cubes
        self._rng = np.random.default_rng(seed)

        # Create simulator for controllers.
        self._sim = ObjectCentricTransport3DEnv(
            num_cubes=num_cubes, use_gui=False, allow_state_access=True
        )

        # Create controllers with settings for motion smoothness.
        self._controllers = create_lifted_controllers(
            self._action_space,
            self._sim,
            birrt_extend_num_interp=birrt_extend_num_interp,
            smooth_mp_max_time=smooth_mp_max_time,
            smooth_mp_max_candidate_plans=smooth_mp_max_candidate_plans,
        )

        # State tracking.
        self._current_controller: Any = None
        self._skill_sequence: list[tuple[str, tuple[Object, ...], NDArray]] = []
        self._skill_index = 0
        self._initialized = False

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        self._current_controller = None
        self._skill_sequence = []
        self._skill_index = 0
        self._initialized = False

    def _build_skill_sequence(self, state: ObjectCentricState) -> None:
        """Build the skill sequence based on current state."""
        robot = state.get_object_from_name("robot")
        box0 = state.get_object_from_name("box0")
        table = state.get_object_from_name("table")

        # Build the skill sequence:
        # 1. For each cube: pick and place in box0
        # 2. Pick box0 and place on table
        self._skill_sequence = []

        # Place params for cubes - offset them within the box.
        place_offsets = [
            np.array([0.0, -0.06], dtype=np.float32),
            np.array([0.0, 0.06], dtype=np.float32),
        ]

        for i in range(self._num_cubes):
            cube = state.get_object_from_name(f"cube{i}")

            # Pick the cube.
            pick_params = np.array([0.5, 0.0], dtype=np.float32)
            self._skill_sequence.append(("pick", (robot, cube), pick_params))

            # Place the cube in box0.
            place_params = place_offsets[i % len(place_offsets)]
            self._skill_sequence.append(("place", (robot, cube, box0), place_params))

        # Finally, pick box0 and place on table.
        pick_box_params = np.array([0.5, 0.0], dtype=np.float32)
        self._skill_sequence.append(("pick", (robot, box0), pick_box_params))

        place_box_params = np.array([0.0, 0.0], dtype=np.float32)
        self._skill_sequence.append(("place", (robot, box0, table), place_box_params))

    def _start_next_skill(self, state: ObjectCentricState) -> bool:
        """Start the next skill in the sequence.

        Returns True if a skill was started, False if sequence is complete.
        """
        if self._skill_index >= len(self._skill_sequence):
            return False

        skill_name, objects, params = self._skill_sequence[self._skill_index]
        lifted_controller = self._controllers[skill_name]
        self._current_controller = lifted_controller.ground(objects)
        try:
            self._current_controller.reset(state, params)
        except TrajectorySamplingFailure:
            raise PolicyFailure("Sampling failed in reset().")
        return True

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute action using scripted controller sequence."""
        # Devectorize observation.
        state = self._observation_space.devectorize(observation)

        # Sync sim.
        assert isinstance(state, Kinematic3DObjectCentricState)
        self._sim.set_state(state)

        # Initialize on first call.
        if not self._initialized:
            self._build_skill_sequence(state)
            self._start_next_skill(state)
            self._initialized = True

        # Observe the result of the last action.
        assert self._current_controller is not None
        self._current_controller.observe(state)

        # Check if current controller is done, move to next skill if so.
        while self._current_controller.terminated():
            self._skill_index += 1
            if not self._start_next_skill(state):
                # All skills complete - return zero action.
                shape = self._action_space.shape
                assert shape is not None
                return np.zeros(shape, dtype=np.float32)

        # Get action from current controller.
        assert self._current_controller is not None
        try:
            action = self._current_controller.step()
        except TrajectorySamplingFailure:
            raise PolicyFailure("Sampling failed in step().")
        return action


def create_domain_specific_policy(
    observation_space: ObjectCentricBoxSpace,
    action_space: Kinematic3DRobotActionSpace,
    num_cubes: int = 2,
    seed: int = 123,
    birrt_extend_num_interp: int = 25,
    smooth_mp_max_time: float = 10.0,
    smooth_mp_max_candidate_plans: int = 1,
) -> StatefulPolicy:
    """Create a domain-specific policy for Transport3D.

    Args:
        observation_space: The observation space used to devectorize observations.
        action_space: The action space (required for controller creation).
        num_cubes: Number of cubes in the environment.
        seed: Random seed for controller parameter sampling.
        birrt_extend_num_interp: Number of interpolation steps for BiRRT extend.
            Defaults to 25.
        smooth_mp_max_time: Maximum time for motion planning smoothing.
            Defaults to 120.0.
        smooth_mp_max_candidate_plans: Maximum candidate plans for smoothing.
            Defaults to 20.

    Returns:
        A policy that maps observations to actions.
    """
    policy = Transport3DScriptedPolicy(
        observation_space=observation_space,
        action_space=action_space,
        num_cubes=num_cubes,
        seed=seed,
        birrt_extend_num_interp=birrt_extend_num_interp,
        smooth_mp_max_time=smooth_mp_max_time,
        smooth_mp_max_candidate_plans=smooth_mp_max_candidate_plans,
    )

    return policy
