"""Utilities."""

from dataclasses import dataclass
from typing import Any, Collection, TypeVar

import gymnasium
import numpy as np
from bilevel_planning.structs import LiftedParameterizedController
from numpy.typing import NDArray
from prbench.core import ObjectCentricPRBenchEnv
from prpl_utils.spaces import FunctionalSpace
from relational_structs import (
    Object,
    ObjectCentricState,
)

# Use object-centric states, as in PRBench.
_ObsType = TypeVar("_ObsType", bound=ObjectCentricState)


@dataclass(frozen=True)
class ParameterizedSkillReference:
    """A symbolic reference to a parameterized skill.

    This does NOT include the policy or termination functions.
    """

    name: str
    objects: list[Object]
    params: dict[str, Any]


# NOTE: actions are symbolic references to parameterized skills.
_ActType = TypeVar("_ActType", bound=ParameterizedSkillReference)


class PRBenchParameterizedSkillEnv(gymnasium.Env[_ObsType, _ActType]):
    """A PRBench env where the actions are references to parameterized skills."""

    def __init__(
        self,
        sim: ObjectCentricPRBenchEnv,
        parameterized_skills: Collection[LiftedParameterizedController],
    ) -> None:
        super().__init__()
        self._sim = sim
        self._name_to_parameterized_skill = {s.name: s for s in parameterized_skills}
        self._current_state: _ObsType | None = None
        self.observation_space = self._sim.observation_space
        self.action_space = FunctionalSpace(
            contains_fn=lambda a: isinstance(a, ParameterizedSkillReference)
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[_ObsType, dict]:
        self._current_state, info = self._sim.reset(seed=seed, options=options)
        assert self._current_state is not None
        return self._current_state.copy(), info

    def step(self, action: _ActType) -> tuple[_ObsType, float, bool, bool, dict]:
        # The action is a parameterized skill reference.
        # We need to look up the parameterized skill and execute it
        # in the simulation.

        # For example, "Pick".
        parameterized_skill = self._name_to_parameterized_skill[action.name]

        # For example, "Pick(block1)".
        skill = parameterized_skill.ground(action.objects)

        # For example, "Pick(block1, continuous_grasp)".
        assert self._current_state is not None
        skill.reset(self._current_state, action.params)

        # Execute the skill in the sim until termination.
        total_rewards = 0.0
        sim_terminated = False
        while not skill.terminated():
            action = skill.step()
            self._current_state, reward, sim_terminated, _, _ = self._sim.step(action)
            skill.observe(self._current_state)
            total_rewards += reward
            if sim_terminated:
                break

        # Return the total rewards and latest object centric state.
        return self._current_state.copy(), total_rewards, sim_terminated, False, {}

    def render(self) -> NDArray[np.uint8]:  # type: ignore
        return self._sim.render()
