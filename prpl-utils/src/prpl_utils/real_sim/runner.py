"""Driver that runs an agent on a real environment via a simulator state space.

The :class:`Runner` ties together four components:

* a real ``gymnasium.Env`` that produces real observations and consumes real actions,
* a :class:`prpl_utils.real_sim.perceiver.Perceiver` that converts real observations
  to simulator states,
* a :class:`prpl_utils.gym_agent.Agent` that consumes simulator states and produces
  simulator actions, and
* a :class:`prpl_utils.real_sim.action_grounder.ActionGrounder` that converts
  simulator actions to real actions.

The runner deliberately does not touch any simulator. If the agent uses a simulator
for planning, it owns that simulator and calls ``set_state`` on it as needed.
Subclasses can override :meth:`on_step` to add visualization or logging.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import gymnasium

from prpl_utils.gym_agent import Agent
from prpl_utils.real_sim.action_grounder import ActionGrounder
from prpl_utils.real_sim.perceiver import Perceiver

_RealObsType = TypeVar("_RealObsType")
_RealActType = TypeVar("_RealActType")
_StateType = TypeVar("_StateType")
_SimActType = TypeVar("_SimActType")


class Runner(Generic[_RealObsType, _RealActType, _StateType, _SimActType]):
    """Run an agent on a real environment via a perceiver and action grounder."""

    def __init__(
        self,
        real_env: gymnasium.Env[_RealObsType, _RealActType],
        perceiver: Perceiver[_RealObsType, _StateType],
        agent: Agent[_StateType, _SimActType],
        action_grounder: ActionGrounder[_SimActType, _RealActType, _StateType],
    ) -> None:
        self._real_env = real_env
        self._perceiver = perceiver
        self._agent = agent
        self._action_grounder = action_grounder
        self._last_state: _StateType | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> _StateType:
        """Reset the real environment, perceiver, and agent; return the initial
        state."""
        real_obs, info = self._real_env.reset(seed=seed, options=options)
        state = self._perceiver.reset(real_obs, info)
        self._agent.reset(state, info)
        self._last_state = state
        return state

    def step(self) -> tuple[_StateType, float, bool, bool, dict[str, Any]]:
        """Execute one (agent → ground → env → perceive → update) cycle."""
        if self._last_state is None:
            raise RuntimeError("Runner.step() called before Runner.reset()")
        sim_action = self._agent.step()
        real_action = self._action_grounder(sim_action, self._last_state)
        real_obs, reward, terminated, truncated, info = self._real_env.step(real_action)
        reward = float(reward)
        state = self._perceiver.step(real_obs, info)
        self._agent.update(state, reward, terminated or truncated, info)
        self.on_step(
            state, sim_action, real_action, reward, terminated, truncated, info
        )
        self._last_state = state
        return state, reward, terminated, truncated, info

    def run(self, max_steps: int) -> float:
        """Step until termination, truncation, or ``max_steps`` is reached.

        Returns the cumulative reward. Subclass and override :meth:`on_step` (or wrap
        :meth:`step`) if you need richer trajectory data.
        """
        total_reward = 0.0
        for _ in range(max_steps):
            _, reward, terminated, truncated, _ = self.step()
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward

    def on_step(
        self,
        state: _StateType,
        sim_action: _SimActType,
        real_action: _RealActType,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Hook called after each :meth:`step` for logging, viz, etc."""


__all__ = ["Runner"]
