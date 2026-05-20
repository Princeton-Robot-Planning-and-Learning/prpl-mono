"""Driver that runs a planning agent on a real environment via a sim state space.

The :class:`Runner` ties together four components:

* a real ``gymnasium.Env`` that produces real observations and consumes real
  actions,
* a :class:`prpl_utils.real_sim.perceiver.Perceiver` that converts real
  observations to simulator states,
* a :class:`prpl_utils.planning_agent.PlanningAgent` that consumes simulator
  states and produces simulator state-action trajectories, and
* a :class:`prpl_utils.real_sim.plan_executor.PlanExecutor` that tracks each
  planned trajectory in closed loop with the real environment.

One outer :meth:`step` call corresponds to one planned trajectory: the agent
plans, the executor is handed that trajectory, and the runner ticks the
executor against the real env until the executor reports ``done`` (or the env
terminates or truncates). The agent's per-tick :meth:`Agent.update` still fires
on every real-env tick, with the planned sim action the executor was tracking
at that tick recorded as the agent's last action.

The runner deliberately does not touch any simulator. If the agent uses a
simulator for planning, it owns that simulator. Subclasses can override
:meth:`on_step` to add per-tick visualization or logging.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import gymnasium

from prpl_utils.planning_agent import PlanningAgent
from prpl_utils.real_sim.perceiver import Perceiver
from prpl_utils.real_sim.plan_executor import PlanExecutor

_RealObsType = TypeVar("_RealObsType")
_RealActType = TypeVar("_RealActType")
_StateType = TypeVar("_StateType")
_SimActType = TypeVar("_SimActType")


class Runner(Generic[_RealObsType, _RealActType, _StateType, _SimActType]):
    """Run a planning agent on a real environment via a perceiver and executor."""

    def __init__(
        self,
        real_env: gymnasium.Env[_RealObsType, _RealActType],
        perceiver: Perceiver[_RealObsType, _StateType],
        agent: PlanningAgent[_StateType, _SimActType, _StateType],
        plan_executor: PlanExecutor[_SimActType, _RealActType, _StateType],
    ) -> None:
        self._real_env = real_env
        self._perceiver = perceiver
        self._agent = agent
        self._plan_executor = plan_executor
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
        """Execute one planned trajectory end-to-end.

        Returns the aggregated transition: the final state after the trajectory
        (or after early termination/truncation), the summed reward over all
        ticks, the latest ``terminated`` / ``truncated`` flags, and the latest
        ``info`` dict.
        """
        if self._last_state is None:
            raise RuntimeError("Runner.step() called before Runner.reset()")
        trajectory = self._agent.plan()
        self._plan_executor.set_trajectory(trajectory)
        state = self._last_state
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        while not self._plan_executor.done(state):
            real_action, current_sim_action = self._plan_executor.step(state)
            real_obs, reward, terminated, truncated, info = self._real_env.step(
                real_action
            )
            reward = float(reward)
            total_reward += reward
            state = self._perceiver.step(real_obs, info)
            self._agent.record_trajectory_step(
                current_sim_action, state, reward, terminated or truncated, info
            )
            self.on_step(
                state,
                current_sim_action,
                real_action,
                reward,
                terminated,
                truncated,
                info,
            )
            if terminated or truncated:
                break
        self._last_state = state
        return state, total_reward, terminated, truncated, info

    def run(self, max_steps: int) -> float:
        """Execute up to ``max_steps`` planned trajectories.

        Returns the cumulative reward across all executed trajectories. Stops
        early on termination or truncation. Note that ``max_steps`` counts
        outer trajectories, not inner real-env ticks.
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
        """Hook called after every inner real-env tick for logging, viz, etc."""


__all__ = ["Runner"]
