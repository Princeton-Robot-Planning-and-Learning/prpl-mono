"""An Agent subclass for agents that plan state-action trajectories.

A :class:`PlanningAgent` adds a :meth:`plan` method that returns a full state-
action trajectory, separate from the per-step :meth:`Agent.step` contract on
the base class. Callers that drive the agent through trajectories (e.g.
:class:`prpl_utils.real_sim.Runner`) call :meth:`plan`; callers that drive it
one action at a time keep using :meth:`Agent.step`.

The two entry points are independent contracts: a planning agent may implement
both, but mixing them in the same rollout is unsupported. Concrete subclasses
choose what state representation pairs with their actions.
"""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

from prpl_utils.gym_agent import Agent

_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")
_StateType = TypeVar("_StateType")


class PlanningAgent(Agent[_ObsType, _ActType], Generic[_ObsType, _ActType, _StateType]):
    """An Agent that can produce a state-action trajectory in one shot.

    The planning trajectory starts from the agent's current state estimate. A
    Runner that drives the agent via :meth:`plan` then feeds each tick's
    observation back through :meth:`Agent.update`, so updates remain per-tick
    even though planning is per-trajectory.
    """

    @abc.abstractmethod
    def plan(self) -> list[tuple[_StateType, _ActType]]:
        """Return a state-action trajectory from the current state estimate.

        Each pair ``(s, a)`` is interpreted as "from state ``s``, take action
        ``a``". The trajectory length is up to the agent.
        """

    def record_trajectory_step(
        self,
        sim_action: _ActType,
        obs: _ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Tell the agent it just took ``sim_action`` and observed the result.

        Wires up the per-tick (state, action, next_state, reward) signal that
        :meth:`Agent.update` consumes from inside trajectory execution. A
        Runner driving the agent via :meth:`plan` does not call
        :meth:`Agent.step`, so ``_last_action`` is set here before
        :meth:`Agent.update` reads it.
        """
        self._last_action = sim_action
        self.update(obs, reward, done, info)


__all__ = ["PlanningAgent"]
