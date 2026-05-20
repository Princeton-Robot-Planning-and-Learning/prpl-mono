"""Plan executors track a state-action trajectory in the real environment.

A :class:`PlanExecutor` is stateful: a Runner calls :meth:`set_trajectory` with
a freshly planned trajectory, then drives :meth:`step` once per real-env tick
until :meth:`done` returns ``True``. On each tick the executor returns the next
real-env action to command together with the planned sim action it is currently
tracking toward (so the caller can pass that sim action back to the agent's
update path).

The interface is deliberately minimal: concrete executors decide how to map
the trajectory into commands (one-shot pass-through, settle-then-advance,
pure-pursuit, MPC, …) and when to declare the trajectory complete.
"""

from __future__ import annotations

import abc
from typing import Generic, TypeVar

_SimActType = TypeVar("_SimActType")
_RealActType = TypeVar("_RealActType")
_StateType = TypeVar("_StateType")


class PlanExecutor(Generic[_SimActType, _RealActType, _StateType], abc.ABC):
    """Stateful follower for a planned state-action trajectory."""

    @abc.abstractmethod
    def set_trajectory(self, trajectory: list[tuple[_StateType, _SimActType]]) -> None:
        """Begin tracking ``trajectory`` from scratch."""

    @abc.abstractmethod
    def step(self, sim_state: _StateType) -> tuple[_RealActType, _SimActType]:
        """Return ``(real_action, current_sim_action)`` for this tick.

        ``current_sim_action`` is the planned action the executor is currently
        tracking toward; it may repeat across consecutive ticks (e.g. while
        settling) and need not appear in trajectory order.
        """

    @abc.abstractmethod
    def done(self, sim_state: _StateType) -> bool:
        """Return ``True`` once the trajectory has been tracked to completion."""


__all__ = ["PlanExecutor"]
