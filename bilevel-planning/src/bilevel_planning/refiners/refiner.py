"""Abstract base class for refiners."""

import abc
from typing import Generic, Hashable, TypeVar

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import Plan

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class Refiner(abc.ABC, Generic[_X, _U, _S, _A]):
    """Abstract base class for refiners."""

    @abc.abstractmethod
    def __call__(
        self,
        x0: _X,
        s_plan: list[_S],
        a_plan: list[_A],
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
    ) -> Plan | None:
        """Returns a plan or None if none are found."""
