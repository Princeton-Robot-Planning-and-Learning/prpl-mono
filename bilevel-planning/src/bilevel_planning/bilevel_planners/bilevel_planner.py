"""Abstract base class for a bilevel planner."""

import abc
from typing import Callable, Generic, Hashable, Iterable, TypeVar

import numpy as np

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import Plan, PlanningProblem

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class BilevelPlanner(abc.ABC, Generic[_X, _U, _S, _A]):
    """Abstract base class for a bilevel planner."""

    def __init__(
        self,
        abstract_successor_function: Callable[[_S], Iterable[tuple[_A, _S]]],
        state_abstractor: Callable[[_X], _S],
        seed: int,
    ) -> None:
        self._abstract_successor_function = abstract_successor_function
        self._state_abstractor = state_abstractor
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def run(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> tuple[Plan | None, BilevelPlanningGraph]:
        """Run planning until timeout (sec) and return a Plan or None and the graph."""
