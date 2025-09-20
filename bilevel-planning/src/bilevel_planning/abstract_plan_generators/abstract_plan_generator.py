"""Abstract base class for abstract plan generators."""

import abc
from typing import Callable, Generic, Hashable, Iterable, Iterator, TypeVar

import numpy as np

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import Goal

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class AbstractPlanGenerator(abc.ABC, Generic[_X, _S, _A]):
    """Abstract base class for abstract plan generators."""

    def __init__(
        self,
        abstract_successor_function: Callable[[_S], Iterable[tuple[_A, _S]]],
        seed: int,
    ) -> None:
        self._abstract_successor_function = abstract_successor_function
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def __call__(
        self,
        x0: _X,
        s0: _S,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
    ) -> Iterator[tuple[list[_S], list[_A]]]:
        """Generate abstract plans."""
