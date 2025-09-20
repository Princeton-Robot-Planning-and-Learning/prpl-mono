"""Abstract base class for a trajectory sampler."""

import abc
from typing import Generic, Hashable, TypeVar

import numpy as np

from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class TrajectorySamplingFailure(BaseException):
    """Raised when trajectory sampling fails."""


class TrajectorySampler(abc.ABC, Generic[_X, _U, _S, _A]):
    """Abstract base class for a trajectory sampler."""

    @abc.abstractmethod
    def __call__(
        self,
        x: _X,
        s: _S,
        a: _A,
        ns: _S,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
        rng: np.random.Generator,
    ) -> tuple[list[_X], list[_U]]:
        """Samples a trajectory or raises TrajectorySamplingFailure().

        Updates bpg in-place with every transition sampled.
        """
