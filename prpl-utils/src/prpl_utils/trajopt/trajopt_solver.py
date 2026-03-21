"""Base class for trajopt solvers."""

import abc
from typing import Any, TypeAlias

import numpy as np

from prpl_utils.trajopt.trajopt_problem import (
    TrajOptProblem,
    TrajOptState,
    TrajOptTraj,
)

_TrajOptSolution: TypeAlias = Any  # intermediate representation of a solution


class TrajOptSolver(abc.ABC):
    """A solver for a trajectory optimization problem."""

    def __init__(self, seed: int, warm_start: bool = True) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._problem: TrajOptProblem | None = None
        self._warm_start = warm_start
        self._last_solution: _TrajOptSolution | None = None

    def reset(
        self,
        problem: TrajOptProblem,
    ) -> None:
        """Reset the memory."""
        self._problem = problem
        self._last_solution = None

    def solve(
        self,
        initial_state: TrajOptState | None = None,
        horizon: int | None = None,
    ) -> TrajOptTraj:
        """Get the most recent state and return an action."""
        assert self._problem is not None, "Call reset() before solve()"
        if initial_state is None:
            initial_state = self._problem.initial_state
        if horizon is None:
            horizon = self._problem.horizon
        solution = self._solve(initial_state, horizon)
        self._last_solution = solution
        traj = self._solution_to_trajectory(solution, initial_state, horizon)
        return traj

    @abc.abstractmethod
    def _solve(
        self,
        initial_state: TrajOptState,
        horizon: int,
    ) -> _TrajOptSolution:
        """The main logic for the solver."""
        raise NotImplementedError

    @abc.abstractmethod
    def _solution_to_trajectory(
        self,
        solution: _TrajOptSolution,
        initial_state: TrajOptState,
        horizon: int,
    ) -> TrajOptTraj:
        """Different solvers have different intermediate representations."""
        raise NotImplementedError
